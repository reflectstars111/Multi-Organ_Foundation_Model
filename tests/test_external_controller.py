"""Q04 dependency barriers, durable locks, retry policy and scheduling."""
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts.run_external_baselines import Controller, METHODS, SEEDS
from scripts.run_organ_weighted import exclusive_lock, write_json


@pytest.fixture
def controller(tmp_path):
    root = tmp_path / "q04"
    reference = tmp_path / "q01"
    evaluation = tmp_path / "metrics"
    for path in (root, reference, evaluation):
        path.mkdir()
    write_json(root / "protocol.json", dict(protocol_sha256="a" * 64,
        reference_protocol_sha256="b" * 64,
        config=dict(methods=list(METHODS), seeds=list(SEEDS), epochs=80,
                    reference_root=str(reference), evaluation_reference_root=str(evaluation),
                    mofo_prior=dict(epochs=80))))
    (root / "source_snapshot" / "unet_moe").mkdir(parents=True)
    (root / "source_snapshot" / "unet_moe" / "external_baseline_suite.py").write_text("")
    result = Controller(root, reference, evaluation, poll_seconds=0.01)
    result.gpu_lock_path = lambda gpu: tmp_path / f"gpu{gpu}.lock"
    result.gpu_memory = lambda gpu: 48000
    return result


def make_reference_complete(controller):
    root = controller.reference_root
    write_json(root / "protocol.json", dict(protocol_sha256="b" * 64))
    runs = []
    for kind in ("shared", "region", "task_id"):
        for arm in ("image_mean", "organ_exposure", "organ_exposure_fg_bg"):
            for seed in SEEDS:
                row = dict(kind=kind, arm=arm, seed=seed, epochs_completed=80,
                           test_evaluated=True, protocol_sha256="b" * 64)
                folder = root / arm / kind / f"seed_{seed}"
                write_json(folder / "run_summary.json", row)
                for name in ("last.pt", "best.pt", "metrics.jsonl"):
                    (folder / name).write_text("fixture artifact")
                runs.append(row)
    write_json(root / "summary.json", dict(complete=True, protocol_sha256="b" * 64, runs=runs))


def valid_preflight(controller, method):
    return dict(protocol_sha256=controller.protocol_sha256, method=method, passed=True,
                required_free_mib=20000, real_optimizer_updates=2, effective_batch=16,
                size=512, real_query_images=32, strict_model_optimizer_rng_reload=True,
                identical_reload_prediction=True)


def test_missing_dependencies_block_without_gpu_admission(controller):
    controller.gpu_memory = lambda _: pytest.fail("Dependency check must not touch a GPU")
    ready, state, reason = controller.prerequisites()
    assert not ready and state == "blocked_reference"
    assert "missing" in reason


def test_q01_summary_without_per_run_evidence_is_not_completion(controller):
    make_reference_complete(controller)
    (controller.reference_root / "image_mean/shared/seed_42/last.pt").unlink()
    ready, state, reason = controller.prerequisites()
    assert not ready and state == "blocked_reference"
    assert "checkpoint" in reason


def test_incomplete_q01_test_matrix_keeps_barrier_closed(controller):
    make_reference_complete(controller)
    path = controller.reference_root / "summary.json"
    summary = json.loads(path.read_text())
    summary["runs"][0]["test_evaluated"] = False
    write_json(path, summary)
    ready, state, _ = controller.prerequisites()
    assert not ready and state == "waiting_q01"


@pytest.mark.parametrize("evaluation,expected", [
    ({"state": "running", "failed_jobs": 0}, "waiting_metrics"),
    ({"state": "complete_with_failures", "failed_jobs": 2}, "blocked_reference"),
    ({"state": "failed", "failed_jobs": 0}, "blocked_reference"),
    ({"state": "complete", "failed_jobs": 0}, "dependencies_complete"),
])
def test_metric_backfill_barrier_requires_success(controller, evaluation, expected):
    make_reference_complete(controller)
    write_json(controller.evaluation_root / "status.json", evaluation)
    ready, state, _ = controller.prerequisites()
    assert state == expected
    assert ready is (expected == "dependencies_complete")


def test_preflight_requires_real_full_resolution_optimizer_and_reload(controller):
    report = valid_preflight(controller, "lanzhou")
    report["real_optimizer_updates"] = 0
    write_json(controller.root / "preflight/lanzhou.json", report)
    with pytest.raises(ValueError, match="real 512"):
        controller.read_preflight("lanzhou")
    report["real_optimizer_updates"] = 2
    write_json(controller.root / "preflight/lanzhou.json", report)
    controller.read_preflight("lanzhou")
    assert controller.required_mib["lanzhou"] == 20000


def test_mofo_prior_precedes_its_preflight_and_each_seed_training(controller):
    tasks = controller.pending_tasks()
    assert ("mofo", 42, "mofo-prior") in tasks
    assert ("mofo", 42, "preflight") not in tasks
    controller._job("mofo", 42)["prior_state"] = "complete"
    assert ("mofo", 42, "preflight") in controller.pending_tasks()
    controller.preflights["mofo"]["state"] = "complete"
    tasks = controller.pending_tasks()
    assert ("mofo", 42, "train") in tasks
    assert ("mofo", 43, "mofo-prior") in tasks
    assert ("mofo", 43, "train") not in tasks


def test_failed_method_does_not_block_independent_training_or_open_test(controller):
    for method in METHODS:
        controller.preflights[method]["state"] = "complete"
    controller._job("lanzhou", 42)["train_state"] = "failed"
    tasks = controller.pending_tasks()
    assert ("lanzhou", 42, "train") not in tasks
    assert ("aimagelab", 42, "train") in tasks
    assert not any(task[2] == "evaluate" for task in tasks)
    for job in controller.jobs:
        job["train_state"] = "complete"
        if job["method"] == "mofo":
            job["prior_state"] = "complete"
    tasks = controller.pending_tasks()
    assert len(tasks) == 15 and all(task[2] == "evaluate" for task in tasks)


def test_retry_failed_is_explicit_across_restart(controller):
    controller._job("lanzhou", 42)["train_state"] = "failed"
    controller.preflights["smart"]["state"] = "failed"
    controller.update("blocked_failed_tasks")
    restored = Controller(controller.root, controller.reference_root, controller.evaluation_root)
    restored.recover()
    assert restored._job("lanzhou", 42)["train_state"] == "failed"
    assert restored.preflights["smart"]["state"] == "failed"
    retried = Controller(controller.root, controller.reference_root, controller.evaluation_root, retry_failed=True)
    retried.recover()
    assert retried._job("lanzhou", 42)["train_state"] == "pending"
    assert retried.preflights["smart"]["state"] == "pending"


def test_gpu_lock_and_memory_both_required(controller):
    task = ("lanzhou", 42, "preflight")
    with exclusive_lock(controller.gpu_lock_path(0)):
        assert controller.admit(0, [task]) is None
    controller.gpu_memory = lambda _: 30000
    assert controller.admit(0, [task]) is None
    controller.gpu_memory = lambda _: 48000
    actual, lock, free = controller.admit(0, [task])
    assert actual == task and free == 48000
    lock.close()


def test_evaluation_launch_cannot_bypass_training_barrier(controller):
    task = ("lanzhou", 42, "evaluate")
    with pytest.raises(ValueError, match="all 15"):
        controller.launch(0, task, None, 48000)


def test_worker_nonzero_exit_records_failure_and_keeps_other_jobs(controller):
    task = ("lanzhou", 42, "train")
    job = controller._job("lanzhou", 42)
    job.update(train_state="running", log="fixture.log", pid=99)
    lock = (controller.root / "fake.lock").open("a+")
    controller.active[0] = dict(process=SimpleNamespace(poll=lambda: 17),
                                task=task, lock=lock, log=io.StringIO())
    controller.reap()
    assert job["train_state"] == "failed" and job["pid"] is None
    assert controller._job("aimagelab", 42)["train_state"] == "pending"
    failure = json.loads((controller.root / "failures/lanzhou_42_train.json").read_text())
    assert failure["returncode"] == 17


def test_worker_inherits_both_locks_even_when_controller_descriptors_close(controller):
    task = ("lanzhou", 42, "preflight")
    # The real subprocess is a stand-in, not a model or a GPU consumer.
    controller.suite_prefix = [sys.executable, "-c", "import time; time.sleep(20)"]
    try:
        with exclusive_lock(controller.root / ".controller.lock") as root_lock:
            controller.root_lock = root_lock
            actual, lock, free = controller.admit(0, [task])
            controller.launch(0, actual, lock, free)
            lock.close()
        for path in (controller.root / ".controller.lock", controller.gpu_lock_path(0)):
            with pytest.raises(RuntimeError, match="already owns"):
                with exclusive_lock(path):
                    pass
        process = controller.active[0]["process"]
        assert process.poll() is None
    finally:
        controller.stop()
    for path in (controller.root / ".controller.lock", controller.gpu_lock_path(0)):
        with exclusive_lock(path):
            pass


def test_evaluation_verifier_requests_full_test_hash_checks(controller):
    commands = []
    controller.run_cpu = lambda command, label: commands.append(command)
    folder = controller.folder(controller._job("lanzhou", 42))
    write_json(folder / "run_summary.json", dict(method="lanzhou", seed=42,
        protocol_sha256=controller.protocol_sha256, epochs_completed=80, test_evaluated=True))
    controller.validate_task(("lanzhou", 42, "evaluate"))
    assert "--require-test" in commands[0]
    assert "--device" in commands[0] and "cpu" in commands[0]
