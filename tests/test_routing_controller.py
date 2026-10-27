"""Q02 queue dependency, retry, test-barrier and inherited-lock integration tests."""
import io
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.run_routing_optimization import (
    Controller, MODELS, ARMS, SEEDS, REFERENCE_MODELS, REFERENCE_ARMS, PRECEDING_METHODS,
)
from scripts.run_organ_weighted import exclusive_lock, write_json


@pytest.fixture
def controller(tmp_path):
    root, reference, previous = (tmp_path / name for name in ("q02", "q01", "q04"))
    for path in (root, reference, previous):
        path.mkdir()
    write_json(root / "protocol.json", dict(protocol_sha256="a" * 64,
        reference_protocol_sha256="b" * 64, after_protocol_sha256="c" * 64,
        config=dict(models=list(MODELS), arms=list(ARMS), seeds=list(SEEDS), epochs=80,
                    reference_root=str(reference), after_root=str(previous))))
    (root / "source_snapshot/unet_moe").mkdir(parents=True)
    (root / "source_snapshot/unet_moe/routing_suite.py").write_text("")
    result = Controller(root, reference, previous, poll_seconds=0.01)
    result.gpu_lock_path = lambda gpu: tmp_path / f"gpu{gpu}.lock"
    result.gpu_memory = lambda gpu: 48000
    return result


def select_main_loss(c):
    write_json(c.root / "main_loss_selection.json", dict(protocol_sha256="a" * 64,
        selection_sha256="d" * 64, main_loss_arm="organ_exposure"))
    c.read_selection()


def complete_reference(c):
    write_json(c.reference_root / "protocol.json", dict(protocol_sha256="b" * 64))
    for kind in REFERENCE_MODELS:
        for arm in REFERENCE_ARMS:
            for seed in SEEDS:
                folder = c.reference_root / arm / kind / f"seed_{seed}"
                write_json(folder / "run_summary.json", dict(kind=kind, arm=arm, seed=seed,
                    protocol_sha256="b" * 64, epochs_completed=80, test_evaluated=False))
                (folder / "metrics.jsonl").write_text("validation only fixture\n")


def complete_predecessor(c):
    root, sha = c.after_root, "c" * 64
    write_json(root / "protocol.json", dict(protocol_sha256=sha))
    write_json(root / "controller_status.json", dict(protocol_sha256=sha, state="complete"))
    results = []
    for method in PRECEDING_METHODS:
        for seed in SEEDS:
            folder = root / method / f"seed_{seed}"
            write_json(folder / "run_summary.json", dict(method=method, seed=seed,
                protocol_sha256=sha, epochs_completed=80, test_evaluated=True))
            for name in ("best.pt", "last.pt", "metrics.jsonl", "best_test_metrics.json", "last_test_metrics.json"):
                (folder / name).write_text("fixture artifact")
            for selection in ("best", "last"):
                results.append(dict(method=method, seed=seed, selection=selection,
                                    protocol_sha256=sha, complete=True, split="test"))
    write_json(root / "summary.json", dict(protocol_sha256=sha, complete=True, results=results))


def preflight_report(c, kind="region", arm="control"):
    return dict(protocol_sha256="a" * 64, main_selection_sha256="d" * 64,
        kind=kind, arm=arm, passed=True, required_free_mib=40000,
        real_optimizer_updates=2, effective_batch=16, micro_batch=4,
        size=512, real_query_images=32, main_loss_arm="organ_exposure", gpu_training_admissible=True,
        strict_model_optimizer_rng_reload=True, identical_reload_prediction=True)


def test_main_loss_waits_for_complete_validation_but_not_test(controller):
    complete_reference(controller)
    ready, _, _ = controller.selection_readiness()
    assert ready  # Every run explicitly test_evaluated=False.
    path = controller.reference_root / "organ_exposure_fg_bg/task_id/seed_44/run_summary.json"
    report = json.loads(path.read_text())
    report["epochs_completed"] = 79
    write_json(path, report)
    ready, state, _ = controller.selection_readiness()
    assert not ready and state == "waiting_q01_validation"


def test_main_loss_cannot_use_changed_reference_protocol(controller):
    complete_reference(controller)
    write_json(controller.reference_root / "protocol.json", dict(protocol_sha256="e" * 64))
    assert controller.selection_readiness()[1] == "blocked_reference"


def test_q04_summary_must_have_every_seed_and_both_checkpoints(controller):
    complete_predecessor(controller)
    assert controller.prerequisites()[0]
    path = controller.after_root / "summary.json"
    report = json.loads(path.read_text())
    report["results"].pop()
    write_json(path, report)
    ready, state, reason = controller.prerequisites()
    assert not ready and state == "blocked_predecessor" and "matrix" in reason


def test_q04_complete_flag_alone_is_insufficient(controller):
    complete_predecessor(controller)
    (controller.after_root / "smart/seed_43/last.pt").unlink()
    assert controller.prerequisites()[1] == "blocked_predecessor"


def test_complete_q04_summary_cannot_hide_failed_worker(controller):
    complete_predecessor(controller)
    write_json(controller.after_root / "controller_status.json", dict(
        protocol_sha256="c" * 64, state="complete", jobs=[dict(train_state="failed")]))
    assert controller.prerequisites()[1] == "blocked_predecessor"


def test_waits_for_q04_without_touching_gpu(controller):
    write_json(controller.after_root / "protocol.json", dict(protocol_sha256="c" * 64))
    write_json(controller.after_root / "controller_status.json", dict(protocol_sha256="c" * 64, state="training"))
    controller.gpu_memory = lambda _: pytest.fail("Dependency probe touched GPU")
    assert controller.prerequisites()[1] == "waiting_q04"


def test_once_only_finalizes_selection_and_never_launches_gpu(controller):
    complete_reference(controller)
    complete_predecessor(controller)
    calls = []
    def cpu(command, label):
        calls.append(command)
        select_main_loss(controller)
    controller.run_cpu = cpu
    controller.admit = lambda *args: pytest.fail("--once admitted GPU")
    assert controller.run(once=True) == 0
    assert len(calls) == 1 and "finalize" in calls[0]
    assert all(j["train_state"] == "pending" for j in controller.jobs)


def test_dry_run_has_no_status_or_selection_mutation(controller, capsys):
    assert controller.run(dry_run=True) == 0
    assert not (controller.root / "controller_status.json").exists()
    assert not (controller.root / "main_loss_selection.json").exists()
    assert len(json.loads(capsys.readouterr().out)["jobs"]) == 12


def test_dispatched_training_command_authorizes_suite_checkpoint_resume(controller):
    from unet_moe.routing_suite import parse_args
    command = controller.command(("region", 42, "control", "train"))
    args = parse_args(command[len(controller.suite_prefix):])
    assert args.resume and args.action == "train"
    verify = controller.command(("region", 42, "control", "train"), action="verify", device="cpu")
    assert "--resume" not in verify


def test_preflight_requires_correct_main_loss_and_real_updates(controller):
    select_main_loss(controller)
    report = preflight_report(controller)
    report["real_optimizer_updates"] = 0
    path = controller.root / "preflight/region_control.json"
    write_json(path, report)
    with pytest.raises(ValueError, match="real 512"):
        controller.read_preflight("region", "control")
    report["real_optimizer_updates"] = 2
    report["main_selection_sha256"] = "f" * 64
    write_json(path, report)
    with pytest.raises(ValueError, match="mismatched"):
        controller.read_preflight("region", "control")
    report["main_selection_sha256"] = "d" * 64
    write_json(path, report)
    controller.read_preflight("region", "control")
    assert controller.required_mib["region_control"] == 40000


def test_all_four_preflights_precede_paired_training_and_all_train_precede_test(controller):
    assert len(controller.pending_tasks()) == 4
    controller.preflights["region_control"]["state"] = "complete"
    assert all(t[3] == "preflight" for t in controller.pending_tasks())
    for p in controller.preflights.values():
        p["state"] = "complete"
    assert len(controller.pending_tasks()) == 12
    assert all(t[3] == "train" for t in controller.pending_tasks())
    for job in controller.jobs:
        job["train_state"] = "complete"
    controller.jobs[-1]["train_state"] = "running"
    assert controller.pending_tasks() == []
    controller.jobs[-1]["train_state"] = "complete"
    assert len(controller.pending_tasks()) == 12
    assert all(t[3] == "evaluate" for t in controller.pending_tasks())


def test_failed_stage_persists_and_stops_new_dispatch_until_explicit_retry(controller):
    controller.jobs[0]["train_state"] = "failed"
    controller.update("blocked_failed_tasks")
    assert controller.pending_tasks() == []
    restored = Controller(controller.root, controller.reference_root, controller.after_root)
    restored.recover()
    assert restored.failed() and restored.pending_tasks() == []
    retried = Controller(controller.root, controller.reference_root, controller.after_root, retry_failed=True)
    retried.recover()
    assert not retried.failed()


def test_nonzero_exit_records_durable_failure(controller):
    task = ("region", 42, "control", "train")
    job = controller._job(*task[:3])
    job.update(train_state="running", log="fixture.log", pid=99)
    controller.active[0] = dict(process=SimpleNamespace(poll=lambda: 17), task=task,
        lock=(controller.root / "fake.lock").open("a+"), log=io.StringIO())
    controller.reap()
    assert job["train_state"] == "failed" and job["pid"] is None
    failure = json.loads((controller.root / "failures/region_42_control_train.json").read_text())
    assert failure["returncode"] == 17 and not controller.pending_tasks()


def test_gpu_memory_and_shared_lock_both_required(controller):
    task = ("task_id", 42, "auxiliary", "preflight")
    with exclusive_lock(controller.gpu_lock_path(0)):
        assert controller.admit(0, [task]) is None
    controller.gpu_memory = lambda _: 40000
    assert controller.admit(0, [task]) is None
    controller.gpu_memory = lambda _: 48000
    actual, lock, free = controller.admit(0, [task])
    assert actual == task and free == 48000
    lock.close()


def test_cannot_launch_evaluation_before_training_barrier(controller):
    with pytest.raises(ValueError, match="all 12"):
        controller.launch(0, ("region", 42, "control", "evaluate"), None, 48000)


def test_orphan_worker_keeps_both_locks(controller):
    select_main_loss(controller)
    task = ("region", 42, "control", "preflight")
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
        assert controller.active[0]["process"].poll() is None
    finally:
        controller.stop()
    for path in (controller.root / ".controller.lock", controller.gpu_lock_path(0)):
        with exclusive_lock(path):
            pass


def test_evaluation_verify_requests_test_checks(controller):
    select_main_loss(controller)
    calls = []
    controller.run_cpu = lambda command, label: calls.append(command)
    job = controller._job("region", 42, "control")
    write_json(controller.folder(job) / "run_summary.json", dict(kind="region", seed=42,
        arm="control", protocol_sha256="a" * 64, main_selection_sha256="d" * 64,
        epochs_completed=80, test_evaluated=True))
    controller.validate_task(("region", 42, "control", "evaluate"))
    assert "--require-test" in calls[0] and "cpu" in calls[0]


def test_recovery_checks_all_training_before_any_test_content(controller):
    select_main_loss(controller)
    job = controller.jobs[0]
    write_json(controller.folder(job) / "run_summary.json", dict(epochs_completed=80, test_evaluated=True))
    checked = []
    controller.validate_task = lambda task: checked.append(task)
    with pytest.raises(ValueError, match="all-12-training barrier"):
        controller.recover()
    assert checked and all(task[3] == "train" for task in checked)
