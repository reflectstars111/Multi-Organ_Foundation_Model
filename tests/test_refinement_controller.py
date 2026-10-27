"""Q05 preserves GPU locks and all-training barriers without reselecting Q02 loss."""
import io
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.run_moe_refinement import Controller, MODELS, ARMS, SEEDS, REFERENCE_MODELS, REFERENCE_ARMS
from scripts.run_organ_weighted import exclusive_lock, write_json
from unet_moe import refinement_suite as suite
from test_refinement_suite import provenance


@pytest.fixture
def controller(tmp_path):
    root, previous = tmp_path / 'q05', tmp_path / 'q02'
    provenance(root, previous)
    (root / 'source_snapshot/unet_moe').mkdir(parents=True)
    (root / 'source_snapshot/unet_moe/refinement_suite.py').write_text('')
    result = Controller(root, previous, previous, poll_seconds=.01)
    result.gpu_lock_path = lambda gpu: tmp_path / f'gpu{gpu}.lock'
    result.gpu_memory = lambda gpu: 48000
    return result


def select_main_loss(c):
    c.read_selection()


def complete_predecessor(c):
    sha = c.protocol['after_protocol_sha256']
    write_json(c.after_root / 'controller_status.json', dict(protocol_sha256=sha, state='complete'))
    runs = []
    for kind in REFERENCE_MODELS:
        for seed in SEEDS:
            for arm in REFERENCE_ARMS:
                folder = c.after_root / arm / kind / f'seed_{seed}'
                run = dict(kind=kind, seed=seed, arm=arm, protocol_sha256=sha,
                           main_selection_sha256=c.protocol['reference_selection_sha256'],
                           epochs_completed=80, test_evaluated=True)
                write_json(folder / 'run_summary.json', run)
                for name in ('best.pt', 'last.pt', 'metrics.jsonl', 'test_best.json', 'test_last.json'):
                    (folder / name).write_text('fixture artifact')
                runs.append(run)
    write_json(c.after_root / 'summary.json', dict(protocol_sha256=sha, complete=True, runs=runs,
               main_selection_sha256=c.protocol['reference_selection_sha256']))


def preflight_report(c, kind='region', arm='control'):
    return dict(protocol_sha256=c.protocol_sha256, main_selection_sha256=c.selection['selection_sha256'],
        kind=kind, arm=arm, passed=True, required_free_mib=40000,
        real_optimizer_updates=2, effective_batch=16, micro_batch=4,
        size=512, real_query_images=32, main_loss_arm='organ_exposure_fg_bg', gpu_training_admissible=True,
        strict_model_optimizer_rng_reload=True, identical_reload_prediction=True,
        identical_resume_update=True, continuation_verification_updates=2)


def test_dispatched_training_command_authorizes_suite_checkpoint_resume(controller):
    from unet_moe.refinement_suite import parse_args
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
    report["main_selection_sha256"] = controller.selection["selection_sha256"]
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
    task = ("task_id", 42, "refined", "preflight")
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
        arm="control", protocol_sha256=controller.protocol_sha256, main_selection_sha256=controller.selection["selection_sha256"],
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


def test_frozen_selection_does_not_need_q02_training_or_test_results(controller):
    assert controller.selection_readiness()[0]
    assert controller.selection['main_loss_arm'] == 'organ_exposure_fg_bg'
    assert not (controller.after_root / 'summary.json').exists()


def test_waiting_q02_never_queries_gpu_memory(controller):
    controller.gpu_memory = lambda *a: pytest.fail('dependency check touched GPU')
    write_json(controller.after_root / 'controller_status.json',
               dict(protocol_sha256=controller.protocol['after_protocol_sha256'], state='waiting_q04'))
    ready, state, _ = controller.prerequisites()
    assert not ready and state == 'waiting_q02'


def test_q02_complete_flag_cannot_hide_missing_checkpoint(controller):
    complete_predecessor(controller)
    assert controller.prerequisites()[0]
    (controller.after_root / 'auxiliary/task_id/seed_44/last.pt').unlink()
    assert controller.prerequisites()[1] == 'blocked_predecessor'


def test_q02_summary_requires_all_twelve_pairs(controller):
    complete_predecessor(controller)
    path = controller.after_root / 'summary.json'
    summary = json.loads(path.read_text())
    summary['runs'].pop()
    write_json(path, summary)
    assert controller.prerequisites()[1] == 'blocked_predecessor'


def test_q02_choice_change_blocks_controller_even_if_valid_json(controller):
    complete_predecessor(controller)
    with (controller.after_root / 'main_loss_selection.json').open('a') as stream:
        stream.write(' ')
    assert controller.prerequisites()[1] == 'blocked_predecessor'


def test_once_never_launches_preflight_or_full_predecessor_verification(controller):
    complete_predecessor(controller)
    controller.admit = lambda *a: pytest.fail('--once admitted GPU')
    controller.verify_predecessor = lambda: pytest.fail('--once ran heavy verification')
    assert controller.run(once=True) == 0
    assert all(j['train_state'] == 'pending' for j in controller.jobs)


def test_dry_run_does_not_write_status_selection_or_gpu(controller, capsys):
    path = controller.root / 'main_loss_selection.json'
    before = path.read_bytes()
    controller.gpu_memory = lambda *a: pytest.fail('dry-run touched GPU')
    assert controller.run(dry_run=True) == 0
    assert len(json.loads(capsys.readouterr().out)['jobs']) == 12
    assert not (controller.root / 'controller_status.json').exists()
    assert path.read_bytes() == before


def test_controller_refuses_tampered_protocol_before_scheduling(controller):
    path = controller.root / 'protocol.json'
    value = json.loads(path.read_text())
    value['config']['size'] = 256
    write_json(path, value)
    with pytest.raises(ValueError, match='hash'):
        Controller(controller.root, controller.reference_root, controller.after_root)


def test_only_readonly_frozen_q02_verification_command_is_constructed(controller):
    observed = []
    controller.run_cpu = lambda command, label: observed.append(command)
    controller.verify_predecessor()
    assert len(observed) == 1
    text = ' '.join(observed[0])
    assert 'require_test=True' in text and 'source_snapshot' in text
    assert 'summarize(' not in text and 'run_model(' not in text
