"""The user deferral changes scheduling/test scope, never training or completion facts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_external_baselines as frozen_base
from scripts.q04_deferral_policy import (
    ACTIVE_METHODS, DEFERRED_METHODS, RESUME_POLICY, SEEDS, deferral_digest,
    load_deferral, validate_remaining_summary,
)
from scripts.run_q04_deferred import (
    assert_active_training_complete, controller_class, evaluate_active_run, summarize_active,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def prepared(tmp_path):
    root, q02, q01, metrics = (tmp_path / name for name in ("q04", "q02", "q01", "metrics"))
    protocol = dict(protocol_sha256="a" * 64, config=dict(
        methods=list(frozen_base.METHODS), seeds=list(SEEDS), epochs=80,
        reference_root=str(q01), evaluation_reference_root=str(metrics), mofo_prior=dict(epochs=80)))
    write(root / "protocol.json", protocol)
    write(q02 / "protocol.json", dict(protocol_sha256="b" * 64))
    manifest = dict(schema="q04_user_deferral_v1", q04_root=str(root), q02_root=str(q02),
        q04_protocol_sha256="a" * 64, q02_protocol_sha256="b" * 64,
        active_methods=list(ACTIVE_METHODS), deferred_methods=list(DEFERRED_METHODS), seeds=list(SEEDS),
        created_at="2026-10-14T16:00:00+08:00", reason="User requested postponing all UniverSeg",
        resume_policy=RESUME_POLICY, checkpoint_inventory={"universeg": "preserved"})
    manifest["deferral_sha256"] = deferral_digest(manifest)
    write(root / "queue_deferral.json", manifest)
    controller = controller_class(frozen_base)(root, q01, metrics, poll_seconds=.01)
    return root, protocol, manifest, controller


def test_manifest_accepts_auditable_extensions_and_binds_both_roots(prepared):
    root, _, manifest, _ = prepared
    assert load_deferral(root / "queue_deferral.json", q04_root=root, q02_root=manifest["q02_root"]) == manifest
    with pytest.raises(ValueError, match="q02 root"):
        load_deferral(root / "queue_deferral.json", q02_root=root)
    write(Path(manifest["q02_root"]) / "protocol.json", dict(protocol_sha256="c" * 64))
    with pytest.raises(ValueError, match="q02 protocol"):
        load_deferral(root / "queue_deferral.json")


@pytest.mark.parametrize("key,value", [("active_methods", ["smart"]), ("deferred_methods", []),
    ("seeds", [42]), ("resume_policy", "automatic"), ("created_at", "2026-10-14"), ("reason", "")])
def test_resigned_unauthorized_scope_still_rejected(prepared, key, value):
    root, _, manifest, _ = prepared
    manifest[key] = value
    manifest["deferral_sha256"] = deferral_digest(manifest)
    write(root / "queue_deferral.json", manifest)
    with pytest.raises(ValueError):
        load_deferral(root / "queue_deferral.json")


def test_unsigned_change_rejected(prepared):
    root, _, manifest, _ = prepared
    manifest["reason"] += "modified"
    write(root / "queue_deferral.json", manifest)
    with pytest.raises(ValueError, match="hash"):
        load_deferral(root / "queue_deferral.json")


def test_all_fifteen_jobs_remain_visible_but_universeg_never_schedules(prepared):
    _, _, manifest, controller = prepared
    assert len(controller.jobs) == 15 and len(controller.preflights) == 5
    deferred = [job for job in controller.jobs if job["method"] == "universeg"]
    assert len(deferred) == 3
    assert all(job["train_state"] == job["evaluate_state"] == "deferred" and job["pid"] is None for job in deferred)
    assert controller.preflights["universeg"]["state"] == "deferred"
    assert controller.status["deferral_sha256"] == manifest["deferral_sha256"]
    assert all(task[0] != "universeg" for task in controller.pending_tasks())
    for action in ("train", "evaluate", "preflight", "verify"):
        with pytest.raises(ValueError, match="deferred"):
            controller.command(("universeg", 42, action))


@pytest.mark.parametrize("retry", [False, True])
def test_restart_does_not_resume_deferred_or_clear_other_failures(prepared, retry):
    root, _, _, controller = prepared
    controller._job("universeg", 42).update(train_state="interrupted")
    controller._job("smart", 42).update(train_state="failed")
    controller.update("interrupted")
    restored = controller_class(frozen_base)(root, controller.reference_root, controller.evaluation_root, retry_failed=retry)
    restored.recover()
    assert restored._job("universeg", 42)["train_state"] == "deferred"
    assert restored._job("smart", 42)["train_state"] == ("pending" if retry else "failed")
    assert all(task[0] != "universeg" for task in restored.pending_tasks())


def test_twelve_active_barrier_blocks_on_every_unfinished_run(prepared):
    _, _, _, controller = prepared
    for method in ACTIVE_METHODS:
        controller.preflights[method]["state"] = "complete"
    for job in controller.active_jobs():
        job.update(train_state="complete", prior_state="complete" if job["method"] == "mofo" else "not_applicable")
    assert len(controller.pending_tasks()) == 12
    assert all(task[2] == "evaluate" for task in controller.pending_tasks())
    controller._job("smart", 44)["train_state"] = "failed"
    assert not any(task[2] == "evaluate" for task in controller.pending_tasks())
    with pytest.raises(ValueError, match="twelve"):
        controller.launch(0, ("mofo", 42, "evaluate"), None, 48000)


def test_training_command_uses_original_suite_evaluation_uses_explicit_wrapper(prepared):
    root, _, _, controller = prepared
    train = controller.command(("smart", 42, "train"))
    assert train[1:3] == ["-m", "unet_moe.external_baseline_suite"]
    verify = controller.command(("smart", 42, "evaluate"), action="verify", device="cpu")
    assert verify[1:3] == ["-m", "unet_moe.external_baseline_suite"]
    evaluate = controller.command(("smart", 42, "evaluate"))
    assert Path(evaluate[1]).name == "run_q04_deferred.py"
    assert "--worker-action" in evaluate


def test_test_barrier_fully_verifies_all_twelve_checkpoints_without_universeg(prepared):
    root, protocol, manifest, _ = prepared
    calls = []
    suite = SimpleNamespace(verify_run=lambda *args: calls.append(args[2:4]))
    assert_active_training_complete(root, protocol, manifest, suite)
    assert set(calls) == {(method, seed) for method in ACTIVE_METHODS for seed in SEEDS}
    assert len(calls) == 12


def test_evaluation_failure_does_not_touch_test_paths(prepared):
    root, protocol, manifest, _ = prepared
    def fail(*args):
        raise ValueError("incomplete checkpoint")
    suite = SimpleNamespace(verify_run=fail)
    records = [{"split": "test", "image": "/must/not/open/image", "mask": "/must/not/open/mask"}]
    with pytest.raises(ValueError, match="incomplete checkpoint"):
        evaluate_active_run(root, protocol, records, "smart", 42, "cpu", manifest, suite)
    with pytest.raises(ValueError, match="Deferred"):
        evaluate_active_run(root, protocol, records, "universeg", 42, "cpu", manifest, suite)


def complete_summary(manifest):
    return dict(complete=False, active_scope_complete=True, protocol_sha256=manifest["q04_protocol_sha256"],
        deferral_sha256=manifest["deferral_sha256"], active_methods=list(ACTIVE_METHODS),
        deferred_methods=list(DEFERRED_METHODS), seeds=list(SEEDS),
        results=[dict(method=method, seed=seed, selection=selection, split="test", complete=True,
                      protocol_sha256=manifest["q04_protocol_sha256"], deferral_sha256=manifest["deferral_sha256"])
                 for method in ACTIVE_METHODS for seed in SEEDS for selection in ("best", "last")])


@pytest.mark.parametrize("fault", ["complete", "missing", "duplicate", "universeg", "hash", "val"])
def test_summary_rejects_false_completion_or_wrong_scope(prepared, fault):
    _, _, manifest, _ = prepared
    summary = complete_summary(manifest)
    validate_remaining_summary(summary, manifest)
    if fault == "complete": summary["complete"] = True
    elif fault == "missing": summary["results"].pop()
    elif fault == "duplicate": summary["results"][-1] = summary["results"][0]
    elif fault == "universeg": summary["results"][0]["method"] = "universeg"
    elif fault == "hash": summary["results"][0]["deferral_sha256"] = "c" * 64
    elif fault == "val": summary["results"][0]["split"] = "val"
    with pytest.raises(ValueError):
        validate_remaining_summary(summary, manifest)


def test_summary_writes_four_method_reports_without_original_all_five_summary(prepared):
    root, protocol, manifest, _ = prepared
    expected = complete_summary(manifest)
    for result in expected["results"]:
        result.update(evaluation_scope="remaining_active_methods", surface_metrics=[dict(macro={"positive": {"dsc": 80.}})])
        folder = root / result["method"] / f"seed_{result['seed']}"
        write(folder / f"{result['selection']}_test_metrics.json", result)
        write(folder / f"{result['selection']}_val_metrics.json", dict(result, split="val"))
    def write_reports(path, *args):
        (path / "REPORT.md").write_text("# Q04 五种外部 baseline\n[summary.json](summary.json)\n")
        for name in ("metrics.csv", "per_domain_metrics.csv"):
            (path / name).write_text("method,seeds\nsmart,3\n")
    suite = SimpleNamespace(verify_run=lambda *args, **kwargs: None, read_json=lambda p: json.loads(p.read_text()),
        run_folder=lambda r, m, s: Path(r) / m / f"seed_{s}", COHORTS=("positive",), FIELDS=("dsc",),
        timestamp=lambda: "2026-10-14T08:00:00+00:00", write_reports=write_reports, write_json=write)
    summary = summarize_active(root, protocol, manifest, suite)
    assert summary["complete"] is False and summary["active_scope_complete"] is True
    assert len(summary["results"]) == 24 and len(summary["seed_groups"]) == 8
    assert not (root / "summary.json").exists()
    assert (root / "remaining_summary.json").exists()
    report = (root / "remaining_reports/REPORT.md").read_text()
    assert "UniverSeg 延期" in report and "../remaining_summary.json" in report
    assert "deferral_sha256" in (root / "remaining_reports/metrics.csv").read_text()
