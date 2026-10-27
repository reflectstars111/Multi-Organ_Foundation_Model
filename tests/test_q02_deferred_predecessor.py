"""The explicit Q04 deferral cannot weaken active-run or frozen-code evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_q02_after_deferred_q04 as overlay
from scripts.q04_deferral_policy import deferral_digest
from scripts.run_routing_optimization import Controller as FrozenController


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def mutate(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


@pytest.fixture
def controller(tmp_path):
    q02, q04, q01 = [tmp_path / name for name in ("q02", "q04", "q01")]
    write(q02 / "protocol.json", dict(protocol_sha256="a" * 64,
        after_protocol_sha256="b" * 64,
        config=dict(models=["region", "task_id"], arms=["control", "auxiliary"],
                    seeds=[42, 43, 44], epochs=80, reference_root=str(q01), after_root=str(q04))))
    write(q04 / "protocol.json", dict(protocol_sha256="b" * 64))
    manifest = dict(schema="q04_user_deferral_v1", q04_root=str(q04), q02_root=str(q02),
                    q04_protocol_sha256="b" * 64, q02_protocol_sha256="a" * 64,
                    active_methods=["lanzhou", "aimagelab", "smart", "mofo"],
                    deferred_methods=["universeg"], seeds=[42, 43, 44],
                    created_at="2026-10-14T15:00:00+08:00", reason="User explicitly deferred UniverSeg",
                    resume_policy="manual_review_after_remaining_queue")
    manifest["deferral_sha256"] = deferral_digest(manifest)
    write(q04 / "queue_deferral.json", manifest)
    cls = overlay.controller_class(FrozenController)
    result = cls(q02, q01, q04, deferral=q04 / "queue_deferral.json", q04_python=sys.executable)
    result._verification_calls = []
    result.verify_active_predecessor = lambda m: result._verification_calls.append(m)
    result.gpu_memory = lambda _: pytest.fail("Readiness probe touched a GPU")
    return result


def completed_active_scope(c):
    m = json.loads(c.deferral_path.read_text())
    jobs = []
    for method in m["active_methods"] + m["deferred_methods"]:
        for seed in m["seeds"]:
            state = "deferred" if method == "universeg" else "complete"
            jobs.append(dict(method=method, seed=seed, train_state=state, evaluate_state=state,
                             prior_state="complete" if method == "mofo" else "not_applicable", pid=None))
    status = dict(protocol_sha256=m["q04_protocol_sha256"], deferral_sha256=m["deferral_sha256"],
                  state="complete_with_deferred", jobs=jobs,
                  preflights={method: dict(state="deferred" if method == "universeg" else "complete")
                              for method in m["active_methods"] + m["deferred_methods"]})
    status.update({key: m[key] for key in ("active_methods", "deferred_methods", "resume_policy")})
    write(c.after_root / "controller_status.json", status)
    rows = []
    for method in m["active_methods"]:
        for seed in m["seeds"]:
            folder = c.after_root / method / f"seed_{seed}"
            write(folder / "run_summary.json", dict(method=method, seed=seed,
                  protocol_sha256=m["q04_protocol_sha256"], epochs_completed=80, test_evaluated=True))
            for name in ("best.pt", "last.pt", "metrics.jsonl"):
                (folder / name).write_text("fixture requires independent frozen verifier")
            for selection in ("best", "last"):
                for split in ("val", "test"):
                    row = dict(method=method, seed=seed, selection=selection, split=split,
                               complete=True, protocol_sha256=m["q04_protocol_sha256"],
                               deferral_sha256=m["deferral_sha256"], evaluation_scope="remaining_active_methods")
                    write(folder / f"{selection}_{split}_metrics.json", row)
                    (folder / f"{selection}_{split}_per_image.jsonl").write_text("fixture\n")
                    if split == "test":
                        rows.append(row)
    summary = dict(complete=False, active_scope_complete=True, protocol_sha256=m["q04_protocol_sha256"],
                   deferral_sha256=m["deferral_sha256"], results=rows, seed_groups=[])
    summary.update({key: m[key] for key in ("active_methods", "deferred_methods", "seeds")})
    write(c.after_root / "remaining_summary.json", summary)
    return m


def test_exact_active_scope_requires_frozen_verification(controller):
    completed_active_scope(controller)
    assert controller.prerequisites()[0]
    assert len(controller._verification_calls) == 1
    assert controller._verification_calls[0]["active_methods"] == ["lanzhou", "aimagelab", "smart", "mofo"]


@pytest.mark.parametrize("state", ["preparing_and_training", "evaluating"])
def test_waits_without_gpu_or_checkpoint_verification(controller, state):
    completed_active_scope(controller)
    mutate(controller.after_root / "controller_status.json", lambda s: s.update(state=state))
    assert controller.prerequisites()[1] == "waiting_q04"
    assert controller._verification_calls == []


@pytest.mark.parametrize("target,change", [
    ("controller_status.json", lambda v: v.update(state="complete")),
    ("controller_status.json", lambda v: v["jobs"].pop()),
    ("controller_status.json", lambda v: v["jobs"].append(dict(v["jobs"][0]))),
    ("controller_status.json", lambda v: v["jobs"][-1].update(train_state="complete")),
    ("controller_status.json", lambda v: v["jobs"][-1].update(pid=12345)),
    ("controller_status.json", lambda v: v["jobs"][0].update(evaluate_state="failed")),
    ("controller_status.json", lambda v: v["preflights"]["smart"].update(state="failed")),
    ("controller_status.json", lambda v: v["preflights"].pop("universeg")),
    ("remaining_summary.json", lambda v: v.update(complete=True)),
    ("remaining_summary.json", lambda v: v.update(active_scope_complete=False)),
    ("remaining_summary.json", lambda v: v["results"].pop()),
    ("remaining_summary.json", lambda v: v["results"].append(dict(v["results"][0]))),
    ("remaining_summary.json", lambda v: v["results"][0].update(method="universeg")),
    ("remaining_summary.json", lambda v: v.update(deferral_sha256="f" * 64)),
    ("remaining_summary.json", lambda v: v["results"][0].update(deferral_sha256="f" * 64)),
    ("remaining_summary.json", lambda v: v["results"][0].update(evaluation_scope="all_methods")),
    ("remaining_summary.json", lambda v: v["results"][0].update(extra="not in saved evidence")),
])
def test_partial_forged_or_failed_evidence_blocks(controller, target, change):
    completed_active_scope(controller)
    mutate(controller.after_root / target, change)
    assert controller.prerequisites()[1] == "blocked_predecessor"
    assert controller._verification_calls == []


def test_checkpoint_file_missing_blocks_even_with_complete_flags(controller):
    completed_active_scope(controller)
    (controller.after_root / "smart/seed_43/last.pt").unlink()
    assert controller.prerequisites()[1] == "blocked_predecessor"


@pytest.mark.parametrize("mutator", [
    lambda m: m.update(reason="Changed without rehash"),
    lambda m: m.update(q04_root="/wrong/root"),
    lambda m: m.update(q02_protocol_sha256="d" * 64),
    lambda m: m.update(active_methods=["lanzhou", "aimagelab", "mofo"]),
])
def test_changed_or_invalid_authorization_blocks(controller, mutator):
    completed_active_scope(controller)
    mutate(controller.deferral_path, mutator)
    assert controller.prerequisites()[1] == "blocked_predecessor"


def test_rehashed_authorization_cannot_change_while_controller_runs(controller):
    completed_active_scope(controller)
    def change(m):
        m["reason"] = "Changed after controller startup"
        m["deferral_sha256"] = deferral_digest(m)
    mutate(controller.deferral_path, change)
    assert controller.prerequisites()[1] == "blocked_predecessor"


def test_original_frozen_verifier_failure_blocks(controller):
    completed_active_scope(controller)
    def reject(_):
        raise RuntimeError("Frozen Q04 checkpoint hash verification failed")
    controller.verify_active_predecessor = reject
    ready, state, reason = controller.prerequisites()
    assert not ready and state == "blocked_predecessor" and "checkpoint hash" in reason


def test_verification_launch_uses_q04_interpreter_in_separate_process(controller):
    manifest = completed_active_scope(controller)
    commands = []
    controller.root_lock = object()
    controller.q04_python = Path("/q04/independent/python")
    controller.run_cpu = lambda command, label: commands.append(command)
    overlay.DeferredQ04Mixin.verify_active_predecessor(controller, manifest)
    command = commands[0]
    assert command[0] == sys.executable  # Q02 launches the isolated Q04 interpreter.
    assert command[4] == "/q04/independent/python"
    assert "require_test=True" in command[5]
    assert "verify_protocol(r)" in command[5]
    assert json.loads(command[-2]) == manifest["active_methods"]
    assert "cwd=snapshot" in command[2] and "'CUDA_VISIBLE_DEVICES':''" in command[2]


def test_training_selection_evaluation_and_locks_are_inherited_unchanged(controller):
    for name in ("command", "run", "launch", "admit", "recover", "pending_tasks", "read_selection"):
        assert getattr(type(controller), name) is getattr(FrozenController, name)
    command = controller.command(("region", 42, "control", "train"))
    assert command[:3] == [sys.executable, "-m", "unet_moe.routing_suite"]
    assert "--resume" in command


def test_modified_frozen_q02_source_is_rejected_before_import(tmp_path):
    root = tmp_path / "q02"
    source = root / "source_snapshot/scripts/run_routing_optimization.py"
    source.parent.mkdir(parents=True)
    source.write_text("original")
    protocol = dict(source_files={"scripts/run_routing_optimization.py": overlay.file_sha(source)})
    protocol["protocol_sha256"] = hashlib.sha256(json.dumps(protocol, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    write(root / "protocol.json", protocol)
    source.write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        overlay.load_frozen_controller(root)
