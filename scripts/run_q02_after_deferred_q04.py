"""Run unchanged, frozen Q02 after the explicitly authorized Q04 active scope.

This external scheduling overlay never rewrites either source_snapshot or
protocol. Its sole change to Q02 is the predecessor readiness check. UniverSeg
remains deferred, and Q04's original fifteen-run matrix remains incomplete.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


FAILED_STATES = {"failed", "interrupted", "blocked_failed_tasks"}
Q04_PYTHON = Path("/ssd1/data/Multi-Organ_Foundation_Model/envs/q04/bin/python")


def read_json(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_deferral(path, **kwargs):
    """Load the separately frozen sibling policy, without importing a package."""
    source = Path(__file__).with_name("q04_deferral_policy.py")
    spec = importlib.util.spec_from_file_location("_q02_deferral_policy", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_deferral(path, **kwargs)


def predecessor_metadata(q04_root, manifest):
    """Validate exact active/deferred membership before expensive CPU checks."""
    root = Path(q04_root)
    sha = manifest["q04_protocol_sha256"]
    active, deferred, seeds = (manifest[key] for key in
                               ("active_methods", "deferred_methods", "seeds"))
    path = root / "controller_status.json"
    if not path.exists():
        return False, "waiting_q04", "Q04 has not published controller status"
    status = read_json(path)
    if (status.get("protocol_sha256") != sha
            or status.get("deferral_sha256") != manifest["deferral_sha256"]):
        raise ValueError("Q04 controller protocol/deferral identity mismatch")
    if status.get("state") in FAILED_STATES:
        raise ValueError("Q04 controller has failed or was interrupted")
    if (any(job.get(field) in FAILED_STATES for job in status.get("jobs", [])
            for field in ("prior_state", "train_state", "evaluate_state"))
            or any(p.get("state") in FAILED_STATES
                   for p in status.get("preflights", {}).values())):
        raise ValueError("Q04 still has failed or interrupted stages")
    if status.get("state") != "complete_with_deferred":
        if status.get("state") == "complete":
            raise ValueError("Deferred Q04 cannot claim the original matrix is complete")
        return False, "waiting_q04", f"Q04 active scope is {status.get('state')}"
    for field in ("active_methods", "deferred_methods", "resume_policy"):
        if status.get(field) != manifest[field]:
            raise ValueError(f"Q04 controller differs from authorization: {field}")
    jobs = status.get("jobs", [])
    identities = [(job.get("method"), job.get("seed")) for job in jobs]
    expected = {(method, seed) for method in active + deferred for seed in seeds}
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError("Q04 controller must report exactly twelve active and three deferred runs")
    for job in jobs:
        is_deferred = job["method"] in deferred
        state = "deferred" if is_deferred else "complete"
        if (any(job.get(field) != state for field in ("train_state", "evaluate_state"))
                or job.get("pid") is not None):
            raise ValueError("Q04 active/deferred training and evaluation state mismatch")
        prior = "complete" if job["method"] == "mofo" else "not_applicable"
        if job.get("prior_state") != prior:
            raise ValueError("Q04 prior stage incomplete")
    preflights = status.get("preflights", {})
    if set(preflights) != set(active + deferred):
        raise ValueError("Q04 preflight methods differ from the authorized scope")
    for method, report in preflights.items():
        if report.get("state") != ("deferred" if method in deferred else "complete"):
            raise ValueError("Q04 preflight stage incomplete")
    summary = read_json(root / "remaining_summary.json")
    if (summary.get("complete") is not False
            or summary.get("active_scope_complete") is not True
            or summary.get("protocol_sha256") != sha
            or summary.get("deferral_sha256") != manifest["deferral_sha256"]
            or any(summary.get(field) != manifest[field]
                   for field in ("active_methods", "deferred_methods", "seeds"))):
        raise ValueError("Q04 active-scope summary is incomplete or mislabeled")
    rows = summary.get("results", [])
    expected_rows = {(method, seed, selection) for method in active
                     for seed in seeds for selection in ("best", "last")}
    identities = [(r.get("method"), r.get("seed"), r.get("selection")) for r in rows]
    if len(identities) != len(expected_rows) or set(identities) != expected_rows:
        raise ValueError("Q04 active-scope matrix must contain exactly 24 test results")
    for result in rows:
        if (result.get("complete") is not True or result.get("split") != "test"
                or result.get("protocol_sha256") != sha
                or result.get("deferral_sha256") != manifest["deferral_sha256"]
                or result.get("evaluation_scope") != "remaining_active_methods"):
            raise ValueError("Q04 active-scope test result identity invalid")
        folder = root / result["method"] / f"seed_{result['seed']}"
        if read_json(folder / f"{result['selection']}_test_metrics.json") != result:
            raise ValueError("Q04 summary differs from saved test-result evidence")
    for method in active:
        for seed in seeds:
            folder = root / method / f"seed_{seed}"
            saved = read_json(folder / "run_summary.json")
            expected = dict(method=method, seed=seed, protocol_sha256=sha,
                            epochs_completed=80, test_evaluated=True)
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Q04 run evidence incomplete: {method}/{seed}")
            names = ["best.pt", "last.pt", "metrics.jsonl"]
            names += [f"{selection}_{split}_{suffix}"
                      for selection in ("best", "last") for split in ("val", "test")
                      for suffix in ("metrics.json", "per_image.jsonl")]
            if any(not (folder / name).is_file() or not (folder / name).stat().st_size
                   for name in names):
                raise ValueError(f"Q04 checkpoint/history/results missing: {method}/{seed}")
    return True, "dependencies_complete", "Q04 twelve active runs ready for frozen evidence verification; UniverSeg deferred"


class DeferredQ04Mixin:
    """Only override readiness; frozen Q02 training and all locks are inherited."""

    def __init__(self, *args, deferral, q04_python=Q04_PYTHON, **kwargs):
        super().__init__(*args, **kwargs)
        self.q04_python = Path(q04_python).absolute()
        if not self.q04_python.is_file():
            raise ValueError("Q04's independent Python interpreter is missing")
        self.deferral_path = Path(deferral).resolve()
        manifest = load_deferral(self.deferral_path, q04_root=self.after_root, q02_root=self.root)
        self.deferral_sha256 = manifest["deferral_sha256"]
        self.status.update(deferral_path=str(self.deferral_path),
                           q04_verification_python=str(self.q04_python),
                           deferral_sha256=self.deferral_sha256,
                           q04_deferred_methods=manifest["deferred_methods"],
                           q04_active_methods=manifest["active_methods"],
                           q04_resume_policy=manifest["resume_policy"])

    def prerequisites(self):
        try:
            manifest = load_deferral(self.deferral_path, q04_root=self.after_root, q02_root=self.root)
            if manifest["deferral_sha256"] != self.deferral_sha256:
                raise ValueError("Authorization changed after Q02 controller startup")
            if (self.protocol.get("after_protocol_sha256") != manifest["q04_protocol_sha256"]
                    or self.protocol_sha256 != manifest["q02_protocol_sha256"]):
                raise ValueError("Deferral differs from frozen Q02 predecessor protocol")
            ready, state, reason = predecessor_metadata(self.after_root, manifest)
            if not ready:
                return ready, state, reason
            self.verify_active_predecessor(manifest)
            return True, "dependencies_complete", "Q04 twelve active runs and test evidence independently verified; UniverSeg remains deferred"
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as error:
            return False, "blocked_predecessor", f"Q04 deferral/evidence invalid: {type(error).__name__}: {error}"

    def verify_active_predecessor(self, manifest):
        """Use Q04's own immutable verifier in a separate interpreter, on CPU."""
        code = (
            "import json,sys; from pathlib import Path; "
            "from unet_moe.external_baselines.protocol import verify_protocol; "
            "from unet_moe.external_baseline_suite import verify_run; "
            "r=Path(sys.argv[1]); p,records=verify_protocol(r); "
            "assert p['protocol_sha256']==sys.argv[2]; "
            "methods=json.loads(sys.argv[3]); seeds=json.loads(sys.argv[4]); "
            "[verify_run(r,p,m,s,require_test=True) for m in methods for s in seeds]; "
            "print(json.dumps({'q04_active_runs_verified':12,'test_complete':True,"
            "'full_original_matrix_complete':False,'deferred_methods':['universeg']}))"
        )
        launcher = (
            "import os,subprocess,sys; from pathlib import Path; "
            "snapshot=Path(sys.argv[1])/'source_snapshot'; "
            "result=subprocess.run([sys.argv[2],'-c',sys.argv[3],*sys.argv[4:]],"
            "cwd=snapshot,env={**os.environ,'PYTHONPATH':str(snapshot),'CUDA_VISIBLE_DEVICES':''}); "
            "sys.exit(result.returncode)"
        )
        command = [sys.executable, "-c", launcher, str(self.after_root), str(self.q04_python), code,
                   str(self.after_root), manifest["q04_protocol_sha256"],
                   json.dumps(manifest["active_methods"]), json.dumps(manifest["seeds"])]
        if self.root_lock is None:
            # --dry-run remains read-only, and still cannot rely on status flags.
            result = subprocess.run(command, cwd=self.code_root, capture_output=True, text=True,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
            if result.returncode:
                raise RuntimeError(f"Frozen Q04 verification failed: {result.stderr[-2000:]}")
        else:
            self.run_cpu(command, "verify Q04 authorized active scope with frozen checkpoint/test verifier")


def controller_class(frozen_class):
    class Controller(DeferredQ04Mixin, frozen_class):
        pass
    return Controller


def load_frozen_controller(root):
    """Never import the evolving repository controller or mixed Q04 packages."""
    root = Path(root).resolve()
    snapshot = root / "source_snapshot"
    protocol = read_json(root / "protocol.json")
    unsigned = {key: value for key, value in protocol.items() if key != "protocol_sha256"}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"),
                                       ensure_ascii=False).encode()).hexdigest()
    if digest != protocol["protocol_sha256"]:
        raise ValueError("Frozen Q02 protocol hash mismatch")
    for relative, expected in protocol["source_files"].items():
        if file_sha(snapshot / relative) != expected:
            raise ValueError(f"Frozen Q02 source changed: {relative}")
    for name, module in list(sys.modules.items()):
        if name in {"scripts", "unet_moe"} or name.startswith(("scripts.", "unet_moe.")):
            source = getattr(module, "__file__", None)
            if source is not None and not Path(source).resolve().is_relative_to(snapshot):
                raise ValueError(f"Cannot mix source snapshots in one interpreter: {name}")
    sys.path.insert(0, str(snapshot))
    path = snapshot / "scripts/run_routing_optimization.py"
    spec = importlib.util.spec_from_file_location("_frozen_q02_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Controller


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--after-root", required=True, type=Path)
    parser.add_argument("--deferral", required=True, type=Path)
    parser.add_argument("--q04-python", type=Path, default=Q04_PYTHON)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--poll-seconds", default=15, type=float)
    parser.add_argument("--preflight-min-free-mib", default=40960, type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cls = controller_class(load_frozen_controller(args.output))
    return cls(args.output, args.reference_root, args.after_root, deferral=args.deferral,
               q04_python=args.q04_python,
               gpus=[int(value) for value in args.gpus.split(",")],
               poll_seconds=args.poll_seconds, preflight_min_free_mib=args.preflight_min_free_mib,
               retry_failed=args.retry_failed).run(once=args.once, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"run_q02_after_deferred_q04: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
