"""Run frozen seed42 Q06 then Q07 with shared GPU locks and stage test barriers.

Preparation is explicit and separate. Q06 requires complete Q05 and its bound
validation-only trigger. Q07 additionally waits for the complete Q06 matrix.
Each stage preflights every arm before training, and verifies all training before
opening test. --dry-run is read-only; --once launches no GPU work. Failed work
requires --retry-failed after the cause is repaired.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CODE_ROOT = Path(__file__).resolve().parents[1]
while str(CODE_ROOT) in sys.path:
    sys.path.remove(str(CODE_ROOT))
sys.path.insert(0, str(CODE_ROOT))

from scripts.run_organ_weighted import (  # noqa: E402
    exclusive_lock, gpu_free_mib, gpu_lock_path, read_json,
    terminate_children, timestamp, write_json,
)

VERSION = "q06_q07_mechanism_v1"
STAGE_MATRIX = {
    "q06": (("region",), ("control", "routing")),
    "q07": (("task_id",), ("experts0", "experts1", "experts9")),
}
FAILED_STATES = {"failed", "interrupted", "blocked_failed_tasks", "blocked_predecessor", "blocked_reference"}


def valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def verify_digest(value, field="protocol_sha256"):
    payload = {k: v for k, v in value.items() if k != field}
    if not valid_sha256(value.get(field)) or digest(payload) != value[field]:
        raise ValueError(f"Frozen {field} fingerprint differs")


def selection_for(root, protocol):
    selection = read_json(Path(root) / "main_loss_selection.json")
    verify_digest(selection, "selection_sha256")
    if (selection.get("protocol_sha256") != protocol["protocol_sha256"]
            or selection.get("main_loss_arm") != "organ_exposure_fg_bg"):
        raise ValueError("Frozen C main loss selection differs")
    return selection


class Controller:
    def __init__(self, output, reference_root, after_root, gpus=(0, 1),
                 poll_seconds=15, preflight_min_free_mib=40960, retry_failed=False):
        self.root = Path(output).resolve()
        self.reference_root = Path(reference_root).resolve()
        self.after_root = Path(after_root).resolve()
        self.code_root = self.root / "source_snapshot"
        self.protocol = read_json(self.root / "protocol.json")
        verify_digest(self.protocol)
        self.stage = self.protocol.get("stage")
        if self.protocol.get("protocol_version") != VERSION or self.stage not in STAGE_MATRIX:
            raise ValueError("Expected a frozen Q06/Q07 mechanism protocol")
        self.models, self.arms = STAGE_MATRIX[self.stage]
        self.seeds = (42,)
        self.protocol_sha256 = self.protocol["protocol_sha256"]
        config = self.protocol.get("config", {})
        if (config.get("models") != list(self.models) or config.get("arms") != list(self.arms)
                or config.get("seeds") != [42] or config.get("epochs") != 80):
            raise ValueError("Q06/Q07 require their declared stage matrix, seed42 only and 80 epochs")
        for field, value in (("reference_root", self.reference_root), ("after_root", self.after_root)):
            frozen = self.protocol.get(field, config.get(field))
            if frozen is None or Path(frozen).resolve() != value:
                raise ValueError(f"Controller dependency differs from protocol.{field}")
        if (self.root == self.reference_root or self.root == self.after_root
                or (self.stage == "q06" and self.after_root != self.reference_root)
                or (self.stage == "q07" and self.after_root == self.reference_root)):
            raise ValueError("Q06 requires Q05 predecessor; Q07 requires a distinct Q06 predecessor")
        self.gpus = list(gpus)
        if (not self.gpus or len(set(self.gpus)) != len(self.gpus)
                or any(type(gpu) is not int or gpu < 0 for gpu in self.gpus)):
            raise ValueError("GPU indices must be distinct nonnegative integers")
        if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
            raise ValueError("poll_seconds must be positive and at most 60")
        if type(preflight_min_free_mib) is not int or preflight_min_free_mib < 40960:
            raise ValueError("Preflight admission requires at least 40960 MiB")
        self.poll_seconds = poll_seconds
        self.preflight_min_free_mib = preflight_min_free_mib
        self.retry_failed = bool(retry_failed)
        self.gpu_memory, self.gpu_lock_path = gpu_free_mib, gpu_lock_path
        self.suite_prefix = [sys.executable, "-m", "unet_moe.mechanism_suite"]
        self.jobs = [dict(kind=kind, seed=seed, arm=arm, train_state="pending",
                          evaluate_state="pending", pid=None, gpu=None)
                     for seed in self.seeds for kind in self.models for arm in self.arms]
        self.preflights = {f"{kind}_{arm}": dict(kind=kind, arm=arm, state="pending")
                           for kind in self.models for arm in self.arms}
        self.active, self.required_mib = {}, {}
        self.control_child = self.root_lock = None
        self.selection = None
        self.status = dict(schema="moe_mechanism_controller_v1", stage=self.stage, state="starting",
                           output=str(self.root), reference_root=str(self.reference_root),
                           after_root=str(self.after_root), protocol_sha256=self.protocol_sha256,
                           controller_pid=os.getpid(), started_at=timestamp(), gpu_indices=self.gpus,
                           jobs=self.jobs, preflights=self.preflights,
                           required_free_mib=self.required_mib, gpu_observations={})

    def update(self, state=None, **extra):
        if state is not None:
            self.status["state"] = state
        self.status.update(updated_at=timestamp(), **extra)
        write_json(self.root / "controller_status.json", self.status)

    def folder(self, job):
        return self.root / job["arm"] / job["kind"] / f'seed_{job["seed"]}'

    def _job(self, kind, seed, arm):
        return next(job for job in self.jobs if (job["kind"], job["seed"], job["arm"]) == (kind, seed, arm))

    def _record(self, task):
        kind, seed, arm, action = task
        if action == "preflight":
            return self.preflights[f"{kind}_{arm}"], "state"
        return self._job(kind, seed, arm), f"{action}_state"

    def state(self, task):
        record, field = self._record(task)
        return record[field]

    def command(self, task=None, action=None, device="cuda:0"):
        result = [*self.suite_prefix, "--output", str(self.root), "--device", device]
        if task is not None:
            kind, seed, arm, phase = task
            result += ["--kind", kind, "--seed", str(seed), "--arm", arm]
            action = action or phase
        result += ["--action", action]
        if action == "train":
            # Resume is harmless for a fresh directory and mandatory when an
            # interrupted worker already wrote its run directory/checkpoint.
            result.append("--resume")
        return result

    def selection_readiness(self):
        """Validate the already copied choice; never select from new outcomes."""
        try:
            self.read_selection()
            return True, "selection_frozen", "Q05 C choice retained and cryptographically bound"
        except (OSError, ValueError, KeyError, TypeError) as error:
            return False, "blocked_reference", f"Q05-derived selection provenance invalid: {error}"

    def read_selection(self):
        selection = selection_for(self.root, self.protocol)
        self.selection = selection
        self.status.update(main_loss_arm=selection["main_loss_arm"],
                           selection_sha256=selection["selection_sha256"])
        return selection

    def validate_reference_evidence(self):
        # The suite checks copied validation-only review/trigger hashes; the
        # predicate itself is fixed before this controller can launch a worker.
        from unet_moe.mechanism_suite import validate_q05_evidence
        return validate_q05_evidence(self.root, self.protocol)

    def prerequisites(self):
        """Check stage-specific predecessor metadata without consulting GPU state."""
        predecessor_name = "Q05" if self.stage == "q06" else "Q06"
        waiting = "waiting_q05" if self.stage == "q06" else "waiting_q06"
        try:
            self.validate_reference_evidence()
            predecessor = read_json(self.after_root / "protocol.json")
            verify_digest(predecessor)
            expected_version = "q05_moe_refinement_v1" if self.stage == "q06" else VERSION
            if (predecessor.get("protocol_version") != expected_version
                    or (self.stage == "q07" and predecessor.get("stage") != "q06")):
                return False, "blocked_predecessor", "Predecessor stage/version differs"
            sha = predecessor["protocol_sha256"]
            if sha != self.protocol["after_protocol_sha256"]:
                return False, "blocked_predecessor", "Predecessor frozen protocol fingerprint differs"
            selection = selection_for(self.after_root, predecessor)
            status_path = self.after_root / "controller_status.json"
            if not status_path.exists():
                return False, waiting, f"{predecessor_name} controller has not published status"
            status = read_json(status_path)
            if status.get("protocol_sha256") != sha or status.get("state") in FAILED_STATES:
                return False, "blocked_predecessor", f"{predecessor_name} failed, interrupted or changed protocol"
            if (any(job.get(field) in FAILED_STATES for job in status.get("jobs", [])
                    for field in ("train_state", "evaluate_state"))
                    or any(report.get("state") in FAILED_STATES for report in status.get("preflights", {}).values())):
                return False, "blocked_predecessor", f"{predecessor_name} still has failed or interrupted tasks"
            if status.get("state") != "complete":
                return False, waiting, f"{predecessor_name} controller is {status.get('state')}"
            cfg = predecessor["config"]
            expected_config = ((["region", "task_id"], ["control", "refined"], [42, 43, 44])
                               if self.stage == "q06" else (["region"], ["control", "routing"], [42]))
            if tuple(cfg.get(k) for k in ("models", "arms", "seeds")) != expected_config or cfg.get("epochs") != 80:
                return False, "blocked_predecessor", "Predecessor experiment matrix differs"
            summary = read_json(self.after_root / "summary.json")
            if (summary.get("protocol_sha256") != sha or summary.get("complete") is not True
                    or summary.get("main_selection_sha256") != selection["selection_sha256"]):
                return False, "blocked_predecessor", f"{predecessor_name} complete summary missing or mismatched"
            expected = {(kind, seed, arm) for kind in cfg["models"] for seed in cfg["seeds"] for arm in cfg["arms"]}
            observed = [(r.get("kind"), r.get("seed"), r.get("arm")) for r in summary.get("runs", [])]
            if len(observed) != len(expected) or set(observed) != expected:
                return False, "blocked_predecessor", f"{predecessor_name} full train/test matrix is incomplete"
            for kind, seed, arm in expected:
                folder = self.after_root / arm / kind / f"seed_{seed}"
                saved = read_json(folder / "run_summary.json")
                fields = dict(kind=kind, seed=seed, arm=arm, protocol_sha256=sha,
                              main_selection_sha256=selection["selection_sha256"],
                              epochs_completed=80, test_evaluated=True)
                if any(saved.get(key) != value for key, value in fields.items()):
                    return False, "blocked_predecessor", f"{predecessor_name} per-run evidence incomplete: {folder}"
                if any(not (folder / name).is_file() or not (folder / name).stat().st_size
                       for name in ("best.pt", "last.pt", "metrics.jsonl", "test_best.json", "test_last.json")):
                    return False, "blocked_predecessor", f"{predecessor_name} checkpoint/history/results missing: {folder}"
            return True, "dependencies_complete", f"{predecessor_name} complete matrix metadata verified"
        except (OSError, ValueError, KeyError, TypeError) as error:
            return False, "blocked_predecessor", f"{predecessor_name} dependency evidence invalid: {type(error).__name__}: {error}"

    def verify_predecessor(self):
        """Read the predecessor through its own frozen source, without rewriting it."""
        module = "refinement_suite" if self.stage == "q06" else "mechanism_suite"
        code = (
            "import json,sys; from pathlib import Path; from importlib import import_module; "
            "q=import_module('unet_moe.'+sys.argv[3]); "
            "r=Path(sys.argv[1]); p,records,draws,w=q.load_protocol(r); p=q.with_selection(r,p); "
            "assert p['protocol_sha256']==sys.argv[2]; "
            "[q.validate_run(r,p,draws,k,s,a,require_test=True) for k,s,a in q.identities(p)]; "
            "q.verify_content(q.read_json(r/'data_content.json'),records,splits=('test',)); "
            "print(json.dumps({'verified_predecessor':str(r),'runs':len(q.identities(p)),'test_complete':True}))"
        )
        launcher = (
            "import os,subprocess,sys; from pathlib import Path; "
            "r=Path(sys.argv[1]); snapshot=r/'source_snapshot'; "
            "result=subprocess.run([sys.executable,'-c',sys.argv[2],str(r),sys.argv[3],sys.argv[4]],"
            "cwd=snapshot,env={**os.environ,'PYTHONPATH':str(snapshot),'CUDA_VISIBLE_DEVICES':''}); "
            "sys.exit(result.returncode)"
        )
        self.run_cpu([sys.executable, "-c", launcher, str(self.after_root), code,
                      self.protocol["after_protocol_sha256"], module],
                     "verify frozen predecessor completion without rewriting it")

    def run_cpu(self, command, label):
        path = self.root / "logs" / "verification.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as log:
            log.write(json.dumps(dict(time=timestamp(), label=label, command=command)) + "\n")
            log.flush()
            self.control_child = subprocess.Popen(command, cwd=self.code_root, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(self.code_root),
                     "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"},
                start_new_session=True, pass_fds=(self.root_lock.fileno(),))
            self.update(control_child_pid=self.control_child.pid)
            try:
                result = self.control_child.wait()
            finally:
                if self.control_child.poll() is None:
                    terminate_children([self.control_child])
                self.control_child = None
                self.update(control_child_pid=None)
        if result:
            raise RuntimeError(f"{label} exited {result}; see {path}")

    def read_preflight(self, kind, arm):
        if self.selection is None:
            self.read_selection()
        key = f"{kind}_{arm}"
        path = self.root / "preflight" / f"{key}.json"
        from unet_moe.mechanism_suite import check_preflight, with_selection
        # Use the suite's full scoped deterministic admission validator, not a
        # duplicate weaker interpretation of its proof or numerical flags.
        report = check_preflight(self.root, with_selection(self.root, self.protocol), kind, arm)
        required = report.get("required_free_mib")
        if type(required) is not int or required <= 0:
            raise ValueError("Preflight memory admission must be a positive integer")
        self.required_mib[key] = required
        self.preflights[key].update(state="complete", path=str(path), required_free_mib=required)
        return report

    def validate_task(self, task):
        kind, seed, arm, action = task
        if action == "preflight":
            return self.read_preflight(kind, arm)
        command = self.command(task, action="verify", device="cpu")
        if action == "evaluate":
            command.append("--require-test")
        self.run_cpu(command, f"verify {task}")
        report = read_json(self.folder(self._job(kind, seed, arm)) / "run_summary.json")
        if self.selection is None:
            self.read_selection()
        fields = dict(kind=kind, seed=seed, arm=arm, protocol_sha256=self.protocol_sha256,
                      main_selection_sha256=self.selection["selection_sha256"], epochs_completed=80)
        if any(report.get(k) != v for k, v in fields.items()):
            raise ValueError(f"Incomplete or mismatched training evidence: {task}")
        if action == "evaluate" and report.get("test_evaluated") is not True:
            raise ValueError(f"Evaluation exited without complete test evidence: {task}")
        return report

    def fail(self, task, error, returncode=None):
        record, field = self._record(task)
        record.update({field: "failed", "pid": None})
        failure = dict(time=timestamp(), task=list(task), error=str(error), returncode=returncode)
        record.setdefault("failures", []).append(failure)
        write_json(self.root / "failures" / ("_".join(map(str, task)) + ".json"),
                   dict(protocol_sha256=self.protocol_sha256, **failure))
        self.update()

    def failed(self):
        return (any(p["state"] == "failed" for p in self.preflights.values())
                or any(j[field] == "failed" for j in self.jobs for field in ("train_state", "evaluate_state")))

    def recover(self):
        path = self.root / "controller_status.json"
        if path.exists():
            previous = read_json(path)
            if previous.get("protocol_sha256") != self.protocol_sha256:
                raise ValueError("Previous controller status belongs to another protocol")
            # A controller/finalization failure also requires an explicit retry.
            if previous.get("state") == "failed" and not self.retry_failed:
                raise ValueError("Previous controller failed; inspect its error and use --retry-failed after repair")
            for old in previous.get("jobs", []):
                job = self._job(old["kind"], old["seed"], old["arm"])
                for field in ("train_state", "evaluate_state"):
                    if old.get(field) == "failed" and not self.retry_failed:
                        job[field] = "failed"
                if old.get("failures"):
                    job["failures"] = old["failures"]
            for key, old in previous.get("preflights", {}).items():
                if key not in self.preflights:
                    raise ValueError("Unknown preflight in controller status")
                if old.get("failures"):
                    self.preflights[key]["failures"] = old["failures"]
                if old.get("state") == "failed" and not self.retry_failed:
                    self.preflights[key].update(old, pid=None)
        if (self.root / "main_loss_selection.json").exists():
            self.read_selection()
        for kind in self.models:
            for arm in self.arms:
                key = f"{kind}_{arm}"
                if (self.root / "preflight" / f"{key}.json").exists() and self.preflights[key]["state"] != "failed":
                    try:
                        self.read_preflight(kind, arm)
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        if not self.retry_failed:
                            self.fail((kind, 42, arm, "preflight"), error)
        test_jobs = []
        for job in self.jobs:
            path = self.folder(job) / "run_summary.json"
            if not path.exists() or job["train_state"] == "failed":
                continue
            report = read_json(path)
            if report.get("epochs_completed") != 80:
                continue
            task = (job["kind"], job["seed"], job["arm"], "train")
            try:
                self.validate_task(task)
                job["train_state"] = "complete"
                if report.get("test_evaluated") is True:
                    test_jobs.append(job)
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                self.fail(task, error)
        if test_jobs and not all(job["train_state"] == "complete" for job in self.jobs):
            raise ValueError("Existing test evaluation violates the all-training barrier")
        # Test verification can read test-content hashes; perform it only after
        # the entire training/checkpoint-selection matrix has been verified.
        for job in test_jobs:
            if job["evaluate_state"] == "failed":
                continue
            task = (job["kind"], job["seed"], job["arm"], "evaluate")
            try:
                self.validate_task(task)
                job["evaluate_state"] = "complete"
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                self.fail(task, error)

    def pending_tasks(self):
        if self.failed():
            return []
        # Every arm must pass preflight before any arm trains; stage matrices
        # have two or three arms, with seed42 only.
        if not all(p["state"] == "complete" for p in self.preflights.values()):
            return [(kind, 42, arm, "preflight") for kind in self.models for arm in self.arms
                    if self.preflights[f"{kind}_{arm}"]["state"] == "pending"]
        tasks = [(j["kind"], j["seed"], j["arm"], "train") for j in self.jobs if j["train_state"] == "pending"]
        if all(j["train_state"] == "complete" for j in self.jobs):
            tasks += [(j["kind"], j["seed"], j["arm"], "evaluate")
                      for j in self.jobs if j["evaluate_state"] == "pending"]
        return tasks

    def admit(self, gpu, tasks):
        lock = self.gpu_lock_path(gpu).open("a+")
        try:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.status["gpu_observations"][str(gpu)] = dict(state="locked", time=timestamp())
                lock.close()
                return None
            free = self.gpu_memory(gpu)
            self.status["gpu_observations"][str(gpu)] = dict(state="available", free_mib=free, time=timestamp())
            for task in tasks:
                required = self.preflight_min_free_mib if task[3] == "preflight" else self.required_mib[f"{task[0]}_{task[2]}"]
                if free >= required:
                    lock.seek(0)
                    lock.truncate()
                    lock.write(json.dumps(dict(pid=os.getpid(), output=str(self.root), gpu=gpu, task=task)) + "\n")
                    lock.flush()
                    return task, lock, free
            self.status["gpu_observations"][str(gpu)]["state"] = "waiting_for_memory"
            lock.close()
            return None
        except BaseException:
            lock.close()
            raise

    def launch(self, gpu, task, lock, free):
        if self.state(task) != "pending" or any(item["task"] == task for item in self.active.values()):
            raise ValueError("Task is not pending or already has a live worker")
        if task[3] == "evaluate" and not all(j["train_state"] == "complete" for j in self.jobs):
            raise ValueError("Cannot open test before all training runs are verified")
        if self.selection is None:
            raise ValueError("Cannot launch a GPU task before main-loss selection is frozen")
        path = self.root / "logs" / ("_".join(map(str, task)) + ".log")
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open("a")
        command = self.command(task)
        try:
            log.write(json.dumps(dict(time=timestamp(), task=task, command=command, gpu=gpu, free_mib=free)) + "\n")
            log.flush()
            worker_env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(self.code_root),
                          "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
                          "MECHANISM_LOCK_FDS": f"{self.root_lock.fileno()},{lock.fileno()}"}
            # Strict verification uses a workspace only in preflight workers.
            # Formal training always retains the previously frozen numerical mode.
            worker_env.pop("CUBLAS_WORKSPACE_CONFIG", None)
            if task[3] == "preflight":
                worker_env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            process = subprocess.Popen(command, cwd=self.code_root, stdout=log, stderr=subprocess.STDOUT,
                env=worker_env,
                start_new_session=True, pass_fds=(self.root_lock.fileno(), lock.fileno()))
        except BaseException:
            log.close()
            lock.close()
            raise
        self.active[gpu] = dict(process=process, task=task, lock=lock, log=log)
        record, field = self._record(task)
        record.update({field: "running", "pid": process.pid, "gpu": gpu, "log": str(path),
                       "free_mib_at_admission": free})
        self.update()

    def reap(self):
        for gpu, item in list(self.active.items()):
            code = item["process"].poll()
            if code is None:
                continue
            del self.active[gpu]
            item["log"].close()
            item["lock"].close()
            task = item["task"]
            record, field = self._record(task)
            record["pid"] = None
            if code:
                self.fail(task, f"Worker exited {code}; see {record['log']}", code)
                continue
            try:
                self.validate_task(task)
                record.update({field: "complete", "completed_at": timestamp()})
                self.update()
            except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
                self.fail(task, error, code)

    def stop(self):
        processes = [item["process"] for item in self.active.values()]
        if self.control_child is not None:
            processes.append(self.control_child)
        terminate_children(processes)
        self.control_child = None
        for item in self.active.values():
            item["lock"].close()
            item["log"].close()
            record, field = self._record(item["task"])
            record.update({field: "interrupted", "pid": None})
        self.active.clear()

    def dependency_tick(self):
        if self.selection is None:
            ready, state, reason = self.selection_readiness()
            self.update(state, selection_reason=reason)
            if not ready:
                return False
        ready, state, reason = self.prerequisites()
        self.update(state, dependency_reason=reason)
        return ready

    def run(self, once=False, dry_run=False):
        if dry_run:
            print(json.dumps(dict(protocol_sha256=self.protocol_sha256,
                                  selection_readiness=self.selection_readiness(),
                                  queue_prerequisites=self.prerequisites(),
                                  gpu_tasks=[self.command(task) for task in self.pending_tasks()],
                                  jobs=self.jobs), indent=2))
            return 0
        if not (self.code_root / "unet_moe" / "mechanism_suite.py").is_file():
            raise ValueError("Frozen source snapshot is missing the mechanism suite")
        with exclusive_lock(self.root / ".controller.lock") as root_lock:
            self.root_lock = root_lock
            handlers = {}

            def interrupted(signum, frame):
                raise InterruptedError(f"Controller received signal {signum}")

            try:
                for signum in (signal.SIGTERM, signal.SIGINT):
                    handlers[signum] = signal.signal(signum, interrupted)
                self.recover()
                (self.root / "controller.pid").write_text(f"{os.getpid()}\n")
                if self.failed():
                    self.update("blocked_failed_tasks", error="Inspect failures/; explicitly use --retry-failed after repair")
                    return 2
                while True:
                    ready = self.dependency_tick()
                    if once:
                        return 0
                    if ready:
                        break
                    time.sleep(self.poll_seconds)
                self.verify_predecessor()
                self.update(dependencies_verified_at=timestamp())
                while True:
                    self.reap()
                    pending = self.pending_tasks()
                    if not pending and not self.active:
                        break
                    for gpu in self.gpus:
                        if gpu in self.active or not pending:
                            continue
                        admitted = self.admit(gpu, pending)
                        if admitted is None:
                            continue
                        task, lock, free = admitted
                        self.launch(gpu, task, lock, free)
                        pending.remove(task)
                    phase = ("draining_after_failure" if self.failed() else
                             "evaluating" if all(j["train_state"] == "complete" for j in self.jobs) else
                             "preflight" if not all(p["state"] == "complete" for p in self.preflights.values()) else "training")
                    self.update(phase, waiting_tasks=len(pending))
                    time.sleep(self.poll_seconds)
                if not all(j["evaluate_state"] == "complete" for j in self.jobs):
                    self.update("blocked_failed_tasks", error="Inspect failures/; explicitly use --retry-failed after repair")
                    return 2
                self.run_cpu(self.command(action="summarize", device="cpu"), f"summarize {self.stage.upper()}")
                summary = read_json(self.root / "summary.json")
                if (summary.get("complete") is not True or summary.get("protocol_sha256") != self.protocol_sha256
                        or summary.get("main_selection_sha256") != self.selection["selection_sha256"]):
                    raise ValueError("Summary did not validate the complete frozen mechanism matrix")
                self.update("complete", completed_at=timestamp(), waiting_tasks=0)
                return 0
            except BaseException as error:
                for signum in handlers:
                    signal.signal(signum, signal.SIG_IGN)
                self.stop()
                self.update("interrupted" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                            error=f"{type(error).__name__}: {error}", control_child_pid=None)
                raise
            finally:
                for signum, handler in handlers.items():
                    signal.signal(signum, handler)
                self.root_lock = None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--after-root", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--poll-seconds", default=15, type=float)
    parser.add_argument("--preflight-min-free-mib", default=40960, type=int)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--once", action="store_true", help="Refresh frozen selection/dependency status once; never launch a GPU worker")
    parser.add_argument("--dry-run", action="store_true", help="Read-only queue plan; no workers or status mutation")
    args = parser.parse_args(argv)
    # A convenient workspace invocation must still execute the validators and
    # controller from the immutable stage source, including in read-only mode.
    frozen_entry = args.output.resolve() / "source_snapshot/scripts/run_moe_mechanisms.py"
    if Path(__file__).resolve() != frozen_entry:
        if not frozen_entry.is_file():
            raise ValueError("Frozen mechanism controller source snapshot is missing")
        forwarded = sys.argv[1:] if argv is None else argv
        os.execv(sys.executable, [sys.executable, str(frozen_entry), *forwarded])
    return Controller(args.output, args.reference_root, args.after_root,
                      gpus=[int(value) for value in args.gpus.split(",")], poll_seconds=args.poll_seconds,
                      preflight_min_free_mib=args.preflight_min_free_mib,
                      retry_failed=args.retry_failed).run(once=args.once, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"run_moe_mechanisms: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
