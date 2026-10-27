"""Run Q04 after the complete Q01 matrix and three-metric backfill.

Launch this file from ROOT/source_snapshot. The controller and every worker
inherit the same durable controller/GPU locks; a crashed controller cannot
start a duplicate worker while an orphan still owns its descriptors. Failed
tasks are retained and require --retry-failed on a subsequent invocation.
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
    exclusive_lock, gpu_free_mib, gpu_lock_path, read_json, terminate_children,
    timestamp, write_json,
)
from scripts.run_region_normalization import matrix_complete  # noqa: E402


METHODS = ("lanzhou", "aimagelab", "universeg", "smart", "mofo")
SEEDS = (42, 43, 44)
REFERENCE_MODELS = ("shared", "region", "task_id")
REFERENCE_ARMS = ("image_mean", "organ_exposure", "organ_exposure_fg_bg")


def valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class Controller:
    def __init__(self, output, after_root, after_evaluation_root, gpus=(0, 1),
                 poll_seconds=15, preflight_min_free_mib=40960, retry_failed=False):
        self.root = Path(output).resolve()
        self.reference_root = Path(after_root).resolve()
        self.evaluation_root = Path(after_evaluation_root).resolve()
        self.code_root = self.root / "source_snapshot"
        self.protocol = read_json(self.root / "protocol.json")
        self.protocol_sha256 = self.protocol.get("protocol_sha256")
        config = self.protocol.get("config", {})
        if (not valid_sha256(self.protocol_sha256)
                or config.get("methods") != list(METHODS)
                or config.get("seeds") != list(SEEDS) or config.get("epochs") != 80):
            raise ValueError("Q04 requires the frozen five-method, three-seed, 80-epoch protocol")
        for field, value in [("reference_root", self.reference_root),
                             ("evaluation_reference_root", self.evaluation_root)]:
            frozen_value = config.get(field, self.protocol.get(field))
            if frozen_value is not None and Path(frozen_value).resolve() != value:
                raise ValueError(f"Controller dependency differs from protocol.{field}")
        if len({self.root, self.reference_root, self.evaluation_root}) != 3:
            raise ValueError("Q04 and its two dependencies must have distinct roots")
        self.gpus = list(gpus)
        if (not self.gpus or len(self.gpus) != len(set(self.gpus))
                or any(type(gpu) is not int or gpu < 0 for gpu in self.gpus)):
            raise ValueError("GPU indices must be distinct nonnegative integers")
        if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
            raise ValueError("poll_seconds must be positive and at most 60")
        if type(preflight_min_free_mib) is not int or preflight_min_free_mib < 40960:
            raise ValueError("preflight admission requires at least 40960 MiB")
        self.poll_seconds = poll_seconds
        self.preflight_min_free_mib = preflight_min_free_mib
        self.retry_failed = bool(retry_failed)
        self.gpu_memory = gpu_free_mib
        self.gpu_lock_path = gpu_lock_path
        self.suite_prefix = [sys.executable, "-m", "unet_moe.external_baseline_suite"]
        self.jobs = [dict(method=method, seed=seed,
                          prior_state="pending" if method == "mofo" else "not_applicable",
                          train_state="pending", evaluate_state="pending", pid=None, gpu=None)
                     for method in METHODS for seed in SEEDS]
        self.preflights = {method: dict(method=method, state="pending") for method in METHODS}
        self.active = {}
        self.control_child = None
        self.root_lock = None
        self.required_mib = {}
        self.status = dict(schema="external_baseline_controller_v1", state="starting",
                           output=str(self.root), reference_root=str(self.reference_root),
                           evaluation_reference_root=str(self.evaluation_root),
                           protocol_sha256=self.protocol_sha256, controller_pid=os.getpid(),
                           started_at=timestamp(), gpu_indices=self.gpus,
                           jobs=self.jobs, preflights=self.preflights,
                           required_free_mib=self.required_mib, gpu_observations={})

    def update(self, state=None, **extra):
        if state is not None:
            self.status["state"] = state
        self.status.update(updated_at=timestamp(), **extra)
        write_json(self.root / "controller_status.json", self.status)

    def folder(self, job):
        return self.root / job["method"] / f'seed_{job["seed"]}'

    def _job(self, method, seed):
        return next(job for job in self.jobs if job["method"] == method and job["seed"] == seed)

    def _record(self, task):
        method, seed, action = task
        if action == "preflight":
            return self.preflights[method], "state"
        return self._job(method, seed), "prior_state" if action == "mofo-prior" else f"{action}_state"

    def state(self, task):
        record, field = self._record(task)
        return record[field]

    def command(self, task, action=None, device="cuda:0"):
        method, seed, phase = task
        return [*self.suite_prefix, "--output", str(self.root), "--method", method,
                "--seed", str(seed), "--action", action or phase, "--device", device]

    def prerequisites(self):
        """Return evidence-based readiness without taking GPU locks or editing dependencies."""
        try:
            reference = read_json(self.reference_root / "protocol.json")
            reference_sha = reference.get("protocol_sha256")
            configured_sha = self.protocol.get("reference_protocol_sha256")
            if not valid_sha256(reference_sha) or (configured_sha and reference_sha != configured_sha):
                return False, "blocked_reference", "Q01 protocol fingerprint mismatch"
            expected = {(kind, seed, arm) for kind in REFERENCE_MODELS for seed in SEEDS for arm in REFERENCE_ARMS}
            controller_path = self.reference_root / "controller_status.json"
            if controller_path.exists():
                controller = read_json(controller_path)
                if controller.get("state") in {"failed", "interrupted", "blocked_failed_tasks"}:
                    return False, "blocked_reference", f"Q01 controller is {controller['state']}"
            summary_path = self.reference_root / "summary.json"
            if not summary_path.exists():
                return False, "waiting_q01", "Q01 complete train/test summary not yet available"
            summary = read_json(summary_path)
            if not matrix_complete(summary, expected, reference_sha):
                return False, "waiting_q01", "Q01 27-run train/test matrix is incomplete"
            for run in summary["runs"]:
                folder = self.reference_root / run["arm"] / run["kind"] / f'seed_{run["seed"]}'
                saved = read_json(folder / "run_summary.json")
                keys = ("kind", "seed", "arm", "protocol_sha256", "epochs_completed", "test_evaluated")
                if any(saved.get(key) != run[key] for key in keys):
                    return False, "blocked_reference", f"Q01 run evidence differs: {folder}"
                if any(not (folder / name).is_file() or (folder / name).stat().st_size == 0
                       for name in ("last.pt", "best.pt", "metrics.jsonl")):
                    return False, "blocked_reference", f"Q01 checkpoint/history missing: {folder}"
            evaluation = read_json(self.evaluation_root / "status.json")
            evaluation_sha = self.protocol.get("evaluation_reference_protocol_sha256")
            if evaluation_sha and evaluation.get("protocol_sha256") != evaluation_sha:
                return False, "blocked_reference", "Metric-backfill protocol fingerprint mismatch"
            failed = evaluation.get("failed_jobs", evaluation.get("counts", {}).get("failed", 0))
            state = evaluation.get("state", "")
            if failed != 0 or state in {"failed", "error", "interrupted"}:
                return False, "blocked_reference", f"Metric backfill has failures or interruption: {state}, failed={failed}"
            if not isinstance(state, str) or not state.startswith("complete"):
                return False, "waiting_metrics", f"Metric backfill is {state}"
            return True, "dependencies_complete", "Q01 train/test and metric backfill are complete"
        except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError, KeyError) as error:
            return False, "blocked_reference", f"Dependency evidence missing or invalid: {type(error).__name__}: {error}"

    def read_preflight(self, method):
        path = self.root / "preflight" / f"{method}.json"
        report = read_json(path)
        required = report.get("required_free_mib")
        if (report.get("protocol_sha256") != self.protocol_sha256 or report.get("method") != method
                or report.get("passed") is not True or type(required) is not int or required <= 0):
            raise ValueError(f"Invalid or failed {method} preflight: {path}")
        if (report.get("real_optimizer_updates") != 2 or report.get("effective_batch") != 16
                or report.get("size") != 512 or report.get("real_query_images") != 32
                or report.get("strict_model_optimizer_rng_reload") is not True
                or report.get("identical_reload_prediction") is not True):
            raise ValueError(f"Preflight did not establish real 512/batch16 optimization and restoration: {path}")
        self.required_mib[method] = required
        self.preflights[method].update(state="complete", path=str(path), required_free_mib=required)
        return report

    def run_cpu(self, command, label):
        path = self.root / "logs" / "verification.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as log:
            log.write(json.dumps(dict(time=timestamp(), label=label, command=command)) + "\n")
            log.flush()
            self.control_child = subprocess.Popen(
                command, cwd=self.code_root, stdout=log, stderr=subprocess.STDOUT,
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

    def validate_task(self, task):
        method, seed, action = task
        if action == "preflight":
            return self.read_preflight(method)
        if action == "mofo-prior":
            path = self.root / "mofo_prior" / f"seed_{seed}" / "prior_complete.json"
            report = read_json(path)
            if (report.get("protocol_sha256") != self.protocol_sha256
                    or report.get("seed") != seed or report.get("complete") is not True
                    or report.get("epochs_completed") != self.protocol["config"]["mofo_prior"]["epochs"]):
                raise ValueError(f"Invalid MOFO prior evidence: {path}")
            checkpoint = path.parent / "last.pt"
            with checkpoint.open("rb") as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if report.get("checkpoint_sha256") != actual_hash:
                raise ValueError(f"MOFO prior checkpoint fingerprint mismatch: {checkpoint}")
            return report
        verify_command = self.command(task, action="verify", device="cpu")
        if action == "evaluate":
            verify_command.append("--require-test")
        self.run_cpu(verify_command, f"verify {task}")
        path = self.folder(self._job(method, seed)) / "run_summary.json"
        report = read_json(path)
        if (report.get("protocol_sha256") != self.protocol_sha256
                or report.get("method") != method or report.get("seed") != seed
                or report.get("epochs_completed") != 80):
            raise ValueError(f"Incomplete or mismatched training evidence: {path}")
        if action == "evaluate" and report.get("test_evaluated") is not True:
            raise ValueError(f"Evaluation exited without complete test evidence: {path}")
        return report

    def fail(self, task, error, returncode=None):
        record, field = self._record(task)
        record.update({field: "failed", "pid": None})
        failure = dict(time=timestamp(), task=list(task), error=str(error), returncode=returncode)
        record.setdefault("failures", []).append(failure)
        path = self.root / "failures" / f"{task[0]}_{task[1]}_{task[2]}.json"
        write_json(path, dict(protocol_sha256=self.protocol_sha256, **failure))
        self.update()

    def recover(self):
        previous_path = self.root / "controller_status.json"
        if previous_path.exists():
            previous = read_json(previous_path)
            if previous.get("protocol_sha256") != self.protocol_sha256:
                raise ValueError("Previous controller status belongs to a different protocol")
            for old in previous.get("jobs", []):
                job = self._job(old["method"], old["seed"])
                for field in ("prior_state", "train_state", "evaluate_state"):
                    if old.get(field) == "failed" and not self.retry_failed:
                        job[field] = "failed"
                if old.get("failures"):
                    job["failures"] = old["failures"]
            for method, old in previous.get("preflights", {}).items():
                if method not in self.preflights:
                    raise ValueError("Unknown method in controller status")
                if old.get("state") == "failed" and not self.retry_failed:
                    self.preflights[method].update(old, pid=None)
        for method in METHODS:
            if (self.root / "preflight" / f"{method}.json").exists():
                if self.preflights[method]["state"] == "failed":
                    continue
                try:
                    self.read_preflight(method)
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    if not self.retry_failed:
                        self.fail((method, 42, "preflight"), error)
        any_test = False
        for job in self.jobs:
            method, seed = job["method"], job["seed"]
            prior_path = self.root / "mofo_prior" / f"seed_{seed}" / "prior_complete.json"
            if method == "mofo" and prior_path.exists() and job["prior_state"] != "failed":
                try:
                    self.validate_task((method, seed, "mofo-prior"))
                    job["prior_state"] = "complete"
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    self.fail((method, seed, "mofo-prior"), error)
            path = self.folder(job) / "run_summary.json"
            if not path.exists() or job["train_state"] == "failed":
                continue
            report = read_json(path)
            if report.get("epochs_completed") == 80:
                try:
                    self.validate_task((method, seed, "train"))
                    job["train_state"] = "complete"
                    if report.get("test_evaluated") is True:
                        if job["evaluate_state"] != "failed":
                            self.validate_task((method, seed, "evaluate"))
                            job["evaluate_state"] = "complete"
                        any_test = True
                except (ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as error:
                    self.fail((method, seed, "train"), error)
        if any_test and not all(job["train_state"] == "complete" for job in self.jobs):
            raise ValueError("Existing test evaluation violates the all-15-training barrier")

    def pending_tasks(self):
        """Independent method failures never block the remaining training methods."""
        tasks = []
        for method in METHODS:
            preflight = self.preflights[method]["state"]
            if preflight == "failed":
                continue
            if method == "mofo" and self._job(method, 42)["prior_state"] != "complete":
                if self._job(method, 42)["prior_state"] == "pending":
                    tasks.append((method, 42, "mofo-prior"))
                continue
            if preflight == "pending":
                tasks.append((method, 42, "preflight"))
        for job in self.jobs:
            method, seed = job["method"], job["seed"]
            if self.preflights[method]["state"] != "complete":
                continue
            if job["prior_state"] in {"failed", "running"}:
                continue
            if job["prior_state"] == "pending":
                tasks.append((method, seed, "mofo-prior"))
            elif job["train_state"] == "pending":
                tasks.append((method, seed, "train"))
        if all(job["train_state"] == "complete" for job in self.jobs):
            tasks.extend((job["method"], job["seed"], "evaluate")
                         for job in self.jobs if job["evaluate_state"] == "pending")
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
                required = self.preflight_min_free_mib if task[2] in {"preflight", "mofo-prior"} else self.required_mib[task[0]]
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
        if task[2] == "evaluate" and not all(job["train_state"] == "complete" for job in self.jobs):
            raise ValueError("Cannot open test before all 15 training runs are verified")
        path = self.root / "logs" / f"{task[0]}_seed{task[1]}_{task[2]}.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        log = path.open("a")
        command = self.command(task)
        try:
            log.write(json.dumps(dict(time=timestamp(), task=task, command=command, gpu=gpu, free_mib=free)) + "\n")
            log.flush()
            process = subprocess.Popen(
                command, cwd=self.code_root, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(self.code_root),
                     "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1",
                     "Q04_LOCK_FDS": f"{self.root_lock.fileno()},{lock.fileno()}"},
                start_new_session=True, pass_fds=(self.root_lock.fileno(), lock.fileno()))
        except BaseException:
            log.close()
            lock.close()
            raise
        self.active[gpu] = dict(process=process, task=task, lock=lock, log=log)
        record, field = self._record(task)
        record.update({field: "running", "pid": process.pid, "gpu": gpu,
                       "log": str(path), "free_mib_at_admission": free})
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
                record[field] = "complete"
                record["completed_at"] = timestamp()
                self.update()
            except (ValueError, RuntimeError, OSError, KeyError, TypeError, json.JSONDecodeError) as error:
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

    def run(self):
        if not (self.code_root / "unet_moe" / "external_baseline_suite.py").is_file():
            raise ValueError("Frozen source snapshot is missing the Q04 suite")
        with exclusive_lock(self.root / ".controller.lock") as root_lock:
            self.root_lock = root_lock
            handlers = {}

            def interrupted(signum, frame):
                raise InterruptedError(f"Controller received signal {signum}")

            try:
                for signum in (signal.SIGTERM, signal.SIGINT):
                    handlers[signum] = signal.signal(signum, interrupted)
                # Read prior status before writing a fresh one; otherwise failed
                # tasks would silently become pending when the daemon restarts.
                self.recover()
                (self.root / "controller.pid").write_text(f"{os.getpid()}\n")
                while True:
                    ready, state, reason = self.prerequisites()
                    self.update(state, dependency_reason=reason)
                    if ready:
                        break
                    time.sleep(self.poll_seconds)
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
                    phase = "evaluating" if all(job["train_state"] == "complete" for job in self.jobs) else "preparing_and_training"
                    self.update(phase, waiting_tasks=len(pending))
                    time.sleep(self.poll_seconds)
                if not all(job["evaluate_state"] == "complete" for job in self.jobs):
                    self.update("blocked_failed_tasks", error="Failed stages remain; inspect failures/ and explicitly use --retry-failed after repair")
                    return 2
                self.run_cpu([*self.suite_prefix, "--output", str(self.root), "--action", "summarize", "--device", "cpu"], "summarize Q04")
                summary = read_json(self.root / "summary.json")
                if summary.get("complete") is not True or summary.get("protocol_sha256") != self.protocol_sha256:
                    raise ValueError("Final summary did not validate the complete frozen matrix")
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
    parser.add_argument("--after-root", required=True, type=Path)
    parser.add_argument("--after-evaluation-root", required=True, type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--poll-seconds", default=15, type=float)
    parser.add_argument("--preflight-min-free-mib", default=40960, type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    return Controller(args.output, args.after_root, args.after_evaluation_root,
                      gpus=[int(value) for value in args.gpus.split(",")],
                      poll_seconds=args.poll_seconds,
                      preflight_min_free_mib=args.preflight_min_free_mib,
                      retry_failed=args.retry_failed).run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"run_external_baselines: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
