"""Run the explicitly authorized four-method Q04 remainder from its frozen code.

Deploy this file beside q04_deferral_policy.py outside source_snapshot. The
snapshot, original protocol and all optimization algorithms remain untouched.
UniverSeg is retained as deferred and requires a later manual decision.
"""
from __future__ import annotations

import argparse
import csv
import gc
import importlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

try:
    from q04_deferral_policy import load_deferral, validate_remaining_summary
except ModuleNotFoundError:
    from scripts.q04_deferral_policy import load_deferral, validate_remaining_summary


def load_frozen(root):
    snapshot = Path(root).resolve() / "source_snapshot"
    sys.path.insert(0, str(snapshot))
    spec = importlib.util.spec_from_file_location(
        "_q04_deferred_frozen_controller", snapshot / "scripts/run_external_baselines.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = importlib.import_module("unet_moe.external_baseline_suite")
    if not Path(suite.__file__).resolve().is_relative_to(snapshot):
        raise ValueError("Q04 worker imported mutable source instead of the frozen snapshot")
    return module, suite


def assert_active_training_complete(root, protocol, manifest, suite):
    """Full checkpoint/history verification for all twelve active runs, before test I/O."""
    current = load_deferral(Path(root) / "queue_deferral.json", q04_root=root)
    if current != manifest or protocol["protocol_sha256"] != current["q04_protocol_sha256"]:
        raise ValueError("Deferral or Q04 protocol changed")
    for method in manifest["active_methods"]:
        for seed in manifest["seeds"]:
            suite.verify_run(root, protocol, method, seed)


def evaluate_active_run(root, protocol, records, method, seed, device, manifest, suite):
    """Original evaluation algorithm with an explicit, recorded twelve-run barrier."""
    if method not in manifest["active_methods"] or seed not in manifest["seeds"]:
        raise ValueError("Deferred/unknown method cannot be evaluated by the remainder queue")
    assert_active_training_complete(root, protocol, manifest, suite)
    folder = suite.run_folder(root, method, seed)
    content = {}
    for path in sorted({r[k] for r in records if r["split"] == "test" for k in ("image", "mask")}):
        content[path] = dict(bytes=Path(path).stat().st_size, sha256=suite.file_sha(path))
    if suite.read_json(Path(root) / "test_content_expected.json") != content:
        raise ValueError("Test images/masks changed since the frozen Q01 data protocol")
    test_path = Path(root) / "test_content_manifest.json"
    if test_path.exists() and suite.read_json(test_path) != content:
        raise ValueError("Test content changed after evaluation began")
    suite.write_json(test_path, content)
    bank = suite._bank(root, records, method, protocol["config"]["size"])
    for selection in ("best", "last"):
        cp = suite.torch.load(folder / f"{selection}.pt", map_location="cpu", weights_only=True)
        suite.assert_checkpoint(cp, protocol, method, seed)
        model = suite.make_model(root, protocol, method, seed)
        model.load_state_dict(cp["model"], strict=True)
        epoch = cp["epoch"]
        del cp
        model.to(device)
        checkpoint_sha = suite.file_sha(folder / f"{selection}.pt")
        for split in ("val", "test"):
            dataset = suite.UnionDataset([r for r in records if r["split"] == split], protocol["config"]["size"])
            result, rows = suite.validate(model, method, dataset, device,
                protocol["config"]["micro_batch_sizes"][method], bank, surfaces=True)
            per_image = folder / f"{selection}_{split}_per_image.jsonl"
            suite._write_history(per_image, rows)
            result.update(complete=True, method=method, seed=seed, split=split, selection=selection,
                protocol_sha256=protocol["protocol_sha256"], checkpoint_sha256=checkpoint_sha,
                checkpoint_epoch=epoch, per_image_sha256=suite.file_sha(per_image),
                test_content_sha256=suite._digest(content), updated_at=suite.timestamp(),
                deferral_sha256=manifest["deferral_sha256"], evaluation_scope="remaining_active_methods")
            suite.write_json(folder / f"{selection}_{split}_metrics.json", result)
        del model
        gc.collect()
        if suite.torch.cuda.is_available():
            suite.torch.cuda.empty_cache()
    summary = suite.read_json(folder / "run_summary.json")
    summary.update(test_evaluated=True, evaluation_deferral_sha256=manifest["deferral_sha256"],
                   evaluation_scope="remaining_active_methods")
    suite.write_json(folder / "run_summary.json", summary)


def verify_active_evaluation(root, method, seed, manifest, suite):
    for selection in ("best", "last"):
        for split in ("val", "test"):
            result = suite.read_json(suite.run_folder(root, method, seed) / f"{selection}_{split}_metrics.json")
            if (result.get("deferral_sha256") != manifest["deferral_sha256"]
                    or result.get("evaluation_scope") != "remaining_active_methods"):
                raise ValueError("Evaluation is not bound to the user deferral")


def summarize_active(root, protocol, manifest, suite):
    assert_active_training_complete(root, protocol, manifest, suite)
    results, groups = [], []
    for method in manifest["active_methods"]:
        for seed in manifest["seeds"]:
            suite.verify_run(root, protocol, method, seed, require_test=True)
            verify_active_evaluation(root, method, seed, manifest, suite)
            for selection in ("best", "last"):
                results.append(suite.read_json(suite.run_folder(root, method, seed) / f"{selection}_test_metrics.json"))
        for selection in ("best", "last"):
            runs = [r for r in results if r["method"] == method and r["selection"] == selection]
            for cohort in suite.COHORTS:
                group = dict(method=method, selection=selection, cohort=cohort, seeds=len(runs), metrics={})
                for field in suite.FIELDS:
                    values = [r["surface_metrics"][0]["macro"][cohort][field] for r in runs]
                    valid = all(value is not None for value in values)
                    group["metrics"][field] = dict(mean=statistics.mean(values) if valid else None,
                        sample_std=statistics.stdev(values) if valid and len(values) > 1 else None)
                groups.append(group)
    summary = dict(complete=False, active_scope_complete=True,
        protocol_sha256=protocol["protocol_sha256"], deferral_sha256=manifest["deferral_sha256"],
        active_methods=manifest["active_methods"], deferred_methods=manifest["deferred_methods"],
        seeds=manifest["seeds"], resume_policy=manifest["resume_policy"],
        results=results, seed_groups=groups, updated_at=suite.timestamp())
    validate_remaining_summary(summary, manifest)
    report_root = Path(root) / "remaining_reports"
    report_root.mkdir(exist_ok=True)
    suite.write_reports(report_root, protocol, results, groups)
    report_path = report_root / "REPORT.md"
    report = report_path.read_text().replace("Q04 五种外部 baseline", "Q04 四种已完成外部 baseline（UniverSeg 延期）")
    report = report.replace("[summary.json](summary.json)", "[remaining_summary.json](../remaining_summary.json)")
    note = (f"用户授权延期记录：`{manifest['deferral_sha256']}`。仅报告 Lanzhou、AImageLab、SMART、MOFO "
            "各三种子，共12项训练；UniverSeg三项延期，未计作完成。原五方法协议尚未全部完成，"
            "后续由用户手动决定是否恢复。\n\n")
    first, rest = report.split("\n", 1)
    report_path.write_text(first + "\n\n" + note + rest)
    for filename in ("metrics.csv", "per_domain_metrics.csv"):
        path = report_root / filename
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = list(reader.fieldnames), list(reader)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["evaluation_scope", "deferral_sha256", *fields])
            writer.writeheader()
            writer.writerows(dict(evaluation_scope="four_active_methods_universeg_deferred",
                                  deferral_sha256=manifest["deferral_sha256"], **row) for row in rows)
    # Publish completion last. Never write the original all-five summary.json.
    suite.write_json(Path(root) / "remaining_summary.json", summary)
    return summary


def controller_class(base):
    class DeferredController(base.Controller):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.manifest = load_deferral(self.root / "queue_deferral.json", q04_root=self.root)
            self.wrapper_path = Path(__file__).resolve()
            self.active_methods = self.manifest["active_methods"]
            self.status.update(schema="q04_deferred_controller_v1", deferral_sha256=self.manifest["deferral_sha256"],
                active_methods=self.active_methods, deferred_methods=self.manifest["deferred_methods"],
                resume_policy=self.manifest["resume_policy"])
            for job in self.jobs:
                if job["method"] in self.manifest["deferred_methods"]:
                    job.update(train_state="deferred", evaluate_state="deferred", deferred_at=self.manifest["created_at"])
                    inventory = self.manifest.get("checkpoint_inventory", [])
                    if isinstance(inventory, list):
                        saved = next((item for item in inventory if item.get("seed") == job["seed"]), None)
                        if saved is not None:
                            job["completed_epochs"] = saved["completed_epochs"]
            self.preflights["universeg"]["state"] = "deferred"

        def active_jobs(self):
            return [job for job in self.jobs if job["method"] in self.active_methods]

        def all_trained(self):
            return all(job["train_state"] == "complete" for job in self.active_jobs())

        def command(self, task, action=None, device="cuda:0"):
            if task[0] not in self.active_methods:
                raise ValueError("UniverSeg is deferred and cannot be automatically restarted")
            if (action or task[2]) == "evaluate":
                return [sys.executable, str(self.wrapper_path), "--output", str(self.root),
                        "--worker-action", "evaluate", "--method", task[0], "--seed", str(task[1]), "--device", device]
            return super().command(task, action, device)

        def recover(self):
            previous_path = self.root / "controller_status.json"
            if previous_path.exists():
                previous = base.read_json(previous_path)
                if previous.get("protocol_sha256") != self.protocol_sha256:
                    raise ValueError("Previous status protocol differs")
                if previous.get("deferral_sha256", self.manifest["deferral_sha256"]) != self.manifest["deferral_sha256"]:
                    raise ValueError("Previous deferral differs")
                for old in previous.get("jobs", []):
                    job = self._job(old["method"], old["seed"])
                    if job["method"] not in self.active_methods:
                        job["previous_train_state"] = old.get("train_state")
                        continue
                    for field in ("prior_state", "train_state", "evaluate_state"):
                        if old.get(field) == "failed" and not self.retry_failed:
                            job[field] = "failed"
                    if old.get("failures"):
                        job["failures"] = old["failures"]
                for method, old in previous.get("preflights", {}).items():
                    if method not in self.preflights:
                        raise ValueError("Unknown previous preflight")
                    if method in self.active_methods and old.get("state") == "failed" and not self.retry_failed:
                        self.preflights[method].update(old, pid=None)
            for method in self.active_methods:
                if ((self.root / "preflight" / f"{method}.json").exists()
                        and self.preflights[method]["state"] != "failed"):
                    try:
                        self.read_preflight(method)
                    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                        if not self.retry_failed:
                            self.fail((method, 42, "preflight"), error)
            any_test = False
            for job in self.active_jobs():
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
                report = base.read_json(path)
                if report.get("epochs_completed") == 80:
                    try:
                        self.validate_task((method, seed, "train"))
                        job["train_state"] = "complete"
                        if report.get("test_evaluated") is True:
                            any_test = True
                            if job["evaluate_state"] != "failed":
                                self.validate_task((method, seed, "evaluate"))
                                job["evaluate_state"] = "complete"
                    except (ValueError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as error:
                        self.fail((method, seed, "train"), error)
            if any_test and not self.all_trained():
                raise ValueError("Existing tests violate the all-twelve-active-training barrier")

        def validate_task(self, task):
            if task[0] not in self.active_methods:
                raise ValueError("Cannot validate a deferred task as completed")
            result = super().validate_task(task)
            if task[2] == "evaluate":
                for selection in ("best", "last"):
                    for split in ("val", "test"):
                        metrics = base.read_json(self.folder(self._job(task[0], task[1])) / f"{selection}_{split}_metrics.json")
                        if (metrics.get("deferral_sha256") != self.manifest["deferral_sha256"]
                                or metrics.get("evaluation_scope") != "remaining_active_methods"):
                            raise ValueError("Evaluation scope mismatch")
            return result

        def pending_tasks(self):
            tasks = []
            for method in self.active_methods:
                state = self.preflights[method]["state"]
                if state == "failed":
                    continue
                if method == "mofo" and self._job(method, 42)["prior_state"] != "complete":
                    if self._job(method, 42)["prior_state"] == "pending":
                        tasks.append((method, 42, "mofo-prior"))
                    continue
                if state == "pending":
                    tasks.append((method, 42, "preflight"))
            for job in self.active_jobs():
                method, seed = job["method"], job["seed"]
                if self.preflights[method]["state"] != "complete" or job["prior_state"] in {"failed", "running"}:
                    continue
                if job["prior_state"] == "pending":
                    tasks.append((method, seed, "mofo-prior"))
                elif job["train_state"] == "pending":
                    tasks.append((method, seed, "train"))
            if self.all_trained():
                tasks.extend((job["method"], job["seed"], "evaluate")
                             for job in self.active_jobs() if job["evaluate_state"] == "pending")
            return tasks

        def launch(self, gpu, task, lock, free):
            if task[0] not in self.active_methods:
                raise ValueError("Deferred task launch rejected")
            if self.state(task) != "pending" or any(item["task"] == task for item in self.active.values()):
                raise ValueError("Task is not pending or already running")
            if task[2] == "evaluate" and not self.all_trained():
                raise ValueError("Cannot open test before all twelve active training runs are verified")
            path = self.root / "logs" / f"{task[0]}_seed{task[1]}_{task[2]}.log"
            path.parent.mkdir(parents=True, exist_ok=True)
            log = path.open("a")
            command = self.command(task)
            try:
                log.write(json.dumps(dict(time=base.timestamp(), task=task, command=command, gpu=gpu,
                    free_mib=free, deferral_sha256=self.manifest["deferral_sha256"])) + "\n")
                log.flush()
                process = subprocess.Popen(command, cwd=self.code_root, stdout=log, stderr=subprocess.STDOUT,
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

        def run(self):
            if not (self.code_root / "unet_moe/external_baseline_suite.py").is_file():
                raise ValueError("Frozen Q04 suite is missing")
            with base.exclusive_lock(self.root / ".controller.lock") as root_lock:
                self.root_lock = root_lock
                handlers = {}

                def interrupted(signum, frame):
                    raise InterruptedError(f"Controller received signal {signum}")

                try:
                    for signum in (signal.SIGTERM, signal.SIGINT):
                        handlers[signum] = signal.signal(signum, interrupted)
                    self.recover()
                    (self.root / "controller.pid").write_text(f"{os.getpid()}\n")
                    while True:
                        ready, state, reason = self.prerequisites()
                        self.update(state, dependency_reason=reason)
                        if ready:
                            break
                        time.sleep(self.poll_seconds)
                    self.update(dependencies_verified_at=base.timestamp())
                    while True:
                        if load_deferral(self.root / "queue_deferral.json", q04_root=self.root) != self.manifest:
                            raise ValueError("Deferral changed during execution")
                        self.reap()
                        pending = self.pending_tasks()
                        if not pending and not self.active:
                            break
                        for gpu in self.gpus:
                            if gpu in self.active or not pending:
                                continue
                            admitted = self.admit(gpu, pending)
                            if admitted is not None:
                                task, lock, free = admitted
                                self.launch(gpu, task, lock, free)
                                pending.remove(task)
                        self.update("evaluating" if self.all_trained() else "preparing_and_training", waiting_tasks=len(pending))
                        time.sleep(self.poll_seconds)
                    if not all(job["evaluate_state"] == "complete" for job in self.active_jobs()):
                        self.update("blocked_failed_tasks", error="Failed active stages require explicit --retry-failed after repair")
                        return 2
                    self.run_cpu([sys.executable, str(self.wrapper_path), "--output", str(self.root),
                                  "--worker-action", "summarize", "--device", "cpu"], "summarize Q04 remainder")
                    validate_remaining_summary(base.read_json(self.root / "remaining_summary.json"), self.manifest)
                    self.update("complete_with_deferred", completed_at=base.timestamp(), waiting_tasks=0,
                                active_scope_complete=True, complete=False)
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
    return DeferredController


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--after-root", type=Path)
    parser.add_argument("--after-evaluation-root", type=Path)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--poll-seconds", type=float, default=15)
    parser.add_argument("--preflight-min-free-mib", type=int, default=40960)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--check-config", action="store_true", help="Validate immutable code/protocol and print scope without writing status or starting workers")
    parser.add_argument("--worker-action", choices=("evaluate", "summarize"))
    parser.add_argument("--method")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    manifest = load_deferral(args.output / "queue_deferral.json", q04_root=args.output)
    base, suite = load_frozen(args.output)
    suite.torch.set_num_threads(4)
    protocol, records = suite.verify_protocol(args.output, verify_content=bool(args.worker_action))
    if args.worker_action:
        if args.check_config:
            parser.error("--check-config cannot execute a worker")
        if args.worker_action == "evaluate":
            evaluate_active_run(args.output, protocol, records, args.method, args.seed, args.device, manifest, suite)
        else:
            summarize_active(args.output, protocol, manifest, suite)
        return 0
    if args.after_root is None or args.after_evaluation_root is None:
        parser.error("Controller requires --after-root and --after-evaluation-root")
    controller = controller_class(base)(args.output, args.after_root, args.after_evaluation_root,
        gpus=[int(value) for value in args.gpus.split(",")], poll_seconds=args.poll_seconds,
        preflight_min_free_mib=args.preflight_min_free_mib, retry_failed=args.retry_failed)
    if args.check_config:
        print(json.dumps(dict(valid=True, protocol_sha256=protocol["protocol_sha256"],
            deferral_sha256=manifest["deferral_sha256"], active_methods=manifest["active_methods"],
            deferred_methods=manifest["deferred_methods"], seeds=manifest["seeds"],
            source_root=str(controller.code_root), gpu_indices=controller.gpus), ensure_ascii=False))
        return 0
    return controller.run()


if __name__ == "__main__":
    sys.exit(main())
