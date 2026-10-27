"""Read-only, validation-only Q02/Q05 follow-up evidence watcher.

Never launches training, chooses a winning arm, reads test metric artifacts, or
rewrites an experiment. Each completed twelve-run matrix is independently
verified by its own frozen suite before reporting. The standard-library parent
process delegates checkpoint verification to an isolated CPU subprocess.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

STAGES = {"Q02": ("routing_suite", "q02_routing_v1", ("control", "auxiliary")),
          "Q05": ("refinement_suite", "q05_moe_refinement_v1", ("control", "refined"))}
MODELS = ("region", "task_id")
SEEDS = (42, 43, 44)
ROUTING_EPOCHS = (60, 80)
METRICS = ("macro_positive_dice", "macro_positive_miss_rate", "macro_positive_zero_overlap_rate",
           "macro_empty_false_positive_rate", "macro_empty_false_positive_area_fraction",
           "abdomen_macro_positive_dice")
STRUCTURE_METRICS = ("positive_dice", "positive_miss_rate", "positive_zero_overlap_rate",
                     "empty_false_positive_rate", "empty_mean_false_positive_area_fraction")
FAILED = {"failed", "interrupted", "blocked_failed_tasks", "blocked_predecessor", "blocked_reference"}
SCHEMA = "moe_followup_validation_v1"


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def canonical_digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2 ** 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    return json.loads(Path(path).read_text(), parse_constant=reject)


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def finite_fraction(value, label, nullable=False):
    if value is None and nullable:
        return None
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError(f"Invalid finite fraction: {label}")
    return value


def positive_integer(value, label, allow_zero=False):
    if type(value) is not int or value < (0 if allow_zero else 1):
        raise ValueError(f"Invalid integer: {label}")
    return value


def expected_identities(protocol, stage):
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    c = protocol["config"]
    if (c.get("models") != list(MODELS) or c.get("seeds") != list(SEEDS)
            or c.get("arms") != list(STAGES[stage][2]) or c.get("epochs") != 80
            or c.get("size") != 512 or c.get("experts") != 9 or c.get("top_k") != 1):
        raise ValueError(f"{stage} differs from the authorized frozen matrix")
    return {(kind, seed, arm) for kind in MODELS for seed in SEEDS for arm in c["arms"]}


def validate_stage_status(status, protocol, stage):
    """Return False while pending; fail closed on malformed/failed completion."""
    expected = expected_identities(protocol, stage)
    if status is None:
        return False
    if status.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise ValueError(f"{stage} controller protocol mismatch")
    if status.get("state") in FAILED:
        raise ValueError(f"{stage} controller requires repair: {status['state']}")
    jobs = status.get("jobs", [])
    identities = [(j.get("kind"), j.get("seed"), j.get("arm")) for j in jobs]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError(f"{stage} controller job matrix is malformed")
    if any(j.get(field) in FAILED for j in jobs for field in ("train_state", "evaluate_state")):
        raise ValueError(f"{stage} has failed/interrupted jobs")
    if any(p.get("state") in FAILED for p in status.get("preflights", {}).values()):
        raise ValueError(f"{stage} has failed/interrupted preflight")
    if status.get("state") != "complete":
        return False
    if any(j.get(field) != "complete" for j in jobs for field in ("train_state", "evaluate_state")):
        raise ValueError(f"{stage} claims complete with unfinished train/evaluation jobs")
    if any(j.get("pid") is not None for j in jobs):
        raise ValueError(f"{stage} complete job still records an active PID")
    return True


def summarize_route(route, experts=9, top_k=1):
    """Validate one route population and return concentration evidence."""
    n = positive_integer(route.get("queries"), "queries", allow_zero=True)
    counts = route.get("selected_counts")
    if not isinstance(counts, list) or len(counts) != experts:
        raise ValueError("Routing selected_counts does not match expert count")
    for count in counts:
        positive_integer(count, "selected_count", allow_zero=True)
        if count > n:
            raise ValueError("An expert is selected more times than available queries")
    if sum(counts) != n * top_k:
        raise ValueError("Routing selected_counts/queries mismatch")
    active = sum(v > 0 for v in counts)
    if type(route.get("experts_selected")) is not int or route["experts_selected"] != active:
        raise ValueError("Routing active expert count mismatch")
    for key, value in route.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError(f"Non-finite route value: {key}")
        if isinstance(value, list):
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
                raise ValueError(f"Invalid route vector: {key}")
    for key in ("mean_top1_probability",):
        if key in route:
            finite_fraction(route[key], key, nullable=n == 0)
    entropy = route.get("mean_softmax_entropy_nats")
    if entropy is not None and (not isinstance(entropy, (int, float)) or isinstance(entropy, bool)
                                or not math.isfinite(entropy) or entropy < 0
                                or entropy > math.log(experts) + 1e-5):
        raise ValueError("Invalid route entropy")
    probabilities = route.get("mean_probability")
    if probabilities is not None:
        if not isinstance(probabilities, list) or len(probabilities) != experts:
            raise ValueError("Invalid route probability shape")
        for value in probabilities:
            finite_fraction(value, "mean_probability")
        if n and not math.isclose(sum(probabilities), 1., rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Route probabilities do not sum to one")
    return dict(queries=n, selected_counts=counts, active_experts=active,
                maximum_share=max(counts) / (n * top_k) if n else None,
                strictly_single_expert=bool(n and active == 1),
                dominant_expert=counts.index(max(counts)) if n else None,
                mean_softmax_entropy_nats=entropy,
                mean_top1_probability=route.get("mean_top1_probability"))


def validate_metrics(metrics):
    result = {key: finite_fraction(metrics.get(key), key) for key in METRICS}
    structures = metrics.get("per_structure")
    if not isinstance(structures, dict) or len(structures) != 16:
        raise ValueError("Expected exactly 16 structure metric tables")
    result["per_structure"] = {}
    for name, row in sorted(structures.items()):
        positive = positive_integer(row.get("positive_samples"), name + "/positive_samples", allow_zero=True)
        empty = positive_integer(row.get("empty_samples"), name + "/empty_samples", allow_zero=True)
        samples = positive_integer(row.get("samples"), name + "/samples", allow_zero=True)
        if positive + empty != samples:
            raise ValueError("Structure positive/empty counts do not sum to samples")
        values = {key: finite_fraction(row.get(key), name + "/" + key,
                                      nullable=(empty == 0 if key.startswith("empty_") else positive == 0))
                  for key in STRUCTURE_METRICS}
        for key, value in values.items():
            applicable = empty > 0 if key.startswith("empty_") else positive > 0
            if not applicable and value is not None:
                raise ValueError("Undefined structure metric must remain null")
        result["per_structure"][name] = dict(samples=samples, positive_samples=positive,
                                            empty_samples=empty, **values)
    return result


def moments(values):
    if not values or any(value is None for value in values):
        return dict(n=len(values), mean=None, sample_std=None, complete=False)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Non-finite paired metric")
    return dict(n=len(values), mean=statistics.mean(values),
                sample_std=statistics.stdev(values) if len(values) > 1 else None, complete=True)


def summarize_routing(routing, kind, expected_images=None):
    result = {}
    if set(routing) != {str(e) for e in ROUTING_EPOCHS}:
        raise ValueError("Both epoch60 and epoch80 route reports are required")
    for epoch, value in routing.items():
        images = positive_integer(value.get("images"), "validation images")
        if expected_images is not None and images != expected_images:
            raise ValueError("Routing validation image count differs from frozen records")
        if value.get("split") != "val":
            raise ValueError("Routing report must be validation-only")
        layers = value.get("layers")
        if not isinstance(layers, list) or len(layers) != 4:
            raise ValueError("Expected four decoder routing layers")
        out_layers = []
        for layer in layers:
            overall = summarize_route(layer["all"])
            if kind == "region" and overall["queries"] != images * 16:
                raise ValueError("Region global routes must cover all 16 proposals per image")
            domains = {name: summarize_route(row) for name, row in layer["by_domain"].items()}
            structures = {name: {cohort: summarize_route(row) for cohort, row in rows.items()}
                          for name, rows in layer["by_structure"].items()}
            if len(domains) != 9 or len(structures) != 16:
                raise ValueError("Expected nine route domains and sixteen structures")
            if sum(v["queries"] for v in domains.values()) != overall["queries"]:
                raise ValueError("Domain route query counts do not sum to global queries")
            if any(sum(v["selected_counts"][i] for v in domains.values()) != overall["selected_counts"][i]
                   for i in range(9)):
                raise ValueError("Domain route expert counts do not sum to global counts")
            for cohorts in structures.values():
                if set(cohorts) != {"all", "positive", "empty"}:
                    raise ValueError("Structure route cohorts incomplete")
                if cohorts["all"]["queries"] != cohorts["positive"]["queries"] + cohorts["empty"]["queries"]:
                    raise ValueError("Structure positive/empty query counts inconsistent")
                if any(cohorts["all"]["selected_counts"][i] != cohorts["positive"]["selected_counts"][i]
                       + cohorts["empty"]["selected_counts"][i] for i in range(9)):
                    raise ValueError("Structure positive/empty expert counts inconsistent")
            if kind == "task_id" and sum(v["all"]["queries"] for v in structures.values()) != overall["queries"]:
                raise ValueError("Task-ID annotated query counts mismatch")
            out_layers.append(dict(all=overall, by_domain=domains, by_structure=structures))
        result[epoch] = dict(split="val", images=images, checkpoint_selection="fixed_epoch_diagnostic",
                             epoch=int(epoch), layer_order="coarsest to finest decoder", layers=out_layers)
    return result


def build_validation_review(stage, protocol, runs):
    """Build a full three-seed validation comparison; never select a winner."""
    expected = expected_identities(protocol, stage)
    ids = [(r.get("kind"), r.get("seed"), r.get("arm")) for r in runs]
    if len(ids) != len(expected) or set(ids) != expected:
        raise ValueError("Cannot review an incomplete or duplicate twelve-run matrix")
    reviewed = []
    for run in runs:
        selected = {}
        if set(run["selections"]) != {"best", "last", "guardrail"}:
            raise ValueError("Missing checkpoint selection evidence")
        for selection, record in run["selections"].items():
            if record is None:
                if selection != "guardrail":
                    raise ValueError("Best/last checkpoint cannot be absent")
                selected[selection] = None
                continue
            epoch = positive_integer(record["epoch"], "checkpoint epoch")
            if epoch > 80 or (selection == "last" and epoch != 80):
                raise ValueError("Invalid checkpoint epoch")
            if type(record.get("guardrail_passed")) is not bool:
                raise ValueError("Checkpoint-specific guardrail evidence is required")
            if selection == "guardrail" and not record["guardrail_passed"]:
                raise ValueError("Selected guardrail checkpoint fails its own criteria")
            selected[selection] = {**record, "metrics": validate_metrics(record["metrics"])}
        reviewed.append(dict(kind=run["kind"], seed=run["seed"], arm=run["arm"], selections=selected,
                             routing=summarize_routing(run["routing"], run["kind"], run.get("validation_images")),
                             evidence=run.get("evidence", [])))
    paired = []
    treatment = STAGES[stage][2][1]
    for kind in MODELS:
        index = {(r["arm"], r["seed"]): r for r in reviewed if r["kind"] == kind}
        for selection in ("best", "last", "guardrail"):
            for metric in METRICS:
                control, modified, deltas = [], [], []
                for seed in SEEDS:
                    pair = [index[(arm, seed)]["selections"][selection] for arm in ("control", treatment)]
                    a, b = [entry["metrics"][metric] if entry is not None else None for entry in pair]
                    control.append(a)
                    modified.append(b)
                    deltas.append(b - a if a is not None and b is not None else None)
                paired.append(dict(kind=kind, checkpoint_selection=selection, metric=metric,
                                   unit="fraction", control=moments(control), treatment=moments(modified),
                                   paired_delta=moments(deltas), seeds=list(SEEDS),
                                   per_seed_delta=dict(zip(map(str, SEEDS), deltas))))
    return dict(schema=SCHEMA, stage=stage, complete=True, state="review_complete", split="val",
                test_metrics_used=False, winner=None, protocol_sha256=protocol["protocol_sha256"],
                treatment_arm=treatment, seeds=list(SEEDS), run_reviews=reviewed, paired_metrics=paired,
                routing_scope="Region all/domain include all16 proposals; structure cohorts are annotated-only; Task-ID is annotated queries only",
                routing_selection="fixed epochs60/80; not best checkpoint unless its selected epoch equals that epoch",
                guardrail_semantics="Each reported checkpoint is checked independently; existence of any guardrail checkpoint does not certify best")


def decide_next_steps(q05_review):
    pending = dict(schema=SCHEMA, test_metrics_used=False,
                   q06=dict(state="waiting_q05_validation_review", triggered=None, training_started=False),
                   q07=dict(state="planned_after_q06_decision", training_started=False))
    if q05_review is None:
        return pending
    if q05_review.get("complete") is not True:
        raise ValueError("Incomplete Q05 review cannot make a decision")
    if q05_review.get("stage") != "Q05" or q05_review.get("test_metrics_used") is not False:
        raise ValueError("Q06 decision requires a complete validation-only Q05 review")
    rows = [r for r in q05_review["run_reviews"] if r["kind"] == "region" and r["arm"] == "refined"]
    if len(rows) != 3 or {r["seed"] for r in rows} != set(SEEDS):
        raise ValueError("Q06 decision requires all three refined Region seeds")
    evidence = []
    for row in sorted(rows, key=lambda r: r["seed"]):
        epochs = {str(e): row["routing"][str(e)]["layers"][3]["all"] for e in ROUTING_EPOCHS}
        matches = all(v["queries"] > 0 and v["active_experts"] == 1 for v in epochs.values())
        evidence.append(dict(seed=row["seed"], strict_exclusivity_at_both_epochs=matches, epochs=epochs))
    qualifying = sum(r["strict_exclusivity_at_both_epochs"] for r in evidence)
    pending["q06"] = dict(state="condition_met_pending_implementation" if qualifying >= 2 else "not_triggered",
                           training_started=False, triggered=qualifying >= 2, qualifying_seeds=qualifying, required_seeds=2,
                           criterion="Q05 refined Region finest decoder layer: active_experts==1 at both epochs60 and80 in at least2of3 seeds",
                           diagnostic_only=True, performance_improvement_claimed=False, evidence=evidence)
    return pending


def render_review(review):
    lines = [f"# {review['stage']} 验证集配对复核", "", "仅采用训练与验证证据；不读取或据测试指标决策。三种子完整后生成，不宣称某组已获胜。", "",
             "| 模型 | 检查点 | 指标 | 对照均值±SD (%) | 改进均值±SD (%) | 配对差值 (百分点) |", "|---|---|---|---:|---:|---:|"]
    def fmt(stat):
        return "NA" if not stat["complete"] else f"{100*stat['mean']:.3f} ± {100*(stat['sample_std'] or 0):.3f}"
    for row in review["paired_metrics"]:
        lines.append(f"| {row['kind']} | {row['checkpoint_selection']} | {row['metric']} | {fmt(row['control'])} | {fmt(row['treatment'])} | {fmt(row['paired_delta'])} |")
    lines += ["", "DSC 越高越好；完全空预测率、零重叠率、空标签误检率和误检面积占比越低越好。护栏检查与最佳 DSC 检查点分别报告。", "",
              "| 模型/组/种子 | best轮 | best通过护栏 | last通过护栏 | 独立guardrail轮 |", "|---|---:|---|---|---:|"]
    for run in review["run_reviews"]:
        s = run["selections"]
        lines.append(f"| {run['kind']}/{run['arm']}/{run['seed']} | {s['best']['epoch']} | {s['best']['guardrail_passed']} | {s['last']['guardrail_passed']} | {s['guardrail']['epoch'] if s['guardrail'] else '无'} |")
    lines += ["", "路由统计来自固定第60、80轮；不能当作任意best轮的路由。层序从粗到细，L4为最细层。", "",
              "| 模型/组/种子 | 轮 | L1 活动专家/最大占比 | L2 | L3 | L4 |", "|---|---:|---:|---:|---:|---:|"]
    for run in review["run_reviews"]:
        for epoch in ("60", "80"):
            cells = []
            for layer in run["routing"][epoch]["layers"]:
                row = layer["all"]
                cells.append(f"{row['active_experts']}/9；{100*row['maximum_share']:.2f}%" if row['maximum_share'] is not None else 'NA')
            lines.append(f"| {run['kind']}/{run['arm']}/{run['seed']} | {epoch} | " + " | ".join(cells) + " |")
    lines += ["", "Region 全局包括所有16个proposal；Task-ID全局仅包括有标注查询。逐器官阳性/空标签路由、漏检和误检、证据文件SHA256均保存在同名JSON。", ""]
    return "\n".join(lines)


# The child accesses only training/validation fields of run_summary. The frozen
# verifier parses that JSON too, but require_test=False never inspects test data.
CHILD_CODE = r'''
import json,sys,hashlib
from pathlib import Path
from importlib import import_module
stage,root_path=sys.argv[1:3]
root=Path(root_path)
suite=import_module("unet_moe." + ("routing_suite" if stage=="Q02" else "refinement_suite"))
p,records,draws,weights=suite.load_protocol(root,check_content=True)
p=suite.with_selection(root,p)
metric_keys=json.loads(sys.argv[3]); structure_keys=json.loads(sys.argv[4])
def digest(path):
    return dict(path=str(path),sha256=suite.file_sha(path))
def white_metrics(row):
    return {**{k:row[k] for k in metric_keys},"per_structure":{name:{k:values[k] for k in ["samples","positive_samples","empty_samples",*structure_keys]} for name,values in row["per_structure"].items()}}
val_dataset=suite.fair.RegionDataset([r for r in records if r["split"]=="val"],p["config"]["size"],p["config"]["task_ids"])
rows=[]
for kind,seed,arm in suite.identities(p):
    summary=suite.validate_run(root,p,draws,kind,seed,arm,require_test=False)
    folder=suite.run_folder(root,kind,seed,arm)
    history=[json.loads(line) for line in (folder/"metrics.jsonl").read_text().splitlines()]
    selections={}
    evidence=[digest(folder/"metrics.jsonl")]
    for name,epoch in (("best",summary["best_epoch"]),("last",80),("guardrail",summary["guardrail_epoch"])):
        if epoch is None:
            selections[name]=None
            continue
        metric=history[epoch-1]
        passed=suite.guardrail_passes(metric,p["guardrail"])
        if metric.get("guardrail_passed") is not passed:
            raise ValueError("Recorded guardrail flag disagrees with checkpoint-specific verification")
        cp=folder/(name+".pt")
        sha=digest(cp); evidence.append(sha)
        selections[name]=dict(epoch=epoch,guardrail_passed=passed,metrics=white_metrics(metric),checkpoint_sha256=sha["sha256"])
    routes={}
    for epoch in (60,80):
        path=folder/"diagnostics"/f"epoch_{epoch:03d}.json"
        diagnostic=json.loads(path.read_text())
        value=diagnostic["validation"]
        routes[str(epoch)]={k:value[k] for k in ("split","images","layers")}
        # Proposal metrics are validation data, but not needed by this reviewer.
        for layer in routes[str(epoch)]["layers"]:
            layer.pop("proposal_metrics",None)
        evidence.append(digest(path))
    rows.append(dict(kind=kind,seed=seed,arm=arm,selections=selections,routing=routes,
                     validation_images=len(val_dataset),evidence=evidence))
print(json.dumps(dict(protocol_sha256=p["protocol_sha256"],runs=rows,test_metrics_used=False),allow_nan=False))
'''


def bind_stage(root, stage):
    root = Path(root).resolve()
    path = root / "protocol.json"
    protocol = read_json(path)
    unsigned = dict(protocol)
    sha = unsigned.pop("protocol_sha256", None)
    if (not sha or canonical_digest(unsigned) != sha
            or protocol.get("protocol_version") != STAGES[stage][1]):
        raise ValueError(f"{stage} protocol digest/version is invalid")
    # The frozen modules enforce exact version/config; record the claimed version
    # here and bind bytes before importing any experiment code.
    expected_identities(protocol, stage)
    sources = protocol.get("source_files")
    if not isinstance(sources, dict) or not sources or canonical_digest(sources) != protocol.get("source_sha256"):
        raise ValueError(f"{stage} source manifest is invalid")
    snapshot = root / "source_snapshot"
    for relative, expected in sources.items():
        source = (snapshot / relative).resolve()
        if not source.is_relative_to(snapshot.resolve()) or file_sha(source) != expected:
            raise ValueError(f"{stage} frozen source differs: {relative}")
    selection_path = root / "main_loss_selection.json"
    binding = dict(root=str(root), protocol_sha256=sha, protocol_file_sha256=file_sha(path),
                   source_sha256=protocol["source_sha256"], source_files=sources,
                   main_loss_selection_file_sha256=file_sha(selection_path))
    return protocol, binding


def assert_bindings(manifest):
    for stage, binding in manifest["stages"].items():
        root = Path(binding["root"])
        if file_sha(root / "protocol.json") != binding["protocol_file_sha256"]:
            raise ValueError(f"{stage} protocol bytes changed after binding")
        if file_sha(root / "main_loss_selection.json") != binding["main_loss_selection_file_sha256"]:
            raise ValueError(f"{stage} frozen loss selection changed after binding")
    if file_sha(Path(__file__).resolve()) != manifest["reviewer_sha256"]:
        raise ValueError("Reviewer code changed while bound; stop and audit before restarting")


def verify_and_review(stage, root, protocol, output, manifest):
    assert_bindings(manifest)
    _, current_binding = bind_stage(root, stage)
    if current_binding != manifest["stages"][stage]:
        raise ValueError("Stage binding changed before verification")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root / "source_snapshot"),
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1"}
    command = [sys.executable, "-c", CHILD_CODE, stage, str(root), json.dumps(METRICS), json.dumps(STRUCTURE_METRICS)]
    result = subprocess.run(command, cwd=root / "source_snapshot", env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    log_path = output / f"{stage.lower()}_verification.log"
    atomic_write(log_path, json.dumps(dict(time=timestamp(), returncode=result.returncode,
                                         cwd=str(root / "source_snapshot"), cpu_only=True)) + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{stage} frozen verifier failed; see {log_path}")
    payload = json.loads(result.stdout)
    if payload.get("protocol_sha256") != protocol["protocol_sha256"] or payload.get("test_metrics_used") is not False:
        raise ValueError("Verifier returned invalid protocol or test-use identity")
    assert_bindings(manifest)
    review = build_validation_review(stage, protocol, payload["runs"])
    review.update(created_at=timestamp(), manifest_sha256=manifest["manifest_sha256"],
                  verification_log_sha256=file_sha(log_path), reviewer_sha256=manifest["reviewer_sha256"])
    review["report_sha256"] = canonical_digest(review)
    atomic_write(output / f"{stage.lower()}_validation_review.json", review)
    atomic_write(output / f"{stage.lower()}_validation_review.md", render_review(review))
    return review


def load_cached_review(path, stage, protocol, manifest):
    if not path.exists():
        return None
    report = read_json(path)
    unsigned = dict(report)
    sha = unsigned.pop("report_sha256", None)
    if (not sha or canonical_digest(unsigned) != sha or report.get("stage") != stage
            or report.get("protocol_sha256") != protocol["protocol_sha256"]
            or report.get("manifest_sha256") != manifest["manifest_sha256"]
            or report.get("complete") is not True or report.get("test_metrics_used") is not False):
        raise ValueError("Existing follow-up report integrity/provenance differs")
    for run in report["run_reviews"]:
        for evidence in run["evidence"]:
            if file_sha(evidence["path"]) != evidence["sha256"]:
                raise ValueError("Previously reviewed training/validation evidence changed")
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q02-root", type=Path, required=True)
    parser.add_argument("--q05-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--watch", action="store_true")
    modes.add_argument("--once", action="store_true", help="Default: inspect once and exit")
    parser.add_argument("--poll-seconds", type=float, default=60.)
    args = parser.parse_args(argv)
    if not math.isfinite(args.poll_seconds) or not 0 < args.poll_seconds <= 60:
        parser.error("--poll-seconds must be positive and at most60")
    return args


def run(args):
    roots = {"Q02": args.q02_root.resolve(), "Q05": args.q05_root.resolve()}
    output = args.output.resolve()
    if roots["Q02"] == roots["Q05"] or any(output.is_relative_to(r) or r.is_relative_to(output) for r in roots.values()):
        raise ValueError("Follow-up output and both frozen runs must be distinct, non-nested directories")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".review.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another follow-up reviewer owns this output") from error
        status = dict(schema=SCHEMA, state="starting", controller_pid=os.getpid(), started_at=timestamp(),
                      output=str(output), training_started=False, test_metrics_used=False, stages={})
        def publish(state, **extra):
            status.update(state=state, updated_at=timestamp(), **extra)
            atomic_write(output / "controller_status.json", status)
        try:
            protocols, bindings = {}, {}
            for stage, root in roots.items():
                protocols[stage], bindings[stage] = bind_stage(root, stage)
            if protocols["Q05"].get("reference_protocol_sha256") != protocols["Q02"]["protocol_sha256"]:
                raise ValueError("Q05 is bound to a different Q02 protocol")
            manifest = dict(schema=SCHEMA, stages=bindings, reviewer_sha256=file_sha(Path(__file__).resolve()),
                            policy=dict(epochs=list(ROUTING_EPOCHS), layer_index=3, active_experts=1,
                                        required_seeds=2, total_seeds=3, test_metrics_used=False,
                                        training_auto_launch=False))
            manifest["manifest_sha256"] = canonical_digest(manifest)
            manifest_path = output / "review_manifest.json"
            if manifest_path.exists():
                if read_json(manifest_path) != manifest:
                    raise ValueError("Existing review manifest binds different roots/protocol/source/code; refusing silent switch")
            else:
                atomic_write(manifest_path, manifest)
            status["manifest_sha256"] = manifest["manifest_sha256"]
            reviews = {stage: load_cached_review(output / f"{stage.lower()}_validation_review.json", stage,
                                                protocols[stage], manifest) for stage in roots}
            if reviews["Q05"] and not reviews["Q02"]:
                raise ValueError("Q05 review exists without required Q02 review")
            while True:
                assert_bindings(manifest)
                for stage, root in roots.items():
                    path = root / "controller_status.json"
                    original = read_json(path) if path.exists() else None
                    ready = validate_stage_status(original, protocols[stage], stage)
                    if reviews[stage]:
                        if not ready:
                            raise ValueError(f"{stage} reverted from previously complete status")
                        status["stages"][stage] = dict(state="review_complete", report_sha256=reviews[stage]["report_sha256"])
                        continue
                    counts = Counter(j.get("train_state") for j in (original or {}).get("jobs", []))
                    status["stages"][stage] = dict(state="pending", controller_state=(original or {}).get("state"),
                                                 training_counts=dict(counts), decision=None)
                    if stage == "Q05" and not reviews["Q02"]:
                        status["stages"][stage]["reason"] = "waiting_q02_review"
                        continue
                    if ready:
                        publish("verifying_" + stage.lower())
                        reviews[stage] = verify_and_review(stage, root, protocols[stage], output, manifest)
                        status["stages"][stage] = dict(state="review_complete", report_sha256=reviews[stage]["report_sha256"])
                decision = decide_next_steps(reviews["Q05"])
                decision.update(updated_at=timestamp(), manifest_sha256=manifest["manifest_sha256"],
                                q02_review_sha256=reviews["Q02"]["report_sha256"] if reviews["Q02"] else None,
                                q05_review_sha256=reviews["Q05"]["report_sha256"] if reviews["Q05"] else None)
                atomic_write(output / "next_steps.json", decision)
                if reviews["Q05"]:
                    publish("complete")
                    return 0
                publish("waiting_q05" if reviews["Q02"] else "waiting_q02")
                if not args.watch:
                    return 0
                time.sleep(args.poll_seconds)
        except Exception as error:
            publish("failed", error=f"{type(error).__name__}: {error}")
            raise


def main(argv=None):
    try:
        return run(parse_args(argv))
    except Exception as error:
        print(f"follow-up reviewer failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
