"""Evidence gates must not turn incomplete or invalid runs into follow-up decisions."""
from copy import deepcopy
import math

import pytest

from scripts import review_moe_followup as review


def protocol(stage="Q05"):
    return {
        "protocol_sha256": "a" * 64,
        "config": {
            "models": ["region", "task_id"],
            "arms": ["control", "refined" if stage == "Q05" else "auxiliary"],
            "seeds": [42, 43, 44], "epochs": 80, "size": 512,
            "experts": 9, "top_k": 1,
        },
    }


def status(stage="Q05"):
    p = protocol(stage)
    return {
        "state": "complete", "protocol_sha256": p["protocol_sha256"],
        "jobs": [
            {"kind": kind, "arm": arm, "seed": seed,
             "train_state": "complete", "evaluate_state": "complete", "pid": None}
            for kind in p["config"]["models"] for arm in p["config"]["arms"]
            for seed in p["config"]["seeds"]
        ],
    }


def route(counts=None):
    counts = counts if counts is not None else [7, 3] + [0] * 7
    n = sum(counts)
    return {
        "queries": n, "selected_counts": counts,
        "probability_sum": [n / 9] * 9,
        "mean_probability": [1 / 9] * 9 if n else None,
        "entropy_sum": n * math.log(9),
        "top1_probability_sum": n / 9,
        "mean_softmax_entropy_nats": math.log(9) if n else None,
        "mean_top1_probability": 1 / 9 if n else None,
        "experts_selected": sum(c > 0 for c in counts),
    }


@pytest.mark.parametrize("stage", ["Q02", "Q05"])
def test_complete_exact_matrix_is_ready(stage):
    assert review.validate_stage_status(status(stage), protocol(stage), stage)


@pytest.mark.parametrize("state", ["training", "evaluating", "waiting_q02"])
def test_unfinished_stage_never_counts_as_ready(state):
    s = status()
    s["state"] = state
    assert not review.validate_stage_status(s, protocol(), "Q05")


@pytest.mark.parametrize("mutation", [
    lambda s: s["jobs"].pop(),
    lambda s: s["jobs"].append(deepcopy(s["jobs"][0])),
    lambda s: s["jobs"][0].update(seed=45),
    lambda s: s["jobs"][0].update(arm="auxiliary"),
    lambda s: s.update(protocol_sha256="b" * 64),
    lambda s: s["jobs"][0].update(pid=1234),
    lambda s: s["jobs"][0].update(evaluate_state="pending"),
    lambda s: s["jobs"][0].update(train_state="running"),
])
def test_forged_completion_is_rejected(mutation):
    s = status()
    mutation(s)
    with pytest.raises(ValueError):
        review.validate_stage_status(s, protocol(), "Q05")


@pytest.mark.parametrize("state", ["failed", "interrupted", "blocked_failed_tasks"])
def test_failed_predecessor_is_not_silently_waited_or_completed(state):
    s = status()
    s["state"] = state
    with pytest.raises(ValueError):
        review.validate_stage_status(s, protocol(), "Q05")


def test_valid_routing_preserves_concentration_and_empty_denominator():
    r = review.summarize_route(route())
    assert r["active_experts"] == 2
    assert r["maximum_share"] == pytest.approx(.7)
    empty = review.summarize_route(route([0] * 9))
    assert empty["active_experts"] == 0
    assert empty["maximum_share"] is None


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(queries=11),
    lambda r: r["selected_counts"].pop(),
    lambda r: r["selected_counts"].__setitem__(0, -1),
    lambda r: r["selected_counts"].__setitem__(0, 7.0),
    lambda r: r.update(experts_selected=1),
    lambda r: r.update(mean_softmax_entropy_nats=float("nan")),
    lambda r: r.update(mean_top1_probability=float("inf")),
    lambda r: r["mean_probability"].__setitem__(0, float("nan")),
])
def test_invalid_route_statistics_cannot_trigger_an_experiment(mutation):
    r = route()
    mutation(r)
    with pytest.raises(ValueError):
        review.summarize_route(r)


def decision_input(collapsed_seeds=(42, 43)):
    runs = []
    for seed in (42, 43, 44):
        layers = [{"all": review.summarize_route(route())} for _ in range(4)]
        if seed in collapsed_seeds:
            layers[-1]["all"] = review.summarize_route(route([10] + [0] * 8))
        runs.append({"kind": "region", "arm": "refined", "seed": seed,
                     "routing": {"60": {"layers": deepcopy(layers)},
                                 "80": {"layers": deepcopy(layers)}}})
    return {"stage": "Q05", "complete": True, "test_metrics_used": False, "run_reviews": runs}


def test_two_replicated_persistent_exclusive_routes_trigger_preparation_only():
    result = review.decide_next_steps(decision_input())
    assert result["q06"]["state"] == "condition_met_pending_implementation"
    assert result["q06"]["triggered"] is True
    assert result["q07"]["state"] != "training"


def test_single_seed_is_not_enough_for_followup_trigger():
    result = review.decide_next_steps(decision_input((42,)))
    assert result["q06"]["triggered"] is False


def test_exclusive_only_at_final_epoch_is_not_persistent():
    report = decision_input()
    report["run_reviews"][1]["routing"]["60"]["layers"][-1]["all"] = review.summarize_route(route())
    assert not review.decide_next_steps(report)["q06"]["triggered"]


def test_near_exclusivity_does_not_silently_change_predefined_rule():
    report = decision_input()
    for row in report["run_reviews"]:
        for epoch in ("60", "80"):
            row["routing"][epoch]["layers"][-1]["all"] = review.summarize_route(route([999, 1] + [0] * 7))
    assert not review.decide_next_steps(report)["q06"]["triggered"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(complete=False),
    lambda r: r.update(stage="Q02"),
    lambda r: r["run_reviews"].pop(),
    lambda r: r["run_reviews"].append(deepcopy(r["run_reviews"][0])),
])
def test_incomplete_wrong_stage_or_duplicate_evidence_cannot_make_decision(mutation):
    report = decision_input()
    mutation(report)
    with pytest.raises(ValueError):
        review.decide_next_steps(report)


def full_run(kind, arm, seed):
    metrics = {key: .1 for key in review.METRICS}
    metrics["macro_positive_dice"] = .8
    metrics["per_structure"] = {
        f"organ_{i}": dict(samples=1, positive_samples=1, empty_samples=0,
                           positive_dice=.8, positive_miss_rate=.1,
                           positive_zero_overlap_rate=.1,
                           empty_false_positive_rate=None,
                           empty_mean_false_positive_area_fraction=None)
        for i in range(16)
    }
    def selection(epoch, passed):
        return dict(epoch=epoch, guardrail_passed=passed,
                    metrics=deepcopy(metrics), checkpoint_sha256="c" * 64)
    overall_n = 144 if kind == "region" else 16
    domain_sizes = [16] * 9 if kind == "region" else [2] * 7 + [1] * 2
    layer = {
        "all": route([overall_n] + [0] * 8),
        "by_domain": {f"domain_{i}": route([n] + [0] * 8) for i, n in enumerate(domain_sizes)},
        "by_structure": {
            f"organ_{i}": {"all": route([1] + [0] * 8),
                             "positive": route([1] + [0] * 8), "empty": route([0] * 9)}
            for i in range(16)
        },
    }
    return dict(kind=kind, arm=arm, seed=seed, validation_images=9,
                selections={"best": selection(20, False), "last": selection(80, False),
                            "guardrail": selection(10, True)},
                routing={str(e): dict(split="val", images=9, layers=[deepcopy(layer) for _ in range(4)])
                         for e in (60, 80)}, evidence=[])


def full_runs():
    return [full_run(kind, arm, seed) for kind in ("region", "task_id")
            for arm in ("control", "refined") for seed in (42, 43, 44)]


def test_separate_guardrail_checkpoint_never_certifies_best():
    report = review.build_validation_review("Q05", protocol(), full_runs())
    assert report["winner"] is None
    assert report["test_metrics_used"] is False
    for run in report["run_reviews"]:
        assert run["selections"]["guardrail"]["guardrail_passed"]
        assert not run["selections"]["best"]["guardrail_passed"]
        assert run["selections"]["best"]["epoch"] == 20


def test_missing_guardrail_remains_missing_in_paired_summary():
    runs = full_runs()
    runs[0]["selections"]["guardrail"] = None
    report = review.build_validation_review("Q05", protocol(), runs)
    row = next(r for r in report["paired_metrics"] if r["kind"] == "region"
               and r["checkpoint_selection"] == "guardrail"
               and r["metric"] == "macro_positive_dice")
    assert row["paired_delta"]["complete"] is False
    assert row["paired_delta"]["mean"] is None


def test_missing_seed_rejects_formal_report_even_if_other_runs_are_complete():
    with pytest.raises(ValueError):
        review.build_validation_review("Q05", protocol(), full_runs()[:-1])


def test_route_scope_must_reconcile_domains_with_overall():
    runs = full_runs()
    runs[0]["routing"]["60"]["layers"][0]["by_domain"]["domain_0"] = route([15] + [0] * 8)
    with pytest.raises(ValueError):
        review.build_validation_review("Q05", protocol(), runs)


def test_unobserved_negative_structure_metric_must_not_be_filled_with_zero():
    runs = full_runs()
    runs[0]["selections"]["best"]["metrics"]["per_structure"]["organ_0"]["empty_false_positive_rate"] = 0.
    with pytest.raises(ValueError):
        review.build_validation_review("Q05", protocol(), runs)
