'Fixed three-level objective and finite single-Agent controls.'

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .operators.types import (
    DESTROY_FEATURE_NAMES,
    DESTROY_OPERATOR_NAMES,
    INITIAL_INSERTION_OPERATOR_NAMES,
    ALNS_REPAIR_OPERATOR_NAMES,
    INSERTION_TASK_FEATURE_NAMES,
)

OBJECTIVE_FORM = "fixed_three_level_lexicographic"
SERVICE_METRICS: Tuple[str, ...] = (
    "missed_priority",
    "unassigned_count",
)
QUALITY_METRICS: Tuple[str, ...] = SERVICE_METRICS + ("energy_total",)
CONSTRAINT_METRICS: Tuple[str, ...] = (
    "violation_total",
    "violation_capability",
    "violation_time_window",
    "violation_energy",
)

ACCEPTANCE_MODES: Tuple[str, ...] = ("greedy", "tolerant", "simulated_annealing")
ACCEPTANCE_MODE_MAP: Dict[str, str] = {
    "greedy": "greedy",
    "tolerant": "threshold",
    "simulated_annealing": "sa",
}
ORDERED_PRIORITY_WEIGHTS: tuple[int, ...] = (10, 5, 2)
PRIORITY_RANK_CAPACITY: int = len(ORDERED_PRIORITY_WEIGHTS)
DETERMINISTIC_INFORMATIVE_TRIALS = 10
STOCHASTIC_INFORMATIVE_TRIALS = 20
AGENT_ACTION_TRIALS = 100

FORMAL_FOCUS_IDS: Tuple[str, ...] = ("global", "service_bottleneck")
# Focus is a bounded soft preference inside one common candidate pool.
SOFT_FOCUS_GREEDY_BONUS: float = 1.0
SOFT_FOCUS_RANDOM_WEIGHT: float = 1.5
REMOVE_RATIO_OPTIONS: list[float] = [0.20, 0.25, 0.30, 0.40, 0.50]
TOLERANCE_VALUE_BASE: list[float] = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def remaining_iters(remaining: Mapping[str, Any]) -> int:
    return max(0, int(remaining.get("iters", 0) or 0))


def tolerance_value_options(low: float, high: float) -> list[float]:
    low_value = float(low)
    high_value = float(high)
    values = [value for value in TOLERANCE_VALUE_BASE if low_value <= float(value) <= high_value]
    return values or [low_value]




def terms_from_order(metrics: Iterable[Any]) -> List[Dict[str, str]]:
    return [{"metric": str(metric), "direction": "min"} for metric in metrics]


def metric_names_from_terms(terms: Iterable[Mapping[str, Any]]) -> List[str]:
    return [str(dict(term).get("metric", "")) for term in terms if str(dict(term).get("metric", ""))]








def normalize_global_objective_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {
        "objective_form": str(contract.get("objective_form", OBJECTIVE_FORM)),
        "service_baseline_order": [str(item) for item in contract.get("service_baseline_order", []) or []],
        "preference_order": [str(item) for item in contract.get("preference_order", []) or []],
    }
    fixed = [str(item) for item in contract.get("fixed_metric_order", []) or []]
    if fixed:
        normalized["fixed_metric_order"] = fixed
    if bool(contract.get("fixed_by_experiment", False)):
        normalized["fixed_by_experiment"] = True
    return normalized


def public_global_objective(
    contract: Mapping[str, Any],
    compiled_terms: Iterable[Mapping[str, Any]],
) -> Dict[str, Any]:
    normalized = normalize_global_objective_contract(contract)
    return {
        **normalized,
        "compiled_global_metric_order": metric_names_from_terms(compiled_terms),
    }
