'Live legal controls and focus targets; no policy selection or stage controller.'
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping

from .agent_types import RuntimeContract
from .decision_landscape import build_decision_landscape
from .domain import (
    ACCEPTANCE_MODES,
    DESTROY_FEATURE_NAMES,
    FORMAL_FOCUS_IDS,
    INSERTION_TASK_FEATURE_NAMES,
    REMOVE_RATIO_OPTIONS,
    TOLERANCE_VALUE_BASE,
)
from .operators.types import (
    ALNS_REPAIR_OPERATOR_NAMES,
    DESTROY_OPERATOR_NAMES,
    INITIAL_INSERTION_OPERATOR_NAMES,
    SELECTOR_MODES,
)
from .solution import AssignmentSolution




class RuntimeContractBuilder:

    @staticmethod
    def step(
        *,
        instance: Any,
        run_state: Any,
        landscape: Mapping[str, Any],
    ) -> RuntimeContract:
        global_remaining = _global_remaining(run_state)
        remaining = dict(global_remaining)
        focus_expansions = _search_focus_expansions(instance, run_state, landscape)
        focus_id = "global"

        controls = {
            "active_focus_id": focus_id,
            "active_destroy_feature_priority": [],
            "active_task_feature_priority": [],
            "destroy_operators": list(DESTROY_OPERATOR_NAMES),
            "repair_operators": list(ALNS_REPAIR_OPERATOR_NAMES),
            "destroy_features": list(DESTROY_FEATURE_NAMES),
            "task_features": list(INSERTION_TASK_FEATURE_NAMES),
            "selector_modes": list(SELECTOR_MODES),
            "acceptance_modes": list(ACCEPTANCE_MODES),
            "max_worsening_tolerance": max(float(v) for v in TOLERANCE_VALUE_BASE),
            "remove_ratio_options": [float(v) for v in REMOVE_RATIO_OPTIONS],
            "priority_must_cover_pool": False,
        }
        can_run = run_state.working_solution is not None and int(remaining["iters"]) > 0
        return RuntimeContract(
            allowed_actions=["run_alns"] if can_run else [],
            remaining=remaining,
            search_focus_expansions=focus_expansions,
            controls=controls,
            decision_landscape=dict(landscape),
        )


def _global_remaining(run_state: Any) -> Dict[str, int]:
    remaining = run_state.global_budget.remaining()
    return {"iters": max(0, int(remaining.get("iters", 0) or 0))}




def build_runtime_decision_landscape(instance: Any, run_state: Any) -> Dict[str, Any]:
    working = getattr(run_state, "working_solution", None)
    if working is None:
        working = AssignmentSolution.empty_from_instance(instance, put_all_unassigned=True)
    structure_key = working.structure_key()
    objective_key = tuple(
        (str(item.get("metric", "")), str(item.get("direction", "")))
        for item in getattr(run_state, "global_objective_terms", []) or []
    )
    cache = dict(getattr(run_state, "decision_landscape_cache", {}) or {})
    if (
        cache.get("structure_key") == structure_key
        and tuple(cache.get("objective_key", ())) == objective_key
        and isinstance(cache.get("landscape"), Mapping)
    ):
        return deepcopy(dict(cache["landscape"]))
    landscape = build_decision_landscape(
        working,
        instance,
        run_state.config,
        objective_terms=run_state.global_objective_terms,
    )
    if hasattr(run_state, "decision_landscape_cache"):
        run_state.decision_landscape_cache = {
            "structure_key": structure_key,
            "objective_key": objective_key,
            "landscape": deepcopy(landscape),
        }
    return landscape










def _service_bottleneck_targets(instance: Any, run_state: Any, landscape: Mapping[str, Any], *, limit: int = 6) -> List[int]:
    working = getattr(run_state, "working_solution", None)
    unresolved = set(int(tid) for tid in (working.unassigned if working is not None else instance.all_task_ids()))
    facts = [
        dict(row)
        for row in dict(landscape.get("insertion_facts", {}) or {}).get("unassigned_task_facts", []) or []
        if isinstance(row, Mapping) and row.get("task_id") is not None
    ]
    facts = [row for row in facts if int(row["task_id"]) in unresolved]
    def rank(row: Mapping[str, Any]) -> tuple[int, float, int]:
        tid = int(row["task_id"])
        feasible = int(row.get("feasible_position_count", 10**9) or 0)
        priority = float(instance.task_by_id(tid).priority)
        return (feasible, -priority, tid)
    facts.sort(key=rank)
    targets = [int(row["task_id"]) for row in facts[: max(1, int(limit))]]
    if targets:
        return targets
    return sorted(
        unresolved,
        key=lambda tid: (-float(instance.task_by_id(int(tid)).priority), int(tid)),
    )[: max(1, int(limit))]


def _search_focus_expansions(instance: Any, run_state: Any, landscape: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    targets = _service_bottleneck_targets(instance, run_state, landscape)
    return {
        "global": {"focus_id": "global", "focus_task_ids": []},
        "service_bottleneck": {
            "focus_id": "service_bottleneck",
            "focus_task_ids": [int(tid) for tid in targets],
            "minimum_target_hits": 1,
        },
    }
