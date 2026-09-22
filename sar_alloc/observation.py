'Four factual observation blocks for the single-agent controller.'
from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List, Mapping

from .agent_types import RuntimeContract
from .domain import QUALITY_METRICS, SERVICE_METRICS, SOFT_FOCUS_RANDOM_WEIGHT
from .evaluator import build_objective_keys, evaluate
from .solution import AssignmentSolution
from .focus_feedback import FOCUS_FEEDBACK_REVISION, build_focus_review
from .search_evidence import evidence_feedback

_MAX_TASK_ROWS = 8
_MAX_ROUTE_ROWS = 6




def build_step_observation(
    *,
    instance: Any,
    run_state: Any,
    contract: RuntimeContract,
) -> Dict[str, Any]:
    summary = _working_summary(instance, run_state)
    best = getattr(run_state, "best_summary", None)
    landscape = dict(contract.decision_landscape or {})
    feedback = evidence_feedback(
        {"focus_review": build_focus_review(run_state.search_memory.recent_controls(limit=12))},
        run_state.search_memory, current_phase=_phase(best or summary), revision="random_start_v3")
    if len(instance.tasks) <= 50 and _phase(best or summary) == 'energy_refinement':
        from .energy_guidance import exposure
        feedback['search_evidence']['repair_exposure'] = exposure(run_state.search_memory)
    blocks = [
        _block(
            "STEP01_state",
            "state",
            "Current objective phase, quality, maturity and run-budget position.",
            _state_signals(summary, best, run_state, n_tasks=len(instance.tasks)),
        ),
        _block(
            "STEP02_bottleneck",
            "bottleneck_opportunity",
            "Insertability and route-pressure facts available to the low-level ALNS decision.",
            _bottleneck_signals(landscape, contract, instance=instance),
        ),
        _block(
            "STEP04_boundary",
            "control_boundary",
            "Step chooses one focus and ALNS controls for the next fixed 100-trial window; both feature vectors remain zero.",
            _step_boundary(run_state, contract),
        ),
    ]
    if bool(feedback.get("history_available")):
        blocks.insert(
            2,
            _block(
                "STEP03_feedback",
                "feedback",
                "Recent low-level strategy outcomes and compact search history.",
                feedback,
            ),
        )
    _validate_observation_layout(blocks)
    observation = {
        "role": "step",
        "decision_contract": {
            "allowed_actions": list(contract.allowed_actions),
            "active_search_policy": {
                "focus_id": "global",
                "destroy_feature_priority": [],
                "task_feature_priority": [],
            },
            "fixed_execution_budget_trials": min(100, int(contract.remaining.get("iters", 0) or 0)),
        },
        "observation_blocks": blocks,
    }
    if contract.controls.get("step_focus_options"):
        observation["decision_contract"]["active_search_policy"].pop("focus_id")
        observation["decision_contract"]["focus_owner"] = "step"
    return observation


def solution_summary(
    solution: AssignmentSolution,
    instance: Any,
    config: Any,
    *,
    objective_terms: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    ev = evaluate(solution, instance, config, update_solution_schedule=True)
    quality = {str(k): float(v) for k, v in ev.quality_metrics.items()}
    terms = [dict(item) for item in (objective_terms or [])]
    return {
        "quality": quality,
        "hard_constraints": {
            "is_feasible": bool(ev.is_feasible),
            "violation_total": float(ev.constraint_report.violation_total),
            "violation_by_type": dict(ev.constraint_report.violation_by_type),
        },
        "objective_keys": build_objective_keys(ev, terms) if terms else None,
        "assigned_count": int(sum(len(route) for route in solution.routes.values())),
        "unassigned_count": int(len(solution.unassigned)),
        "unassigned_task_ids": sorted(int(tid) for tid in solution.unassigned),
        "route_task_counts": {str(aid): int(len(route)) for aid, route in sorted(solution.routes.items())},
    }


def _working_summary(instance: Any, run_state: Any) -> Dict[str, Any]:
    solution = getattr(run_state, "working_solution", None)
    if solution is None:
        solution = AssignmentSolution.empty_from_instance(instance, put_all_unassigned=True)
    return solution_summary(
        solution,
        instance,
        run_state.config,
        objective_terms=getattr(run_state, "global_objective_terms", None),
    )


def _phase(summary: Mapping[str, Any]) -> str:
    q = dict(summary.get("quality", {}) or {})
    return (
        "service_recovery"
        if float(q.get("missed_priority", 0.0) or 0.0) > 1e-12
        or int(q.get("unassigned_count", summary.get("unassigned_count", 0)) or 0) > 0
        else "energy_refinement"
    )


def _state_signals(summary: Mapping[str, Any], best: Mapping[str, Any] | None, run_state: Any, *, n_tasks: int) -> Dict[str, Any]:
    q = dict(summary.get("quality", {}) or {})
    bq = dict((best or {}).get("quality", {}) or {}) if best else {}
    metric_order = [str(item.get("metric")) for item in getattr(run_state, "global_objective_terms", []) or []]
    if not metric_order:
        metric_order = list(QUALITY_METRICS)
    used = int(getattr(run_state.global_budget, "iters_used", 0) or 0)
    limit = max(1, int(getattr(run_state.global_budget, "max_iters", 1) or 1))
    since_best = int(getattr(run_state.run_progress, "iters_since_global_best", 0) or 0)
    gap = {
        name: round(float(q.get(name, 0.0) or 0.0) - float(bq.get(name, q.get(name, 0.0)) or 0.0), 6)
        for name in metric_order
    }
    assigned_count = int(summary.get("assigned_count", 0) or 0)
    unassigned_count = int(summary.get("unassigned_count", 0) or 0)
    return {
        "phase": _phase(summary),
        "task_count": int(n_tasks),
        "metric_order": metric_order,
        "current_quality": {name: float(q.get(name, 0.0) or 0.0) for name in metric_order},
        "run_best_quality": {name: float(bq.get(name, 0.0) or 0.0) for name in metric_order} if bq else None,
        "current_minus_run_best": gap if bq else None,
        "current_matches_run_best": bool(bq) and all(abs(value) <= 1e-9 for value in gap.values()),
        "hard_feasible": bool(dict(summary.get("hard_constraints", {}) or {}).get("is_feasible", False)),
        "assigned_task_count": assigned_count,
        "unassigned_task_count": unassigned_count,
        "energy_per_assigned_task": round(
            float(q.get("energy_total", 0.0) or 0.0) / max(1, assigned_count),
            6,
        ),
        "missed_priority_per_unassigned_task": round(
            float(q.get("missed_priority", 0.0) or 0.0)
            / max(1, unassigned_count),
            6,
        ),
        "global_trial": used,
        "budget_fraction_used": round(used / limit, 6),
        "trials_since_run_best_update": since_best,
    }


def _bottleneck_signals(
    landscape: Mapping[str, Any],
    contract: RuntimeContract,
    *,
    instance: Any,
) -> Dict[str, Any]:
    insertion = dict(landscape.get("insertion_facts", {}) or {})
    rows = [dict(v) for v in insertion.get("unassigned_task_facts", []) or [] if isinstance(v, Mapping)]
    rows.sort(key=lambda row: (int(row.get("feasible_position_count", 0) or 0), int(row.get("task_id", 0) or 0)))
    counts = [int(row.get("feasible_position_count", 0) or 0) for row in rows]
    capable_agent_counts = [
        sum(
            1
            for agent in instance.agents
            if set(instance.task_by_id(int(row["task_id"])).skill_req)
            <= set(agent.skills)
        )
        for row in rows
    ]
    route_rows = [dict(v) for v in landscape.get("candidate_routes", []) or [] if isinstance(v, Mapping)]
    # Preserve route-landscape terminology but avoid copying a large route table into the LLM context.
    route_sample = route_rows[:_MAX_ROUTE_ROWS]
    expansions = {}
    if not expansions:
        expansions = dict(contract.search_focus_expansions or {})
    service_focus = dict(expansions.get("service_bottleneck", {}) or {})
    at_most_one = sum(1 for value in counts if value <= 1)
    route_margins = [
        float(row.get("bottleneck_margin", 0.0) or 0.0)
        for row in route_rows
    ]
    route_time_slacks = [
        float(row["min_time_slack"])
        for row in route_rows
        if row.get("min_time_slack") is not None
    ]
    route_energy_slacks = [
        float(row["energy_slack_ratio"])
        for row in route_rows
        if row.get("energy_slack_ratio") is not None
    ]
    return {
        "unassigned_task_count": len(rows),
        "zero_feasible_task_count": sum(1 for value in counts if value <= 0),
        "at_most_one_feasible_task_count": at_most_one,
        "at_most_one_feasible_ratio": round(
            at_most_one / max(1, len(rows)), 6
        ),
        "feasible_position_count_distribution": _distribution(counts),
        "route_bottleneck_margin_distribution": _float_distribution(
            route_margins
        ),
        "route_time_slack_distribution": _float_distribution(
            route_time_slacks
        ),
        "route_energy_slack_ratio_distribution": _float_distribution(
            route_energy_slacks
        ),
        "capable_agent_count_distribution": _distribution(capable_agent_counts),
        "single_capability_task_count": sum(
            value <= 1 for value in capable_agent_counts
        ),
        "hardest_unassigned_tasks": [
            {
                "task_id": int(row.get("task_id", 0) or 0),
                "feasible_position_count": int(row.get("feasible_position_count", 0) or 0),
                "features": {str(k): round(float(v), 6) for k, v in dict(row.get("features", {}) or {}).items()},
            }
            for row in rows[:_MAX_TASK_ROWS]
        ],
        "service_bottleneck_target_ids": [int(v) for v in service_focus.get("focus_task_ids", []) or []],
        "route_pressure_sample": route_sample,
    }










def _step_boundary(run_state: Any, contract: RuntimeContract) -> Dict[str, Any]:
    from .operators.destroy import compute_destroy_strength
    expansion = dict(contract.search_focus_expansions.get("global", {}) or {})
    boundary = {
        "active_search_policy": {
            "focus_id": "global",
            "focus_task_ids": [int(v) for v in expansion.get("focus_task_ids", []) or []],
            "destroy_feature_priority": [],
            "task_feature_priority": [],
        },
        "run_budget_total": {"iters": run_state.global_budget.max_iters},
        "run_budget_remaining": dict(run_state.global_budget.remaining()),
        "fixed_step_execution_budget_trials": min(100, int(contract.remaining.get("iters", 0) or 0)),
        "legal_destroy_operators": list(contract.controls.get("destroy_operators", []) or []),
        "legal_repair_operators": list(contract.controls.get("repair_operators", []) or []),
        "legal_remove_ratios": list(contract.controls.get("remove_ratio_options", []) or []),
        "actual_destroy_counts_at_current_state": [
            {"remove_ratio":ratio,"target_removed_tasks":compute_destroy_strength(run_state.working_solution,ratio).target_k}
            for ratio in contract.controls.get('remove_ratio_options',[])],
        "destroy_count_semantics": "The unchanged solver uses round(remove_ratio * min(assigned_count,100)), capped to available assigned tasks and recomputed each trial. The displayed counts describe the state before the next action; they can change within its 100 trials. This is not an additional control.",
        "legal_selectors": list(contract.controls.get("selector_modes", []) or []),
        "legal_acceptance_modes": list(contract.controls.get("acceptance_modes", []) or []),
        "ownership": "Step only: Destroy/Repair + remove ratio + selectors + acceptance.",
    }
    if contract.controls.get("step_focus_options"):
        boundary["active_search_policy"].pop("focus_id")
        boundary["active_search_policy"].pop("focus_task_ids")
        boundary["legal_focus_ids"] = list(contract.controls["step_focus_options"])
        boundary["available_focus_targets"] = {
            name: list(contract.search_focus_expansions[name].get("focus_task_ids", []))
            for name in boundary["legal_focus_ids"]
        }
        boundary["focus_owner"] = "step"
        boundary["focus_random_weight"] = SOFT_FOCUS_RANDOM_WEIGHT
        boundary["ownership"] = "Step only: one focus_id + Destroy/Repair + remove ratio + exposed selectors + acceptance. Both feature vectors stay zero."
    return boundary


def _distribution(values: List[int]) -> Dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(int(v) for v in values)
    return {
        "count": len(ordered),
        "min": int(ordered[0]),
        "mean": round(sum(ordered) / len(ordered), 6),
        "median": float(median(ordered)),
        "max": int(ordered[-1]),
    }


def _float_distribution(values: List[float]) -> Dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
        "median": round(float(median(ordered)), 6),
        "max": round(ordered[-1], 6),
    }


def _block(block_id: str, block_type: str, summary: str, signals: Mapping[str, Any]) -> Dict[str, Any]:
    return {"block_id": block_id, "block_type": block_type, "summary": summary, "signals": dict(signals)}


def _validate_observation_layout(blocks: List[Mapping[str, Any]]) -> None:
    ids = [str(block.get("block_id", "")) for block in blocks]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("observation block_id values must be non-empty and unique")
    if len(blocks) not in (3, 4):
        raise ValueError("agent observation must contain three live blocks and at most one history block")
    for block in blocks:
        if not isinstance(block.get("signals"), Mapping):
            raise ValueError(f"observation block {block.get('block_id')} signals must be an object")
