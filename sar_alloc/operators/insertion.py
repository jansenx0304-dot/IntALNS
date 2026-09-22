"""Random pending-task selection with bounded focus bias and two position operators.

Insertion landscape features are observation facts, never Agent control weights.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..evaluator import (
    QualityMetricsCache,
    build_lex_key,
    evaluate,
    evaluate_quality_metrics,
    route_is_feasible,
)
from ..models import Instance
from ..solution import AssignmentSolution
from .features import (
    InsertionRouteContext,
    basic_insertion_feasibility_filter,
    build_insertion_route_contexts,
    feasibility_filtered_positions_for_agent,
    insertion_lower_bound_filter,
)
from .focus import partition_repair_tasks
from .route_metrics import (
    RouteInsertionContext,
    RouteMetrics,
    build_route_insertion_context,
    simulate_insertion_route,
    simulate_route,
)
from .selector import select_item
from .types import (
    ALNS_REPAIR_OPERATOR_NAMES,
    INITIAL_INSERTION_OPERATOR_NAMES,
    INSERTION_TASK_FEATURE_NAMES,
    InsertPosition,
    InsertionPolicy,
    LandscapeFeatures,
)

_EPS = 1e-9
_DEFAULT_OBJECTIVE_TERMS: Tuple[Mapping[str, str], ...] = (
    {"metric": "missed_priority", "direction": "min"},
    {"metric": "unassigned_count", "direction": "min"},
)


@dataclass(frozen=True, slots=True)
class InsertionContext:
    kind: str = "alns"
    focus_task_ids: Tuple[int, ...] = ()
    focus_soft_greedy_bonus: float = 1.0
    focus_soft_random_weight: float = 1.5


@dataclass(frozen=True, slots=True)
class InsertionCandidate:
    tid: int
    agent_id: int
    position: int
    objective_key: Tuple[float, ...]
    delta_distance: float
    delta_energy: float
    bottleneck_margin: float
    relatedness: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_id": int(self.tid),
            "agent_id": int(self.agent_id),
            "position": int(self.position),
            "objective_key": [float(value) for value in self.objective_key],
            "delta_distance": float(self.delta_distance),
            "delta_energy": float(self.delta_energy),
            "bottleneck_margin": float(self.bottleneck_margin),
            "relatedness": float(self.relatedness),
        }


@dataclass(frozen=True, slots=True)
class TaskInsertionStats:
    tid: int
    candidates: Tuple[InsertionCandidate, ...]
    candidate_count: int
    feasible_count: int
    features: LandscapeFeatures = LandscapeFeatures()
    task_score: float = 0.0

    @property
    def sorted_feasible_candidates(self) -> Tuple[InsertionCandidate, ...]:
        return self.candidates


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    success: bool
    task_id: int
    chosen_agent_id: Optional[int] = None
    chosen_position: Optional[int] = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class _CandidateKeyContext:
    base_quality: Mapping[str, float]
    objective_specs: Tuple[Tuple[str, float], ...]


# ---------------------------------------------------------------------------
# Candidate generation and task features
# ---------------------------------------------------------------------------



def _effective_objective_terms(
    objective_terms: Optional[Sequence[Mapping[str, Any]]],
) -> Tuple[Mapping[str, Any], ...]:
    terms = tuple(objective_terms or _DEFAULT_OBJECTIVE_TERMS)
    if not terms:
        raise ValueError("repair requires at least one run-global objective term")
    return terms


def _route_metrics_map(sol: AssignmentSolution, instance: Instance, config: Config) -> Dict[int, RouteMetrics]:
    return {
        int(aid): simulate_route(instance, config, int(aid), sol.routes.get(int(aid), []))
        for aid in instance.all_agent_ids()
    }


def _candidate_quality_metrics(
    *,
    sol: AssignmentSolution,
    tid: int,
    aid: int,
    after_route_metrics: RouteMetrics,
    before_route_metrics: Mapping[int, RouteMetrics],
    base_quality: Mapping[str, float],
    instance: Instance,
) -> Dict[str, float]:
    task = instance.task_by_id(int(tid))
    before = before_route_metrics[int(aid)]
    metrics = {str(name): float(value) for name, value in base_quality.items()}
    if int(tid) in sol.unassigned:
        metrics["missed_priority"] = float(metrics.get("missed_priority", 0.0)) - float(task.priority)
        metrics["unassigned_count"] = float(metrics.get("unassigned_count", 0.0)) - 1.0
    metrics["energy_total"] = float(metrics.get("energy_total", 0.0)) + float(after_route_metrics.energy - before.energy)
    return metrics


def compute_insertion_candidate(
    sol: AssignmentSolution,
    tid: int,
    position: InsertPosition,
    instance: Instance,
    config: Config,
    objective_terms: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    base_quality: Optional[Mapping[str, float]] = None,
    route_metrics_before: Optional[Mapping[int, RouteMetrics]] = None,
    route_insertion_context: Optional[RouteInsertionContext] = None,
    position_prefiltered: bool = False,
    defer_objective_key: bool = False,
) -> Optional[InsertionCandidate]:
    aid = int(position.agent_id)
    route = list(sol.routes.get(aid, []))
    route.insert(int(position.position), int(tid))
    if not position_prefiltered and not route_is_feasible(aid, route, instance, config):
        return None

    terms = _effective_objective_terms(objective_terms)
    before_map = dict(route_metrics_before or _route_metrics_map(sol, instance, config))
    quality = dict(
        base_quality
        or evaluate_quality_metrics(sol, instance, config)
    )
    after_metrics = (
        simulate_insertion_route(
            instance,
            config,
            route_insertion_context,
            int(tid),
            int(position.position),
        )
        if route_insertion_context is not None
        else simulate_route(instance, config, aid, route)
    )
    if position_prefiltered and (
        float(after_metrics.min_time_slack) < -_EPS
        or float(after_metrics.energy_slack_ratio) < -_EPS
    ):
        return None
    before_metrics = before_map[aid]
    if defer_objective_key:
        objective_key: Tuple[float, ...] = ()
    else:
        after_quality = _candidate_quality_metrics(
            sol=sol,
            tid=int(tid),
            aid=aid,
            after_route_metrics=after_metrics,
            before_route_metrics=before_map,
            base_quality=quality,
            instance=instance,
        )
        objective_key = tuple(float(v) for v in build_lex_key(after_quality, terms))
    return InsertionCandidate(
        tid=int(tid),
        agent_id=aid,
        position=int(position.position),
        objective_key=objective_key,
        delta_distance=float(after_metrics.distance - before_metrics.distance),
        delta_energy=float(after_metrics.energy - before_metrics.energy),
        bottleneck_margin=float(after_metrics.bottleneck_margin),
        relatedness=float(_position_relatedness(sol, int(tid), position, instance)),
    )


def collect_task_insertion_stats(
    sol: AssignmentSolution,
    tid: int,
    instance: Instance,
    config: Config,
    *,
    objective_terms: Optional[Sequence[Mapping[str, Any]]] = None,
    base_eval: Optional[Any] = None,
    base_quality: Optional[Mapping[str, float]] = None,
    route_metrics_before: Optional[Mapping[int, RouteMetrics]] = None,
    route_contexts_before: Optional[Mapping[int, InsertionRouteContext]] = None,
    route_insertion_contexts_before: Optional[Mapping[int, RouteInsertionContext]] = None,
) -> Tuple[TaskInsertionStats, Dict[str, Any]]:
    base_quality_map = dict(
        base_quality
        or getattr(base_eval, "quality_metrics", None)
        or evaluate_quality_metrics(sol, instance, config)
    )
    route_map = dict(route_metrics_before or _route_metrics_map(sol, instance, config))
    route_contexts = dict(
        route_contexts_before or build_insertion_route_contexts(sol, instance)
    )
    route_insertion_contexts = dict(route_insertion_contexts_before or {})
    candidates: List[InsertionCandidate] = []
    generated = 0
    for aid in instance.all_agent_ids():
        route_candidates, route_generated = _collect_task_route_candidates(
            sol=sol,
            tid=int(tid),
            aid=int(aid),
            instance=instance,
            config=config,
            objective_terms=objective_terms,
            base_quality=base_quality_map,
            route_metrics_before=route_map,
            route_context=route_contexts[int(aid)],
            route_insertion_context=(
                route_insertion_contexts.get(int(aid))
                or build_route_insertion_context(
                    instance, config, int(aid), sol.routes.get(int(aid), [])
                )
            ),
        )
        generated += int(route_generated)
        candidates.extend(route_candidates)
    candidates.sort(key=lambda item: (item.objective_key, item.delta_energy, item.agent_id, item.position))
    stats = TaskInsertionStats(
        tid=int(tid),
        candidates=tuple(candidates),
        candidate_count=int(generated),
        feasible_count=len(candidates),
    )
    return stats, {
        "positions_generated": int(generated),
        "positions_strict_checked": int(generated),
        "strict_feasible_positions": len(candidates),
    }


def _collect_task_route_candidates(
    *,
    sol: AssignmentSolution,
    tid: int,
    aid: int,
    instance: Instance,
    config: Config,
    objective_terms: Optional[Sequence[Mapping[str, Any]]],
    base_quality: Mapping[str, float],
    route_metrics_before: Mapping[int, RouteMetrics],
    route_context: InsertionRouteContext,
    route_insertion_context: Optional[RouteInsertionContext] = None,
    defer_objective_key: bool = False,
) -> Tuple[Tuple[InsertionCandidate, ...], int]:
    positions = feasibility_filtered_positions_for_agent(
        sol,
        int(tid),
        int(aid),
        instance,
        config,
        route_context=route_context,
    )
    candidates: List[InsertionCandidate] = []
    for position in positions:
        candidate = compute_insertion_candidate(
            sol,
            int(tid),
            position,
            instance,
            config,
            objective_terms,
            base_quality=base_quality,
            route_metrics_before=route_metrics_before,
            route_insertion_context=route_insertion_context,
            position_prefiltered=True,
            defer_objective_key=bool(defer_objective_key),
        )
        if candidate is not None:
            candidates.append(candidate)
    candidates.sort(key=lambda item: (item.objective_key, item.delta_energy, item.agent_id, item.position))
    return tuple(candidates), len(positions)


def _refresh_candidate_objective_key(
    candidate: InsertionCandidate,
    *,
    sol: AssignmentSolution,
    key_context: _CandidateKeyContext,
    instance: Instance,
) -> InsertionCandidate:
    aid = int(candidate.agent_id)
    tid = int(candidate.tid)
    base = key_context.base_quality
    task = instance.task_by_id(tid)
    is_unassigned = tid in sol.unassigned
    values = {
        "missed_priority": float(base.get("missed_priority", 0.0))
        - (float(task.priority) if is_unassigned else 0.0),
        "unassigned_count": float(base.get("unassigned_count", 0.0))
        - (1.0 if is_unassigned else 0.0),
        "energy_total": float(base.get("energy_total", 0.0))
        + float(candidate.delta_energy),
    }
    return replace(
        candidate,
        objective_key=tuple(
            float(direction * float(values.get(metric, base.get(metric, 0.0))))
            for metric, direction in key_context.objective_specs
        ),
    )


def _build_candidate_key_context(
    *,
    base_quality: Mapping[str, float],
    objective_terms: Sequence[Mapping[str, Any]],
) -> _CandidateKeyContext:
    objective_specs = tuple(
        (
            str(term["metric"]),
            -1.0 if str(term.get("direction", "min")) == "max" else 1.0,
        )
        for term in objective_terms
    )
    return _CandidateKeyContext(
        base_quality=base_quality,
        objective_specs=objective_specs,
    )


def _operating_insertion_cost(candidate: InsertionCandidate) -> float:
    """Pure marginal route cost used only for task-ordering features."""

    return float(candidate.delta_distance + candidate.delta_energy)


def score_candidate_tasks(
    stats_by_tid: Mapping[int, TaskInsertionStats],
    insertion_policy: InsertionPolicy,
) -> Dict[int, TaskInsertionStats]:
    """Score pending tasks using only the features enabled by this action.

    Plain Basic ALNS does not call cross-task semantic scoring: it samples one
    pending task uniformly and lets a primitive Repair choose only that task's
    insertion position.  The observation builder may call this routine after its middleware has
    supplied a bounded guidance/focus pool and activated task features.  The
    selected public Repair still owns insertion-position choice.
    """

    stats = {int(tid): row for tid, row in stats_by_tid.items()}
    active = [
        name for name in INSERTION_TASK_FEATURE_NAMES
        if abs(float(insertion_policy.task_feature_weights.get(name, 0))) > 1e-12
    ]
    raw_rows: Dict[int, Dict[str, float]] = {int(tid): {} for tid in stats}

    if {"cheapest_score", "regret_score"} & set(active):
        best_cost_by_tid: Dict[int, float] = {}
        regret_by_tid: Dict[int, float] = {}
        single_position_tasks: list[int] = []
        for tid, row in stats.items():
            costs = sorted(_operating_insertion_cost(candidate) for candidate in row.candidates)
            if costs:
                best_cost_by_tid[int(tid)] = float(costs[0])
            if len(costs) >= 2:
                regret_by_tid[int(tid)] = max(0.0, float(costs[1] - costs[0]))
            elif len(costs) == 1:
                single_position_tasks.append(int(tid))
            else:
                regret_by_tid[int(tid)] = 0.0
        if "regret_score" in active:
            single_position_regret = max(regret_by_tid.values(), default=0.0) + 1.0
            for tid in single_position_tasks:
                regret_by_tid[int(tid)] = float(single_position_regret)
            for tid in stats:
                raw_rows[int(tid)]["regret_score"] = float(regret_by_tid.get(int(tid), 0.0))
        if "cheapest_score" in active:
            cheapest_rank: Dict[int, float] = {}
            ordered = sorted(best_cost_by_tid.items(), key=lambda item: (item[1], item[0]))
            denom = max(1, len(ordered) - 1)
            for rank, (tid, _cost) in enumerate(ordered):
                cheapest_rank[int(tid)] = (
                    1.0 - float(rank) / float(denom) if len(ordered) > 1 else 1.0
                )
            for tid in stats:
                raw_rows[int(tid)]["cheapest_score"] = float(cheapest_rank.get(int(tid), 0.0))

    if "scarcity_score" in active:
        for tid, row in stats.items():
            raw_rows[int(tid)]["scarcity_score"] = 1.0 / (1.0 + float(row.feasible_count))

    normalized = _normalize_feature_rows(raw_rows, active) if active else {
        int(tid): {} for tid in stats
    }
    out: Dict[int, TaskInsertionStats] = {}
    for tid, row in stats.items():
        values = normalized[int(tid)]
        features = LandscapeFeatures(
            **{name: float(values.get(name, 0.0)) for name in INSERTION_TASK_FEATURE_NAMES}
        )
        score = sum(
            float(insertion_policy.task_feature_weights.get(name, 0))
            * float(values.get(name, 0.0))
            for name in active
        )
        out[int(tid)] = replace(row, features=features, task_score=float(score))
    return out


def _normalize_feature_rows(
    rows: Mapping[int, Mapping[str, float]], names: Sequence[str]
) -> Dict[int, Dict[str, float]]:
    out = {int(tid): {str(name): 0.0 for name in names} for tid in rows}
    for name in names:
        values = [float(row.get(name, 0.0)) for row in rows.values()]
        lo = min(values, default=0.0)
        hi = max(values, default=0.0)
        for tid, row in rows.items():
            value = float(row.get(name, 0.0))
            if hi - lo <= _EPS:
                out[int(tid)][name] = 0.0
            else:
                out[int(tid)][name] = (value - lo) / (hi - lo)
    return out


# ---------------------------------------------------------------------------
# Repair position operators
# ---------------------------------------------------------------------------

def _choose_position(
    operator_name: str,
    stats: TaskInsertionStats,
    rng: random.Random,
) -> Optional[InsertionCandidate]:
    candidates = list(stats.candidates)
    if not candidates:
        return None
    if operator_name == "best_insertion":
        return min(candidates, key=lambda item: (item.objective_key, item.delta_energy, item.agent_id, item.position))
    if operator_name == "random_insertion":
        # Primitive diversification repair: uniformly choose one strict-feasible
        # position for the already selected pending task.
        return rng.choice(candidates)
    raise ValueError(f"unknown repair operator: {operator_name!r}")


def _apply_direct_candidate(
    sol: AssignmentSolution,
    candidate: InsertionCandidate,
) -> RepairOutcome:
    sol.add_task(candidate.agent_id, candidate.tid, position=candidate.position)
    return RepairOutcome(
        success=True,
        task_id=int(candidate.tid),
        chosen_agent_id=int(candidate.agent_id),
        chosen_position=int(candidate.position),
    )




# ---------------------------------------------------------------------------
# Complete repair loop
# ---------------------------------------------------------------------------

def run_insertion_kernel(
    partial_solution: AssignmentSolution,
    candidate_tasks: Sequence[int],
    insertion_policy: InsertionPolicy,
    context: InsertionContext,
    *,
    selected_operator_name: Optional[str] = None,
    instance: Instance,
    config: Config,
    rng: random.Random,
    objective_terms: Optional[Sequence[Mapping[str, Any]]] = None,
) -> AssignmentSolution:
    terms = _effective_objective_terms(objective_terms)
    operator_name = _resolve_fixed_insertion_operator(insertion_policy, context, selected_operator_name, rng)
    out = partial_solution.clone_search_state()
    pending = {
        int(tid) for tid in candidate_tasks if int(tid) not in out.all_assigned_tasks()
    }
    out.unassigned.update(pending)
    out.normalize(instance)
    diagnostics = _new_diagnostics(context, insertion_policy, operator_name, len(pending))
    started = time.perf_counter()
    agent_ids = tuple(int(aid) for aid in instance.all_agent_ids())
    focus_ids = set(int(x) for x in context.focus_task_ids)
    route_candidate_cache: Dict[int, Dict[int, Tuple[InsertionCandidate, ...]]] = {}
    route_generated_cache: Dict[int, Dict[int, int]] = {}
    dirty_agent_ids: set[int] = set(agent_ids)
    route_map: Dict[int, RouteMetrics] = {}
    route_insertion_contexts: Dict[int, RouteInsertionContext] = {}
    route_context_cache: Dict[int, InsertionRouteContext] = {}
    quality_cache = QualityMetricsCache()

    while pending:
        base_quality = quality_cache.evaluate(
            out, instance, config, dirty_agent_ids=dirty_agent_ids
        )
        if dirty_agent_ids:
            refreshed_insertion_contexts = build_insertion_route_contexts(
                out,
                instance,
                agent_ids=dirty_agent_ids,
            )
            route_context_cache.update(refreshed_insertion_contexts)
            for aid in sorted(dirty_agent_ids):
                route_insertion_context = build_route_insertion_context(
                    instance, config, int(aid), out.routes.get(int(aid), [])
                )
                route_insertion_contexts[int(aid)] = route_insertion_context
                route_map[int(aid)] = route_insertion_context.base_metrics
        route_contexts = {
            int(aid): route_context_cache[int(aid)] for aid in dirty_agent_ids
        }
        key_context = _build_candidate_key_context(
            base_quality=base_quality,
            objective_terms=terms,
        )
        # The current protocol always keeps the ordinary pending-task pool intact.
        # Focus is applied only at final task selection as a bounded preference.
        selectable_pending = set(pending)
        diagnostics["task_selection_scope"] = "common_pending_pool"

        # Primitive random task order is intentionally cheap and semantics-free:
        # choose one pending task first, then enumerate insertion positions only
        # for that task.  This is the ordinary Basic path.  Guided/semantic
        # actions activate feature scoring and therefore analyze their bounded
        # selectable pool instead.
        random_task_only = bool(
            insertion_policy.task_selector_mode == "random"
            and all(
                abs(float(insertion_policy.task_feature_weights.get(name, 0.0) or 0.0)) <= 1e-12
                for name in INSERTION_TASK_FEATURE_NAMES
            )
        )
        preselected_tid = None
        if random_task_only and selectable_pending and not focus_ids:
            preselected_tid = int(rng.choice(sorted(selectable_pending)))
            analysis_pending = {preselected_tid}
            diagnostics["primitive_random_task_draw_count"] += 1
        else:
            analysis_pending = selectable_pending
        stats_by_tid: Dict[int, TaskInsertionStats] = {}
        for tid in sorted(analysis_pending):
            by_agent = route_candidate_cache.setdefault(int(tid), {})
            generated_by_agent = route_generated_cache.setdefault(int(tid), {})
            refresh_agent_ids = set(dirty_agent_ids)
            refresh_agent_ids.update(aid for aid in agent_ids if aid not in by_agent)
            for aid in sorted(refresh_agent_ids):
                candidates, generated = _collect_task_route_candidates(
                    sol=out,
                    tid=int(tid),
                    aid=int(aid),
                    instance=instance,
                    config=config,
                    objective_terms=terms,
                    base_quality=base_quality,
                    route_metrics_before=route_map,
                    route_context=route_contexts.get(int(aid))
                    or route_context_cache[int(aid)],
                    route_insertion_context=route_insertion_contexts[int(aid)],
                    defer_objective_key=True,
                )
                by_agent[int(aid)] = tuple(candidates)
                generated_by_agent[int(aid)] = int(generated)
                diagnostics["route_task_recomputations"] += 1
                diagnostics["positions_generated"] += int(generated)
                diagnostics["positions_strict_checked"] += int(generated)
                diagnostics["strict_feasible_positions"] += len(candidates)

            refreshed_candidates = [
                _refresh_candidate_objective_key(
                    candidate,
                    sol=out,
                    key_context=key_context,
                    instance=instance,
                )
                for aid in agent_ids
                for candidate in by_agent.get(int(aid), ())
            ]
            refreshed_candidates.sort(
                key=lambda item: (item.objective_key, item.delta_energy, item.agent_id, item.position)
            )
            stats_by_tid[int(tid)] = TaskInsertionStats(
                tid=int(tid),
                candidates=tuple(refreshed_candidates),
                candidate_count=sum(
                    int(generated_by_agent.get(int(aid), 0)) for aid in agent_ids
                ),
                feasible_count=len(refreshed_candidates),
            )
            diagnostics["tasks_analyzed"] += 1
        scored = score_candidate_tasks(stats_by_tid, insertion_policy)
        _accumulate_feature_usage(diagnostics, scored)
        active_ids = list(scored.keys())
        if not active_ids:
            break
        if preselected_tid is not None:
            selected_stats = scored[int(preselected_tid)]
        else:
            baseline_rng = random.Random(); baseline_rng.setstate(rng.getstate())
            baseline_stats = select_item(
                [scored[int(tid)] for tid in active_ids],
                mode=insertion_policy.task_selector_mode,
                score_getter=lambda item: item.task_score,
                tie_key=lambda item: int(item.tid),
                rng=baseline_rng,
            )
            selected_stats = _select_task_with_soft_focus(
                [scored[int(tid)] for tid in active_ids],
                mode=insertion_policy.task_selector_mode,
                focus_ids=focus_ids,
                greedy_bonus=float(context.focus_soft_greedy_bonus),
                random_weight=float(context.focus_soft_random_weight),
                rng=rng,
            )
            if int(selected_stats.tid) != int(baseline_stats.tid):
                diagnostics["focus_changed_selection_count"] += 1
            diagnostics["no_focus_selected_task_id"] = int(baseline_stats.tid)
        if int(selected_stats.tid) in focus_ids:
            diagnostics["focus_matched_task_selection_count"] += 1
            if int(selected_stats.feasible_count) > 0:
                diagnostics["focus_feasible_task_count"] += 1
        tid = int(selected_stats.tid)
        diagnostics["task_selection_order"].append(tid)
        if tid in focus_ids:
            _append_unique(diagnostics["search_focus_tasks_attempted"], tid)

        candidate = _choose_position(operator_name, selected_stats, rng)
        if candidate is not None:
            outcome = _apply_direct_candidate(out, candidate)
        else:
            outcome = RepairOutcome(False, tid, reason="no_strict_feasible_position")

        pending.discard(tid)
        route_candidate_cache.pop(int(tid), None)
        route_generated_cache.pop(int(tid), None)
        if outcome.success:
            diagnostics["inserted_count"] += 1
            diagnostics["operator_task_selection_count"][operator_name] += 1
            if tid in focus_ids:
                _append_unique(diagnostics["search_focus_tasks_inserted"], tid)
            diagnostics["last_selected_task"] = tid
            diagnostics["last_selected_position"] = {
                "agent_id": outcome.chosen_agent_id,
                "position": outcome.chosen_position,
            }
            if outcome.chosen_agent_id is not None:
                dirty_agent_ids = {int(outcome.chosen_agent_id)}
            else:
                dirty_agent_ids = set(agent_ids)
        else:
            diagnostics["failure_breakdown"][outcome.reason] = (
                diagnostics["failure_breakdown"].get(outcome.reason, 0) + 1
            )
            dirty_agent_ids = set()

    diagnostics["unassigned_after"] = len(out.unassigned)
    diagnostics["failed_count"] = len(out.unassigned)
    focus_set = set(int(tid) for tid in context.focus_task_ids)
    diagnostics["search_focus_tasks_failed"] = sorted(focus_set & set(out.unassigned))[:20]
    diagnostics["top_failed_tasks"] = [
        {
            "task_id": int(tid),
            "priority": float(instance.task_by_id(int(tid)).priority),
            "reason": "not_inserted",
        }
        for tid in sorted(out.unassigned, key=lambda value: (-float(instance.task_by_id(int(value)).priority), int(value)))[:20]
    ]
    diagnostics["time_ms"] = round((time.perf_counter() - started) * 1000.0, 4)
    out.solver_diagnostics = dict(out.solver_diagnostics or {})
    out.solver_diagnostics["last_insertion"] = diagnostics
    out.normalize(instance)
    return out


def select_insertion_operator(insertion_policy: InsertionPolicy, rng: random.Random) -> str:
    positive = [(str(name), float(weight)) for name, weight in insertion_policy.operator_weights.items() if float(weight) > 0.0]
    if not positive:
        raise ValueError("all repair operators are disabled")
    total = sum(weight for _, weight in positive)
    threshold = rng.random() * total
    cumulative = 0.0
    for name, weight in positive:
        cumulative += weight
        if cumulative >= threshold:
            return name
    return positive[-1][0]


def _resolve_fixed_insertion_operator(
    policy: InsertionPolicy,
    context: InsertionContext,
    selected_operator_name: Optional[str],
    rng: random.Random,
) -> str:
    name = str(selected_operator_name or select_insertion_operator(policy, rng))
    allowed = INITIAL_INSERTION_OPERATOR_NAMES if context.kind == "initial" else ALNS_REPAIR_OPERATOR_NAMES
    if name not in allowed:
        raise ValueError(f"repair operator {name!r} is not valid for {context.kind}")
    if float(policy.operator_weights.get(name, 0.0)) <= 0.0:
        raise ValueError(f"selected repair operator {name!r} is disabled")
    return name


# ---------------------------------------------------------------------------
# Observation support
# ---------------------------------------------------------------------------

def build_insertion_landscape(
    solution: AssignmentSolution,
    instance: Instance,
    config: Config,
    *,
    objective_terms: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    default_policy = InsertionPolicy(
        operator_weights={name: 1 for name in ALNS_REPAIR_OPERATOR_NAMES},
        task_feature_weights={name: 1 for name in INSERTION_TASK_FEATURE_NAMES},
        task_selector_mode="random",
    )
    base_quality = evaluate_quality_metrics(solution, instance, config)
    route_contexts = build_insertion_route_contexts(solution, instance)
    route_insertion_contexts = {
        int(aid): build_route_insertion_context(
            instance, config, int(aid), solution.routes.get(int(aid), [])
        )
        for aid in instance.all_agent_ids()
    }
    route_map = {
        int(aid): ctx.base_metrics for aid, ctx in route_insertion_contexts.items()
    }
    stats: Dict[int, TaskInsertionStats] = {}
    for tid in sorted(int(tid) for tid in solution.unassigned):
        row, _diag = collect_task_insertion_stats(
            solution,
            tid,
            instance,
            config,
            objective_terms=objective_terms,
            base_quality=base_quality,
            route_metrics_before=route_map,
            route_contexts_before=route_contexts,
            route_insertion_contexts_before=route_insertion_contexts,
        )
        stats[tid] = row
    scored = score_candidate_tasks(stats, default_policy)
    rows = [
        {
            "task_id": int(tid),
            "feasible_position_count": int(row.feasible_count),
            "features": {
                name: float(getattr(row.features, name))
                for name in INSERTION_TASK_FEATURE_NAMES
            },
        }
        for tid, row in sorted(scored.items())
    ]
    counts = [int(row["feasible_position_count"]) for row in rows]
    return {
        "insertion_facts": {
            "unassigned_task_count": len(rows),
            "unassigned_task_facts": rows,
            "feasible_position_count_distribution": _distribution(counts),
        },
    }


def _select_task_with_soft_focus(
    items: Sequence[TaskInsertionStats],
    *,
    mode: str,
    focus_ids: set[int],
    greedy_bonus: float,
    random_weight: float,
    rng: random.Random,
) -> TaskInsertionStats:
    rows = list(items)
    if not rows:
        raise ValueError("task selection requires at least one candidate")
    if not focus_ids:
        return select_item(rows, mode=mode, score_getter=lambda item: item.task_score, tie_key=lambda item: int(item.tid), rng=rng)
    weights = [max(1.0, float(random_weight)) if int(item.tid) in focus_ids else 1.0 for item in rows]
    threshold = rng.random() * sum(weights); acc = 0.0
    for item, weight in zip(rows, weights):
        acc += weight
        if acc >= threshold:
            return item
    return rows[-1]


# ---------------------------------------------------------------------------
# Diagnostics helpers
# ---------------------------------------------------------------------------

def _new_diagnostics(
    context: InsertionContext,
    policy: InsertionPolicy,
    operator_name: str,
    unassigned_before: int,
) -> Dict[str, Any]:
    return {
        "kind": str(context.kind),
        "selected_operator": str(operator_name),
        "operator_selection_scope": "one_operator_fixed_for_trial",
        "operator_trial_use": {str(operator_name): 1},
        "operator_task_selection_count": {name: 0 for name in ALNS_REPAIR_OPERATOR_NAMES},
        "task_selector_mode": str(policy.task_selector_mode),
        "task_feature_weights": dict(policy.task_feature_weights),
        "unassigned_before": int(unassigned_before),
        "unassigned_after": int(unassigned_before),
        "tasks_analyzed": 0,
        "route_task_recomputations": 0,
        "candidate_cache_mode": "exact_route_incremental",
        "positions_generated": 0,
        "positions_strict_checked": 0,
        "strict_feasible_positions": 0,
        "inserted_count": 0,
        "failed_count": 0,
        "failure_breakdown": {},
        "focus_task_ids": [int(tid) for tid in context.focus_task_ids],
        "task_selection_scope": "common_pending_pool",
        "primitive_random_task_draw_count": 0,
        "search_focus_tasks_attempted": [],
        "search_focus_tasks_inserted": [],
        "search_focus_tasks_failed": [],
        "focus_feasible_task_count": 0,
        "focus_changed_selection_count": 0,
        "focus_matched_task_selection_count": 0,
        "no_focus_selected_task_id": None,
        "task_selection_order": [],
        "feature_usage": {
            name: {"positive_count": 0, "observed_count": 0, "weighted_score_sum": 0.0}
            for name in INSERTION_TASK_FEATURE_NAMES
        },
        "ejection_count": 0,
        "ejected_task_ids": [],
        "top_failed_tasks": [],
    }


def _accumulate_feature_usage(diagnostics: Dict[str, Any], stats: Mapping[int, TaskInsertionStats]) -> None:
    for row in stats.values():
        for name in INSERTION_TASK_FEATURE_NAMES:
            bucket = diagnostics["feature_usage"][name]
            value = float(getattr(row.features, name))
            bucket["observed_count"] += 1
            bucket["positive_count"] += 1 if value > _EPS else 0
            bucket["weighted_score_sum"] += value * float(diagnostics["task_feature_weights"].get(name, 0))


def _position_relatedness(
    sol: AssignmentSolution,
    tid: int,
    position: InsertPosition,
    instance: Instance,
) -> float:
    task = instance.task_by_id(int(tid))
    route = list(sol.routes.get(int(position.agent_id), []))
    neighbor_ids: List[int] = []
    if int(position.position) > 0:
        neighbor_ids.append(int(route[int(position.position) - 1]))
    if int(position.position) < len(route):
        neighbor_ids.append(int(route[int(position.position)]))
    if not neighbor_ids:
        return 0.0
    task_center = 0.5 * (float(task.tw_start) + float(task.tw_end))
    values = []
    for neighbor_id in neighbor_ids:
        neighbor = instance.task_by_id(neighbor_id)
        spatial = float(instance.distance(task.loc, neighbor.loc))
        neighbor_center = 0.5 * (float(neighbor.tw_start) + float(neighbor.tw_end))
        temporal = abs(task_center - neighbor_center) / max(1.0, float(task.tw_end) - float(task.tw_start))
        values.append(1.0 / (1.0 + spatial + temporal))
    return sum(values) / len(values)


def _append_unique(target: List[int], value: int) -> None:
    if int(value) not in target:
        target.append(int(value))


def _distribution(values: Sequence[int]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "min": ordered[0],
        "median": ordered[(len(ordered) - 1) // 2],
        "max": ordered[-1],
    }


__all__ = [
    "InsertionCandidate",
    "InsertionContext",
    "RepairOutcome",
    "TaskInsertionStats",
    "build_insertion_landscape",
    "collect_task_insertion_stats",
    "compute_insertion_candidate",
    "run_insertion_kernel",
    "score_candidate_tasks",
    "select_insertion_operator",
]
