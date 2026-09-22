"""Shared Destroy operators for ordinary ALNS and Agent-guided ALNS.

Every registered operator owns one fixed neighbourhood-generation behaviour and
never reads Agent focus or semantic score weights.  Ordinary Basic ALNS samples
one move directly from the selected operator through
:func:`sample_intrinsic_destroy_move`; no shared semantic candidate scoring is
involved. The Agent uses the same candidates and its action-local focus. Feature
scores are fixed to zero; focus does not add or remove candidate moves.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..config import Config
from ..evaluator import route_is_feasible
from ..models import Instance
from ..solution import AssignmentSolution
from .features import basic_insertion_feasibility_filter, insertion_lower_bound_filter
from .route_metrics import (
    build_route_preference_landscape,
    build_route_insertion_context,
    simulate_removal_route,
    simulate_route,
)
from .types import DESTROY_FEATURE_NAMES, DESTROY_OPERATOR_NAMES, DestroyPolicy, LandscapeFeatures

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class DestroyStrength:
    target_k: int


@dataclass(frozen=True, slots=True)
class DestroyContext:
    """Focus targets used by shared destroy scoring, never by generators."""

    focus_task_ids: Tuple[int, ...] = ()
    score_features: bool = True

    @property
    def target_task_ids(self) -> Tuple[int, ...]:
        return tuple(dict.fromkeys(int(tid) for tid in self.focus_task_ids))


@dataclass(frozen=True, slots=True)
class DestroyMove:
    operator_name: str
    shape: str
    task_ids: Tuple[int, ...]
    affected_routes: Tuple[int, ...]
    features: LandscapeFeatures
    score: float
    metadata: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operator_name": self.operator_name,
            "shape": self.shape,
            "task_ids": [int(tid) for tid in self.task_ids],
            "affected_routes": [int(aid) for aid in self.affected_routes],
            "features": self.features.as_dict(),
            "score": float(self.score),
            "metadata": dict(self.metadata),
        }


DestroyOperator = Callable[
    [
        AssignmentSolution,
        Instance,
        Config,
        DestroyPolicy,
        DestroyStrength,
        random.Random,
        Optional[DestroyContext],
    ],
    List[DestroyMove],
]


def sample_intrinsic_destroy_move(
    moves: Sequence[DestroyMove], rng: random.Random
) -> DestroyMove:
    """Select one move using the ordinary behaviour of a chosen Destroy operator.

    The operator has already defined the neighbourhood by generating ``moves``.
    Basic ALNS performs no cross-candidate semantic scoring: it simply samples
    one feasible neighbourhood realization with the run RNG.  Keeping this
    helper in the Destroy module makes that responsibility explicit while
    preserving the exact distribution previously produced by ``rng.choice``.
    """
    pool = list(moves)
    if not pool:
        raise ValueError("cannot sample an intrinsic destroy move from an empty pool")
    return rng.choice(pool)


def compute_destroy_strength(sol: AssignmentSolution, strength_ratio: float) -> DestroyStrength:
    assigned_count = len(sol.all_assigned_tasks())
    if assigned_count <= 0:
        return DestroyStrength(0)
    # The ratio controls a bounded large-neighborhood base.  T50/T100 retain
    # the ordinary percentage semantics; larger cases cap the base at 100
    # tasks so exact greedy repair does not receive 70--130 simultaneously
    # removed tasks.  Stronger Agent actions still scale transparently: at
    # The capped base keeps large-instance destroy neighborhoods tractable while preserving ratio ordering.
    destroy_base = min(assigned_count, 100)
    target = _clamp_int(round(float(strength_ratio) * destroy_base), 1, assigned_count)
    return DestroyStrength(target_k=target)


# ---------------------------------------------------------------------------
# Pure candidate generators
# ---------------------------------------------------------------------------

def enumerate_random_removal(
    sol: AssignmentSolution,
    instance: Instance,
    config: Config,
    policy: DestroyPolicy,
    strength: DestroyStrength,
    rng: random.Random,
    context: Optional[DestroyContext] = None,
) -> List[DestroyMove]:
    del instance, config, policy, context
    assigned = sorted(int(tid) for tid in sol.all_assigned_tasks())
    if not assigned or strength.target_k <= 0:
        return []
    k = min(strength.target_k, len(assigned))
    count = min(8, max(3, math.ceil(len(assigned) / max(1, k))))
    rows: List[DestroyMove] = []
    seen: set[Tuple[int, ...]] = set()
    for index in range(count):
        tasks = tuple(sorted(rng.sample(assigned, k)))
        if tasks in seen:
            continue
        seen.add(tasks)
        rows.append(_make_move("random_removal", "random_set", tasks, _affected_routes_for_tasks(sol, tasks), {"candidate_index": index}))
    return rows


def enumerate_worst_cost_removal(
    sol: AssignmentSolution,
    instance: Instance,
    config: Config,
    policy: DestroyPolicy,
    strength: DestroyStrength,
    rng: random.Random,
    context: Optional[DestroyContext] = None,
) -> List[DestroyMove]:
    del policy, context
    assigned = sorted(int(tid) for tid in sol.all_assigned_tasks())
    if not assigned or strength.target_k <= 0:
        return []
    ranked = sorted(
        assigned,
        key=lambda tid: (-_single_task_route_cost_release(sol, instance, config, tid), tid),
    )
    k = min(strength.target_k, len(ranked))
    starts = sorted({0, min(max(0, len(ranked) - k), max(1, k // 2)), max(0, len(ranked) - k)})
    rows: List[DestroyMove] = []
    for index, start in enumerate(starts):
        tasks = tuple(sorted(ranked[start : start + k]))
        if tasks:
            rows.append(_make_move("worst_cost_removal", "high_local_cost_set", tasks, _affected_routes_for_tasks(sol, tasks), {"rank_window_start": start, "candidate_index": index}))
    # A small randomized top-rank variant is intrinsic to worst removal and does
    # not use focus or shared feature weights.
    top = ranked[: min(len(ranked), max(k, 2 * k))]
    if len(top) >= k:
        tasks = tuple(sorted(rng.sample(top, k)))
        rows.append(_make_move("worst_cost_removal", "high_local_cost_sample", tasks, _affected_routes_for_tasks(sol, tasks), {"top_pool_size": len(top)}))
    return _dedupe_moves(rows)


def enumerate_spatial_related_removal(
    sol: AssignmentSolution,
    instance: Instance,
    config: Config,
    policy: DestroyPolicy,
    strength: DestroyStrength,
    rng: random.Random,
    context: Optional[DestroyContext] = None,
) -> List[DestroyMove]:
    del config, policy, context
    assigned = sorted(int(tid) for tid in sol.all_assigned_tasks())
    if not assigned or strength.target_k <= 0:
        return []
    k = min(strength.target_k, len(assigned))
    seeds = _representative_seeds(assigned, rng, limit=6)
    rows: List[DestroyMove] = []
    for seed in seeds:
        seed_loc = instance.task_by_id(seed).loc
        ordered = sorted(
            assigned,
            key=lambda tid: (float(instance.distance(seed_loc, instance.task_by_id(tid).loc)), tid),
        )
        tasks = tuple(sorted(ordered[:k]))
        rows.append(_make_move("spatial_related_removal", "spatial_neighborhood", tasks, _affected_routes_for_tasks(sol, tasks), {"seed_task_id": seed}))
    return _dedupe_moves(rows)


def enumerate_time_related_removal(
    sol: AssignmentSolution,
    instance: Instance,
    config: Config,
    policy: DestroyPolicy,
    strength: DestroyStrength,
    rng: random.Random,
    context: Optional[DestroyContext] = None,
) -> List[DestroyMove]:
    del config, policy, context
    assigned = sorted(int(tid) for tid in sol.all_assigned_tasks())
    if not assigned or strength.target_k <= 0:
        return []
    k = min(strength.target_k, len(assigned))
    seeds = _representative_seeds(assigned, rng, limit=6)
    rows: List[DestroyMove] = []
    for seed in seeds:
        seed_task = instance.task_by_id(seed)
        seed_center = 0.5 * (float(seed_task.tw_start) + float(seed_task.tw_end))
        ordered = sorted(
            assigned,
            key=lambda tid: (
                abs(0.5 * (float(instance.task_by_id(tid).tw_start) + float(instance.task_by_id(tid).tw_end)) - seed_center),
                tid,
            ),
        )
        tasks = tuple(sorted(ordered[:k]))
        rows.append(_make_move("time_related_removal", "time_neighborhood", tasks, _affected_routes_for_tasks(sol, tasks), {"seed_task_id": seed}))
    return _dedupe_moves(rows)


DESTROY_OPERATORS: Dict[str, DestroyOperator] = {
    "random_removal": enumerate_random_removal,
    "worst_cost_removal": enumerate_worst_cost_removal,
    "spatial_related_removal": enumerate_spatial_related_removal,
    "time_related_removal": enumerate_time_related_removal,
}
assert tuple(DESTROY_OPERATORS) == DESTROY_OPERATOR_NAMES


# ---------------------------------------------------------------------------
# Shared scoring, deliberately outside the generators
# ---------------------------------------------------------------------------


def _score_moves(
    moves: Sequence[DestroyMove],
    sol: AssignmentSolution,
    instance: Instance,
    config: Config,
    policy: DestroyPolicy,
    *,
    context: Optional[DestroyContext] = None,
) -> List[DestroyMove]:
    return [replace(move, features=LandscapeFeatures(), score=0.0) for move in moves]


def route_cost(sol: AssignmentSolution, instance: Instance, config: Config, aid: int) -> float:
    metrics = simulate_route(instance, config, int(aid), sol.routes.get(int(aid), []))
    return float(metrics.distance + metrics.energy)


def _group_removal_cost_release(
    sol: AssignmentSolution,
    instance: Instance,
    config: Config,
    task_ids: Sequence[int],
) -> float:
    affected = _affected_routes_for_tasks(sol, task_ids)
    remove_set = {int(tid) for tid in task_ids}
    release = 0.0
    for aid in affected:
        before = route_cost(sol, instance, config, int(aid))
        after_route = [int(tid) for tid in sol.routes.get(int(aid), []) if int(tid) not in remove_set]
        metrics = simulate_route(instance, config, int(aid), after_route)
        release += before - float(metrics.distance + metrics.energy)
    return max(0.0, float(release))


def _single_task_route_cost_release(
    sol: AssignmentSolution,
    instance: Instance,
    config: Config,
    tid: int,
) -> float:
    return _group_removal_cost_release(sol, instance, config, (int(tid),))


# ---------------------------------------------------------------------------
# Landscape / diagnostics
# ---------------------------------------------------------------------------

def build_destroy_landscape(
    sol: AssignmentSolution,
    instance: Instance,
    config: Config,
    *,
    preference_metric: str,
) -> Dict[str, Any]:
    route_rows = build_route_preference_landscape(
        instance,
        config,
        sol.routes,
        preference_metric=str(preference_metric),
        include_hotspots=not bool(sol.unassigned),
    )
    return {"candidate_routes": route_rows}


# ---------------------------------------------------------------------------
# 候选生成使用的公共辅助接口。
# ---------------------------------------------------------------------------

def _make_move(
    operator_name: str,
    shape: str,
    task_ids: Sequence[int],
    affected_routes: Sequence[int],
    metadata: Optional[Mapping[str, Any]] = None,
) -> DestroyMove:
    return DestroyMove(
        operator_name=str(operator_name),
        shape=str(shape),
        task_ids=tuple(int(tid) for tid in task_ids),
        affected_routes=tuple(int(aid) for aid in affected_routes),
        features=LandscapeFeatures(),
        score=0.0,
        metadata=dict(metadata or {}),
    )


def _affected_routes_for_tasks(sol: AssignmentSolution, task_ids: Sequence[int]) -> Tuple[int, ...]:
    targets = {int(tid) for tid in task_ids}
    return tuple(
        int(aid)
        for aid, route in sorted(sol.routes.items())
        if targets & {int(tid) for tid in route}
    )




def _representative_seeds(assigned: Sequence[int], rng: random.Random, limit: int) -> List[int]:
    if len(assigned) <= limit:
        return list(assigned)
    deterministic = [assigned[0], assigned[len(assigned) // 2], assigned[-1]]
    remaining = [tid for tid in assigned if tid not in deterministic]
    needed = max(0, limit - len(deterministic))
    return list(dict.fromkeys(deterministic + rng.sample(remaining, needed)))


def _dedupe_moves(moves: Sequence[DestroyMove]) -> List[DestroyMove]:
    out: List[DestroyMove] = []
    seen: set[Tuple[int, ...]] = set()
    for move in moves:
        key = tuple(sorted(int(tid) for tid in move.task_ids))
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(move)
    return out


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))


__all__ = [
    "DESTROY_OPERATORS",
    "sample_intrinsic_destroy_move",
    "DestroyContext",
    "DestroyMove",
    "DestroyOperator",
    "DestroyStrength",
    "build_destroy_landscape",
    "compute_destroy_strength",
    "enumerate_random_removal",
    "enumerate_worst_cost_removal",
    "enumerate_spatial_related_removal",
    "enumerate_time_related_removal",
    "route_cost",
    "_affected_routes_for_tasks",
    "_make_move",
    "_score_moves",
]
