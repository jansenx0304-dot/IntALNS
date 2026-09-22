"""Shared low-cost feasibility helpers for ALNS destroy and repair.

Agent-controlled repair features are task-level only. Insertion positions are
strictly filtered here and are later chosen only by the fixed repair position
operator; task scores never influence position selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..config import Config
from ..models import Agent, Instance, Location, Task
from ..solution import AssignmentSolution
from .types import InsertPosition

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class InsertionRouteContext:
    """Reusable exact timing state for one unchanged route.

    The context contains only route-dependent data.  It can therefore be shared
    by every pending task until this route is modified.
    """

    agent_id: int
    route: Tuple[int, ...]
    end_times: Tuple[float, ...]
    latest_starts: Tuple[float, ...]
    timewindow_bounds_feasible: bool


def build_insertion_route_contexts(
    sol: AssignmentSolution,
    instance: Instance,
    *,
    agent_ids: Optional[Iterable[int]] = None,
) -> Dict[int, InsertionRouteContext]:
    selected = (
        [int(aid) for aid in instance.all_agent_ids()]
        if agent_ids is None
        else [int(aid) for aid in agent_ids]
    )
    contexts: Dict[int, InsertionRouteContext] = {}
    for aid in selected:
        route = tuple(int(tid) for tid in sol.routes.get(int(aid), []))
        timing = _simulate_route(instance, int(aid), route)
        _earliest, latest, feasible = _route_timewindow_bounds(
            instance, int(aid), route
        )
        contexts[int(aid)] = InsertionRouteContext(
            agent_id=int(aid),
            route=route,
            end_times=tuple(float(stop.end_time) for stop in timing),
            latest_starts=tuple(float(value) for value in latest),
            timewindow_bounds_feasible=bool(feasible),
        )
    return contexts


def feasibility_filtered_positions_for_agent(
    sol: AssignmentSolution,
    tid: int,
    agent_id: int,
    instance: Instance,
    config: Config,
    *,
    route_context: Optional[InsertionRouteContext] = None,
) -> List[InsertPosition]:
    """Return tasks surviving the shared necessary-condition filters.

    Route timing and latest-start bounds are
    supplied once and reused across all pending tasks.
    """

    aid = int(agent_id)
    task = instance.task_by_id(int(tid))
    agent = instance.agent_by_id(aid)
    if not _agent_can_do_task(agent, task) or not _agent_basic_reach(
        instance, agent, task
    ):
        return []

    depot_lb_energy = _travel_energy(
        instance, agent, instance.depot.loc, task.loc
    )
    depot_lb_energy += _travel_energy(
        instance, agent, task.loc, instance.depot.loc
    )
    if depot_lb_energy - float(agent.init_energy) > _EPS:
        return []

    context = route_context
    if context is None or int(context.agent_id) != aid:
        context = build_insertion_route_contexts(
            sol, instance, agent_ids=(aid,)
        )[aid]
    route = list(context.route)
    if not route or not context.timewindow_bounds_feasible:
        return [
            InsertPosition(agent_id=aid, position=position)
            for position in range(len(route) + 1)
        ]

    survivors: List[InsertPosition] = []
    for position in range(len(route) + 1):
        prev_finish = 0.0
        prev_loc = instance.depot.loc
        if position > 0:
            prev_tid = int(route[position - 1])
            prev_finish = float(context.end_times[position - 1])
            prev_loc = instance.task_by_id(prev_tid).loc

        arr_tid = prev_finish + float(
            instance.travel_time(agent, prev_loc, task.loc)
        )
        start_tid = max(arr_tid, float(task.tw_start))
        if start_tid - float(task.tw_end) > _EPS:
            continue

        if position < len(route):
            next_task = instance.task_by_id(int(route[position]))
            finish_tid = start_tid + float(task.service_time)
            arr_next = finish_tid + float(
                instance.travel_time(agent, task.loc, next_task.loc)
            )
            next_start_lb = max(arr_next, float(next_task.tw_start))
            if next_start_lb - float(context.latest_starts[position]) > _EPS:
                continue

        survivors.append(InsertPosition(agent_id=aid, position=position))
    return survivors


@dataclass(frozen=True, slots=True)
class _RouteStop:
    start_time: float
    end_time: float
    slack: float
    tardiness: float


@dataclass(frozen=True, slots=True)
class _InsertionLowerBound:
    position: InsertPosition
    slack_lb: float
    detour_delta: float
    energy_delta: float
    affected_suffix_ratio: float


def basic_insertion_feasibility_filter(
    sol: AssignmentSolution,
    tid: int,
    instance: Instance,
    config: Config,
    *,
    candidate_positions: Optional[Sequence[InsertPosition]] = None,
) -> List[InsertPosition]:
    """Hard/obvious feasibility pruning for insertion positions.

    This filter only removes positions that fail skill coverage, optimistic depot
    reachability, or trivial route-independent energy lower bounds.
    """
    task = instance.task_by_id(int(tid))
    positions = (
        _all_insert_positions(sol, instance)
        if candidate_positions is None
        else list(candidate_positions)
    )
    if not positions:
        return []
    out: List[InsertPosition] = []

    for position in positions:
        agent = instance.agent_by_id(int(position.agent_id))
        if not _agent_can_do_task(agent, task):
            continue
        if not _agent_basic_reach(instance, agent, task):
            continue

        depot_lb_energy = _travel_energy(
            instance, agent, instance.depot.loc, task.loc
        )
        depot_lb_energy += _travel_energy(
            instance, agent, task.loc, instance.depot.loc
        )
        if depot_lb_energy - float(agent.init_energy) > _EPS:
            continue

        out.append(position)

    return out


def insertion_lower_bound_filter(
    sol: AssignmentSolution,
    tid: int,
    instance: Instance,
    *,
    candidate_positions: Optional[Sequence[InsertPosition]] = None,
) -> List[InsertPosition]:
    """Route-time lower-bound pruning with explicit earliest/latest tests.

    This step uses only fixed, low-cost necessary conditions. It does not score
    candidates or smuggle in extra heuristics.
    """
    task = instance.task_by_id(int(tid))
    positions = (
        _all_insert_positions(sol, instance)
        if candidate_positions is None
        else list(candidate_positions)
    )
    if not positions:
        return []
    survivors: List[InsertPosition] = []
    route_timing_cache = _build_route_timing_cache(sol, instance)
    route_bounds_cache: Dict[int, Tuple[List[float], List[float], bool]] = {}

    for position in positions:
        aid = int(position.agent_id)
        route = list(sol.routes.get(aid, []))
        if not route:
            survivors.append(position)
            continue

        if aid not in route_bounds_cache:
            route_bounds_cache[aid] = _route_timewindow_bounds(instance, aid, route)
        earliest, latest, feasible = route_bounds_cache[aid]

        # If the current route is already infeasible, the bound test is no longer
        # trustworthy as a pruning certificate. Keep the candidate for later
        # insert-score ranking and exact feasibility checking.
        if not feasible:
            survivors.append(position)
            continue

        prev_finish = 0.0
        prev_loc = instance.depot.loc
        if int(position.position) > 0:
            prev_tid = int(route[int(position.position) - 1])
            prev_finish = route_timing_cache[aid][int(position.position) - 1].end_time
            prev_loc = instance.task_by_id(prev_tid).loc

        agent = instance.agent_by_id(aid)
        arr_tid = prev_finish + float(instance.travel_time(agent, prev_loc, task.loc))
        start_tid = max(arr_tid, float(task.tw_start))
        if start_tid - float(task.tw_end) > _EPS:
            continue

        if int(position.position) < len(route):
            next_task = instance.task_by_id(int(route[int(position.position)]))
            finish_tid = start_tid + float(task.service_time)
            arr_next = finish_tid + float(
                instance.travel_time(agent, task.loc, next_task.loc)
            )
            next_start_lb = max(arr_next, float(next_task.tw_start))
            if next_start_lb - float(latest[int(position.position)]) > _EPS:
                continue

        survivors.append(position)

    return survivors


def dominated_position_filter(
    sol: AssignmentSolution,
    tid: int,
    instance: Instance,
    config: Config,
    *,
    candidate_positions: Optional[Sequence[InsertPosition]] = None,
) -> List[InsertPosition]:
    """Pareto-style pruning on transparent lower bounds.

    A candidate is dropped only if another candidate is no worse in detour, energy,
    affected suffix ratio, and reachable slack lower bound, with at least one strict
    improvement. No learned or weighted score is used here.
    """
    positions = (
        _all_insert_positions(sol, instance)
        if candidate_positions is None
        else list(candidate_positions)
    )
    if not positions:
        return []
    lower_bounds = _lower_bounds_by_position(
        sol=sol,
        tid=tid,
        positions=positions,
        instance=instance,
        config=config,
    )
    keep: List[InsertPosition] = []

    for key, current in lower_bounds.items():
        dominated = False
        for other_key, other in lower_bounds.items():
            if other_key == key:
                continue
            if _dominates(other, current):
                dominated = True
                break
        if not dominated:
            keep.append(key)

    return keep





def _build_route_timing_cache(
    sol: AssignmentSolution, instance: Instance
) -> Dict[int, List[_RouteStop]]:
    return {
        int(aid): _simulate_route(instance, int(aid), list(route))
        for aid, route in sol.routes.items()
    }


def _simulate_route(
    instance: Instance, aid: int, route: Sequence[int]
) -> List[_RouteStop]:
    agent = instance.agent_by_id(int(aid))
    current_loc = instance.depot.loc
    current_time = 0.0
    out: List[_RouteStop] = []

    for tid in route:
        task = instance.task_by_id(int(tid))
        current_time += float(instance.travel_time(agent, current_loc, task.loc))
        if current_time < float(task.tw_start):
            current_time = float(task.tw_start)

        start_time = float(current_time)
        tardiness = max(0.0, start_time - float(task.tw_end))
        slack = float(task.tw_end) - start_time
        end_time = start_time + float(task.service_time)
        out.append(
            _RouteStop(
                start_time=start_time,
                end_time=end_time,
                slack=slack,
                tardiness=tardiness,
            )
        )

        current_time = end_time
        current_loc = task.loc

    return out


def _route_timewindow_bounds(
    instance: Instance,
    aid: int,
    route: Sequence[int],
) -> Tuple[List[float], List[float], bool]:
    if not route:
        return [], [], True

    agent = instance.agent_by_id(int(aid))
    earliest: List[float] = [0.0] * len(route)
    latest: List[float] = [0.0] * len(route)

    current_loc = instance.depot.loc
    current_time = 0.0
    feasible = True
    for index, tid in enumerate(route):
        task = instance.task_by_id(int(tid))
        current_time += float(instance.travel_time(agent, current_loc, task.loc))
        if current_time < float(task.tw_start):
            current_time = float(task.tw_start)
        earliest[index] = float(current_time)
        if earliest[index] - float(task.tw_end) > _EPS:
            feasible = False
            break
        current_time += float(task.service_time)
        current_loc = task.loc

    if not feasible:
        return earliest, latest, False

    last_task = instance.task_by_id(int(route[-1]))
    latest[-1] = float(last_task.tw_end)
    if latest[-1] + _EPS < float(last_task.tw_start):
        return earliest, latest, False

    for index in range(len(route) - 2, -1, -1):
        cur = instance.task_by_id(int(route[index]))
        nxt = instance.task_by_id(int(route[index + 1]))
        bound = (
            latest[index + 1]
            - float(cur.service_time)
            - float(instance.travel_time(agent, cur.loc, nxt.loc))
        )
        latest[index] = min(float(cur.tw_end), float(bound))
        if latest[index] + _EPS < float(cur.tw_start):
            feasible = False
            break

    return earliest, latest, feasible


def _lower_bounds_by_position(
    sol: AssignmentSolution,
    tid: int,
    positions: Sequence[InsertPosition],
    instance: Instance,
    config: Config,
) -> Dict[InsertPosition, _InsertionLowerBound]:
    task = instance.task_by_id(int(tid))
    route_timing_cache = _build_route_timing_cache(sol, instance)
    out: Dict[InsertPosition, _InsertionLowerBound] = {}

    for position in dict.fromkeys(positions):
        aid = int(position.agent_id)
        route = list(sol.routes.get(aid, []))
        prev_finish = 0.0
        prev_loc = instance.depot.loc
        if int(position.position) > 0:
            prev_tid = int(route[int(position.position) - 1])
            prev_finish = route_timing_cache[aid][int(position.position) - 1].end_time
            prev_loc = instance.task_by_id(prev_tid).loc

        agent = instance.agent_by_id(aid)
        arrival_lb = prev_finish + float(
            instance.travel_time(agent, prev_loc, task.loc)
        )
        start_lb = max(arrival_lb, float(task.tw_start))

        out[position] = _InsertionLowerBound(
            position=position,
            slack_lb=float(task.tw_end) - start_lb,
            detour_delta=_insert_distance_delta(
                instance, config, route, int(position.position), task
            ),
            energy_delta=_insert_energy_delta(
                instance, config, aid, route, int(position.position), task
            ),
            affected_suffix_ratio=_affected_suffix_ratio(route, int(position.position)),
        )

    return out


def _dominates(lhs: _InsertionLowerBound, rhs: _InsertionLowerBound) -> bool:
    no_worse = (
        lhs.detour_delta <= rhs.detour_delta + _EPS
        and lhs.energy_delta <= rhs.energy_delta + _EPS
        and lhs.affected_suffix_ratio <= rhs.affected_suffix_ratio + _EPS
        and lhs.slack_lb + _EPS >= rhs.slack_lb
    )
    strictly_better = (
        lhs.detour_delta < rhs.detour_delta - _EPS
        or lhs.energy_delta < rhs.energy_delta - _EPS
        or lhs.affected_suffix_ratio < rhs.affected_suffix_ratio - _EPS
        or lhs.slack_lb > rhs.slack_lb + _EPS
    )
    return no_worse and strictly_better


def _all_insert_positions(
    sol: AssignmentSolution, instance: Instance
) -> List[InsertPosition]:
    out: List[InsertPosition] = []
    for aid in instance.all_agent_ids():
        route_len = len(sol.routes.get(int(aid), []))
        for pos in range(route_len + 1):
            out.append(InsertPosition(agent_id=int(aid), position=int(pos)))
    return out


def _agent_can_do_task(agent: Agent, task: Task) -> bool:
    return set(task.skill_req).issubset(set(agent.skills))


def _agent_basic_reach(instance: Instance, agent: Agent, task: Task) -> bool:
    earliest_arrival = float(instance.travel_time(agent, instance.depot.loc, task.loc))
    earliest_start = max(float(task.tw_start), earliest_arrival)
    return earliest_start <= float(task.tw_end) + _EPS


def _insert_distance_delta(
    instance: Instance,
    config: Config,
    route: Sequence[int],
    insert_pos: int,
    task: Task,
) -> float:
    prev_loc, next_loc = _insertion_neighbor_locs(
        instance, config, route, int(insert_pos)
    )
    base = 0.0 if next_loc is None else float(instance.distance(prev_loc, next_loc))
    added = float(instance.distance(prev_loc, task.loc)) + (
        0.0 if next_loc is None else float(instance.distance(task.loc, next_loc))
    )
    return added - base


def _insert_energy_delta(
    instance: Instance,
    config: Config,
    aid: int,
    route: Sequence[int],
    insert_pos: int,
    task: Task,
) -> float:
    agent = instance.agent_by_id(int(aid))
    prev_loc, next_loc = _insertion_neighbor_locs(
        instance, config, route, int(insert_pos)
    )
    base = (
        0.0
        if next_loc is None
        else _travel_energy(instance, agent, prev_loc, next_loc)
    )
    added = _travel_energy(instance, agent, prev_loc, task.loc) + (
        0.0
        if next_loc is None
        else _travel_energy(instance, agent, task.loc, next_loc)
    )
    return added - base


def _insertion_neighbor_locs(
    instance: Instance,
    config: Config,
    route: Sequence[int],
    insert_pos: int,
) -> Tuple[Location, Optional[Location]]:
    prev_loc = (
        instance.depot.loc
        if int(insert_pos) <= 0
        else instance.task_by_id(int(route[int(insert_pos) - 1])).loc
    )
    if int(insert_pos) < len(route):
        next_loc: Optional[Location] = instance.task_by_id(
            int(route[int(insert_pos)])
        ).loc
    else:
        next_loc = instance.depot.loc
    return prev_loc, next_loc


def _affected_suffix_ratio(route: Sequence[int], insert_pos: int) -> float:
    return float(len(route) - int(insert_pos) + 1) / max(1.0, float(len(route) + 1))


def _travel_energy(
    instance: Instance, agent: Agent, a: Location, b: Location
) -> float:
    # Travel-energy semantics are fixed project-wide: rate multiplied by distance.
    return float(agent.travel_energy_rate) * float(instance.distance(a, b))





__all__ = [
    "InsertionRouteContext",
    "basic_insertion_feasibility_filter",
    "build_insertion_route_contexts",
    "dominated_position_filter",
    "feasibility_filtered_positions_for_agent",
    "insertion_lower_bound_filter",
]
