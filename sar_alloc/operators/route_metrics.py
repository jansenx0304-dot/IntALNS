"""Small exact route metrics shared by destroy and repair operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ..config import Config
from ..models import Instance

_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class RouteMetrics:
    distance: float
    energy: float
    duration: float
    min_time_slack: float
    energy_slack_ratio: float

    @property
    def bottleneck_margin(self) -> float:
        # Both terms are dimensionless and clipped.  The minimum represents the
        # single tightest route constraint rather than a weighted composite.
        time_ratio = self.min_time_slack / (1.0 + abs(self.min_time_slack))
        return float(min(time_ratio, self.energy_slack_ratio))


@dataclass(frozen=True, slots=True)
class RoutePrefixState:
    """Exact state after an unchanged route prefix, before any depot return leg."""

    current_loc: Any
    current_time: float
    distance_total: float
    energy_total: float
    min_slack: float


@dataclass(frozen=True, slots=True)
class RouteInsertionContext:
    """Reusable exact prefix states for insertion delta evaluation on one route."""

    agent_id: int
    route: tuple[int, ...]
    prefix_states: tuple[RoutePrefixState, ...]
    base_metrics: RouteMetrics


def build_route_insertion_context(
    instance: Instance,
    config: Config,
    agent_id: int,
    route: Sequence[int],
) -> RouteInsertionContext:
    """Build exact prefix states once for all insertion positions on a route.

    Prefix states reproduce the arithmetic order of :func:`simulate_route`.
    A candidate inserted at position ``p`` can therefore resume from the state
    after the unchanged prefix ``route[:p]`` and simulate only the inserted task
    plus the affected suffix.
    """
    aid = int(agent_id)
    agent = instance.agent_by_id(aid)
    normalized = tuple(int(tid) for tid in route)
    current_loc = instance.depot.loc
    current_time = 0.0
    distance_total = 0.0
    energy_total = 0.0
    min_slack = float("inf")
    states = [
        RoutePrefixState(
            current_loc=current_loc,
            current_time=current_time,
            distance_total=distance_total,
            energy_total=energy_total,
            min_slack=min_slack,
        )
    ]

    for tid in normalized:
        task = instance.task_by_id(int(tid))
        distance = float(instance.distance(current_loc, task.loc))
        travel_time = float(instance.travel_time(agent, current_loc, task.loc))
        distance_total += distance
        current_time += travel_time
        energy_total += float(agent.travel_energy_rate) * distance
        if current_time < float(task.tw_start):
            wait = float(task.tw_start) - current_time
            current_time += wait

        min_slack = min(min_slack, float(task.tw_end) - current_time)
        current_time += float(task.service_time)

        current_loc = task.loc
        states.append(
            RoutePrefixState(
                current_loc=current_loc,
                current_time=float(current_time),
                distance_total=float(distance_total),
                energy_total=float(energy_total),
                min_slack=float(min_slack),
            )
        )

    final_loc = current_loc
    final_time = current_time
    final_distance = distance_total
    final_energy = energy_total
    if normalized:
        distance = float(instance.distance(final_loc, instance.depot.loc))
        travel_time = float(instance.travel_time(agent, final_loc, instance.depot.loc))
        final_distance += distance
        final_time += travel_time
        final_energy += float(agent.travel_energy_rate) * distance
    final_slack = 0.0 if min_slack == float("inf") else float(min_slack)
    energy_slack_ratio = (
        float(agent.init_energy) - final_energy
    ) / max(float(agent.init_energy), _EPS)
    base_metrics = RouteMetrics(
        distance=float(final_distance),
        energy=float(final_energy),
        duration=float(final_time),
        min_time_slack=float(final_slack),
        energy_slack_ratio=float(energy_slack_ratio),
    )
    return RouteInsertionContext(
        agent_id=aid,
        route=normalized,
        prefix_states=tuple(states),
        base_metrics=base_metrics,
    )


def simulate_insertion_route(
    instance: Instance,
    config: Config,
    context: RouteInsertionContext,
    task_id: int,
    position: int,
) -> RouteMetrics:
    """Exactly simulate one insertion by recomputing only the affected suffix."""
    aid = int(context.agent_id)
    route = context.route
    pos = int(position)
    if pos < 0 or pos > len(route):
        raise IndexError(f"insertion position {pos} outside route of length {len(route)}")
    agent = instance.agent_by_id(aid)
    state = context.prefix_states[pos]
    current_loc = state.current_loc
    current_time = float(state.current_time)
    distance_total = float(state.distance_total)
    energy_total = float(state.energy_total)
    min_slack = float(state.min_slack)

    for tid in (int(task_id), *route[pos:]):
        task = instance.task_by_id(int(tid))
        distance = float(instance.distance(current_loc, task.loc))
        travel_time = float(instance.travel_time(agent, current_loc, task.loc))
        distance_total += distance
        current_time += travel_time
        energy_total += float(agent.travel_energy_rate) * distance
        if current_time < float(task.tw_start):
            wait = float(task.tw_start) - current_time
            current_time += wait

        min_slack = min(min_slack, float(task.tw_end) - current_time)
        current_time += float(task.service_time)

        current_loc = task.loc

    distance = float(instance.distance(current_loc, instance.depot.loc))
    travel_time = float(instance.travel_time(agent, current_loc, instance.depot.loc))
    distance_total += distance
    current_time += travel_time
    energy_total += float(agent.travel_energy_rate) * distance

    if min_slack == float("inf"):
        min_slack = 0.0
    energy_slack_ratio = (
        float(agent.init_energy) - energy_total
    ) / max(float(agent.init_energy), _EPS)
    return RouteMetrics(
        distance=float(distance_total),
        energy=float(energy_total),
        duration=float(current_time),
        min_time_slack=float(min_slack),
        energy_slack_ratio=float(energy_slack_ratio),
    )


def simulate_removal_route(
    instance: Instance,
    config: Config,
    context: RouteInsertionContext,
    removed_task_ids: Sequence[int],
) -> RouteMetrics:
    """Exactly simulate deletion by recomputing only from the first changed stop."""
    remove_set = {int(tid) for tid in removed_task_ids}
    if not remove_set:
        return context.base_metrics
    route = context.route
    changed_positions = [idx for idx, tid in enumerate(route) if int(tid) in remove_set]
    if not changed_positions:
        return context.base_metrics
    first = int(changed_positions[0])
    aid = int(context.agent_id)
    agent = instance.agent_by_id(aid)
    state = context.prefix_states[first]
    current_loc = state.current_loc
    current_time = float(state.current_time)
    distance_total = float(state.distance_total)
    energy_total = float(state.energy_total)
    min_slack = float(state.min_slack)

    kept_suffix = [int(tid) for tid in route[first:] if int(tid) not in remove_set]
    for tid in kept_suffix:
        task = instance.task_by_id(int(tid))
        distance = float(instance.distance(current_loc, task.loc))
        travel_time = float(instance.travel_time(agent, current_loc, task.loc))
        distance_total += distance
        current_time += travel_time
        energy_total += float(agent.travel_energy_rate) * distance
        if current_time < float(task.tw_start):
            wait = float(task.tw_start) - current_time
            current_time += wait

        min_slack = min(min_slack, float(task.tw_end) - current_time)
        current_time += float(task.service_time)

        current_loc = task.loc

    remaining_count = first + len(kept_suffix)
    if remaining_count > 0:
        distance = float(instance.distance(current_loc, instance.depot.loc))
        travel_time = float(instance.travel_time(agent, current_loc, instance.depot.loc))
        distance_total += distance
        current_time += travel_time
        energy_total += float(agent.travel_energy_rate) * distance

    if min_slack == float("inf"):
        min_slack = 0.0
    energy_slack_ratio = (
        float(agent.init_energy) - energy_total
    ) / max(float(agent.init_energy), _EPS)
    return RouteMetrics(
        distance=float(distance_total),
        energy=float(energy_total),
        duration=float(current_time),
        min_time_slack=float(min_slack),
        energy_slack_ratio=float(energy_slack_ratio),
    )


def simulate_route(
    instance: Instance,
    config: Config,
    agent_id: int,
    route: Sequence[int],
) -> RouteMetrics:
    agent = instance.agent_by_id(int(agent_id))
    current_loc = instance.depot.loc
    current_time = 0.0
    distance_total = 0.0
    energy_total = 0.0
    min_slack = float("inf")

    for tid in route:
        task = instance.task_by_id(int(tid))
        distance = float(instance.distance(current_loc, task.loc))
        travel_time = float(instance.travel_time(agent, current_loc, task.loc))
        distance_total += distance
        current_time += travel_time
        energy_total += float(agent.travel_energy_rate) * distance
        if current_time < float(task.tw_start):
            wait = float(task.tw_start) - current_time
            current_time += wait

        min_slack = min(min_slack, float(task.tw_end) - current_time)
        current_time += float(task.service_time)

        current_loc = task.loc

    if route:
        distance = float(instance.distance(current_loc, instance.depot.loc))
        travel_time = float(instance.travel_time(agent, current_loc, instance.depot.loc))
        distance_total += distance
        current_time += travel_time
        energy_total += float(agent.travel_energy_rate) * distance

    if min_slack == float("inf"):
        min_slack = 0.0
    energy_slack_ratio = (
        float(agent.init_energy) - energy_total
    ) / max(float(agent.init_energy), _EPS)
    return RouteMetrics(
        distance=float(distance_total),
        energy=float(energy_total),
        duration=float(current_time),
        min_time_slack=float(min_slack),
        energy_slack_ratio=float(energy_slack_ratio),
    )


ROUTE_REBUILD_METRICS = ("energy_total",)


def _route_rebuild_value(metrics: RouteMetrics, metric_name: str) -> float:
    if metric_name == "energy_total":
        return float(metrics.energy)
    raise ValueError(f"unsupported route-rebuild preference metric: {metric_name!r}")


def build_route_preference_landscape(
    instance: Instance,
    config: Config,
    routes: Mapping[int, Sequence[int]],
    *,
    preference_metric: str,
    max_hotspot_tasks: int = 12,
    include_hotspots: bool = True,
) -> list[dict[str, Any]]:
    """Return exact route pressure and task-level releases for one preference.

    The values are computed from the same route simulator used by ALNS. They
    are resource observations, not objective terms or focus decisions. The
    conference observation uses ``energy_total`` solely as a local diagnostic.
    """

    metric_name = str(preference_metric or "")
    if metric_name not in ROUTE_REBUILD_METRICS:
        raise ValueError(
            f"unsupported route-rebuild preference metric: {metric_name!r}"
        )

    rows: list[dict[str, Any]] = []
    for aid in instance.all_agent_ids():
        route = [int(tid) for tid in routes.get(int(aid), ())]
        base = simulate_route(instance, config, int(aid), route)
        releases: list[dict[str, Any]] = []
        route_metric_value = _route_rebuild_value(base, metric_name)
        if include_hotspots:
            for position, tid in enumerate(route):
                reduced = route[:position] + route[position + 1 :]
                after = simulate_route(instance, config, int(aid), reduced)
                metric_release = (
                    _route_rebuild_value(base, metric_name)
                    - _route_rebuild_value(after, metric_name)
                )
                releases.append(
                    {
                        "task_id": int(tid),
                        "metric_release": round(float(metric_release), 6),
                        "distance_release": round(
                            float(base.distance - after.distance), 6
                        ),
                    }
                )
        releases.sort(
            key=lambda row: (
                -float(row["metric_release"]),
                -float(row["distance_release"]),
                int(row["task_id"]),
            )
        )
        target_count = min(
            max(0, int(max_hotspot_tasks)),
            len(route),
            (
                max(2, int(math.ceil(0.20 * len(route))))
                if route and include_hotspots
                else 0
            ),
        )
        hotspots = [
            row for row in releases if float(row["metric_release"]) > _EPS
        ][:target_count]
        rows.append(
            {
                "agent_id": int(aid),
                "task_count": len(route),
                "task_ids": list(route),
                "distance": round(float(base.distance), 6),
                "energy": round(float(base.energy), 6),
                "duration": round(float(base.duration), 6),
                "route_cost": round(float(base.distance + base.energy), 6),
                "min_time_slack": round(float(base.min_time_slack), 6),
                "energy_slack_ratio": round(
                    float(base.energy_slack_ratio), 6
                ),
                "bottleneck_margin": round(
                    float(base.bottleneck_margin), 6
                ),
                "rebuild_metric": metric_name,
                "rebuild_metric_value": round(float(route_metric_value), 6),
                "rebuild_hotspot_task_ids": [
                    int(row["task_id"]) for row in hotspots
                ],
                "rebuild_hotspot_releases": [dict(row) for row in hotspots],
            }
        )
    metric_total = sum(
        max(0.0, float(row["rebuild_metric_value"])) for row in rows
    )
    for row in rows:
        row["rebuild_metric_share"] = round(
            max(0.0, float(row["rebuild_metric_value"]))
            / max(_EPS, metric_total),
            6,
        )
    rows.sort(
        key=lambda row: (
            -float(row["rebuild_metric_value"]),
            int(row["agent_id"]),
        )
    )
    return rows


__all__ = [
    "RouteMetrics",
    "RoutePrefixState",
    "RouteInsertionContext",
    "build_route_insertion_context",
    "simulate_insertion_route",
    "simulate_removal_route",
    "simulate_route",
    "ROUTE_REBUILD_METRICS",
    "build_route_preference_landscape",
]
