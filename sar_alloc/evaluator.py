from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .config import Config
from .domain import CONSTRAINT_METRICS, QUALITY_METRICS
from .models import Agent, Instance, Task
from .solution import AssignmentSolution, EvalResult

_EPS = 1e-9


@dataclass(frozen=True)
class ConstraintReport:
    is_feasible: bool
    violation_total: float
    violation_capability: float
    violation_time_window: float
    violation_energy: float
    violation_by_type: Dict[str, float] = field(default_factory=dict)
    violation_by_task: Dict[str, Dict[str, float]] = field(default_factory=dict)
    violation_by_route: Dict[str, Dict[str, float]] = field(default_factory=dict)
    violation_ratio_by_type: Dict[str, float] = field(default_factory=dict)
    violation_details_by_type: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)


def check_constraints(
    solution: AssignmentSolution,
    instance: Instance,
    config: Config,
    *,
    update_solution_schedule: bool = True,
) -> Tuple[ConstraintReport, Dict[str, Tuple[float, float]]]:
    min_time_window_ref = _reference_floor(config.eval.min_time_window_ref, "min_time_window_ref")
    min_energy_ref = _reference_floor(config.eval.min_energy_ref, "min_energy_ref")
    schedule: Dict[str, Tuple[float, float]] = {}

    by_type = {"capability": 0.0, "time_window": 0.0, "energy": 0.0}
    by_task: Dict[str, Dict[str, float]] = {}
    by_route: Dict[str, Dict[str, float]] = {}
    time_window_details: List[Dict[str, float]] = []
    energy_details: List[Dict[str, float]] = []
    time_window_lateness_sum = 0.0
    time_window_ref_sum = 0.0
    energy_over_sum = 0.0
    energy_ref_sum = 0.0

    for aid in instance.all_agent_ids():
        agent = instance.agent_by_id(aid)
        route = [int(tid) for tid in solution.routes.get(aid, [])]
        route_key = str(aid)
        by_route[route_key] = {"capability": 0.0, "time_window": 0.0, "energy": 0.0}

        cur_loc = instance.depot.loc
        cur_time = 0.0
        agent_energy = 0.0
        task_energy: List[Tuple[int, float]] = []

        for tid in route:
            task = instance.task_by_id(tid)
            task_key = str(tid)
            by_task.setdefault(task_key, {"capability": 0.0, "time_window": 0.0, "energy": 0.0})

            cap_def = _capability_deficit(agent, task)
            if cap_def > 0:
                _add_violation(by_type, by_task, by_route, task_key, route_key, "capability", cap_def)

            dist = instance.distance(cur_loc, task.loc)
            t_travel = instance.travel_time(agent, cur_loc, task.loc)
            cur_time += t_travel
            e_travel = agent.travel_energy_rate * dist
            agent_energy += e_travel
            task_energy.append((tid, e_travel))

            if cur_time < task.tw_start:
                wait = task.tw_start - cur_time
                cur_time += wait


            service_start = cur_time
            service_end = service_start + task.service_time
            cur_time = service_end
            late = max(0.0, service_start - task.tw_end)
            time_ref = max(float(task.tw_end) - float(task.tw_start), min_time_window_ref)
            time_window_lateness_sum += late
            time_window_ref_sum += time_ref
            time_window_details.append(
                {
                    "task_id": int(task.id),
                    "lateness": float(late),
                    "time_ref": float(time_ref),
                    "ratio": float(late / time_ref),
                }
            )
            if late > 0:
                _add_violation(by_type, by_task, by_route, task_key, route_key, "time_window", late)

            schedule[f"{aid}:{tid}"] = (service_start, service_end)
            cur_loc = task.loc

        if route:
            dist_back = instance.distance(cur_loc, instance.depot.loc)
            t_back = instance.travel_time(agent, cur_loc, instance.depot.loc)
            e_back = agent.travel_energy_rate * dist_back
            agent_energy += e_back
            task_energy[-1] = (task_energy[-1][0], task_energy[-1][1] + e_back)

        excess = max(0.0, agent_energy - agent.init_energy)
        energy_ref = max(float(agent.init_energy), min_energy_ref)
        energy_over_sum += excess
        energy_ref_sum += energy_ref
        energy_details.append(
            {
                "agent_id": int(agent.id),
                "energy_over": float(excess),
                "energy_ref": float(energy_ref),
                "ratio": float(excess / energy_ref),
            }
        )
        if excess > _EPS and task_energy:
            total_task_energy = sum(max(0.0, energy) for _, energy in task_energy)
            if total_task_energy <= _EPS:
                share = excess / float(len(task_energy))
                for tid, _ in task_energy:
                    _add_violation(by_type, by_task, by_route, str(tid), route_key, "energy", share)
            else:
                for tid, energy in task_energy:
                    portion = excess * max(0.0, energy) / total_task_energy
                    _add_violation(by_type, by_task, by_route, str(tid), route_key, "energy", portion)

    by_task = {tid: _clean_violation_map(values) for tid, values in by_task.items() if sum(values.values()) > _EPS}
    by_route = {aid: _clean_violation_map(values) for aid, values in by_route.items() if sum(values.values()) > _EPS}
    by_type = _clean_violation_map(by_type)

    capability = float(by_type.get("capability", 0.0))
    time_window = float(by_type.get("time_window", 0.0))
    energy = float(by_type.get("energy", 0.0))
    total = capability + time_window + energy
    ratio_by_type = {
        "time_window": float(time_window_lateness_sum / time_window_ref_sum) if time_window_ref_sum > _EPS else 0.0,
        "energy": float(energy_over_sum / energy_ref_sum) if energy_ref_sum > _EPS else 0.0,
    }

    if update_solution_schedule:
        solution.schedule = {
            (int(pair.split(":", 1)[0]), int(pair.split(":", 1)[1])): value
            for pair, value in schedule.items()
        }

    return (
        ConstraintReport(
            is_feasible=bool(total <= _EPS),
            violation_total=float(total),
            violation_capability=float(capability),
            violation_time_window=float(time_window),
            violation_energy=float(energy),
            violation_by_type=by_type,
            violation_by_task=by_task,
            violation_by_route=by_route,
            violation_ratio_by_type=ratio_by_type,
            violation_details_by_type={"time_window": time_window_details, "energy": energy_details},
        ),
        schedule,
    )


def route_is_feasible(
    agent_id: int,
    route: Iterable[int],
    instance: Instance,
    config: Config,
) -> bool:
    """Check the hard constraints of one route with evaluator-equivalent rules.

    This is an exact route-local counterpart of ``check_constraints``.  It is
    suitable for checking one route without re-evaluating unchanged routes.
    """
    agent = instance.agent_by_id(int(agent_id))
    cur_loc = instance.depot.loc
    cur_time = 0.0
    energy = 0.0
    normalized_route = [int(tid) for tid in route]

    for tid in normalized_route:
        task = instance.task_by_id(int(tid))
        if _capability_deficit(agent, task) > _EPS:
            return False

        distance = instance.distance(cur_loc, task.loc)
        travel_time = instance.travel_time(agent, cur_loc, task.loc)
        cur_time += travel_time
        energy += agent.travel_energy_rate * distance

        if cur_time < task.tw_start:
            wait = float(task.tw_start) - cur_time
            cur_time += wait

        service_start = cur_time
        if service_start - float(task.tw_end) > _EPS:
            return False
        cur_time = service_start + float(task.service_time)
        cur_loc = task.loc

    if normalized_route:
        distance = instance.distance(cur_loc, instance.depot.loc)
        travel_time = instance.travel_time(agent, cur_loc, instance.depot.loc)
        energy += agent.travel_energy_rate * distance

    return bool(energy - float(agent.init_energy) <= _EPS)



@dataclass(frozen=True, slots=True)
class _QualityTaskContribution:
    distance: float
    travel_time: float
    travel_energy: float
    wait_time: float
    service_time: float


@dataclass(frozen=True, slots=True)
class _QualityRouteTrace:
    task_count: int
    contributions: Tuple[_QualityTaskContribution, ...]
    depot_distance: float
    depot_travel_time: float
    depot_travel_energy: float
    duration: float


class QualityMetricsCache:
    """Exact route-local cache for repeated quality evaluation during repair.

    Only dirty routes rebuild their task-level distance/time/energy contributions.
    Global metrics are then replayed from cached scalars in the exact same agent
    and task order as :func:`evaluate_quality_metrics`, preserving floating-point
    results while avoiding repeated geometry/energy calculations on unchanged routes.
    """

    def __init__(self) -> None:
        self._routes: Dict[int, _QualityRouteTrace] = {}

    def evaluate(
        self,
        solution: AssignmentSolution,
        instance: Instance,
        config: Config,
        *,
        dirty_agent_ids: Optional[Iterable[int]] = None,
    ) -> Dict[str, float]:
        selected = (
            [int(aid) for aid in instance.all_agent_ids()]
            if dirty_agent_ids is None
            else [int(aid) for aid in dirty_agent_ids]
        )
        for aid in selected:
            self._routes[int(aid)] = _build_quality_route_trace(
                solution, instance, config, int(aid)
            )
        # Defensive fill for first use or a caller that supplies a partial dirty set.
        for aid in instance.all_agent_ids():
            if int(aid) not in self._routes:
                self._routes[int(aid)] = _build_quality_route_trace(
                    solution, instance, config, int(aid)
                )

        travel_energy = 0.0
        missed_priority = sum(
            float(instance.task_by_id(int(tid)).priority)
            for tid in solution.unassigned
        )
        unassigned_count = float(len(solution.unassigned))

        for aid in instance.all_agent_ids():
            trace = self._routes[int(aid)]
            for item in trace.contributions:
                travel_energy += float(item.travel_energy)
            if trace.task_count:
                travel_energy += float(trace.depot_travel_energy)

        return {
            "missed_priority": float(missed_priority),
            "unassigned_count": float(unassigned_count),
            "energy_total": float(travel_energy),
        }


def _build_quality_route_trace(
    solution: AssignmentSolution,
    instance: Instance,
    config: Config,
    agent_id: int,
) -> _QualityRouteTrace:
    aid = int(agent_id)
    agent = instance.agent_by_id(aid)
    route = [int(tid) for tid in solution.routes.get(aid, [])]
    cur_loc = instance.depot.loc
    cur_time = 0.0
    rows: List[_QualityTaskContribution] = []
    for tid in route:
        task = instance.task_by_id(tid)
        dist = instance.distance(cur_loc, task.loc)
        t_travel = instance.travel_time(agent, cur_loc, task.loc)
        cur_time += t_travel
        e_travel = agent.travel_energy_rate * dist
        wait = 0.0

        if cur_time < task.tw_start:
            wait = task.tw_start - cur_time
            cur_time += wait

        service_time = task.service_time
        cur_time = cur_time + service_time
        rows.append(
            _QualityTaskContribution(
                distance=float(dist),
                travel_time=float(t_travel),
                travel_energy=float(e_travel),
                wait_time=float(wait),
                service_time=float(service_time),
            )
        )
        cur_loc = task.loc

    depot_distance = 0.0
    depot_travel_time = 0.0
    depot_travel_energy = 0.0
    if route:
        depot_distance = instance.distance(cur_loc, instance.depot.loc)
        depot_travel_time = instance.travel_time(agent, cur_loc, instance.depot.loc)
        depot_travel_energy = agent.travel_energy_rate * depot_distance
        cur_time += depot_travel_time
    return _QualityRouteTrace(
        task_count=len(route),
        contributions=tuple(rows),
        depot_distance=float(depot_distance),
        depot_travel_time=float(depot_travel_time),
        depot_travel_energy=float(depot_travel_energy),
        duration=float(cur_time),
    )


def evaluate_quality_metrics(
    solution: AssignmentSolution,
    instance: Instance,
    config: Config,
) -> Dict[str, float]:
    """Compute service objectives and resource diagnostics without constraint reports.

    This is the exact quality-metric portion of :func:`evaluate`.  Repair
    candidate scoring often needs only these values; running ``check_constraints``
    there duplicated a full route traversal whose report was immediately
    discarded.
    """
    travel_energy = 0.0

    missed_priority = sum(
        float(instance.task_by_id(int(tid)).priority) for tid in solution.unassigned
    )
    unassigned_count = float(len(solution.unassigned))

    for aid in instance.all_agent_ids():
        agent = instance.agent_by_id(aid)
        route = [int(tid) for tid in solution.routes.get(aid, [])]
        cur_loc = instance.depot.loc
        cur_time = 0.0

        for tid in route:
            task = instance.task_by_id(tid)
            dist = instance.distance(cur_loc, task.loc)
            t_travel = instance.travel_time(agent, cur_loc, task.loc)
            cur_time += t_travel

            e_travel = agent.travel_energy_rate * dist
            travel_energy += e_travel

            if cur_time < task.tw_start:
                wait = task.tw_start - cur_time
                cur_time += wait

            service_start = cur_time
            cur_time = service_start + task.service_time
            cur_loc = task.loc

        if route:
            dist_back = instance.distance(cur_loc, instance.depot.loc)
            t_back = instance.travel_time(agent, cur_loc, instance.depot.loc)
            cur_time += t_back
            travel_energy += agent.travel_energy_rate * dist_back

    return {
        "missed_priority": float(missed_priority),
        "unassigned_count": float(unassigned_count),
        "energy_total": float(travel_energy),
    }


def evaluate(
    solution: AssignmentSolution,
    instance: Instance,
    config: Config,
    update_solution_schedule: bool = True,
) -> EvalResult:
    quality_metrics = evaluate_quality_metrics(solution, instance, config)
    constraint_report, _ = check_constraints(
        solution,
        instance,
        config,
        update_solution_schedule=update_solution_schedule,
    )
    ev = EvalResult(
        quality_metrics=quality_metrics, constraint_report=constraint_report
    )
    solution.eval = ev
    return ev


def compare_quality(
    eval_a: EvalResult, eval_b: EvalResult, objective_terms: Iterable[Any]
) -> int:
    for layer in _normalize_objective_terms(objective_terms):
        metric = str(layer["metric"])
        if metric in CONSTRAINT_METRICS:
            raise ValueError(
                f"constraint metric is not allowed in quality comparison: {metric}"
            )
        if metric not in QUALITY_METRICS:
            raise ValueError(f"unknown quality metric: {metric}")
        va = float(eval_a.get_quality_metric(metric))
        vb = float(eval_b.get_quality_metric(metric))
        diff = va - vb
        if str(layer["direction"]) == "max":
            diff = -diff
        # Formal search uses the complete lexicographic vector exactly.  A tie
        # is therefore possible only when this metric is numerically identical;
        # feasibility tolerances remain separate and still use _EPS above.
        if diff < 0.0:
            return -1
        if diff > 0.0:
            return 1
    return 0


def build_lex_key(
    metrics: Dict[str, float], objective_terms: Iterable[Any]
) -> Tuple[float, ...]:
    key_values: List[float] = []
    for layer in _normalize_objective_terms(objective_terms):
        metric = str(layer["metric"])
        if metric in CONSTRAINT_METRICS:
            raise ValueError(f"constraint metric is not allowed in objective: {metric}")
        value = float(metrics.get(metric, 0.0))
        if str(layer["direction"]) == "max":
            value = -value
        key_values.append(value)
    return tuple(key_values)


def build_objective_keys(
    evaluation: EvalResult,
    objective_terms: Iterable[Any],
) -> Dict[str, Any]:
    """Return the single run-global lexicographic key used everywhere."""
    terms = _normalize_objective_terms(objective_terms)
    metrics = {
        item["metric"]: float(evaluation.get_quality_metric(item["metric"]))
        for item in terms
    }
    return {
        "terms": [item["metric"] for item in terms],
        "key": list(build_lex_key(metrics, terms)),
    }


def _normalize_objective_terms(objective_terms: Iterable[Any]) -> List[Dict[str, str]]:
    terms: List[Dict[str, str]] = []
    for raw in objective_terms:
        if isinstance(raw, Mapping):
            metric = str(raw.get("metric", ""))
            direction = str(raw.get("direction", ""))
        else:
            metric = str(getattr(raw, "metric", ""))
            direction = str(getattr(raw, "direction", ""))
        if not metric:
            continue
        if direction not in {"min", "max"}:
            raise ValueError(f"illegal objective direction: {direction}")
        terms.append({"metric": metric, "direction": direction})
    expected = [{"metric": "missed_priority", "direction": "min"},
                {"metric": "unassigned_count", "direction": "min"},
                {"metric": "energy_total", "direction": "min"}]
    if terms != expected:
        raise ValueError("The conference objective is fixed: minimize missed priority, then unassigned count, then transfer energy.")
    return terms


def _add_violation(
    by_type: Dict[str, float],
    by_task: Dict[str, Dict[str, float]],
    by_route: Dict[str, Dict[str, float]],
    task_key: str,
    route_key: str,
    violation_type: str,
    value: float,
) -> None:
    amount = float(value)
    by_type[violation_type] = by_type.get(violation_type, 0.0) + amount
    by_task.setdefault(task_key, {"capability": 0.0, "time_window": 0.0, "energy": 0.0})
    by_route.setdefault(route_key, {"capability": 0.0, "time_window": 0.0, "energy": 0.0})
    by_task[task_key][violation_type] = by_task[task_key].get(violation_type, 0.0) + amount
    by_route[route_key][violation_type] = by_route[route_key].get(violation_type, 0.0) + amount


def _clean_violation_map(values: Dict[str, float]) -> Dict[str, float]:
    return {key: float(value) for key, value in values.items() if abs(float(value)) > _EPS}


def _capability_deficit(agent: Agent, task: Task) -> float:
    return float(len(task.skill_req - agent.skills))





def _reference_floor(raw_value: float, name: str) -> float:
    value = float(raw_value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"EvaluationConfig.{name} must be a positive finite number")
    return value
