from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from .models import Instance

AgentID = int
TaskID = int


def _obj_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "_asdict"):
        return value._asdict()
    if hasattr(value, "__dict__"):
        return vars(value)
    return value


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Objective-neutral evaluation result."""

    quality_metrics: Dict[str, float] = field(default_factory=dict)
    constraint_report: Any = None

    @property
    def is_feasible(self) -> bool:
        return bool(getattr(self.constraint_report, "is_feasible", False))

    @property
    def metrics(self) -> Dict[str, float]:
        return dict(self.quality_metrics)

    @property
    def violations(self) -> Dict[str, float]:
        return dict(getattr(self.constraint_report, "violation_by_type", {}) or {})

    def get_quality_metric(self, name: str) -> float:
        return float(self.quality_metrics.get(str(name), 0.0))

    def get_constraint_metric(self, name: str) -> float:
        report = self.constraint_report
        if report is None:
            return 0.0
        mapping = {
            "violation_total": "violation_total",
            "violation_capability": "violation_capability",
            "violation_time_window": "violation_time_window",
            "violation_energy": "violation_energy",
        }
        attr = mapping.get(str(name))
        return 0.0 if attr is None else float(getattr(report, attr))

    def get_metric(self, name: str) -> float:
        if str(name) in self.quality_metrics:
            return self.get_quality_metric(name)
        return self.get_constraint_metric(name)


@dataclass(slots=True)
class AssignmentSolution:
    routes: Dict[AgentID, List[TaskID]] = field(default_factory=dict)
    unassigned: Set[TaskID] = field(default_factory=set)
    schedule: Dict[Tuple[AgentID, TaskID], Tuple[float, float]] = field(
        default_factory=dict
    )
    eval: Optional[EvalResult] = None
    solver_diagnostics: Dict[str, Any] = field(default_factory=dict)
    run_summary: Dict[str, Any] = field(default_factory=dict)
    run_artifact: Dict[str, Any] = field(default_factory=dict)

    def structure_key(self) -> tuple[tuple[tuple[int, tuple[int, ...]], ...], tuple[int, ...]]:
        return (
            tuple((int(aid), tuple(int(tid) for tid in route)) for aid, route in sorted(self.routes.items())),
            tuple(sorted(int(tid) for tid in self.unassigned)),
        )

    def clone(self, deep: bool = True) -> "AssignmentSolution":
        if not deep:
            return AssignmentSolution(
                routes=self.routes,
                unassigned=self.unassigned,
                schedule=self.schedule,
                eval=self.eval,
                solver_diagnostics=self.solver_diagnostics,
                run_summary=self.run_summary,
                run_artifact=self.run_artifact,
            )
        return AssignmentSolution(
            routes={
                int(aid): [int(tid) for tid in seq] for aid, seq in self.routes.items()
            },
            unassigned={int(tid) for tid in self.unassigned},
            schedule=dict(self.schedule),
            eval=self.eval,
            solver_diagnostics=deepcopy(self.solver_diagnostics),
            run_summary=deepcopy(self.run_summary),
            run_artifact=deepcopy(self.run_artifact),
        )

    def clone_search_state(self) -> "AssignmentSolution":
        """Clone only state that can affect search decisions or evaluation.

        Solver diagnostics and final-run artifacts are reporting payloads.  They
        can become large after a Agent ALNS action and were previously deep-copied
        into every temporary destroy/repair candidate of the next action.  This
        search-state clone deliberately omits those reporting-only payloads while
        preserving routes, unassigned tasks, schedule, and the immutable
        evaluation reference.  Callers that need diagnostics continue to use
        ``clone(deep=True)``.
        """
        return AssignmentSolution(
            routes={
                int(aid): [int(tid) for tid in seq] for aid, seq in self.routes.items()
            },
            unassigned={int(tid) for tid in self.unassigned},
            schedule=dict(self.schedule),
            eval=self.eval,
        )

    @staticmethod
    def empty_from_instance(
        instance: Instance, put_all_unassigned: bool = True
    ) -> "AssignmentSolution":
        sol = AssignmentSolution(
            routes={int(aid): [] for aid in instance.all_agent_ids()}
        )
        if put_all_unassigned:
            sol.unassigned = {int(tid) for tid in instance.all_task_ids()}
        return sol

    def all_assigned_tasks(self) -> Set[TaskID]:
        out: Set[TaskID] = set()
        for seq in self.routes.values():
            out.update(int(tid) for tid in seq)
        return out

    def add_task(
        self, agent_id: AgentID, task_id: TaskID, position: Optional[int] = None
    ) -> None:
        aid = int(agent_id)
        tid = int(task_id)
        self.routes.setdefault(aid, [])
        if position is None or int(position) >= len(self.routes[aid]):
            self.routes[aid].append(tid)
        else:
            self.routes[aid].insert(int(position), tid)
        self.unassigned.discard(tid)
        self.eval = None

    def remove_task(
        self, agent_id: AgentID, task_id: TaskID, to_unassigned: bool = True
    ) -> None:
        aid = int(agent_id)
        tid = int(task_id)
        seq = self.routes.get(aid, [])
        if tid in seq:
            seq.remove(tid)
            if to_unassigned:
                self.unassigned.add(tid)
            self.schedule.pop((aid, tid), None)
            self.eval = None

    def move_task(
        self,
        from_agent: AgentID,
        to_agent: AgentID,
        task_id: TaskID,
        to_position: Optional[int] = None,
    ) -> None:
        self.remove_task(from_agent, task_id, to_unassigned=False)
        self.add_task(to_agent, task_id, position=to_position)

    def normalize(self, instance: Instance) -> None:
        for aid in instance.all_agent_ids():
            self.routes.setdefault(int(aid), [])
        valid_tasks = {int(tid) for tid in instance.all_task_ids()}
        for aid, seq in list(self.routes.items()):
            self.routes[int(aid)] = [int(tid) for tid in seq if int(tid) in valid_tasks]
        self.unassigned = {
            int(tid) for tid in self.unassigned if int(tid) in valid_tasks
        }
        self.unassigned.difference_update(self.all_assigned_tasks())
        valid_pairs = {
            (int(aid), int(tid)) for aid, seq in self.routes.items() for tid in seq
        }
        self.schedule = {
            pair: value for pair, value in self.schedule.items() if pair in valid_pairs
        }
        self.eval = None

    def to_dict(self) -> Dict[str, Any]:
        ev_dict = _obj_to_dict(self.eval)
        return {
            "routes": {
                str(aid): [int(tid) for tid in seq] for aid, seq in self.routes.items()
            },
            "unassigned": [int(tid) for tid in sorted(self.unassigned)],
            "schedule": {
                f"{aid}:{tid}": [float(st), float(en)]
                for (aid, tid), (st, en) in self.schedule.items()
            },
            "eval": ev_dict,
            "solver_diagnostics": deepcopy(self.solver_diagnostics),
            "run_summary": deepcopy(self.run_summary),
        }

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> "AssignmentSolution":
        routes = {
            int(aid): [int(tid) for tid in seq]
            for aid, seq in dict(data.get("routes", {}) or {}).items()
        }
        unassigned = {int(tid) for tid in list(data.get("unassigned", []) or [])}
        schedule: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for key, value in dict(data.get("schedule", {}) or {}).items():
            aid_s, tid_s = str(key).split(":", 1)
            schedule[(int(aid_s), int(tid_s))] = (float(value[0]), float(value[1]))
        return AssignmentSolution(
            routes=routes,
            unassigned=unassigned,
            schedule=schedule,
            solver_diagnostics=dict(data.get("solver_diagnostics", {}) or {}),
            run_summary=dict(data.get("run_summary", {}) or {}),
        )
