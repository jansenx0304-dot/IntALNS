# sar_alloc/models.py
# 文件位置: sar_alloc/models.py (数据模型定义)
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, Mapping, Optional, Tuple, Set

Location = Tuple[float, float]  # (x, y)


@dataclass(frozen=True, slots=True)
class Task:
    id: int
    loc: Location
    tw_start: float
    tw_end: float
    service_time: float
    skill_req: Set[str] = field(default_factory=set)
    priority: float = 0.0

    def __post_init__(self) -> None:
        _validate_location(self.loc, f"task {self.id} location")
        for name, value in (
            ("tw_start", self.tw_start),
            ("tw_end", self.tw_end),
            ("service_time", self.service_time),
            ("priority", self.priority),
        ):
            _require_finite(value, f"task {self.id} {name}")
        if float(self.tw_end) < float(self.tw_start):
            raise ValueError(f"task {self.id} has tw_end < tw_start")
        if float(self.service_time) < 0.0:
            raise ValueError(f"task {self.id} service_time must be non-negative")
        if float(self.priority) < 0.0:
            raise ValueError(f"task {self.id} priority must be non-negative")


@dataclass(frozen=True, slots=True)
class Agent:
    """Heterogeneous platform with transfer energy per distance and a closed-route budget."""

    id: int
    init_energy: float
    skills: Set[str] = field(default_factory=set)

    # 每个智能体自己的速度（距离单位/时间单位）
    speed: float = 1.0

    # 行驶能耗：每距离单位能耗，项目内统一采用该口径。
    travel_energy_rate: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("init_energy", self.init_energy),
            ("speed", self.speed),
            ("travel_energy_rate", self.travel_energy_rate),
        ):
            _require_finite(value, f"agent {self.id} {name}")
        if float(self.init_energy) < 0.0:
            raise ValueError(f"agent {self.id} init_energy must be non-negative")
        if float(self.speed) < 0.0:
            raise ValueError(f"agent {self.id} speed must be non-negative")
        if float(self.travel_energy_rate) < 0.0:
            raise ValueError(f"agent {self.id} energy rates must be non-negative")


@dataclass(frozen=True, slots=True)
class Depot:
    id: int
    loc: Location

    def __post_init__(self) -> None:
        _validate_location(self.loc, "depot location")


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def _validate_location(value: Location, name: str) -> None:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two coordinates")
    _require_finite(value[0], f"{name}[0]")
    _require_finite(value[1], f"{name}[1]")


def euclidean(a: Location, b: Location) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


@dataclass(frozen=True, slots=True)
class Instance:
    """
    问题实例：任务、智能体、救援中心（depot），以及距离/时间计算函数。

    - 若提供 travel_time_fn：优先用它
    - 否则：travel_time = distance / agent.speed（speed<=0 时回退 default_speed）
    """

    tasks: Tuple[Task, ...]
    agents: Tuple[Agent, ...]
    depot: Depot

    distance_fn: Callable[[Location, Location], float] = euclidean
    travel_time_fn: Optional[Callable[[Agent, Location, Location], float]] = None

    # 兜底速度（当 agent.speed <= 0 时使用）
    default_speed: float = 1.0

    # 运行时缓存，避免在 ALNS 试探评估时重复计算。Instance 是 frozen
    # dataclass，但这些缓存容器可在运行时填充。自定义 travel_time_fn 不缓存，
    # 因为它可能依赖外部状态；默认静态 travel time 才安全缓存。
    _dist_cache: Dict[Tuple[Location, Location], float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _tt_cache: Dict[Tuple[int, Location, Location], float] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _task_index_cache: Dict[int, Task] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _agent_index_cache: Dict[int, Agent] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        task_ids = [int(task.id) for task in self.tasks]
        agent_ids = [int(agent.id) for agent in self.agents]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task ids must be unique")
        if len(agent_ids) != len(set(agent_ids)):
            raise ValueError("agent ids must be unique")
        if not agent_ids:
            raise ValueError("instance must contain at least one agent")
        _require_finite(self.default_speed, "default_speed")
        if float(self.default_speed) <= 0.0:
            raise ValueError("default_speed must be positive")

    def task_by_id(self, tid: int) -> Task:
        # cached index
        if not self._task_index_cache:
            self._task_index_cache.update({t.id: t for t in self.tasks})
        return self._task_index_cache[tid]

    def agent_by_id(self, aid: int) -> Agent:
        # cached index
        if not self._agent_index_cache:
            self._agent_index_cache.update({a.id: a for a in self.agents})
        return self._agent_index_cache[aid]

    def all_task_ids(self) -> Tuple[int, ...]:
        return tuple(t.id for t in self.tasks)

    def all_agent_ids(self) -> Tuple[int, ...]:
        return tuple(a.id for a in self.agents)

    def distance(self, a: Location, b: Location) -> float:
        # fast-path: memoize symmetric distance
        key = (a, b)
        v = self._dist_cache.get(key)
        if v is not None:
            return v

        d = float(self.distance_fn(a, b))
        if not math.isfinite(d) or d < 0.0:
            raise ValueError(f"distance_fn returned invalid distance: {d}")
        self._dist_cache[key] = d
        # Only the built-in Euclidean metric is guaranteed symmetric. A custom
        # distance function may represent directed travel costs.
        if self.distance_fn is euclidean:
            self._dist_cache[(b, a)] = d
        return d

    def travel_time(self, agent: Agent, a: Location, b: Location) -> float:
        if self.travel_time_fn is not None:
            # Custom travel time may be state/time dependent; never cache it
            # behind a key that omits that state.
            value = float(self.travel_time_fn(agent, a, b))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"travel_time_fn returned invalid travel time: {value}")
            return value

        tkey = (agent.id, a, b)
        tv = self._tt_cache.get(tkey)
        if tv is not None:
            return tv

        dist = self.distance(a, b)
        speed = float(agent.speed) if float(agent.speed) > 0.0 else float(self.default_speed)
        t = dist / speed
        self._tt_cache[tkey] = t
        return t

    def clear_caches(self) -> None:
        """手动清空缓存（例如 travel_time_fn 随环境变化时）。"""
        self._dist_cache.clear()
        self._tt_cache.clear()
        self._task_index_cache.clear()
        self._agent_index_cache.clear()
