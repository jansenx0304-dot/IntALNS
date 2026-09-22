"""Lightweight benchmark instance loading without importing the LLM stack."""
from __future__ import annotations

import json
from pathlib import Path

from .models import Agent, Depot, Instance, Task


def _location(raw: dict) -> tuple[float, float]:
    if "loc" in raw:
        return tuple(float(value) for value in raw["loc"])
    return (float(raw["x"]), float(raw["y"]))


def load_instance_from_json(path: str | Path) -> Instance:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    depot_raw = dict(data["depot"])
    depot = Depot(id=int(depot_raw["id"]), loc=_location(depot_raw))
    platform_fields = {"id", "init_energy", "skills", "speed", "travel_energy_rate"}
    for raw in data["agents"]:
        if set(raw) != platform_fields:
            raise ValueError("Platform fields must match the transfer-only schema exactly")
    agents = tuple(
        Agent(
            id=int(raw["id"]),
            init_energy=float(raw["init_energy"]),
            skills=set(raw["skills"]),
            speed=float(raw["speed"]),
            travel_energy_rate=float(raw["travel_energy_rate"]),
        )
        for raw in data["agents"]
    )
    tasks = tuple(
        Task(
            id=int(raw["id"]),
            loc=_location(raw),
            tw_start=float(raw["tw_start"]),
            tw_end=float(raw["tw_end"]),
            service_time=float(raw["service_time"]),
            skill_req=set(raw["skill_req"]),
            priority=float(raw["priority"]),
        )
        for raw in data["tasks"]
    )
    return Instance(
        tasks=tasks,
        agents=agents,
        depot=depot,
        default_speed=float(data["default_speed"]),
    )


__all__ = ["load_instance_from_json"]
