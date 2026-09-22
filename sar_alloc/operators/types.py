"""Typed lower-layer controls for the refactored ALNS.

The public controls deliberately separate four responsibilities:

* operators generate a neighbourhood or choose an insertion position;
* focus marks targets within the common pending-task pool;
* feature-control weights remain zero;
* selectors make a random choice, with a bounded focus bias when active.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Tuple

DESTROY_OPERATOR_NAMES: Tuple[str, ...] = (
    "random_removal",
    "worst_cost_removal",
    "spatial_related_removal",
    "time_related_removal",
)

INITIAL_INSERTION_OPERATOR_NAMES: Tuple[str, ...] = (
    "best_insertion",
    "random_insertion",
)

ALNS_REPAIR_OPERATOR_NAMES: Tuple[str, ...] = (
    "best_insertion",
    "random_insertion",
)
DESTROY_FEATURE_NAMES: Tuple[str, ...] = (
    "route_cost_release",
    "target_feasibility_gain",
    "bottleneck_relief",
)

INSERTION_TASK_FEATURE_NAMES: Tuple[str, ...] = (
    "cheapest_score",
    "regret_score",
    "scarcity_score",
)

SELECTOR_MODES: Tuple[str, ...] = ("random",)

INTERNAL_ACCEPTANCE_MODES: Tuple[str, ...] = ("greedy", "threshold", "sa")


@dataclass(frozen=True, slots=True)
class DestroyPolicy:
    operator_weights: Dict[str, int]
    feature_weights: Dict[str, int]
    remove_ratio: float
    selector_mode: str = "random"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operator_weights": dict(self.operator_weights),
            "feature_weights": dict(self.feature_weights),
            "remove_ratio": float(self.remove_ratio),
            "selector_mode": str(self.selector_mode),
        }


@dataclass(frozen=True, slots=True)
class InsertionPolicy:
    operator_weights: Dict[str, int]
    task_feature_weights: Dict[str, int]
    task_selector_mode: str = "random"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "operator_weights": dict(self.operator_weights),
            "task_feature_weights": dict(self.task_feature_weights),
            "task_selector_mode": str(self.task_selector_mode),
            "position_selection": "selected_repair_operator",
        }


@dataclass(frozen=True, slots=True)
class AcceptancePolicy:
    mode: str
    worsening_tolerance: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "worsening_tolerance": self.worsening_tolerance,
        }


@dataclass(frozen=True, slots=True)
class CompiledALNSPolicy:
    destroy_policy: DestroyPolicy
    insertion_policy: InsertionPolicy
    acceptance_policy: AcceptancePolicy
    reaction_factor: float = 0.20

    def as_dict(self) -> Dict[str, Any]:
        return {
            "destroy_policy": self.destroy_policy.as_dict(),
            "insertion_policy": self.insertion_policy.as_dict(),
            "acceptance_policy": self.acceptance_policy.as_dict(),
            "reaction_factor": self.reaction_factor,
        }


@dataclass(frozen=True, slots=True)
class LandscapeFeatures:
    route_cost_release: float = 0.0
    target_feasibility_gain: float = 0.0
    bottleneck_relief: float = 0.0
    cheapest_score: float = 0.0
    regret_score: float = 0.0
    scarcity_score: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {field.name: float(getattr(self, field.name)) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class InsertPosition:
    agent_id: int
    position: int

    def as_dict(self) -> Dict[str, int]:
        return {"agent_id": self.agent_id, "position": self.position}


__all__ = [
    "INTERNAL_ACCEPTANCE_MODES",
    "SELECTOR_MODES",
    "DESTROY_OPERATOR_NAMES",
    "DESTROY_FEATURE_NAMES",
    "INITIAL_INSERTION_OPERATOR_NAMES",
    "ALNS_REPAIR_OPERATOR_NAMES",
    "INSERTION_TASK_FEATURE_NAMES",
    "DestroyPolicy",
    "InsertionPolicy",
    "AcceptancePolicy",
    "CompiledALNSPolicy",
    "LandscapeFeatures",
    "InsertPosition",
]
