"""Shared decision landscape for observation and runtime contract building."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict

from .config import Config
from .models import Instance
from .operators.destroy import build_destroy_landscape
from .operators.insertion import build_insertion_landscape
from .solution import AssignmentSolution


def build_decision_landscape(
    solution: AssignmentSolution,
    instance: Instance,
    config: Config,
    *,
    objective_terms: list[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    terms = [dict(item) for item in (objective_terms or [])]
    preference_metric = "energy_total"  # fixed third-level objective
    destroy = build_destroy_landscape(
        solution,
        instance,
        config,
        preference_metric=preference_metric,
    )
    insertion = build_insertion_landscape(
        solution, instance, config, objective_terms=terms
    )
    return {
        "insertion_facts": dict(insertion.get("insertion_facts", {}) or {}),
        "candidate_routes": [
            dict(item)
            for item in destroy.get("candidate_routes", []) or []
            if isinstance(item, dict)
        ],
    }


__all__ = ["build_decision_landscape"]
