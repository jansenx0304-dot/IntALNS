"""Single-source acceptance normalization shared by execution and AgentInput."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
from typing import Dict
from .evaluator import EvalResult
from .models import Instance


def build_acceptance_scales(
    instance: Instance,
    action_start_eval: EvalResult,
    objective_terms: Sequence[Mapping[str, object]],
) -> Dict[str, float]:
    total_task_count = max(1.0, float(len(instance.tasks)))
    total_priority = max(1.0, sum(max(0.0, float(task.priority)) for task in instance.tasks))
    scales: Dict[str, float] = {}
    for layer in objective_terms:
        metric = str(layer.get("metric", ""))
        if metric == "missed_priority":
            scale = total_priority
        elif metric == "unassigned_count":
            scale = total_task_count
        else:
            scale = max(1.0, abs(float(action_start_eval.get_quality_metric(metric))))
        scales[metric] = float(scale)
    return scales

__all__ = ["build_acceptance_scales"]
