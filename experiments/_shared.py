from __future__ import annotations
from typing import Any
from sar_alloc.config import Config
from sar_alloc.evaluator import evaluate
from sar_alloc.models import Instance
from sar_alloc.instance_io import load_instance_from_json
from sar_alloc.paths import project_path
from sar_alloc.solution import AssignmentSolution
from experiments._objective import SEARCH_OBJECTIVE_METRICS as QUALITY_FIELDS

def load_case(row: dict[str, Any]) -> Instance:
    return load_instance_from_json(project_path(str(row["instance_path"])))


def quality_dict(solution: AssignmentSolution, instance: Instance, cfg: Config) -> dict[str, float]:
    q = evaluate(solution, instance, cfg, update_solution_schedule=False).quality_metrics
    return {name: float(q[name]) for name in QUALITY_FIELDS}


def compact_solution_dict(
    solution: AssignmentSolution, instance: Instance, cfg: Config
) -> dict[str, Any]:
    """Serialize the final assignment without duplicating runtime diagnostics."""
    evaluate(solution, instance, cfg, update_solution_schedule=True)
    payload = solution.to_dict()
    return {
        "routes": dict(payload["routes"]),
        "unassigned": list(payload["unassigned"]),
        "schedule": dict(payload["schedule"]),
    }
