"""Code-faithful static semantics for the decision roles."""
from __future__ import annotations
from typing import Any, Dict
from .domain import SOFT_FOCUS_RANDOM_WEIGHT

from .operator_catalog import (
    build_acceptance_cards,
    build_destroy_operator_cards,
    build_repair_operator_cards,
)




def build_step_decision_catalog(contract: Any) -> Dict[str, Any]:
    catalog = {
        "fixed_policy": {
            "focus_id": str(contract.controls.get("active_focus_id", "global")),
            "destroy_feature_priority": list(contract.controls.get("active_destroy_feature_priority", []) or []),
            "task_feature_priority": list(contract.controls.get("active_task_feature_priority", []) or []),
        },
        "ownership": "Step chooses only Destroy/Repair priorities, remove ratio, selectors, and acceptance for the next fixed 100-trial action.",
        "destroy_operators": build_destroy_operator_cards(contract.controls.get("destroy_operators", [])),
        "repair_operators": build_repair_operator_cards(contract.controls.get("repair_operators", [])),
        "remove_ratio_options": [float(v) for v in contract.controls.get("remove_ratio_options", []) or []],
        "selector_options": list(contract.controls.get("selector_modes", []) or []),
        "acceptance": build_acceptance_cards(contract.controls.get("acceptance_modes", [])),
    }
    focus_options = list(contract.controls.get("step_focus_options", []) or [])
    if focus_options:
        fixed = catalog.pop("fixed_policy")
        fixed.pop("focus_id")
        catalog["fixed_search_policy"] = fixed
        catalog["step_focus_control"] = True
        catalog["ownership"] = "Step chooses one focus_id plus Destroy/Repair priorities, remove ratio, the exposed selectors, and acceptance for each fixed 100-trial action. Both semantic feature vectors remain zero."
        catalog["focus_options"] = focus_options
        catalog["focus_selection_rule"] = (
            "global uses ordinary uniform random selection. service_bottleneck uses the live target set shown in the boundary: "
            "at most six currently unassigned tasks, ranked by feasible insertion count ascending then priority descending. "
            f"Target repair tasks receive sampling weight {SOFT_FOCUS_RANDOM_WEIGHT}, all other tasks weight 1. "
            "The common candidate pool stays intact. Targets are fixed for this action and resolved tasks lose their bias; "
            "the next Step may choose a new focus using refreshed targets. Focus does not identify blocking assigned tasks for removal."
        )
    if contract.controls.get("focus_feedback_kind"):
        catalog["focus_feedback_kind"] = contract.controls["focus_feedback_kind"]
    if contract.controls.get('search_evidence_kind'):
        catalog['search_evidence_kind'] = contract.controls['search_evidence_kind']
    return catalog
