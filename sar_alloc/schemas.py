"""Small live JSON schemas for the frozen ownership split."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Dict

from .agent_types import RuntimeContract
from .domain import PRIORITY_RANK_CAPACITY, remaining_iters




def step_schema_from_contract(
    contract: RuntimeContract,
    *,
    observation_blocks: Iterable[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    blocks = list(observation_blocks or [])
    branches: list[Dict[str, Any]] = []
    if "run_alns" in contract.allowed_actions:
        branches.append(_run_alns(contract, blocks))
    return _root("step_decision", branches)








def _run_alns(contract: RuntimeContract, blocks: list[Mapping[str, Any]]) -> Dict[str, Any]:
    destroy = [str(v) for v in contract.controls.get("destroy_operators", []) or []]
    repair = [str(v) for v in contract.controls.get("repair_operators", []) or []]
    exact_pool = bool(contract.controls.get("priority_must_cover_pool", False))
    selectors = [str(v) for v in contract.controls.get("selector_modes", []) or []]
    control = _object(
        {
            # Operator priorities may rank every currently legal operator.  Their
            # length is bounded by the live enum itself, not by the historical
            # three-entry weight preset used for feature priorities.
            "destroy_priority": _ordered(destroy, max_items=None, exact=exact_pool),
            "repair_priority": _ordered(repair, max_items=None, exact=exact_pool),
            "search_bias": _object(
                {
                    "destroy_selector": {"type": "string", "enum": selectors},
                    "task_selector": {"type": "string", "enum": selectors},
                },
                ["destroy_selector", "task_selector"],
            ),
            "acceptance": _acceptance(contract),
            "remove_ratio": {"type": "number", "enum": [float(v) for v in contract.controls.get("remove_ratio_options", []) or []]},
        },
        ["destroy_priority", "repair_priority", "search_bias", "acceptance", "remove_ratio"],
    )
    props: Dict[str, Any] = {"action": {"type": "string", "const": "run_alns"}, "control": control}
    focus_options = list(contract.controls.get("step_focus_options", []) or [])
    if focus_options:
        control["properties"]["focus_id"] = {"type": "string", "enum": focus_options}
        control["required"].append("focus_id")
    required = ["action", "control"]
    props["rationale"] = _rationale(blocks); required.append("rationale")
    return _object(props, required)


def _acceptance(contract: RuntimeContract) -> Dict[str, Any]:
    modes = [str(v) for v in contract.controls.get("acceptance_modes", []) or []]
    positive = [float(v) for v in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20) if float(v) <= float(contract.controls.get("max_worsening_tolerance", 0.20) or 0.20)]
    branches = []
    if "greedy" in modes:
        branches.append(_object({"mode": {"type": "string", "const": "greedy"}}, ["mode"]))
    for mode in ("tolerant", "simulated_annealing"):
        if mode in modes:
            branches.append(_object({"mode": {"type": "string", "const": mode}, "worsening_tolerance": {"type": "number", "enum": positive}}, ["mode", "worsening_tolerance"]))
    return {"oneOf": branches}


def _rationale(blocks: list[Mapping[str, Any]]) -> Dict[str, Any]:
    refs: list[str] = []
    for block in blocks:
        bid = str(block.get("block_id", "") or "")
        for name in dict(block.get("signals", {}) or {}):
            refs.append(f"{bid}.{name}")
    return _object(
        {
            "reason": {"type": "string"},
            "refs": {"type": "array", "uniqueItems": True, "items": {"type": "string", "enum": refs}},
        },
        ["reason", "refs"],
    )


def _ordered(values: Iterable[Any], *, max_items: int | None, exact: bool, allow_empty: bool = False) -> Dict[str, Any]:
    vals = [str(v) for v in values]
    schema: Dict[str, Any] = {
        "type": "array",
        "minItems": len(vals) if exact else (0 if allow_empty else 1),
        "uniqueItems": True,
        "items": {"type": "string", "enum": vals},
    }
    if exact:
        schema["maxItems"] = len(vals)
    elif max_items is not None:
        schema["maxItems"] = min(int(max_items), len(vals))
    return schema


def _object(properties: Mapping[str, Any], required: list[str]) -> Dict[str, Any]:
    return {"type": "object", "properties": dict(properties), "required": list(required), "additionalProperties": False}


def _root(key: str, branches: list[Dict[str, Any]]) -> Dict[str, Any]:
    if not branches:
        branches = [_object({}, [])]
    return _object({key: {"oneOf": branches}}, [key])
