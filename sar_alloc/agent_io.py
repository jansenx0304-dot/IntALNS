"""Strict JSON parsing, validation and compilation for the agent protocol."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping

try:
    from jsonschema.validators import validator_for
except ImportError:  # pragma: no cover
    from .schema_validation import validator_for

from .agent_types import (
    CompiledSearchFocus,
    RuntimeContract,
    RuntimeControl,
    StepDecision,
)
from .domain import (
    ACCEPTANCE_MODE_MAP,
    AGENT_ACTION_TRIALS,
    OBJECTIVE_FORM,
    ORDERED_PRIORITY_WEIGHTS,
    SERVICE_METRICS,
    SOFT_FOCUS_GREEDY_BONUS,
    SOFT_FOCUS_RANDOM_WEIGHT,
    remaining_iters,
    tolerance_value_options,
)
from .decision_rationale import resolve_decision_rationale, validate_rationale
from .operators import AcceptancePolicy, DestroyPolicy, InsertionPolicy


class AgentIOError(ValueError):
    def __init__(self, message: str, *, validation: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.validation = validation


def _new_validation_report() -> Dict[str, Any]:
    return {
        "ok": False,
        "parse_ok": False,
        "schema_ok": False,
        "semantic_ok": False,
        "rationale_ok": False,
        "compile_ok": False,
        "resolved_rationale": [],
        "compiled_control": None,
        "errors": [],
    }


def _fail(validation: Dict[str, Any], stage: str, message: str) -> None:
    validation["ok"] = False
    validation.setdefault("errors", []).append({"stage": stage, "message": message})


def _validation_stage(validation: Mapping[str, Any]) -> str:
    if not validation.get("parse_ok"): return "parse"
    if not validation.get("schema_ok"): return "schema"
    if not validation.get("rationale_ok"): return "rationale"
    if not validation.get("semantic_ok"): return "semantic"
    return "compile"




def parse_validate_compile_step(
    *, raw_text: str, schema: Dict[str, Any], observation: Dict[str, Any],
    contract: RuntimeContract,
) -> tuple[StepDecision, RuntimeControl, Dict[str, Any]]:
    validation = _new_validation_report()
    try:
        payload = parse_json(raw_text); validation["parse_ok"] = True
        _validate_schema(payload, schema); validation["schema_ok"] = True
        root = dict(payload["step_decision"])
        _validate_action_rationale(root, observation, field="step_decision")
        validation["resolved_rationale"] = resolve_decision_rationale(observation, root)
        validation["rationale_ok"] = True
        validate_step_decision(root, contract=contract); validation["semantic_ok"] = True
        decision = StepDecision(action=str(root["action"]), raw=dict(root))
        control = compile_step_decision(root, contract=contract)
        validation["compile_ok"] = True; validation["compiled_control"] = control.as_dict(); validation["ok"] = True
        return decision, control, validation
    except Exception as exc:
        message = str(exc); _fail(validation, _validation_stage(validation), message)
        raise AgentIOError(message, validation=validation) from exc


def parse_json(raw_text: str) -> Dict[str, Any]:
    try:
        value = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AgentIOError(f"LLM output is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentIOError("LLM output must be one JSON object")
    return value






def validate_step_decision(root: Mapping[str, Any], *, contract: RuntimeContract) -> None:
    action = str(root.get("action", ""))
    if action not in set(contract.allowed_actions):
        raise AgentIOError(f"step action is not allowed now: {action}")
    if action != "run_alns":
        raise AgentIOError(f"unsupported step action: {action}")

    control = dict(root.get("control", {}) or {})
    focus_options = list(contract.controls.get("step_focus_options", []) or [])
    if focus_options and control.get("focus_id") not in focus_options:
        raise AgentIOError("control.focus_id must select one available focus")
    _validate_priority(control.get("destroy_priority", []), contract.controls.get("destroy_operators", []), "control.destroy_priority", allow_empty=False)
    _validate_priority(control.get("repair_priority", []), contract.controls.get("repair_operators", []), "control.repair_priority", allow_empty=False)
    bias = dict(control.get("search_bias", {}) or {})
    for name in ("destroy_selector", "task_selector"):
        if str(bias.get(name, "")) not in {str(v) for v in contract.controls.get("selector_modes", []) or []}:
            raise AgentIOError(f"control.search_bias.{name} is not available")
    ratio = control.get("remove_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise AgentIOError("control.remove_ratio must be numeric")
    if not any(abs(float(ratio) - float(v)) <= 1e-9 for v in contract.controls.get("remove_ratio_options", []) or []):
        raise AgentIOError("control.remove_ratio is not available")
    _validate_acceptance_policy(dict(control.get("acceptance", {}) or {}), contract)




def compile_step_decision(root: Mapping[str, Any], *, contract: RuntimeContract) -> RuntimeControl:
    action = str(root["action"])
    control = dict(root["control"])
    bias = dict(control["search_bias"])
    step_focus_control = bool(contract.controls.get("step_focus_options"))
    focus_id = str(control["focus_id"] if step_focus_control else contract.controls.get("active_focus_id", "global") or "global")
    insertion_input = {
        "repair_priority": [str(v) for v in control["repair_priority"]],
        "task_feature_priority": [],
        "task_selector": str(bias["task_selector"]),
    }
    destroy_input = {
        "destroy_priority": [str(v) for v in control["destroy_priority"]],
        "destroy_feature_priority": [str(v) for v in contract.controls.get("active_destroy_feature_priority", []) or []],
        "destroy_selector": str(bias["destroy_selector"]),
        "remove_ratio": float(control["remove_ratio"]),
    }
    acceptance_input = dict(control["acceptance"])
    insertion_policy, insertion_transform = _compile_insertion_policy(insertion_input, contract)
    destroy_policy, destroy_transform = _compile_destroy_policy(destroy_input, contract)
    execution_budget = {"iters": min(AGENT_ACTION_TRIALS, max(1, remaining_iters(contract.remaining)))}
    high = {
        "focus_id": focus_id,
        "destroy_feature_priority": list(destroy_input["destroy_feature_priority"]),
        "task_feature_priority": list(insertion_input["task_feature_priority"]),
    }
    return RuntimeControl(
        action=action,
        compiled_search_focus=_compile_active_focus(contract, focus_id=focus_id),
        insertion_policy=insertion_policy,
        destroy_policy=destroy_policy,
        alns_acceptance_policy=_compile_acceptance_policy(acceptance_input),
        execution_budget=execution_budget,
        compiler_transform={
            "search_policy": high,
            **({"step_focus_control": True} if step_focus_control else {}),
            **({"focus_feedback_revision": contract.controls["focus_feedback_revision"]} if contract.controls.get("focus_feedback_revision") else {}),
            "rank_weights": list(ORDERED_PRIORITY_WEIGHTS),
            "decision_blocks": {
                "operator_priority": {"destroy_priority": list(destroy_input["destroy_priority"]), "repair_priority": list(insertion_input["repair_priority"])},
                "search_bias": {"destroy_selector": str(destroy_input["destroy_selector"]), "task_selector": str(insertion_input["task_selector"]), "remove_ratio": float(destroy_input["remove_ratio"])},
                "acceptance": acceptance_input,
                "execution_budget": execution_budget,
                "search_policy": high,
            },
            "insertion_transform": insertion_transform,
            "destroy_transform": destroy_transform,
        },
    )


def _compile_insertion_policy(root: Mapping[str, Any], contract: RuntimeContract) -> tuple[InsertionPolicy, Dict[str, Any]]:
    repair = [str(item) for item in root["repair_priority"]]
    task = [str(item) for item in root.get("task_feature_priority", []) or []]
    selector = str(root["task_selector"])
    return (
        InsertionPolicy(
            operator_weights=_rank_weights(repair, contract.controls.get("repair_operators", [])),
            task_feature_weights=_rank_weights(task, contract.controls.get("task_features", [])),
            task_selector_mode=selector,
        ),
        {"repair_priority": repair, "task_feature_priority": task, "task_selector": selector, "rank_weights": list(ORDERED_PRIORITY_WEIGHTS)},
    )


def _compile_destroy_policy(root: Mapping[str, Any], contract: RuntimeContract) -> tuple[DestroyPolicy, Dict[str, Any]]:
    destroy = [str(item) for item in root["destroy_priority"]]
    features = [str(item) for item in root.get("destroy_feature_priority", []) or []]
    selector = str(root["destroy_selector"])
    return (
        DestroyPolicy(
            operator_weights=_rank_weights(destroy, contract.controls.get("destroy_operators", [])),
            feature_weights=_rank_weights(features, contract.controls.get("destroy_features", [])),
            remove_ratio=float(root["remove_ratio"]),
            selector_mode=selector,
        ),
        {"destroy_priority": destroy, "destroy_feature_priority": features, "destroy_selector": selector, "remove_ratio": float(root["remove_ratio"]), "rank_weights": list(ORDERED_PRIORITY_WEIGHTS)},
    )


def _compile_acceptance_policy(raw: Mapping[str, Any]) -> AcceptancePolicy:
    mode = str(raw.get("mode", "greedy"))
    internal = ACCEPTANCE_MODE_MAP.get(mode)
    if internal is None:
        raise AgentIOError(f"unknown acceptance mode: {mode}")
    tolerance = 0.0 if mode == "greedy" else float(raw.get("worsening_tolerance", 0.0) or 0.0)
    return AcceptancePolicy(mode=internal, worsening_tolerance=tolerance)


def _compile_active_focus(contract: RuntimeContract, *, focus_id: str | None = None) -> Dict[str, Any]:
    focus_id = str(focus_id if focus_id is not None else contract.controls.get("active_focus_id", "global") or "global")
    expansion = dict(contract.search_focus_expansions.get(focus_id, {}) or {})
    return CompiledSearchFocus(
        focus_id=focus_id,
        focus_task_ids=_dedupe_ints(expansion.get("focus_task_ids", []) or []),
        minimum_target_hits=max(1, int(expansion.get("minimum_target_hits", 1) or 1)),
        soft_greedy_bonus=SOFT_FOCUS_GREEDY_BONUS,
        soft_random_weight=SOFT_FOCUS_RANDOM_WEIGHT,
    ).as_dict()


def _rank_weights(priority: Iterable[str], allowed_names: Iterable[str]) -> Dict[str, int]:
    ordered = [str(item) for item in priority]
    # Preserve the established 10/5/2 ranking and give any additional legal
    # ranks a small positive tail weight instead of rejecting or disabling them.
    weights = {
        name: int(
            ORDERED_PRIORITY_WEIGHTS[index]
            if index < len(ORDERED_PRIORITY_WEIGHTS)
            else 1
        )
        for index, name in enumerate(ordered)
    }
    return {str(name): int(weights.get(str(name), 0)) for name in allowed_names}


def _validate_priority(raw: Any, allowed: Iterable[Any], field: str, *, allow_empty: bool = True) -> None:
    if not isinstance(raw, list):
        raise AgentIOError(f"{field} must be an array")
    values = [str(v) for v in raw]
    if not allow_empty and not values:
        raise AgentIOError(f"{field} must not be empty")
    if len(values) != len(set(values)):
        raise AgentIOError(f"{field} must contain unique values")
    allowed_set = {str(v) for v in allowed}
    unknown = [v for v in values if v not in allowed_set]
    if unknown:
        raise AgentIOError(f"{field} contains unavailable values: {unknown}")


def _validate_action_rationale(root: Mapping[str, Any], observation: Mapping[str, Any], *, field: str) -> None:
    validate_rationale(root.get("rationale"), observation, field=f"{field}.rationale")




def _validate_acceptance_policy(raw: Mapping[str, Any], contract: RuntimeContract) -> None:
    mode = str(raw.get("mode", ""))
    allowed = {str(v) for v in contract.controls.get("acceptance_modes", []) or []}
    if mode not in allowed:
        raise AgentIOError(f"acceptance.mode is not available: {mode}")
    if mode == "greedy":
        if "worsening_tolerance" in raw:
            raise AgentIOError("greedy acceptance must not include worsening_tolerance")
        return
    value = raw.get("worsening_tolerance")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AgentIOError("acceptance.worsening_tolerance must be numeric")
    max_tol = float(contract.controls.get("max_worsening_tolerance", 0.0) or 0.0)
    if not any(abs(float(value) - float(v)) <= 1e-9 for v in tolerance_value_options(0.0, max_tol) if float(v) > 0.0):
        raise AgentIOError("acceptance.worsening_tolerance is not available")


def _dedupe_ints(values: Iterable[Any]) -> list[int]:
    out: list[int] = []; seen: set[int] = set()
    for value in values:
        if isinstance(value, bool): continue
        item = int(value)
        if item not in seen:
            seen.add(item); out.append(item)
    return out


@lru_cache(maxsize=64)
def _compiled_schema_validator(schema_json: str) -> Any:
    schema = json.loads(schema_json)
    cls = validator_for(schema); cls.check_schema(schema)
    return cls(schema)


def _leaf_schema_errors(error: Any) -> list[Any]:
    context = list(getattr(error, "context", []) or [])
    if not context:
        return [error]
    return [leaf for child in context for leaf in _leaf_schema_errors(child)]


def _validate_schema(payload: Dict[str, Any], schema: Dict[str, Any]) -> None:
    schema_json = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    errors = list(_compiled_schema_validator(schema_json).iter_errors(payload))
    if not errors:
        return
    leaves = [leaf for error in errors for leaf in _leaf_schema_errors(error)]
    error = max(
        leaves,
        key=lambda e: (len(list(e.absolute_path)), len(list(getattr(e, "absolute_schema_path", ())))),
    )
    path = "$" + "".join(f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path)
    raise AgentIOError(f"JSON schema validation failed at {path}: {error.message}")
