'Search memory and validated controls for the single-agent method.'

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping
from .focus_feedback import FOCUS_FEEDBACK_KIND

from .domain import (
    DETERMINISTIC_INFORMATIVE_TRIALS,
    SERVICE_METRICS,
    STOCHASTIC_INFORMATIVE_TRIALS,
    public_global_objective,
)








@dataclass(slots=True)
class RunProgress:
    """Factual action and stagnation counters; no stage or stopping controller."""
    executable_actions: int = 0
    alns_actions: int = 0
    global_best_updates: int = 0
    actions_since_global_best: int = 0
    iters_since_global_best: int = 0

    def apply_result(self, result):
        self.executable_actions += 1
        self.alns_actions += 1
        updates = int(result["outcome_counts"].get("global_best_update_count", 0))
        self.global_best_updates += updates
        if updates:
            self.actions_since_global_best = self.iters_since_global_best = 0
        else:
            self.actions_since_global_best += 1
            self.iters_since_global_best += int(result["iters_used"])

    def as_dict(self, *, max_iters):
        return {"executable_actions": self.executable_actions, "alns_actions": self.alns_actions,
            "global_best_updates": self.global_best_updates,
            "actions_since_global_best": self.actions_since_global_best,
            "iters_since_global_best": self.iters_since_global_best}


@dataclass(slots=True)
class SearchMemory:
    """Compact cross-action factual memory shared by every formal variant."""

    adaptive_destroy_weights: Dict[str, float] = field(default_factory=dict)
    adaptive_repair_weights: Dict[str, float] = field(default_factory=dict)
    adaptive_destroy_scores: Dict[str, float] = field(default_factory=dict)
    adaptive_destroy_uses: Dict[str, int] = field(default_factory=dict)
    adaptive_repair_scores: Dict[str, float] = field(default_factory=dict)
    adaptive_repair_uses: Dict[str, int] = field(default_factory=dict)
    adaptation_trials_elapsed: int = 0
    operator_pair_history: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    acceptance_history: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    control_history: List[Dict[str, Any]] = field(default_factory=list)
    control_actions: int = 0

    CONTROL_HISTORY_LIMIT = 12

    def adaptation_state(self) -> Dict[str, Any]:
        return {
            "adaptive_destroy_weights": dict(self.adaptive_destroy_weights),
            "adaptive_repair_weights": dict(self.adaptive_repair_weights),
            "adaptive_destroy_scores": dict(self.adaptive_destroy_scores),
            "adaptive_destroy_uses": dict(self.adaptive_destroy_uses),
            "adaptive_repair_scores": dict(self.adaptive_repair_scores),
            "adaptive_repair_uses": dict(self.adaptive_repair_uses),
            "adaptation_trials_elapsed": int(self.adaptation_trials_elapsed),
        }

    def apply_result(self, result: Mapping[str, Any]) -> None:
        for field_name, cast in (
            ("adaptive_destroy_weights", float),
            ("adaptive_repair_weights", float),
            ("adaptive_destroy_scores", float),
            ("adaptive_destroy_uses", int),
            ("adaptive_repair_scores", float),
            ("adaptive_repair_uses", int),
        ):
            key = f"{field_name}_after"
            if key in result:
                setattr(self, field_name, {str(k): cast(v) for k, v in dict(result.get(key, {}) or {}).items()})
        if "adaptation_trials_elapsed_after" in result:
            self.adaptation_trials_elapsed = int(result.get("adaptation_trials_elapsed_after", 0) or 0)

        feedback = dict(result.get("decision_feedback", {}) or {})
        for pair, raw in dict(dict(feedback.get("operator_priority", {}) or {}).get("operator_pairs", {}) or {}).items():
            if not isinstance(raw, Mapping):
                continue
            bucket = self.operator_pair_history.setdefault(str(pair), {})
            for name in ("used", "accepted", "improved", "best_updates", "structurally_changed", "accepted_structurally_changed", "destroy_repair_noop"):
                bucket[name] = int(bucket.get(name, 0) or 0) + int(raw.get(name, 0) or 0)
            bucket["total_reward"] = round(float(bucket.get("total_reward", 0.0) or 0.0) + float(raw.get("total_reward", 0.0) or 0.0), 6)

        acceptance = dict(feedback.get("acceptance", {}) or {})
        selected_acceptance = dict(acceptance.get("selected", {}) or {})
        mode = str(selected_acceptance.get("mode", "") or "")
        if mode:
            tolerance = float(selected_acceptance.get("worsening_tolerance", 0.0) or 0.0)
            key = f"{mode}|{tolerance:.12g}"
            bucket = self.acceptance_history.setdefault(key, {"mode": mode, "worsening_tolerance": tolerance, "actions": 0})
            bucket["actions"] = int(bucket.get("actions", 0) or 0) + 1
            for name in ("candidate_trials", "accepted_improving", "accepted_equal", "accepted_worsening", "rejected", "best_updates"):
                bucket[name] = int(bucket.get(name, 0) or 0) + int(acceptance.get(name, 0) or 0)

        if str(result.get("executed_action", "") or "") == "run_alns" and feedback:
            self._append_control_history(result, feedback)

    def _append_control_history(self, result: Mapping[str, Any], feedback: Mapping[str, Any]) -> None:
        focus = dict(dict(feedback.get("focus", {}) or {}).get("selected", {}) or {})
        operator = dict(dict(feedback.get("operator_priority", {}) or {}).get("selected", {}) or {})
        bias = dict(dict(feedback.get("search_bias", {}) or {}).get("selected", {}) or {})
        acceptance_feedback = dict(feedback.get("acceptance", {}) or {})
        acceptance = dict(acceptance_feedback.get("selected", {}) or {})
        flow = dict(feedback.get("search_flow", {}) or {})
        budget = dict(feedback.get("execution_budget", {}) or {})
        objective = dict(feedback.get("objective_progress", {}) or {})
        high = dict(result.get("search_policy", {}) or {})
        self.control_actions += 1
        row = {
            "action_index": int(self.control_actions),
            "focus_id": str(high.get("focus_id", focus.get("focus_id", "global")) or "global"),
            "destroy_feature_priority": [str(v) for v in high.get("destroy_feature_priority", []) or []],
            "task_feature_priority": [str(v) for v in high.get("task_feature_priority", []) or []],
            "destroy_priority": [str(v) for v in operator.get("destroy_priority", []) or []],
            "repair_priority": [str(v) for v in operator.get("repair_priority", []) or []],
            "destroy_selector": str(bias.get("destroy_selector", "") or ""),
            "task_selector": str(bias.get("task_selector", "") or ""),
            "remove_ratio": float(bias.get("remove_ratio", 0.0) or 0.0),
            "acceptance_mode": str(acceptance.get("mode", "") or ""),
            "acceptance_evidence": {key: deepcopy(acceptance_feedback.get(key, {} if key in {"rejection_reasons", "worsening_by_metric"} else 0))
                                    for key in ("candidate_trials", "accepted_improving", "accepted_equal", "accepted_worsening", "rejected", "rejection_reasons", "worsening_by_metric")},
            "worsening_tolerance": float(acceptance.get("worsening_tolerance", 0.0) or 0.0),
            "used_iters": int(budget.get("used_trials", result.get("iters_used", 0)) or 0),
            "global_trial_start": int(result.get("global_trial_start", 0) or 0),
            "global_trial_end": int(result.get("global_trial_end", 0) or 0),
            "global_best_update_count": int(dict(result.get("outcome_counts", {}) or {}).get("global_best_update_count", 0) or 0),
            "objective_comparison": str(objective.get("comparison", "equal") or "equal"),
            "objective_before": [float(v) for v in objective.get("global_best_before", []) or []],
            "objective_after": [float(v) for v in objective.get("global_best_after", []) or []],
            "noop_rate": float(flow.get("noop_rate", 0.0) or 0.0),
            "structure_change_rate": float(flow.get("structure_change_rate", 0.0) or 0.0),
            "feasible_candidate_rate": float(
                flow.get("feasible_candidate_rate", 0.0) or 0.0
            ),
            "acceptance_rate": float(flow.get("acceptance_rate", 0.0) or 0.0),
            "structural_passage_rate": float(flow.get("structural_passage_rate", 0.0) or 0.0),
            "focus_changed_selections": int(flow.get("focus_changed_selections", 0) or 0),
            "focus_run_best_updates": int(flow.get("focus_run_best_updates", 0) or 0),
        }
        if result.get("focus_feedback_kind") == FOCUS_FEEDBACK_KIND:
            window = deepcopy(result["focus_review_window"])
            if window["action_index"] != self.control_actions or window["focus_id"] != row["focus_id"]:
                raise ValueError("focus feedback window does not match the executed action")
            row["focus_review_window"] = window
            # 汇总字段与逐窗口服务目标证据保持一致。
            row["focus_changed_selections"] = window["selection_difference_diagnostic"]
            row["focus_run_best_updates"] = window["best_updates_while_focus_active"]
        self.control_history.append(row)
        if len(self.control_history) > self.CONTROL_HISTORY_LIMIT:
            del self.control_history[:-self.CONTROL_HISTORY_LIMIT]

    def recent_controls(self, limit: int = 8) -> List[Dict[str, Any]]:
        rows = self.control_history
        return [dict(row) for row in rows[-max(0, int(limit)):]]






    def as_dict(self) -> Dict[str, Any]:
        return {**self.adaptation_state(), "operator_pair_history": {str(k): dict(v) for k, v in self.operator_pair_history.items()}, "acceptance_history": {str(k): dict(v) for k, v in self.acceptance_history.items()}, "recent_control_history": self.recent_controls(limit=self.CONTROL_HISTORY_LIMIT)}




@dataclass(slots=True)
class StepDecision:
    action: str
    raw: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "raw": dict(self.raw)}


@dataclass(slots=True)
class RuntimeControl:
    action: str
    compiled_search_focus: Dict[str, Any] = field(default_factory=dict)
    insertion_policy: Any | None = None
    destroy_policy: Any | None = None
    alns_acceptance_policy: Any | None = None
    execution_budget: Dict[str, Any] = field(default_factory=dict)
    compiler_transform: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "action": self.action,
            "compiler_transform": dict(self.compiler_transform or {}),
        }
        if self.insertion_policy is not None:
            data["insertion_policy"] = _policy_dict(self.insertion_policy)
        if self.action == "run_alns":
            data["compiled_search_focus"] = dict(self.compiled_search_focus)
            data["destroy_policy"] = _policy_dict(self.destroy_policy)
            data["acceptance_policy"] = _policy_dict(self.alns_acceptance_policy)
            data["execution_budget"] = dict(self.execution_budget)
        return data




@dataclass(frozen=True, slots=True)
class CompiledSearchFocus:
    """Resolved semantic focus used as a bounded soft candidate preference."""

    focus_id: str
    focus_task_ids: List[int] = field(default_factory=list)
    minimum_target_hits: int = 1
    soft_greedy_bonus: float = 1.0
    soft_random_weight: float = 1.5

    def as_dict(self) -> Dict[str, Any]:
        return {
            "focus_id": str(self.focus_id),
            "focus_task_ids": [int(tid) for tid in self.focus_task_ids],
            "focus_task_count": len(self.focus_task_ids),
            "minimum_target_hits": max(1, int(self.minimum_target_hits)),
            "soft_greedy_bonus": float(self.soft_greedy_bonus),
            "soft_random_weight": float(self.soft_random_weight),
        }

    def as_executor_focus(self) -> Dict[str, Any]:
        return self.as_dict()


@dataclass(frozen=True, slots=True)
class RuntimeContract:
    allowed_actions: List[str]
    remaining: Dict[str, Any]
    stop_reasons: List[str] = field(default_factory=list)
    search_focus_expansions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    controls: Dict[str, Any] = field(default_factory=dict)
    decision_landscape: Dict[str, Any] = field(default_factory=dict)


def _policy_dict(value: Any) -> Dict[str, Any] | None:
    if value is None:
        return None
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    return dict(value)
