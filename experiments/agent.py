'Recording adapter for the fixed single-agent search loop.'
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Callable


from sar_alloc.config import Budget, Config
from sar_alloc.evaluator import evaluate
from sar_alloc.solution import AssignmentSolution

from experiments._shared import compact_solution_dict, load_case, quality_dict
from experiments._objective import (
    SEARCH_OBJECTIVE,
    SEARCH_OBJECTIVE_METRICS,
    objective_dict,
)





def sample_progress(points: list[dict[str, Any]], checkpoints: tuple[int, ...], *, final_trial: int) -> list[dict[str, Any]]:
    ordered = sorted(points, key=lambda row: int(row.get("trial", 0) or 0))
    if not ordered or int(ordered[0].get("trial", -1)) != 0:
        raise ValueError("convergence trace must begin at the common stored start at trial 0")
    out: list[dict[str, Any]] = []
    pos = 0
    current = ordered[0]
    for checkpoint in checkpoints:
        if checkpoint > final_trial:
            continue
        while pos + 1 < len(ordered) and int(ordered[pos + 1].get("trial", 0) or 0) <= checkpoint:
            pos += 1
            current = ordered[pos]
        out.append({
            "trial": int(checkpoint),
            **{name: float(current[name]) for name in SEARCH_OBJECTIVE_METRICS},
        })
    return out


class TimedClient:
    def __init__(
        self,
        client: Any,
        phase_callback: Callable[[str, int], None] | None = None,
    ) -> None:
        self.client = client
        self.total_sec = 0.0
        self.calls = 0
        self.phase_callback = phase_callback

    def _phase(self, status: str, calls: int) -> None:
        if self.phase_callback is not None:
            self.phase_callback(status, calls)

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        t0 = time.perf_counter()
        try:
            self._phase("WAITING API", self.calls + 1)
            return self.client.chat(messages, **kwargs)
        finally:
            self.total_sec += time.perf_counter() - t0
            self.calls += 1
            self._phase("RUNNING ALNS", self.calls)

    def usage_summary(self) -> dict[str, int]:
        fn = getattr(self.client, "usage_summary", None)
        return dict(fn() if callable(fn) else {"requests": self.calls})


def run_agent(
    row: dict[str, Any],
    start: dict[str, Any],
    *,
    algorithm_seed: int,
    trials: int,
    client: TimedClient,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    from sar_alloc.orchestrator import run_orchestrator

    instance = load_case(row)
    initial = AssignmentSolution.from_dict(dict(start["solution"])); initial.normalize(instance)
    cfg = Config(rng_seed=algorithm_seed)
    start_eval = evaluate(initial, instance, cfg, update_solution_schedule=True)
    if not start_eval.is_feasible:
        raise ValueError("stored Agent start is not hard-feasible")
    start_q = objective_dict(start_eval.quality_metrics)
    progress: list[dict[str, Any]] = [{"trial": 0, **start_q}]
    actions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    last_end = 0
    last_llm_total = 0.0; last_llm_calls = 0

    def callback(record: Any) -> None:
        nonlocal last_end, last_llm_total, last_llm_calls
        agent_input = dict(record.agent_input or {}); agent_output = dict(record.agent_output or {})
        validation = dict(record.validation or {}); decision = dict(agent_output.get("decision", {}) or {})
        context = dict(agent_input.get("context", {}) or {}); observation = dict(context.get("observation", {}) or {})
        compact: dict[str, Any] = {}
        for block in observation.get("observation_blocks", []) or []:
            bid = str(block.get("block_id", "") or ""); sig = dict(block.get("signals", {}) or {})
            if bid.endswith("01_state"):
                compact["state"] = {k: sig.get(k) for k in ("phase", "task_count", "current_quality", "run_best_quality", "current_minus_run_best", "current_matches_run_best", "global_trial", "budget_fraction_used", "trials_since_run_best_update") if k in sig}
            elif bid.endswith("02_bottleneck"):
                compact["bottleneck"] = {k: sig.get(k) for k in ("unassigned_task_count", "zero_feasible_task_count", "at_most_one_feasible_task_count", "feasible_position_count_distribution", "capable_agent_count_distribution", "single_capability_task_count", "service_bottleneck_target_ids") if k in sig}
            elif bid.endswith("03_feedback"):
                compact["feedback"] = {k: sig.get(k) for k in ("history_available", "focus_review", "search_evidence") if k in sig}
            elif bid.endswith("04_boundary"):
                compact["boundary"] = sig
        rationale = dict(decision.get("rationale", {}) or {})
        decisions.append({
            "decision_index": len(decisions)+1, "turn_id": str(record.turn_id), "role": str(record.role),
            "phase": str(record.phase), "action": record.action,
            "example_ids": [str(x.get("example_id")) for x in agent_input.get("decision_examples", []) or [] if isinstance(x, Mapping) and x.get("example_id")],
            "decision": decision, "observation_summary": compact,
            "rationale_refs": [str(v) for v in rationale.get("refs", []) or []],
            "validation": {"ok": bool(validation.get("ok", record.error is None)), "repair_count": int(validation.get("repair_count", 0) or 0), "errors": list(validation.get("errors", []) or []), "repair_attempts":list(validation.get('repair_attempts',[]) or [])},
            "error": record.error,
            "envelope_repair_count": int(validation.get('envelope_repair_count', 0)),
            "envelope_repairs": list(validation.get('envelope_repairs', [])),
        })
        if record.role != "step" or record.action != "run_alns" or not isinstance(record.result, Mapping):
            return
        result = dict(record.result or {}); execution = dict(result.get("execution", {}) or {})
        if execution.get("executed_action") != "run_alns": return
        start_trial = int(execution.get("global_trial_start", -1)); end_trial = int(execution.get("global_trial_end", -1))
        if start_trial != last_end or end_trial <= start_trial:
            raise RuntimeError(f"non-monotonic global trial clock: expected start {last_end}, got {start_trial}->{end_trial}")
        last_end = end_trial
        trace = dict(result.get("executor_trace", {}) or {})
        for point in trace.get("best_progress", []) or []:
            terms = [str(v) for v in point.get("objective_terms", []) or []]; values = [float(v) for v in point.get("objective_key", []) or []]
            if terms == list(SEARCH_OBJECTIVE_METRICS) and len(values) == len(terms):
                progress.append({"trial": int(point.get("trial", 0) or 0), **dict(zip(terms, values))})
        feedback = dict(execution.get("decision_feedback", {}) or {})
        high = dict(execution.get("search_policy", {}) or {})
        boundary = dict(compact.get("boundary", {}) or {})
        selected = dict(dict(feedback.get("search_bias", {}) or {}).get("selected", {}) or {})
        llm_delta = max(0.0, float(client.total_sec) - last_llm_total); call_delta = max(0, int(client.calls) - last_llm_calls)
        last_llm_total = float(client.total_sec); last_llm_calls = int(client.calls)
        actions.append({
            "action_index": len(actions)+1, "turn_id": str(record.turn_id),
            "global_trial_start": start_trial, "global_trial_end": end_trial,
            "iters_used": int(execution.get("iters_used", end_trial-start_trial) or 0),
            "search_policy": high, "focus_id": str(high.get("focus_id", "global") or "global"),
            "destroy_feature_priority": list(high.get("destroy_feature_priority", []) or []),
            "task_feature_priority": list(high.get("task_feature_priority", []) or []),
            "focus_observed": dict(feedback.get("focus", {}) or {}), "search_flow": dict(feedback.get("search_flow", {}) or {}),
            "objective_progress": dict(feedback.get("objective_progress", {}) or {}),
            "remove_ratio": selected.get("remove_ratio"), "destroy_selector": selected.get("destroy_selector"), "task_selector": selected.get("task_selector"),
            "operator_priority": dict(dict(feedback.get("operator_priority", {}) or {}).get("selected", {}) or {}),
            "acceptance": dict(dict(feedback.get("acceptance", {}) or {}).get("selected", {}) or {}),
            "solver_time_sec": float(execution.get("elapsed_sec", 0.0) or 0.0),
            "llm_time_since_previous_action_sec": round(llm_delta, 6), "llm_calls_since_previous_action": call_delta,
            "decision": decision,
        })
        if "focus_id" in dict(decision.get("control", {}) or {}):
            actions[-1]["focus_selection_diagnostics"] = dict(execution.get("search_focus_result", {}) or {})
        if "focus_review_window" in execution:
            actions[-1]["focus_feedback_revision"] = execution["focus_feedback_revision"]
            actions[-1]["focus_review_window"] = dict(execution["focus_review_window"])
        if progress_callback is not None:
            progress_callback(end_trial, int(client.calls))

    t0 = time.perf_counter()
    final = run_orchestrator(
        client, instance, "Minimize missed priority, then unassigned count, then closed-route transfer energy, in fixed lexicographic order.",
        config=cfg, budget=Budget(max_iters=int(trials)), rng_seed=algorithm_seed, trace_callback=callback,
        initial_solution=initial, fixed_objective_terms=SEARCH_OBJECTIVE, require_budget_exhaustion=True,
    )
    total_sec = time.perf_counter() - t0
    if last_end != trials:
        raise RuntimeError(f"Agent global trial clock ended at {last_end}, expected {trials}")
    final_q = quality_dict(final, instance, cfg); progress.append({"trial": trials, **objective_dict(final_q)})
    summary = dict(final.run_summary or {})
    run = {
        "start_quality": start_q, "final_quality": objective_dict(final_q),
        "final_solution": compact_solution_dict(final, instance, cfg),
        "hard_feasible": bool(final.eval.is_feasible if final.eval is not None else evaluate(final, instance, cfg).is_feasible),
        "trials_used": int(dict(summary.get("budget", {}) or {}).get("used", {}).get("iters", last_end)),
        "solver_time_sec": float(summary.get("solver_action_time_sec", 0.0) or 0.0),
        "llm_time_sec": round(float(client.total_sec), 6), "total_time_sec": float(total_sec),
        "llm_usage": client.usage_summary(), "run_alns_actions": int(summary.get("run_alns_actions", len(actions)) or len(actions)),
        "decision_events": len(decisions),
    }
    return run, progress, actions, decisions


def build_client(
    kind: str,
    *,
    model: str,
    reasoning: str,
    base_url: str | None,
    api_key: str | None,
    phase_callback: Callable[[str, int], None] | None = None,
) -> TimedClient:
    from sar_alloc.runner import build_llm_client

    if kind != "real":
        raise ValueError("Experiments require the real API client")
    return TimedClient(
        build_llm_client(
            base_url=base_url,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning,
        ),
        phase_callback=phase_callback,
    )


__all__ = [
    "TimedClient",
    "build_client",
    "run_agent",
    "sample_progress",
]
