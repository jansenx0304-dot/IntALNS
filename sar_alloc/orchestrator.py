"""Single-agent loop: observe, validate a control, execute, and retain feedback."""

from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Protocol

from .agent_contract import RuntimeContractBuilder, build_runtime_decision_landscape
from .agent_io import (
    AgentIOError,
    parse_validate_compile_step,
)
from .agent_turn import PromptBundle, build_step_prompt_bundle
from .agent_types import (
    CompiledSearchFocus,
    RunProgress,
    RuntimeContract,
    RuntimeControl,
    SearchMemory,
)
from .config import Budget, Config
from .decision_catalog import build_step_decision_catalog
from .domain import (
    OBJECTIVE_FORM,
    QUALITY_METRICS,
    SERVICE_METRICS,
    public_global_objective,
    terms_from_order,
)
from .evaluator import build_objective_keys, compare_quality, evaluate
from .focus_feedback import FOCUS_FEEDBACK_KIND, build_focus_window
from .models import Instance
from .observation import build_step_observation, solution_summary
from .operators import (
    ALNS_REPAIR_OPERATOR_NAMES,
    DESTROY_OPERATOR_NAMES,
    CompiledALNSPolicy,
)
from .schemas import step_schema_from_contract
from .solution import AssignmentSolution
from .tools.assign_solvers import solve_assignment
from .trace import AgentTurnRecord, RunTrace


class AgentClient(Protocol):
    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...


class OrchestratorError(RuntimeError):
    pass


MAX_AGENT_OUTPUT_REPAIR_ATTEMPTS = 4


@dataclass(slots=True)
class GlobalBudgetState:
    max_iters: int
    iters_used: int = 0
    agent_turns: int = 0

    def exhausted(self) -> bool:
        return self.iters_used >= self.max_iters

    def remaining(self) -> Dict[str, int]:
        return {"iters": max(0, self.max_iters - self.iters_used)}

    def apply_result(self, result: Mapping[str, Any]) -> None:
        used = dict(result.get("budget_used", {}) or {})
        self.iters_used += int(result.get("iters_used", used.get("iters", 0)) or 0)

    def as_dict(self) -> Dict[str, Any]:
        remaining = self.remaining()
        return {
            "limits": {"iters": self.max_iters},
            "used": {"iters": self.iters_used},
            "remaining": {"iters": int(remaining["iters"])},
            "agent_turns": int(self.agent_turns),
        }


@dataclass(slots=True)
class RunState:
    instance: Instance
    config: Config
    global_budget: GlobalBudgetState
    rng_seed: int
    require_budget_exhaustion: bool = False
    global_objective_contract: Dict[str, Any] = field(default_factory=dict)
    global_objective_terms: list[Dict[str, str]] = field(default_factory=list)
    working_solution: AssignmentSolution | None = None
    best_solution: AssignmentSolution | None = None
    records: RunTrace = field(default_factory=RunTrace)
    last_result: Dict[str, Any] | None = None
    run_progress: RunProgress = field(default_factory=RunProgress)
    search_memory: SearchMemory = field(default_factory=SearchMemory)
    step_index: int = 0
    task_block_analysis_cache: Dict[str, Any] = field(default_factory=dict)
    decision_landscape_cache: Dict[str, Any] = field(default_factory=dict)
    solver_rng: Any = None

    @property
    def working_summary(self) -> Dict[str, Any]:
        solution = self.working_solution or AssignmentSolution.empty_from_instance(
            self.instance, put_all_unassigned=True
        )
        return solution_summary(
            solution,
            self.instance,
            self.config,
            objective_terms=self.global_objective_terms,
        )

    @property
    def best_summary(self) -> Dict[str, Any] | None:
        if self.best_solution is None:
            return None
        return solution_summary(
            self.best_solution,
            self.instance,
            self.config,
            objective_terms=self.global_objective_terms,
        )








def run_orchestrator(
    client: AgentClient,
    instance: Instance,
    user_goal_text: str,
    config: Config | None = None,
    budget: Budget | None = None,
    rng_seed: int = 0,
    trace_callback: Callable[[AgentTurnRecord], None] | None = None,
    initial_solution: AssignmentSolution | None = None,
    fixed_objective_terms: list[Dict[str, str]] | None = None,
    require_budget_exhaustion: bool = False,
) -> AssignmentSolution:
    cfg = config or Config()
    total = budget or Budget(max_iters=500)
    max_iters = int(total.max_iters)
    if max_iters <= 0:
        raise OrchestratorError("global iteration budget must be positive")

    state = RunState(
        instance=instance,
        config=cfg,
        global_budget=GlobalBudgetState(max_iters=max_iters),
        rng_seed=int(rng_seed),
        require_budget_exhaustion=bool(require_budget_exhaustion),
    )
    if fixed_objective_terms is not None:
        terms = [dict(item) for item in fixed_objective_terms]
        if not terms:
            raise OrchestratorError("fixed_objective_terms cannot be empty")
        state.global_objective_terms = terms
        metric_order = [str(item["metric"]) for item in terms]
        state.global_objective_contract = {
            "objective_form": OBJECTIVE_FORM,
            "service_baseline_order": list(metric_order[: len(SERVICE_METRICS)]),
            "preference_order": list(metric_order[len(SERVICE_METRICS):]),
            "fixed_metric_order": list(metric_order),
            "fixed_by_experiment": True,
        }
    state.solver_rng = random.Random(int(rng_seed))
    if initial_solution is not None:
        start = initial_solution.clone(deep=True); start.normalize(instance)
        start_eval = evaluate(start, instance, cfg, update_solution_schedule=True)
        if not start_eval.is_feasible: raise OrchestratorError("provided stored initial solution is not hard-feasible")
        start.eval = start_eval; state.working_solution = start; state.best_solution = start.clone(deep=True)
    started_at = time.time()
    if state.working_solution is None:
        raise OrchestratorError("A feasible stored initial solution is required")

    stop_reason: str | None = None
    while not state.global_budget.exhausted():
        if state.global_budget.exhausted():
            stop_reason = "global_budget_exhausted"
            break

        contract = _build_step_runtime_contract(instance=instance, state=state)
        contract = replace(contract, controls={
            **contract.controls, "selector_modes": ["random"],
            "step_focus_options": [name for name, expansion in contract.search_focus_expansions.items()
                                   if name == "global" or expansion.get("focus_task_ids")],
            "focus_feedback_kind": FOCUS_FEEDBACK_KIND,
            "search_evidence_kind": "control_history",
        })
        if not contract.allowed_actions:
            raise OrchestratorError("no legal Agent action remains")
        observation = build_step_observation(
            instance=instance, run_state=state, contract=contract
        )
        schema = step_schema_from_contract(
            contract,
            observation_blocks=observation["observation_blocks"],
        )
        prompt = build_step_prompt_bundle(
            user_goal=user_goal_text,
            decision_catalog=build_step_decision_catalog(contract),
            observation=observation,
            schema=schema,
            max_examples=1,
            examples_enabled=True,
        )
        repair_messages = [dict(message) for message in prompt.messages]
        repair_attempts: list[Dict[str, Any]] = []
        envelope_repairs: list[Dict[str, Any]] = []
        raw = ""
        protected_decision = None
        while True:
            raw = _chat_with_llm(state, client, repair_messages)
            from .output_repair import repair_rationale_envelope
            raw, envelope_repair = repair_rationale_envelope(raw)
            if envelope_repair is not None:
                envelope_repairs.append(envelope_repair)
            try:
                decision, control, validation = parse_validate_compile_step(
                    raw_text=raw,
                    schema=schema,
                    observation=observation,
                    contract=contract,
                )
                if protected_decision is not None and {
                    "action":decision.action,"control":decision.raw["control"]
                } != protected_decision:
                    raise AgentIOError(
                        "Output self-repair changed an already legal Agent control. Restore original_legal_decision_must_be_preserved exactly; repair only output defects.",
                        validation={"ok":False,"errors":[{"stage":"repair_preservation","message":"Legal Agent decision changed during repair"}]},
                    )
                if repair_attempts:
                    validation["repair_count"] = len(repair_attempts)
                    validation["repair_attempts"] = list(repair_attempts)
                validation["envelope_repair_count"] = len(envelope_repairs)
                validation["envelope_repairs"] = list(envelope_repairs)
                break
            except AgentIOError as exc:
                if protected_decision is None:
                    from .output_repair import legal_decision_for_repair
                    protected_decision = legal_decision_for_repair(
                        raw=raw,schema=schema,observation=observation,contract=contract,
                    )
                repair_attempts.append(
                    {
                        "attempt": len(repair_attempts) + 1,
                        "error": str(exc),
                        "raw_text": raw,
                        "parsed_output": _maybe_parse(raw),
                    }
                )
                if len(repair_attempts) > MAX_AGENT_OUTPUT_REPAIR_ATTEMPTS:
                    if isinstance(exc.validation, Mapping):
                        exc.validation["repair_count"] = len(repair_attempts) - 1
                        exc.validation["repair_attempts"] = list(repair_attempts)
                    _add_record(
                        state,
                        _error_record(
                            state,
                            role="step",
                            phase="step",
                            prompt_bundle=prompt,
                            raw=raw,
                            exc=exc,
                        ),
                        trace_callback,
                    )
                    raise OrchestratorError(f"Step output invalid after repair attempts: {exc}") from exc
                from .output_repair import repair_messages as build_repair_messages
                repair_messages = build_repair_messages(
                    raw=raw, error=str(exc), schema=schema, observation=observation,
                    attempt=len(repair_attempts),
                    previous_errors=[a['error'] for a in repair_attempts],
                    protected_decision=protected_decision,
                )


        executor_result = _execute_alns(
            control, instance=instance, config=cfg, state=state, rng_seed=rng_seed
        )
        executor_result, executor_trace = _split_executor_trace(executor_result)
        if control.action == "run_alns":
            used = dict(executor_result.get("budget_used", {}) or {})
            if int(used.get("iters", 0) or 0) <= 0:
                raise OrchestratorError(
                    f"{control.action} returned without consuming execution budget"
                )
            state.global_budget.apply_result(executor_result)
        state.run_progress.apply_result(executor_result)
        state.search_memory.apply_result(executor_result)
        state.last_result = dict(executor_result)
        _add_record(
            state,
            _turn_record(
                state,
                role="step",
                phase="step",
                action=decision.action,
                prompt_bundle=prompt,
                raw=raw,
                parsed_payload=_maybe_parse(raw),
                decision_root=decision.raw,
                validation=validation,
                result=_step_execution_result(
                    executor_result, executor_trace=executor_trace
                ),
            ),
            trace_callback,
        )
        state.step_index += 1

        if state.global_budget.exhausted():
            stop_reason = "global_budget_exhausted"
            break

    final = state.best_solution or state.working_solution
    if final is None:
        final = AssignmentSolution.empty_from_instance(instance, put_all_unassigned=True)
    stop_reason = stop_reason or "run_finished"
    execution = _execution_time_summary(state.records)
    final.run_summary = {
        "stop_reason": stop_reason,
        "global_objective": public_global_objective(
            state.global_objective_contract, state.global_objective_terms
        ),
        "budget": state.global_budget.as_dict(),
        "run_progress": state.run_progress.as_dict(
            max_iters=state.global_budget.max_iters,
        ),
        "search_memory": state.search_memory.as_dict(),
        "elapsed_sec": round(time.time() - started_at, 6),
        "execution_time_summary": execution,
        "alns_time_sec": execution["alns_time_sec"],
        "alns_iters": execution["alns_iters"],
        "run_alns_actions": execution["run_alns_actions"],
        "solver_action_time_sec": execution["solver_action_time_sec"],
    }
    final.run_artifact = {
        "records": state.records.as_artifact(),
        "final_result": solution_summary(
            final,
            instance,
            cfg,
            objective_terms=state.global_objective_terms,
        ),
    }
    return final










def _search_profile(control: RuntimeControl) -> str:
    """Classify the executed search path for logging only."""
    focus = dict(control.compiled_search_focus or {})
    focus_id = str(focus.get("focus_id", "global") or "global")
    if focus_id != "global" or list(focus.get("focus_task_ids", []) or []):
        return "focused"
    destroy_features = dict(getattr(control.destroy_policy, "feature_weights", {}) or {})
    task_features = dict(getattr(control.insertion_policy, "task_feature_weights", {}) or {})
    semantic_destroy = any(abs(float(value)) > 1e-12 for value in destroy_features.values())
    semantic_task = any(
        abs(float(task_features.get(name, 0.0) or 0.0)) > 1e-12
        for name in ("regret_score", "scarcity_score")
    )
    return "semantic" if semantic_destroy or semantic_task else "broad"


def _execute_alns(
    control: RuntimeControl,
    *,
    instance: Instance,
    config: Config,
    state: RunState,
    rng_seed: int,
) -> Dict[str, Any]:
    if state.working_solution is None:
        raise OrchestratorError("run_alns requested before initial construction")
    before_solution = state.working_solution.clone(deep=True)
    before_eval = evaluate(
        before_solution, instance, config, update_solution_schedule=True
    )
    before_best_solution = (
        state.best_solution.clone(deep=True)
        if state.best_solution is not None
        else before_solution.clone(deep=True)
    )
    before_best_eval = evaluate(
        before_best_solution, instance, config, update_solution_schedule=True
    )
    search_profile = _search_profile(control)
    adaptation_state = state.search_memory.adaptation_state()
    # Global search time is owned by the run budget, never by an adaptation
    # context.  This stays monotonic across focus and neighborhood changes.
    global_trial_start = int(state.global_budget.iters_used)
    result = solve_assignment(
        instance=instance,
        init_solution=state.working_solution,
        config=config,
        budget=Budget(max_iters=int(control.execution_budget["iters"])),
        policy=CompiledALNSPolicy(
            destroy_policy=control.destroy_policy,
            insertion_policy=control.insertion_policy,
            acceptance_policy=control.alns_acceptance_policy,
            reaction_factor=float(config.alns_reaction_factor),
        ),
        objective_terms=list(state.global_objective_terms),
        rng_seed=rng_seed + state.step_index + 1,
        trace_id=f"X{state.step_index + 1}",
        compiled_search_focus=_executor_search_focus(control),
        adaptive_destroy_weights=adaptation_state["adaptive_destroy_weights"],
        adaptive_repair_weights=adaptation_state["adaptive_repair_weights"],
        adaptive_destroy_scores=adaptation_state["adaptive_destroy_scores"],
        adaptive_destroy_uses=adaptation_state["adaptive_destroy_uses"],
        adaptive_repair_scores=adaptation_state["adaptive_repair_scores"],
        adaptive_repair_uses=adaptation_state["adaptive_repair_uses"],
        adaptation_trials_elapsed=adaptation_state["adaptation_trials_elapsed"],
        global_best_solution=state.best_solution,
        rng_stream=state.solver_rng,
        adaptation_horizon_iters=state.global_budget.max_iters,
    )
    state.working_solution = result.working_solution
    best_updated = False
    if result.global_best_feasible is not None:
        candidate_eval = evaluate(
            result.global_best_feasible, instance, config, update_solution_schedule=True
        )
        current_best_eval = (
            None
            if state.best_solution is None
            else evaluate(state.best_solution, instance, config, update_solution_schedule=True)
        )
        if current_best_eval is None or compare_quality(
            candidate_eval, current_best_eval, state.global_objective_terms
        ) < 0:
            state.best_solution = result.global_best_feasible.clone(deep=True)
            best_updated = True
    after_eval = evaluate(state.working_solution, instance, config, update_solution_schedule=True)
    after_best_solution = state.best_solution if state.best_solution is not None else state.working_solution
    after_best_eval = evaluate(
        after_best_solution, instance, config, update_solution_schedule=True
    )
    diagnostics = dict(result.diagnostics or {})
    payload = _build_search_action_result(
        control,
        before_eval=before_eval,
        after_eval=after_eval,
        before_best_eval=before_best_eval,
        after_best_eval=after_best_eval,
        elapsed_sec=float(diagnostics.get("actual_time_used_sec", 0.0) or 0.0),
        iters_used=int(diagnostics.get("actual_iters_used", 0) or 0),
        trace=_globalize_search_trace(
            _search_focus_trace(dict(result.trace), control.compiled_search_focus),
            global_trial_start=global_trial_start,
        ),
        adaptive_destroy_weights_after=result.adaptive_destroy_weights,
        adaptive_repair_weights_after=result.adaptive_repair_weights,
        adaptive_destroy_scores_after=result.adaptive_destroy_scores,
        adaptive_destroy_uses_after=result.adaptive_destroy_uses,
        adaptive_repair_scores_after=result.adaptive_repair_scores,
        adaptive_repair_uses_after=result.adaptive_repair_uses,
        adaptation_trials_elapsed_after=result.adaptation_trials_elapsed,
        global_trial_start=global_trial_start,
        global_trial_end=global_trial_start + int(diagnostics.get("actual_iters_used", 0) or 0),
        objective_terms=list(state.global_objective_terms),
        structure_changed=before_solution.structure_key() != state.working_solution.structure_key(),
        n_tasks=len(instance.tasks),
    )
    payload["search_profile"] = str(search_profile)
    payload["search_policy"] = dict(control.compiler_transform["search_policy"])
    if control.compiler_transform.get("focus_feedback_kind") == FOCUS_FEEDBACK_KIND:
        payload["focus_feedback_kind"] = FOCUS_FEEDBACK_KIND
        payload["focus_review_window"] = build_focus_window(payload, action_index=state.search_memory.control_actions + 1)
    payload["executor_trace"]["search_profile"] = str(search_profile)
    return payload


def _globalize_search_trace(trace: Dict[str, Any], *, global_trial_start: int) -> Dict[str, Any]:
    """Attach the run-global monotonic trial clock to one ALNS action trace.

    The same clock also drives the single shared classic-ALNS adaptation state;
    focus or neighborhood changes never create a private learning timeline.
    """
    out = dict(trace or {})
    offset = int(global_trial_start)
    rows = []
    for raw in list(out.get("best_progress", []) or []):
        row = dict(raw or {})
        local = int(row.get("trial", 0) or 0)
        row["local_trial"] = local
        row["trial"] = offset + local
        rows.append(row)
    out["best_progress"] = rows
    iteration_rows = []
    for raw in list(out.get("iteration_trace", []) or []):
        row = dict(raw or {})
        local = int(row.get("iteration", 0) or 0)
        row["global_trial"] = offset + local
        iteration_rows.append(row)
    if iteration_rows:
        out["iteration_trace"] = iteration_rows
    out["global_trial_start"] = offset
    out["global_trial_end"] = offset + int(out.get("total_iters", len(iteration_rows)) or len(iteration_rows))
    return out




def _build_search_action_result(
    control: RuntimeControl,
    *,
    before_eval: Any,
    after_eval: Any,
    before_best_eval: Any,
    after_best_eval: Any,
    elapsed_sec: float,
    iters_used: int,
    trace: Dict[str, Any],
    adaptive_destroy_weights_after: Mapping[str, float],
    adaptive_repair_weights_after: Mapping[str, float],
    adaptive_destroy_scores_after: Mapping[str, float],
    adaptive_destroy_uses_after: Mapping[str, int],
    adaptive_repair_scores_after: Mapping[str, float],
    adaptive_repair_uses_after: Mapping[str, int],
    adaptation_trials_elapsed_after: int,
    global_trial_start: int,
    global_trial_end: int,
    objective_terms: list[Dict[str, str]],
    structure_changed: bool,
    n_tasks: int,
) -> Dict[str, Any]:
    before_quality, after_quality, _ = _quality_change(before_eval, after_eval)
    best_before_quality, best_after_quality, _ = _quality_change(
        before_best_eval, after_best_eval
    )
    service_components = {
        "missed_priority_before": float(best_before_quality.get("missed_priority", 0.0)),
        "missed_priority_after": float(best_after_quality.get("missed_priority", 0.0)),
        "unassigned_count_before": float(best_before_quality.get("unassigned_count", 0.0)),
        "unassigned_count_after": float(best_after_quality.get("unassigned_count", 0.0)),
    }
    before_key = build_objective_keys(before_best_eval, objective_terms)
    after_key = build_objective_keys(after_best_eval, objective_terms)
    objective_cmp = compare_quality(after_best_eval, before_best_eval, objective_terms)
    decisive_level = None
    decisive_delta = None
    for idx, (name, before_value, after_value) in enumerate(
        zip(before_key["terms"], before_key["key"], after_key["key"]), start=1
    ):
        if abs(float(after_value) - float(before_value)) > 1e-9:
            decisive_level = f"L{idx}:{name}"
            decisive_delta = float(after_value) - float(before_value)
            break
    objective_progress = {
        "metric_order": list(before_key["terms"]),
        "global_best_before": [float(v) for v in before_key["key"]],
        "global_best_after": [float(v) for v in after_key["key"]],
        "comparison": "improved" if objective_cmp < 0 else ("worsened" if objective_cmp > 0 else "equal"),
        "decisive_level": decisive_level,
        "decisive_delta": decisive_delta,
        "improvements_per_100_trials": round(100.0 * (1.0 if objective_cmp < 0 else 0.0) / max(1, int(iters_used)), 6),
    }
    raw_trial_flow = dict(trace.get("trial_flow", {}) or {})
    outcome_counts = {
        "global_best_update_count": int(raw_trial_flow.get("global_best_update_count", 0) or 0),
        "accepted_improving_count": int(raw_trial_flow.get("accepted_improving_count", 0) or 0),
        "accepted_non_improving_count": int(
            raw_trial_flow.get("accepted_non_improving_count", 0) or 0
        ),
    }
    trace["feature_usage_summary"] = _feature_usage_summary(control, trace)
    trace["adaptive_destroy_weights_after"] = {
        str(k): round(float(v), 6) for k, v in adaptive_destroy_weights_after.items()
    }
    trace["adaptive_repair_weights_after"] = {
        str(k): round(float(v), 6) for k, v in adaptive_repair_weights_after.items()
    }
    trace.pop("executor_feature_usage", None)
    decision_feedback = _decision_feedback(
        control=control,
        trace=trace,
        iters_used=iters_used,
        outcome_counts=outcome_counts,
    )
    decision_feedback["service_components"] = service_components
    decision_feedback["objective_progress"] = objective_progress
    return {
        "executed_action": "run_alns",
        "outcome": _result_outcome(**outcome_counts),
        "before_quality": before_quality,
        "after_quality": after_quality,
        "before_feasible": bool(before_eval.is_feasible),
        "after_feasible": bool(after_eval.is_feasible),
        "outcome_counts": outcome_counts,
        "structure_changed": bool(structure_changed),
        "budget_requested": dict(control.execution_budget),
        "budget_used": {"iters": int(iters_used)},
        "elapsed_sec": round(float(elapsed_sec), 6),
        "iters_used": int(iters_used),
        "global_trial_start": int(global_trial_start),
        "global_trial_end": int(global_trial_end),
        "search_focus_result": dict(trace.get("search_focus_result", {}) or {}),
        "search_focus_progress": dict(trace.get("executor_search_focus_progress", {}) or {}),
        "trial_flow": dict(trace.get("trial_flow", {}) or {}),
        "decision_feedback": decision_feedback,
        "operator_result": _operator_result(trace),
        "adaptive_destroy_weights_after": {
            str(k): float(v) for k, v in adaptive_destroy_weights_after.items()
        },
        "adaptive_repair_weights_after": {
            str(k): float(v) for k, v in adaptive_repair_weights_after.items()
        },
        "adaptive_destroy_scores_after": {str(k): float(v) for k, v in adaptive_destroy_scores_after.items()},
        "adaptive_destroy_uses_after": {str(k): int(v) for k, v in adaptive_destroy_uses_after.items()},
        "adaptive_repair_scores_after": {str(k): float(v) for k, v in adaptive_repair_scores_after.items()},
        "adaptive_repair_uses_after": {str(k): int(v) for k, v in adaptive_repair_uses_after.items()},
        "adaptation_trials_elapsed_after": int(adaptation_trials_elapsed_after),
        "executor_error": None,
        "executor_trace": trace,
    }


def _quality_change(before_eval: Any, after_eval: Any) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    before = {str(k): float(v) for k, v in before_eval.quality_metrics.items()}
    after = {str(k): float(v) for k, v in after_eval.quality_metrics.items()}
    delta = {name: after.get(name, 0.0) - before.get(name, 0.0) for name in sorted(set(before) | set(after))}
    return before, after, delta


def _executor_search_focus(control: RuntimeControl) -> Dict[str, Any]:
    focus = dict(control.compiled_search_focus or {})
    return CompiledSearchFocus(
        focus_id=str(focus.get("focus_id", "global")),
        focus_task_ids=[int(v) for v in focus.get("focus_task_ids", []) or []],
        minimum_target_hits=max(1, int(focus.get("minimum_target_hits", 1) or 1)),
        soft_greedy_bonus=float(focus.get("soft_greedy_bonus", 1.0) or 1.0),
        soft_random_weight=float(focus.get("soft_random_weight", 1.5) or 1.5),
    ).as_executor_focus()


def _search_focus_trace(trace: Dict[str, Any], compiled: Mapping[str, Any]) -> Dict[str, Any]:
    engagement = dict(trace.get("executor_search_focus_engagement", {}) or {})
    trace["search_focus_result"] = {
        "focus_id": str(compiled.get("focus_id", "global")),
        "focus_task_count": len(list(compiled.get("focus_task_ids", []) or [])),
        "minimum_target_hits": max(1, int(compiled.get("minimum_target_hits", 1) or 1)),
        "soft_greedy_bonus": float(compiled.get("soft_greedy_bonus", 1.0) or 1.0),
        "soft_random_weight": float(compiled.get("soft_random_weight", 1.5) or 1.5),
        "matched_destroy_candidate_count": int(engagement.get("focus_targeted_destroy_move_count", 0) or 0),
        "selected_focus_destroy_count": int(engagement.get("selected_focus_targeted_destroy_move_count", 0) or 0),
        "focus_changed_selection_count": int(engagement.get("focus_changed_selection_count", 0) or 0),
        "focus_matched_task_selection_count": int(engagement.get("focus_matched_task_selection_count", 0) or 0),
    }
    trace["compiled_search_focus"] = dict(compiled)
    return trace




def _feature_usage_summary(control: RuntimeControl, trace: Mapping[str, Any]) -> list[Dict[str, Any]]:
    usage = dict(trace.get("executor_feature_usage", {}) or {})
    destroy = control.destroy_policy.as_dict() if control.destroy_policy is not None else {}
    insertion = control.insertion_policy.as_dict() if control.insertion_policy is not None else {}
    selected = {
        "destroy_feature_weights": dict(destroy.get("feature_weights", {}) or {}),
        "task_feature_weights": dict(insertion.get("task_feature_weights", {}) or {}),
    }
    groups = (
        ("destroy_feature", "destroy_feature_weights", "destroy_features"),
        ("task_feature", "task_feature_weights", "insertion_task_features"),
    )
    rows: list[Dict[str, Any]] = []
    for kind, selected_key, usage_key in groups:
        for name, score in selected[selected_key].items():
            if float(score or 0.0) <= 0.0:
                continue
            raw = dict(dict(usage.get(usage_key, {}) or {}).get(str(name), {}) or {})
            evaluated = int(raw.get("feature_evaluated_move_count", 0) or 0)
            rows.append({
                "kind": kind,
                "name": str(name),
                "compiled_score_0_to_10": float(score),
                "feature_evaluated_move_count": evaluated,
                "feature_nonzero_move_count": int(raw.get("feature_nonzero_move_count", 0) or 0),
                "used_in_ranking": bool(raw.get("used_in_score", False) and evaluated > 0),
            })
    return rows


def _decision_feedback(
    *,
    control: RuntimeControl,
    trace: Mapping[str, Any],
    iters_used: int,
    outcome_counts: Mapping[str, int],
) -> Dict[str, Any]:
    """Project executor facts back onto the Step decision blocks."""
    selected = dict(control.compiler_transform.get("decision_blocks", {}) or {})
    focus_result = dict(trace.get("search_focus_result", {}) or {})
    focus_progress = dict(trace.get("executor_search_focus_progress", {}) or {})
    engagement = dict(trace.get("executor_search_focus_engagement", {}) or {})
    operator_pairs = {
        str(name): dict(values)
        for name, values in dict(trace.get("operator_pair_summary", {}) or {}).items()
        if isinstance(values, Mapping)
    }
    feature_rows = list(trace.get("feature_usage_summary", []) or [])
    feature_feedback: Dict[str, Dict[str, Any]] = {}
    for row in feature_rows:
        if not isinstance(row, Mapping):
            continue
        key = f"{row.get('kind', 'feature')}::{row.get('name', '')}"
        feature_feedback[key] = {
            "kind": str(row.get("kind", "")),
            "name": str(row.get("name", "")),
            "evaluated_count": int(row.get("feature_evaluated_move_count", 0) or 0),
            "nonzero_count": int(row.get("feature_nonzero_move_count", 0) or 0),
            "used_in_ranking": bool(row.get("used_in_ranking", False)),
        }
    acceptance = dict(trace.get("acceptance_summary", {}) or {})
    progress = dict(trace.get("execution_progress_summary", {}) or {})
    trial_flow = dict(trace.get("trial_flow", {}) or {})
    candidate_trials = int(trial_flow.get("candidate_trials", 0) or 0)
    admissible_trials = int(trial_flow.get("admissible_trials", 0) or 0)
    accepted_trials = int(trial_flow.get("accepted_trials", 0) or 0)
    changed_trials = int(trial_flow.get("structurally_changed_trials", 0) or 0)
    accepted_changed = int(
        trial_flow.get("accepted_structurally_changed_trials", 0) or 0
    )
    noop_trials = int(trial_flow.get("destroy_repair_noop_trials", 0) or 0)
    search_flow = {
        "candidate_trials": candidate_trials,
        "destroy_repair_noop_trials": noop_trials,
        "noop_rate": round(noop_trials / max(1, candidate_trials), 6),
        "structurally_changed_trials": changed_trials,
        "structure_change_rate": round(changed_trials / max(1, candidate_trials), 6),
        "admissible_trials": admissible_trials,
        "feasible_candidate_rate": round(
            admissible_trials / max(1, candidate_trials), 6
        ),
        "accepted_trials": accepted_trials,
        "acceptance_rate": round(
            accepted_trials / max(1, admissible_trials), 6
        ),
        "accepted_structurally_changed_trials": accepted_changed,
        "structural_passage_rate": round(accepted_changed / max(1, changed_trials), 6),
        "global_best_update_count": int(outcome_counts.get("global_best_update_count", 0) or 0),
        "accepted_improving_count": int(outcome_counts.get("accepted_improving_count", 0) or 0),
        "accepted_non_improving_count": int(
            outcome_counts.get("accepted_non_improving_count", 0) or 0
        ),
        "unique_trial_structure_count": int(
            trial_flow.get("unique_trial_structure_count", 0) or 0
        ),
        "unique_accepted_structure_count": int(
            trial_flow.get("unique_accepted_structure_count", 0) or 0
        ),
        "focus_active_trials": int(
            trial_flow.get("focus_active_trials", 0) or 0
        ),
        "focus_inactive_trials": int(
            trial_flow.get("focus_inactive_trials", 0) or 0
        ),
        "focus_active_best_updates": int(
            trial_flow.get("focus_active_best_updates", 0) or 0
        ),
        "focus_inactive_best_updates": int(
            trial_flow.get("focus_inactive_best_updates", 0) or 0
        ),
    }
    focus_task_ids = [
        int(value) for value in dict(control.compiled_search_focus or {}).get("focus_task_ids", []) or []
    ]
    focus_attempted = [int(value) for value in engagement.get("engaged_task_ids", []) or []]
    return {
        "search_policy": dict(control.compiler_transform.get("search_policy", {}) or {}),
        "focus": {
            "selected": {
                "focus_id": str(dict(control.compiled_search_focus or {}).get("focus_id", "global")),
                "focus_task_ids": focus_task_ids[:20],
                "minimum_target_hits": max(
                    1,
                    int(
                        dict(control.compiled_search_focus or {}).get(
                            "minimum_target_hits", 1
                        )
                        or 1
                    ),
                ),
            },
            "observed": {
                "tasks_attempted": focus_attempted[:20],
                "tasks_inserted": [
                    int(value) for value in focus_progress.get("focus_tasks_inserted", []) or []
                ][:20],
                "tasks_still_unassigned": [
                    int(value)
                    for value in focus_progress.get("focus_tasks_still_unassigned", []) or []
                ][:20],
                "focus_targeted_destroy_moves": int(
                    focus_result.get("selected_focus_targeted_destroy_move_count", 0) or 0
                ),
                "destroy_focus_annotations": int(
                    focus_result.get("destroy_focus_annotation_application_count", 0) or 0
                ),
                "focus_matched_task_selections": int(
                    focus_result.get("focus_matched_task_selection_count", 0) or 0
                ),
                "selected_target_hit_count_min": int(
                    focus_result.get("selected_focus_target_hit_count_min", 0)
                    or 0
                ),
                "selected_target_hit_count_max": int(
                    focus_result.get("selected_focus_target_hit_count_max", 0)
                    or 0
                ),
                "selected_target_hit_count_mean": float(
                    focus_result.get("selected_focus_target_hit_count_mean", 0.0)
                    or 0.0
                ),
            },
        },
        "operator_priority": {
            "selected": dict(selected.get("operator_priority", {}) or {}),
            "operator_pairs": operator_pairs,
        },
        "search_bias": {
            "selected": dict(selected.get("search_bias", {}) or {}),
            "feature_usage": feature_feedback,
        },
        "acceptance": {
            "selected": dict(selected.get("acceptance", {}) or {}),
            **acceptance,
            "global_best_update_count": int(
                outcome_counts.get("global_best_update_count", 0) or 0
            ),
        },
        "search_flow": search_flow,
        "execution_budget": {
            "selected": dict(selected.get("execution_budget", {}) or {}),
            "requested_trials": int(
                dict(selected.get("execution_budget", {}) or {}).get("iters", iters_used) or iters_used
            ),
            **progress,
            "outcome_counts": {
                str(name): int(value) for name, value in outcome_counts.items()
            },
        },
    }


def _operator_result(trace: Mapping[str, Any]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for group_name, group in dict(trace.get("operator_usage_summary", {}) or {}).items():
        if not isinstance(group, Mapping):
            continue
        for name, raw in group.items():
            if isinstance(raw, Mapping):
                rows.append({"group": str(group_name), "name": str(name), **dict(raw)})
    return rows


def _result_outcome(
    *,
    global_best_update_count: int,
    accepted_improving_count: int,
    accepted_non_improving_count: int,
) -> str:
    """Classify one complete ALNS action from process-wide result counts only."""
    if int(global_best_update_count) > 0:
        return "global_best_improved"
    if int(accepted_improving_count) > 0:
        return "working_solution_improved"
    if int(accepted_non_improving_count) > 0:
        return "accepted_without_improvement"
    return "no_improvement"


def _build_step_runtime_contract(*, instance: Instance, state: RunState) -> RuntimeContract:
    landscape = build_runtime_decision_landscape(instance, state)
    return RuntimeContractBuilder.step(
        instance=instance,
        run_state=state,
        landscape=landscape,
    )






def _step_execution_result(
    executor_result: Dict[str, Any],
    *,
    executor_trace: Dict[str, Any],
) -> Dict[str, Any]:
    out = {
        "executed_action": executor_result.get("executed_action"),
        "execution": dict(executor_result),
        "error": executor_result.get("executor_error"),
    }
    if executor_result.get("outcome") is not None:
        out["outcome"] = executor_result.get("outcome")
    if executor_result.get("action_status") is not None:
        out["action_status"] = executor_result.get("action_status")
    if executor_trace:
        out["executor_trace"] = dict(executor_trace)
    return out

def _split_executor_trace(result: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    public = dict(result or {})
    trace = public.pop("executor_trace", {})
    return public, dict(trace or {}) if isinstance(trace, Mapping) else {}




def _chat_with_llm(state: RunState, client: AgentClient, messages: list[dict[str, str]]) -> str:
    raw = client.chat(messages, temperature=0.0, extra={'response_format':{'type':'json_object'}})
    state.global_budget.agent_turns += 1
    return raw




def _turn_record(
    state: RunState,
    *,
    role: str,
    phase: str,
    action: str | None,
    prompt_bundle: PromptBundle,
    raw: str,
    parsed_payload: Dict[str, Any] | None,
    decision_root: Dict[str, Any] | None,
    validation: Dict[str, Any],
    result: Dict[str, Any] | None,
    error: str | None = None,
) -> AgentTurnRecord:
    return AgentTurnRecord(
        turn_id=_next_turn_id(state),
        role=role,
        phase=phase,
        action=action,
        agent_input={
            "system_message": prompt_bundle.system_message,
            "role": prompt_bundle.role,
            "instruction": prompt_bundle.instruction,
            "decision_examples": [dict(item) for item in prompt_bundle.decision_examples],
            "context": dict(prompt_bundle.context),
            "output_json_schema": dict(prompt_bundle.output_json_schema),
            "messages": [dict(message) for message in prompt_bundle.messages],
        },
        agent_output={
            "raw_text": raw,
            "parsed_json": parsed_payload,
            "decision": None if decision_root is None else dict(decision_root),
        },
        validation=dict(validation or {}),
        result=None if result is None else dict(result),
        error=error,
    )


def _error_record(
    state: RunState,
    *,
    role: str,
    phase: str,
    prompt_bundle: PromptBundle,
    raw: str,
    exc: AgentIOError,
) -> AgentTurnRecord:
    parsed = _maybe_parse(raw)
    validation = getattr(exc, "validation", None)
    if not isinstance(validation, Mapping):
        validation = {"ok": False, "errors": [{"stage": "unknown", "message": str(exc)}]}
    return _turn_record(
        state,
        role=role,
        phase=phase,
        action=str(_decision_root(parsed).get("action", "") or "") or None,
        prompt_bundle=prompt_bundle,
        raw=raw,
        parsed_payload=parsed,
        decision_root=_decision_root(parsed) or None,
        validation=dict(validation),
        result=None,
        error=str(exc),
    )


def _maybe_parse(raw: str) -> Dict[str, Any] | None:
    try:
        value = json.loads(raw)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _decision_root(parsed: Any) -> Dict[str, Any]:
    if isinstance(parsed, Mapping):
        for key in ("step_decision",):
            value = parsed.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        if parsed.get("action") is not None:
            return dict(parsed)
    return {}


def _next_turn_id(state: RunState) -> str:
    return f"T{len(state.records.turns) + 1:03d}"


def _add_record(state: RunState, record: AgentTurnRecord, callback: Callable[[AgentTurnRecord], None] | None) -> None:
    state.records.add(record)
    if callback is not None:
        callback(record)




def _execution_time_summary(records: RunTrace) -> Dict[str, Any]:
    alns_time = 0.0
    alns_actions = alns_iters = 0
    for record in records.turns:
        result = _execution_result_from_record(record)
        action = str(result.get("executed_action", "") or "")
        used = dict(result.get("budget_used", {}) or {})
        elapsed_sec = float(result.get("elapsed_sec", 0.0) or 0.0)
        if action == "run_alns":
            alns_actions += 1
            alns_time += elapsed_sec
            alns_iters += int(result.get("iters_used", used.get("iters", 0)) or 0)
    return {
        "alns_time_sec": round(alns_time, 6),
        "solver_action_time_sec": round(alns_time, 6),
        "run_alns_actions": alns_actions,
        "alns_iters": alns_iters,
    }


def _execution_result_from_record(record: AgentTurnRecord) -> Dict[str, Any]:
    result = record.result if isinstance(record.result, Mapping) else {}
    execution = result.get("execution") if isinstance(result, Mapping) else None
    return dict(execution) if isinstance(execution, Mapping) else dict(result)
