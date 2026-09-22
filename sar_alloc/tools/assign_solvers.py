"""ALNS executor with explicit operator, focus, score and selector boundaries.

Each trial samples a primitive Destroy operator from Agent rank × shared classic
ALNS history. The operator generates its ordinary candidates; an explicit
action-local focus can only add a bounded selection preference inside that same
pool. Repair follows the same ownership split: the Agent selects the primitive while
focus can only softly prefer matching pending tasks; no trials or candidates are
reserved for a separate focused branch.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from ..acceptance import build_acceptance_scales
from ..config import Budget, Config
from ..domain import DESTROY_FEATURE_NAMES, SERVICE_METRICS
from ..evaluator import build_objective_keys, compare_quality, evaluate as _evaluate_raw
from ..models import Instance
from ..operators import (
    DestroyPolicy,
    CompiledALNSPolicy,
    DESTROY_OPERATOR_NAMES,
    ALNS_REPAIR_OPERATOR_NAMES,
)
from ..operators.destroy import (
    DESTROY_OPERATORS,
    DestroyContext,
    DestroyMove,
    DestroyOperator,
    compute_destroy_strength,
    sample_intrinsic_destroy_move,
    _score_moves,
)
from ..operators.insertion import (
    InsertionContext,
    run_insertion_kernel,
    select_insertion_operator,
)
from ..operators.focus import FocusContext, annotate_destroy_moves_by_focus
from ..operators.selector import select_item
from ..runtime_focus import (
    RuntimeFocusState,
    restrict_runtime_focus,
)
from ..solution import AssignmentSolution, EvalResult


@dataclass
class _EvalStats:
    n: int = 0
    t: float = 0.0


EVAL_STATS = _EvalStats()

SA_FINAL_TEMPERATURE_RATIO = 0.05

# Executor-local policy.  The agent continues to choose the semantic focus; the
# solver only re-resolves that immutable focus against the current accepted
# solution after one action-local focus segment without a new best solution.

@dataclass(frozen=True)
class FeasibilityDecision:
    admissible: bool
    accept_scope: str
    reason: str
    events: List[str]



@dataclass(frozen=True)
class ALNSResult:
    final_current: AssignmentSolution
    action_best_feasible: Optional[AssignmentSolution]
    global_best_feasible: Optional[AssignmentSolution]
    working_solution: AssignmentSolution
    events: List[str]
    trace: Dict[str, Any]
    diagnostics: Dict[str, Any]
    adaptive_destroy_weights: Dict[str, float]
    adaptive_repair_weights: Dict[str, float]
    adaptive_destroy_scores: Dict[str, float]
    adaptive_destroy_uses: Dict[str, int]
    adaptive_repair_scores: Dict[str, float]
    adaptive_repair_uses: Dict[str, int]
    adaptation_trials_elapsed: int



class ExecutorContractError(RuntimeError):
    """Raised when validated compiled control cannot be executed as requested."""


class NoDestroyMoveAvailable(ExecutorContractError):
    """Normal search stop when every explicitly enabled destroy operator is empty."""

    def __init__(self, attempted: Sequence[Mapping[str, Any]]):
        super().__init__("no destroy move available from explicitly enabled operators")
        self.attempted = [dict(item) for item in attempted]


def evaluate(*args, **kwargs):  # noqa: F811
    t0 = time.perf_counter()
    ev = _evaluate_raw(*args, **kwargs)
    EVAL_STATS.n += 1
    EVAL_STATS.t += time.perf_counter() - t0
    return ev


def _check_strict_feasibility(trial_report: Any) -> FeasibilityDecision:
    return _strict_feasibility_decision(trial_report)


def _strict_feasibility_decision(trial: Any) -> FeasibilityDecision:
    if trial.is_feasible:
        return FeasibilityDecision(True, "working_and_best_candidate", "Trial is feasible under strict policy.", [])
    return FeasibilityDecision(
        False,
        "reject",
        "feasibility_rejected_infeasible_trial",
        ["feasibility_rejected_infeasible_trial"],
    )


def solve_assignment(
    instance: Instance,
    init_solution: AssignmentSolution,
    config: Config,
    budget: Budget,
    policy: CompiledALNSPolicy,
    objective_terms: List[Dict[str, Any]],
    rng_seed: int = 0,
    trace_id: str = "X_solver",
    compiled_search_focus: Optional[Dict[str, Any]] = None,
    adaptive_destroy_weights: Optional[Mapping[str, float]] = None,
    adaptive_repair_weights: Optional[Mapping[str, float]] = None,
    adaptive_destroy_scores: Optional[Mapping[str, float]] = None,
    adaptive_destroy_uses: Optional[Mapping[str, int]] = None,
    adaptive_repair_scores: Optional[Mapping[str, float]] = None,
    adaptive_repair_uses: Optional[Mapping[str, int]] = None,
    adaptation_trials_elapsed: int = 0,
    global_best_solution: Optional[AssignmentSolution] = None,
    rng_stream: Optional[random.Random] = None,
    adaptation_horizon_iters: Optional[int] = None,
    operator_pair_bandit: Optional[Mapping[str, Any]] = None,
    best_snapshot_callback: Optional[Callable[[int, AssignmentSolution], None]] = None,
) -> ALNSResult:
    """Run weighted ALNS from an incumbent using an already validated policy.

    ``rng_stream`` and ``adaptation_horizon_iters`` let an Agent run split one
    global search budget into multiple actions without silently resetting randomness
    or changing the low-level ALNS learning time scale.
    """
    if policy.destroy_policy.selector_mode != "random" or policy.insertion_policy.task_selector_mode != "random":
        raise ExecutorContractError("Only random selectors are supported")
    if any(float(v) != 0 for v in policy.destroy_policy.feature_weights.values()) or any(float(v) != 0 for v in policy.insertion_policy.task_feature_weights.values()):
        raise ExecutorContractError("Feature control is disabled in the conference method")
    rng = rng_stream if rng_stream is not None else random.Random(int(rng_seed))

    cur = init_solution.clone_search_state()
    cur.normalize(instance)
    cur_ev = evaluate(cur, instance, config, update_solution_schedule=False)
    terms = list(objective_terms)
    if not terms:
        raise ExecutorContractError("objective_terms must contain the run-global ordering")
    return _solve_weighted_alns(
        instance,
        cur,
        cur_ev,
        config,
        budget,
        policy,
        rng,
        objective_terms=terms,
        trace_id=trace_id,
        compiled_search_focus=dict(compiled_search_focus or {}),
        adaptive_destroy_weights=dict(adaptive_destroy_weights or {}),
        adaptive_repair_weights=dict(adaptive_repair_weights or {}),
        adaptive_destroy_scores=dict(adaptive_destroy_scores or {}),
        adaptive_destroy_uses={str(k): int(v) for k, v in dict(adaptive_destroy_uses or {}).items()},
        adaptive_repair_scores=dict(adaptive_repair_scores or {}),
        adaptive_repair_uses={str(k): int(v) for k, v in dict(adaptive_repair_uses or {}).items()},
        adaptation_trials_elapsed=int(adaptation_trials_elapsed),
        global_best_solution=(
            None if global_best_solution is None else global_best_solution.clone_search_state()
        ),
        adaptation_horizon_iters=adaptation_horizon_iters,
        operator_pair_bandit=dict(operator_pair_bandit or {}),
        best_snapshot_callback=best_snapshot_callback,
    )


def _solve_weighted_alns(
    instance: Instance,
    cur: AssignmentSolution,
    cur_ev: EvalResult,
    config: Config,
    budget: Budget,
    policy: CompiledALNSPolicy,
    rng: random.Random,
    *,
    objective_terms: List[Dict[str, Any]],
    trace_id: str,
    compiled_search_focus: Dict[str, Any],
    adaptive_destroy_weights: Dict[str, float],
    adaptive_repair_weights: Dict[str, float],
    adaptive_destroy_scores: Dict[str, float],
    adaptive_destroy_uses: Dict[str, int],
    adaptive_repair_scores: Dict[str, float],
    adaptive_repair_uses: Dict[str, int],
    adaptation_trials_elapsed: int,
    global_best_solution: Optional[AssignmentSolution],
    adaptation_horizon_iters: Optional[int],
    operator_pair_bandit: Dict[str, Any],
    best_snapshot_callback: Optional[Callable[[int, AssignmentSolution], None]] = None,
) -> ALNSResult:
    enabled_destroy_ops = {
        str(name): operator
        for name, operator in DESTROY_OPERATORS.items()
        if float(policy.destroy_policy.operator_weights.get(str(name), 0) or 0) > 0.0
    }
    if not enabled_destroy_ops:
        raise ExecutorContractError("compiled destroy policy has no enabled operators")
    destroy_priors = {
        name: float(policy.destroy_policy.operator_weights[name])
        for name in enabled_destroy_ops
    }
    destroy_state_names = set(DESTROY_OPERATOR_NAMES) | set(adaptive_destroy_weights) | set(adaptive_destroy_scores) | set(adaptive_destroy_uses)
    d_w = {
        name: max(0.10, min(10.0, float(adaptive_destroy_weights.get(name, 1.0) or 1.0)))
        for name in sorted(destroy_state_names)
    }
    adaptive_before = {name: float(d_w.get(name, 1.0)) for name in enabled_destroy_ops}
    d_score = {name: float(adaptive_destroy_scores.get(name, 0.0) or 0.0) for name in d_w}
    d_used = {name: int(adaptive_destroy_uses.get(name, 0) or 0) for name in d_w}

    enabled_repair_ops = {
        str(name): float(weight)
        for name, weight in policy.insertion_policy.operator_weights.items()
        if float(weight) > 0.0
    }
    if not enabled_repair_ops:
        raise ExecutorContractError("compiled repair policy has no enabled operators")
    repair_priors = dict(enabled_repair_ops)
    repair_state_names = set(ALNS_REPAIR_OPERATOR_NAMES) | set(adaptive_repair_weights) | set(adaptive_repair_scores) | set(adaptive_repair_uses)
    r_w = {
        name: max(0.10, min(10.0, float(adaptive_repair_weights.get(name, 1.0) or 1.0)))
        for name in sorted(repair_state_names)
    }
    adaptive_repair_before = {name: float(r_w.get(name, 1.0)) for name in enabled_repair_ops}
    r_score = {name: float(adaptive_repair_scores.get(name, 0.0) or 0.0) for name in r_w}
    r_used = {name: int(adaptive_repair_uses.get(name, 0) or 0) for name in r_w}
    # Diagnostic-only uniqueness counter.  It never filters or changes move
    # eligibility; standard ALNS adaptation remains the only search memory.
    selected_destroy_move_keys: set[tuple[str, tuple[int, ...]]] = set()

    if isinstance(budget.max_iters, bool) or int(budget.max_iters) <= 0:
        raise ExecutorContractError("budget.max_iters must be a positive integer")
    max_iters = int(budget.max_iters)
    horizon = int(adaptation_horizon_iters or max_iters)
    if horizon <= 0:
        raise ExecutorContractError("adaptation_horizon_iters must be positive when provided")
    adaptation_offset = int(adaptation_trials_elapsed)
    if adaptation_offset < 0:
        raise ExecutorContractError("adaptation_trials_elapsed must be nonnegative")
    segment_len = _clamp_int(int(0.10 * horizon), 10, 200)

    reaction = float(policy.reaction_factor)
    worsening_tolerance = float(policy.acceptance_policy.worsening_tolerance)
    action_start_solution = cur.clone_search_state()
    action_start_eval = cur_ev
    action_start_constraint_report = cur_ev.constraint_report
    acceptance_scales = _build_acceptance_scales(
        instance, action_start_eval, objective_terms
    )
    temperature, cooling = _sa_temperature_schedule(
        worsening_tolerance=worsening_tolerance,
        max_iters=max_iters,
    )
    initial_temperature = float(temperature)
    initial_solution_feasible = bool(cur_ev.is_feasible)
    initial_working_objective_keys = build_objective_keys(cur_ev, objective_terms)

    initial_is_eligible = bool(cur_ev.is_feasible)
    action_best_feasible: Optional[AssignmentSolution] = (
        cur.clone_search_state() if initial_is_eligible else None
    )
    action_best_feasible_ev: Optional[EvalResult] = (
        cur_ev if initial_is_eligible else None
    )
    if global_best_solution is not None:
        global_best_solution.normalize(instance)
        reference_ev = evaluate(
            global_best_solution, instance, config, update_solution_schedule=False
        )
    else:
        reference_ev = None
    if reference_ev is not None and reference_ev.is_feasible:
        global_best_feasible = global_best_solution.clone_search_state()
        global_best_feasible_ev = reference_ev
    else:
        global_best_feasible = cur.clone_search_state() if initial_is_eligible else None
        global_best_feasible_ev = cur_ev if initial_is_eligible else None

    best_update_iters: List[int] = []
    best_update_objective_keys: List[Dict[str, Any]] = []
    last_acceptance_decision: Optional[Dict[str, Any]] = None
    destroy_operator_stats = _init_destroy_operator_stats(d_w.keys())
    insertion_operator_summary = _init_insertion_operator_summary(
        policy.insertion_policy.operator_weights.keys()
    )
    iteration_trace: List[Dict[str, Any]] = []
    last_destroy_move: Optional[Dict[str, Any]] = None
    last_insertion: Optional[Dict[str, Any]] = None
    feasibility_rejection_reasons: Dict[str, int] = {}
    events: List[str] = []
    trial_flow = {
        "candidate_trials": 0,
        "no_reinserted_trials": 0,
        "feasibility_rejected": 0,
        "admissible_trials": 0,
        "acceptance_rejected": 0,
        "accepted_trials": 0,
        "global_best_update_count": 0,
        "accepted_improving_count": 0,
        "accepted_non_improving_count": 0,
        "structurally_changed_trials": 0,
        "accepted_structurally_changed_trials": 0,
        "destroy_repair_noop_trials": 0,
        "focus_active_trials": 0,
        "focus_inactive_trials": 0,
        "focus_active_best_updates": 0,
        "focus_inactive_best_updates": 0,
    }
    visited_trial_structures: set[tuple[Any, ...]] = set()
    accepted_structures: set[tuple[Any, ...]] = {cur.structure_key()}
    insertion_failure_reasons: Dict[str, int] = {}
    removed_task_count_sum = 0.0

    pair_bandit_enabled = bool(operator_pair_bandit.get("enabled", False))
    pair_bandit_exploration = float(
        operator_pair_bandit.get("exploration", 0.0) or 0.0
    )
    if pair_bandit_exploration < 0.0:
        raise ExecutorContractError("operator-pair Bandit exploration must be nonnegative")
    pair_bandit_discount = float(
        operator_pair_bandit.get("discount", 1.0) or 1.0
    )
    if not (0.0 < pair_bandit_discount <= 1.0):
        raise ExecutorContractError("operator-pair Bandit discount must be in (0,1]")
    pair_arms = [
        (destroy, repair)
        for destroy in sorted(enabled_destroy_ops)
        for repair in sorted(enabled_repair_ops)
    ]
    pair_counts = {arm: 0 for arm in pair_arms}
    pair_effective_counts = {arm: 0.0 for arm in pair_arms}
    pair_rewards = {arm: 0.0 for arm in pair_arms}
    pair_order = list(pair_arms)
    pair_rng = random.Random(
        int(operator_pair_bandit.get("seed", int(config.rng_seed)) or 0)
    )
    pair_rng.shuffle(pair_order)
    pair_decisions: List[Dict[str, Any]] = []
    pair_controller_time = 0.0

    def choose_pair_arm() -> tuple[str, str]:
        nonlocal pair_controller_time
        started = time.perf_counter()
        try:
            # AlphaUCB: optimistic unobserved mean, cumulative credit,
            # and the count+1 convention of the public ALNS implementation.
            played = sum(pair_counts.values())
            def alpha_value(candidate):
                count = pair_counts[candidate]
                mean = pair_rewards[candidate] / count if count else 1.0
                return mean + math.sqrt(
                    pair_bandit_exploration * math.log1p(played) / (count + 1))
            return max(pair_arms, key=alpha_value)
        finally:
            pair_controller_time += time.perf_counter() - started

    iteration = 0
    started_at = time.perf_counter()
    stop_reason = "completed"
    no_destroy_move_details: Dict[str, Any] = {}
    runtime_focus = RuntimeFocusState.from_compiled(compiled_search_focus, cur)

    while iteration < max_iters:
        iteration += 1
        pair_arm = (
            choose_pair_arm()
            if pair_bandit_enabled and iteration > int(operator_pair_bandit.get("uniform_warmup", 0))
            else None
        )
        before_unassigned = {int(tid) for tid in cur.unassigned}
        active_search_focus = restrict_runtime_focus(runtime_focus.as_mapping(), cur)
        focus_is_active = FocusContext.from_mapping(active_search_focus).active
        if focus_is_active:
            trial_flow["focus_active_trials"] += 1
        else:
            trial_flow["focus_inactive_trials"] += 1
        compiled_focus_targets = tuple(
            int(tid) for tid in active_search_focus.get("focus_task_ids", []) or []
        )
        destroy_context = DestroyContext(focus_task_ids=compiled_focus_targets)
        active_focus_id = str(active_search_focus.get("focus_id", "global") or "global")
        try:
            move = select_destroy_move(
                sol=cur,
                instance=instance,
                config=config,
                policy=policy.destroy_policy,
                operator=(
                    None if pair_arm is None else DESTROY_OPERATORS[pair_arm[0]]
                ),
                rng=rng,
                compiled_search_focus=active_search_focus,
                destroy_context=destroy_context,
                adaptive_weights=d_w,
            )
        except NoDestroyMoveAvailable as exc:
            stop_reason = "no_destroy_move_available"
            no_destroy_move_details = {
                "attempted_operators": [dict(item) for item in exc.attempted]
            }
            iteration -= 1
            break
        actual_d_name = str(move.operator_name)
        selected_destroy_move_keys.add(_destroy_move_key(move))
        removed = list(int(tid) for tid in move.task_ids)
        last_destroy_move = move.as_dict()
        move_metadata = dict(move.metadata or {})
        selection_phase = str(move_metadata.get("focus_selection_phase", ""))
        partial = cur.clone_search_state()
        _remove_tasks(partial, removed)
        partial.solver_diagnostics = {"last_destroy_move": last_destroy_move}
        partial.normalize(instance)

        candidate_tasks = list(
            dict.fromkeys(removed + sorted(int(tid) for tid in partial.unassigned))
        )
        actual_r_name = (
            pair_arm[1]
            if pair_arm is not None
            else _select_repair_operator(
                repair_priors=repair_priors,
                adaptive_weights=r_w,
                rng=rng,
            )
        )
        if pair_bandit_enabled and pair_arm is None:
            # Uniform warm-up uses the existing sampler and solver RNG, while
            # the Bandit observes the actual executed pair and its reward.
            pair_arm = (actual_d_name, actual_r_name)
        trial = run_insertion_kernel(
            partial_solution=partial,
            candidate_tasks=candidate_tasks,
            insertion_policy=policy.insertion_policy,
            context=InsertionContext(
                kind="alns",
                focus_task_ids=tuple(
                    int(tid) for tid in active_search_focus.get("focus_task_ids", []) or []
                ),
                focus_soft_greedy_bonus=float(active_search_focus.get("soft_greedy_bonus", 1.0) or 1.0),
                focus_soft_random_weight=float(active_search_focus.get("soft_random_weight", 1.5) or 1.5),
            ),
            selected_operator_name=actual_r_name,
            instance=instance,
            config=config,
            rng=rng,
            objective_terms=objective_terms,
        )
        trial.solver_diagnostics["last_destroy_move"] = last_destroy_move
        last_insertion = dict(
            (getattr(trial, "solver_diagnostics", {}) or {}).get("last_insertion", {})
            or {}
        )
        trial.normalize(instance)
        ev_trial = evaluate(trial, instance, config, update_solution_schedule=False)
        iter_search_focus_progress = _search_focus_progress_from_snapshot(
            before_unassigned=before_unassigned,
            after_unassigned={int(tid) for tid in trial.unassigned},
            compiled_search_focus=active_search_focus,
        )
        feasibility_decision = _check_strict_feasibility(ev_trial.constraint_report)
        events.extend(feasibility_decision.events)
        previous_cur_ev = cur_ev
        previous_structure_key = cur.structure_key()
        trial_structure_key = trial.structure_key()
        trial_structure_changed = trial_structure_key != previous_structure_key
        visited_trial_structures.add(trial_structure_key)
        if trial_structure_changed:
            trial_flow["structurally_changed_trials"] += 1
        else:
            trial_flow["destroy_repair_noop_trials"] += 1

        acceptance = _AcceptanceDecision(
            compare_result=0,
            accepted=False,
            accept_mode=policy.acceptance_policy.mode,
            feasibility_admissible=feasibility_decision.admissible,
            accept_scope=feasibility_decision.accept_scope,
            feasibility_reason=feasibility_decision.reason,
        )
        rejection_reason = ""
        if feasibility_decision.admissible:
            trial_flow["admissible_trials"] += 1
            if not trial_structure_changed:
                # A destroy-repair cycle that reconstructs the incumbent is not a
                # search move. Reject it before acceptance and never reinforce it.
                acceptance = _AcceptanceDecision(
                    compare_result=compare_quality(ev_trial, cur_ev, objective_terms),
                    accepted=False,
                    accept_mode=policy.acceptance_policy.mode,
                    feasibility_admissible=True,
                    accept_scope=feasibility_decision.accept_scope,
                    feasibility_reason=feasibility_decision.reason,
                    rejection_reason="destroy_repair_noop_rejected",
                )
            else:
                acceptance = _alns_accept(
                    cur_ev=cur_ev,
                    trial_ev=ev_trial,
                    objective_terms=objective_terms,
                    mode=policy.acceptance_policy.mode,
                    rng=rng,
                    temperature=temperature,
                    worsening_tolerance=worsening_tolerance,
                    feasibility_admissible=True,
                    accept_scope=feasibility_decision.accept_scope,
                    feasibility_reason=feasibility_decision.reason,
                    acceptance_scales=acceptance_scales,
                )
        elif not feasibility_decision.admissible:
            trial_flow["feasibility_rejected"] += 1
            rejection_reason = feasibility_decision.reason
            feasibility_rejection_reasons[feasibility_decision.reason] = (
                feasibility_rejection_reasons.get(feasibility_decision.reason, 0) + 1
            )
        accepted = acceptance.accepted
        last_acceptance_decision = acceptance.as_dict()
        d_used[actual_d_name] += 1
        r_used[actual_r_name] += 1
        removed_task_count_sum += float(len(removed))
        trial_flow["candidate_trials"] += 1
        if int(last_insertion.get("inserted_count", 0) or 0) == 0:
            trial_flow["no_reinserted_trials"] += 1
            insertion_failure_reasons["no_reinserted_task"] = (
                insertion_failure_reasons.get("no_reinserted_task", 0) + 1
            )
        for name, count in dict(
            last_insertion.get("failure_breakdown", {}) or {}
        ).items():
            if int(count):
                insertion_failure_reasons[str(name)] = insertion_failure_reasons.get(
                    str(name), 0
                ) + int(count)
        reward = 0.0
        if accepted:
            cur = trial
            cur_ev = ev_trial
            trial_flow["accepted_trials"] += 1
            accepted_structures.add(trial_structure_key)
            if trial_structure_changed:
                trial_flow["accepted_structurally_changed_trials"] += 1
            if acceptance.compare_result < 0:
                trial_flow["accepted_improving_count"] += 1
            else:
                trial_flow["accepted_non_improving_count"] += 1
            reward = max(reward, 0.2)
        else:
            if feasibility_decision.admissible:
                trial_flow["acceptance_rejected"] += 1
                rejection_reason = acceptance.rejection_reason or "acceptance_rejected"

        if (
            ev_trial.is_feasible
            and (
                action_best_feasible_ev is None
                or compare_quality(
                    ev_trial, action_best_feasible_ev, objective_terms
                )
                < 0
            )
        ):
            action_best_feasible = trial.clone_search_state()
            action_best_feasible_ev = ev_trial

        best_improved = False
        if (
            ev_trial.is_feasible
            and (
                global_best_feasible_ev is None
                or compare_quality(
                    ev_trial, global_best_feasible_ev, objective_terms
                )
                < 0
            )
        ):
            global_best_feasible = trial.clone_search_state()
            global_best_feasible_ev = ev_trial
            # Optional read-only export: isolate the callback from live search state.
            # No RNG calls or changes to search decisions when exporting snapshots.
            if best_snapshot_callback is not None:
                best_snapshot_callback(iteration, global_best_feasible.clone(deep=True))
            best_update_iters.append(iteration)
            best_update_objective_keys.append(
                build_objective_keys(global_best_feasible_ev, objective_terms)
            )
            reward = max(reward, 5.0)
            best_improved = True
            trial_flow["global_best_update_count"] += 1
            if active_focus_id == "global":
                trial_flow["focus_inactive_best_updates"] += 1
            else:
                trial_flow["focus_active_best_updates"] += 1

        reward = _adaptive_operator_reward(
            previous_cur_ev,
            ev_trial,
            accepted=accepted,
            global_best_improved=best_improved,
            structure_changed=trial_structure_changed,
            objective_terms=objective_terms,
        )
        if pair_arm is not None:
            update_started = time.perf_counter()
            pair_reward = max(0.0, min(1.0, float(reward) / 8.0))
            for candidate in pair_arms:
                pair_effective_counts[candidate] *= pair_bandit_discount
                pair_rewards[candidate] *= pair_bandit_discount
            pair_counts[pair_arm] += 1
            pair_effective_counts[pair_arm] += 1.0
            pair_rewards[pair_arm] += pair_reward
            pair_decisions.append({
                "decision_index": len(pair_decisions) + 1,
                "role": "bandit",
                "trial": int(iteration - 1),
                "arm": {"destroy": pair_arm[0], "repair": pair_arm[1]},
                "reward": float(pair_reward),
                "raw_operator_reward": float(reward),
                "global_best_improved": bool(best_improved),
                "count_after": int(pair_counts[pair_arm]),
                "effective_count_after": float(pair_effective_counts[pair_arm]),
                "mean_reward_after": float(
                    pair_rewards[pair_arm] / pair_effective_counts[pair_arm]
                ),
            })
            pair_controller_time += time.perf_counter() - update_started

        iteration_trace.append(
            {
                "iteration": int(iteration),
                "destroy_operator": actual_d_name,
                "insertion_operator": actual_r_name,
                "operator_pair": {
                    "destroy": actual_d_name,
                    "repair": actual_r_name,
                },
                "accepted": bool(accepted),
                "acceptance_decision": acceptance.as_dict(),
                "trial_objective_keys": build_objective_keys(ev_trial, objective_terms),
                "reward": float(reward),
                "trial_structure_changed": bool(trial_structure_changed),
                "global_best_improved": bool(best_improved),
                "active_focus_task_ids": _dedupe_ints(
                    active_search_focus.get("focus_task_ids", []) or []
                ),
                "active_blocking_task_count": int(
                    (active_search_focus.get("block_analysis", {}) or {}).get(
                        "blocking_task_count", 0
                    )
                    or 0
                ),
                "active_blocking_neighborhood_count": int(
                    (active_search_focus.get("block_analysis", {}) or {}).get(
                        "blocking_neighborhood_count", 0
                    )
                    or 0
                ),
                "current_objective_keys": build_objective_keys(cur_ev, objective_terms),
                "action_best_objective_keys": (
                    None
                    if action_best_feasible_ev is None
                    else build_objective_keys(action_best_feasible_ev, objective_terms)
                ),
                "global_best_objective_keys": (
                    None
                    if global_best_feasible_ev is None
                    else build_objective_keys(global_best_feasible_ev, objective_terms)
                ),
                "violation_total": float(ev_trial.get_metric("violation_total")),
                "violation_ratio_by_type": dict(
                    ev_trial.constraint_report.violation_ratio_by_type
                ),
                "feasibility_reason": feasibility_decision.reason,
                "rejection_reason": rejection_reason or None,
                "destroy_search_focus_metadata": dict(
                    (last_destroy_move or {}).get("metadata", {}) or {}
                ),
                "search_focus_insertion": _compact_insertion_scope(last_insertion),
                "search_focus_progress": iter_search_focus_progress,
                "destroy_feature_usage": dict((last_destroy_move or {}).get("metadata", {}).get("feature_usage", {}) or {}),
                "insertion_feature_usage": dict((last_insertion or {}).get("feature_usage", {}) or {}),
            }
        )

        d_score[actual_d_name] += reward
        r_score[actual_r_name] += reward
        _accumulate_destroy_operator_stats(
            destroy_operator_stats,
            move=move,
            accepted=accepted,
            best_improved=best_improved,
            reward=reward,
        )
        _accumulate_insertion_operator_summary(
            insertion_operator_summary,
            operator_name=actual_r_name,
            last_insertion=last_insertion,
            accepted=accepted,
            best_improved=best_improved,
            reward=reward,
        )

        if policy.acceptance_policy.mode == "sa" and temperature > 0.0:
            temperature = temperature * cooling

        global_adaptation_iteration = adaptation_offset + iteration
        if global_adaptation_iteration % segment_len == 0:
            _update_weights(d_w, d_score, d_used, reaction)
            _update_weights(r_w, r_score, r_used, reaction)
            _reset_segment_scores(d_score, d_used)
            _reset_segment_scores(r_score, r_used)

    if iteration >= max_iters and stop_reason == "completed":
        stop_reason = "iteration_budget_exhausted"

    # Do not settle a partial reward segment at an Agent action boundary.
    # Pending scores/uses and the global trial offset are returned so a sequence
    # of identical Basic actions is exactly one continuous ALNS run.

    final_current = cur.clone_search_state()
    evaluate(final_current, instance, config, update_solution_schedule=True)
    if action_best_feasible is not None:
        evaluate(action_best_feasible, instance, config, update_solution_schedule=True)
    if global_best_feasible is not None:
        evaluate(global_best_feasible, instance, config, update_solution_schedule=True)
    # The accepted ALNS trajectory is the working state.  The action/global best
    # remain separate incumbents for reporting and final output.  Returning the
    # action best here would silently undo the last accepted exploratory moves and
    # make tolerant/SA acceptance ineffective across Step actions.
    working_solution = final_current.clone_search_state()
    returned_source = "final_current"
    runtime_focus.restrict_to(final_current)
    actual_time_used_sec = max(0.0, time.perf_counter() - started_at)
    trial_flow["unique_trial_structure_count"] = len(visited_trial_structures)
    trial_flow["unique_accepted_structure_count"] = len(accepted_structures)
    trial_flow["unique_destroy_move_count"] = len(selected_destroy_move_keys)

    trace = _build_execution_trace(
        trace_id=trace_id,
        total_iters=iteration,
        destroy_operator_summary=_finalize_destroy_operator_stats(
            destroy_operator_stats
        ),
        insertion_operator_summary=_finalize_insertion_operator_summary(
            insertion_operator_summary
        ),
        trial_flow=trial_flow,
        rejection_reasons={
            **feasibility_rejection_reasons,
        },
        insertion_failure_reasons=insertion_failure_reasons,
        removed_task_count_sum=removed_task_count_sum,
        operator_prior_trace={
            "llm_prior": dict(destroy_priors),
            "adaptive_weight_before": dict(adaptive_before),
            "adaptive_weight_after": dict(d_w),
            "effective_sampling": "two_level_operator_then_move",
            "repair_llm_prior": dict(repair_priors),
            "repair_adaptive_weight_before": dict(adaptive_repair_before),
            "repair_adaptive_weight_after": dict(r_w),
            "repair_effective_sampling": "prior_x_adaptive_history",
        },
        compiled_search_focus=compiled_search_focus,
        feasibility_rejection_reasons=feasibility_rejection_reasons,
        final_constraint_report=working_solution.eval.constraint_report,
        actual_time_used_sec=actual_time_used_sec,
        state_update_selected_source=returned_source,
        stop_reason=stop_reason,
        iteration_trace=iteration_trace,
        last_insertion=last_insertion,
        before_solution=action_start_solution,
        after_solution=working_solution,
    )
    trace["acceptance_model"] = {
        "model": "lexicographic_first_difference_fixed_scale",
        "worsening_tolerance": float(worsening_tolerance),
        "metric_scales": {
            str(name): float(value) for name, value in acceptance_scales.items()
        },
        "lower_priority_offsets_allowed": False,
        "threshold_limit": (
            float(worsening_tolerance)
            if policy.acceptance_policy.mode == "threshold"
            else None
        ),
        "sa_initial_temperature": (
            float(initial_temperature)
            if policy.acceptance_policy.mode == "sa"
            else None
        ),
        "sa_cooling_factor": (
            float(cooling) if policy.acceptance_policy.mode == "sa" else None
        ),
        "sa_final_temperature_ratio": (
            float(SA_FINAL_TEMPERATURE_RATIO)
            if policy.acceptance_policy.mode == "sa"
            else None
        ),
        "sa_final_temperature": (
            float(temperature) if policy.acceptance_policy.mode == "sa" else None
        ),
    }
    if no_destroy_move_details:
        trace["no_destroy_move"] = dict(no_destroy_move_details)
    trace["timing"] = {
        "elapsed_sec": round(float(actual_time_used_sec), 6),
        "seconds_per_iteration": round(
            float(actual_time_used_sec) / max(1, int(iteration)), 9
        ),
    }
    trace["runtime_focus_refresh"] = runtime_focus.diagnostics(
        enabled=False,
        actual_time_used_sec=actual_time_used_sec,
    )
    if pair_bandit_enabled:
        trace["operator_pair_bandit"] = {
            "algorithm": (
                ("discounted_" if pair_bandit_discount < 1.0 else "")
                + str(operator_pair_bandit.get("sampler", "ucb1"))
            ),
            "prior": float(operator_pair_bandit.get("prior", 1.0)),
            "warmup": int(operator_pair_bandit.get("warmup", 1)),
            "uniform_warmup": int(operator_pair_bandit.get("uniform_warmup", 0)),
            "exploration": float(pair_bandit_exploration),
            "discount": float(pair_bandit_discount),
            "reward_definition": "adaptive_operator_reward_divided_by_8",
            "controller_time_sec": float(pair_controller_time),
            "counts": {f"{d}+{r}": int(pair_counts[(d, r)]) for d, r in pair_arms},
            "mean_rewards": {
                f"{d}+{r}": (
                    float(pair_rewards[(d, r)] / pair_effective_counts[(d, r)])
                    if pair_effective_counts[(d, r)] > 0.0
                    else None
                )
                for d, r in pair_arms
            },
            "effective_counts": {
                f"{d}+{r}": float(pair_effective_counts[(d, r)])
                for d, r in pair_arms
            },
            "decisions": pair_decisions,
        }
    diagnostics = _build_solver_diagnostics(
        policy=policy,
        total_iters=iteration,
        actual_time_used_sec=actual_time_used_sec,
        best_update_iters=best_update_iters,
        best_update_objective_keys=best_update_objective_keys,
        state_update_selected_source=returned_source,
        initial_solution_feasible=initial_solution_feasible,
        returned_solution_feasible=bool(
            getattr(working_solution.eval, "is_feasible", False)
        ),
        last_acceptance_decision=last_acceptance_decision,
        last_destroy_move=last_destroy_move,
        destroy_operator_summary=trace["destroy"]["selected_operator_counts"],
        insertion_operator_summary=trace["insertion"],
        iteration_trace=iteration_trace,
        last_insertion=last_insertion,
        operator_weights={
            "destroy_operators": {
                "adaptive_before": adaptive_before,
                "adaptive_after": d_w,
                "llm_score_prior": destroy_priors,
                "effective_sampling": "two_level_operator_then_move",
            },
            "insertion_operators": {
                "llm_weights": policy.insertion_policy.operator_weights,
                "adaptive_before": adaptive_repair_before,
                "adaptive_after": r_w,
                "effective_sampling": "prior_x_adaptive_history",
            },
        },
        final_constraint_report=working_solution.eval.constraint_report,
        feasibility_rejection_reasons=feasibility_rejection_reasons,
        execution_trace=trace,
        stop_reason=stop_reason,
    )
    diagnostics["solution_flow"] = {
        "initial_working": {"objective_keys": initial_working_objective_keys},
        "final_current": {
            "objective_keys": build_objective_keys(final_current.eval, objective_terms),
            "is_feasible": bool(final_current.eval.is_feasible),
        },
        "action_best_feasible": (
            None
            if action_best_feasible is None
            else {
                "objective_keys": build_objective_keys(action_best_feasible.eval, objective_terms),
                "is_feasible": True,
            }
        ),
        "state_update_selected_source": returned_source,
    }
    final_current.solver_diagnostics = diagnostics
    working_solution.solver_diagnostics = diagnostics
    if action_best_feasible is not None:
        action_best_feasible.solver_diagnostics = diagnostics
    if global_best_feasible is not None:
        global_best_feasible.solver_diagnostics = diagnostics
    return ALNSResult(
        final_current=final_current,
        action_best_feasible=action_best_feasible,
        global_best_feasible=global_best_feasible,
        working_solution=working_solution,
        events=_dedupe_events(events),
        trace=trace,
        diagnostics=diagnostics,
        adaptive_destroy_weights=dict(d_w),
        adaptive_repair_weights=dict(r_w),
        adaptive_destroy_scores=dict(d_score),
        adaptive_destroy_uses={str(k): int(v) for k, v in d_used.items()},
        adaptive_repair_scores=dict(r_score),
        adaptive_repair_uses={str(k): int(v) for k, v in r_used.items()},
        adaptation_trials_elapsed=int(adaptation_offset + iteration),
    )


def _select_repair_operator(
    *,
    repair_priors: Mapping[str, float],
    adaptive_weights: Mapping[str, float],
    rng: random.Random,
) -> str:
    """Sample one repair heuristic from Step prior × classic ALNS history weight."""
    rows = []
    for name, prior in repair_priors.items():
        effective = max(0.0, float(prior)) * max(0.10, float(adaptive_weights.get(str(name), 1.0) or 1.0))
        if effective > 0.0:
            rows.append((str(name), effective))
    if not rows:
        raise ExecutorContractError("all repair operators have zero effective weight")
    total = sum(weight for _, weight in rows)
    threshold = rng.random() * total
    cumulative = 0.0
    for name, weight in rows:
        cumulative += weight
        if cumulative >= threshold:
            return name
    return rows[-1][0]


def select_destroy_move(
    sol: AssignmentSolution,
    instance: Instance,
    config: Config,
    policy: DestroyPolicy,
    operator: Optional[DestroyOperator] = None,
    rng: Optional[random.Random] = None,
    compiled_search_focus: Optional[Dict[str, Any]] = None,
    destroy_context: Optional[DestroyContext] = None,
    adaptive_weights: Optional[Mapping[str, float]] = None,
) -> DestroyMove:
    """Generate pure operator candidates, then focus, score and select once.

    The final candidate selector is random; focus can apply a bounded bias.
    """

    rng = rng or random.Random(int(config.rng_seed))
    search_focus = dict(compiled_search_focus or {})
    focus_context = FocusContext.from_mapping(search_focus)
    context = destroy_context or DestroyContext(
        focus_task_ids=tuple(
            int(tid) for tid in search_focus.get("focus_task_ids", []) or []
        ),
    )
    if not sol.all_assigned_tasks():
        raise NoDestroyMoveAvailable([{"reason": "solution_has_no_assigned_tasks"}])

    strength = compute_destroy_strength(sol, policy.remove_ratio)
    adaptive = {str(name): float(value) for name, value in dict(adaptive_weights or {}).items()}
    effective_weights = {
        str(name): float(weight) * max(0.10, float(adaptive.get(str(name), 1.0)))
        for name, weight in policy.operator_weights.items()
        if float(weight) > 0.0
    }
    if not effective_weights:
        raise ExecutorContractError("all destroy operators are disabled")

    attempted: List[Dict[str, Any]] = []
    generated: List[DestroyMove] = []
    selected_operator = ""
    if operator is not None:
        generated = list(operator(sol, instance, config, policy, strength, rng, context))
        selected_operator = (
            str(generated[0].operator_name)
            if generated
            else str(getattr(operator, "__name__", "provided_operator"))
        )
        attempted.append({
            "operator": selected_operator,
            "effective_weight": float(effective_weights.get(selected_operator, 1.0)),
            "generated_move_count": len(generated),
        })
    else:
        remaining = dict(effective_weights)
        while remaining:
            selected_operator = _weighted_choice_mapping(remaining, rng)
            generator = DESTROY_OPERATORS.get(selected_operator)
            if generator is None:
                raise ExecutorContractError(
                    f"unknown destroy operator in compiled policy: {selected_operator}"
                )
            generated = list(generator(sol, instance, config, policy, strength, rng, context))
            attempted.append({
                "operator": selected_operator,
                "score_0_to_10": int(policy.operator_weights.get(selected_operator, 0)),
                "adaptive_weight": float(adaptive.get(selected_operator, 1.0)),
                "effective_weight": float(remaining[selected_operator]),
                "generated_move_count": len(generated),
            })
            if generated:
                break
            remaining.pop(selected_operator, None)
    if not generated:
        raise NoDestroyMoveAvailable(attempted)

    # Plain/basic ALNS path: with no semantic focus and no destroy features,
    # the chosen destroy operator owns the neighbourhood completely.  We pick
    # directly from that operator's generated moves and do not run the shared
    # semantic candidate scoring/normalization layer.  The Agent uses this exact
    # same path whenever the Agent chooses a basic/global action.
    active_feature_weights = {
        str(name): float(weight)
        for name, weight in policy.feature_weights.items()
        if abs(float(weight)) > 1e-12
    }
    if (not focus_context.active) and (not active_feature_weights):
        selected = sample_intrinsic_destroy_move(generated, rng)
        attempted_rows = [
            {**row, "selected_count": int(str(row.get("operator")) == selected_operator)}
            for row in attempted
        ]
        return _with_search_focus_metadata(
            selected,
            focus_selection_phase="operator_intrinsic_global",
            destroy_selector_mode="operator_intrinsic_random",
            selectable_candidate_count=len(generated),
            **_search_focus_destroy_metadata(selected, search_focus, len(generated), 0),
            destroy_operator_opportunities=attempted_rows,
            operator_effective_weights=effective_weights,
            feature_usage=[],
        )
    # The current protocol keeps one common candidate pool. Focus only adds a bounded
    # selection preference; it never filters or augments candidates.
    partitioned, targeted_count = _partition_destroy_candidates_by_focus(generated, focus=focus_context)
    active_pool = _score_moves(partitioned, sol, instance, config, policy, context=context)
    selectable_pool = active_pool
    baseline = _select_destroy_baseline(selectable_pool, mode=policy.selector_mode, rng=rng)
    selected = _select_destroy_with_soft_focus(
        selectable_pool, mode=policy.selector_mode, focus=focus_context, rng=rng
    )
    selection_phase = "soft_focus_common_pool" if focus_context.active else "global_common_pool"
    selected = _with_search_focus_metadata(
        selected,
        focus_selection_phase=selection_phase,
        destroy_selector_mode=str(policy.selector_mode),
        selectable_candidate_count=len(selectable_pool),
        focus_changed_selection=(_destroy_move_key(selected) != _destroy_move_key(baseline)),
    )
    feature_usage = _destroy_feature_usage(active_pool, policy)
    attempted_rows = [
        {**row, "selected_count": int(str(row.get("operator")) == selected_operator)}
        for row in attempted
    ]
    return _with_search_focus_metadata(
        selected,
        **_search_focus_destroy_metadata(
            selected, search_focus, len(partitioned), targeted_count
        ),
        destroy_operator_opportunities=attempted_rows,
        operator_effective_weights=effective_weights,
        feature_usage=feature_usage,
    )


def _partition_destroy_candidates_by_focus(
    moves: Sequence[DestroyMove],
    *,
    focus: FocusContext,
) -> tuple[List[DestroyMove], int]:
    if not moves:
        return [], 0
    metadata, targeted_count = annotate_destroy_moves_by_focus(
        moves,
        focus,
        task_ids_getter=lambda move: move.task_ids,
    )
    out: List[DestroyMove] = []
    for index, move in enumerate(moves):
        row = dict(metadata.get(index, {}) or {})
        out.append(
            _with_search_focus_metadata(
                move,
                **row,
                focus_annotation_applied=bool(focus.active),
                focus_candidate_group=(
                    "focus_targeted" if row.get("focus_targeted") else "other"
                ),
            )
        )
    # Preserve the generator's candidate order and scores. Focus only annotates
    # membership; the explicit selector makes the final choice.
    return out, targeted_count



def _select_destroy_baseline(moves: Sequence[DestroyMove], *, mode: str, rng: random.Random) -> DestroyMove:
    # Counterfactual baseline uses a cloned RNG state so diagnostics do not consume
    # the production random stream.
    clone = random.Random(); clone.setstate(rng.getstate())
    return select_item(
        list(moves), mode=mode, score_getter=lambda move: float(move.score),
        tie_key=lambda move: tuple(int(tid) for tid in sorted(move.task_ids)), rng=clone,
    )


def _select_destroy_with_soft_focus(moves: Sequence[DestroyMove], *, mode: str, focus: FocusContext, rng: random.Random) -> DestroyMove:
    rows = list(moves)
    if not focus.active:
        return select_item(rows, mode=mode, score_getter=lambda move: float(move.score), tie_key=lambda move: tuple(int(tid) for tid in sorted(move.task_ids)), rng=rng)
    weights = [float(focus.soft_random_weight) if bool(dict(move.metadata or {}).get("focus_targeted", False)) else 1.0 for move in rows]
    total = sum(weights); threshold = rng.random() * total; acc = 0.0
    for move, weight in zip(rows, weights):
        acc += weight
        if acc >= threshold:
            return move
    return rows[-1]


def _destroy_move_key(move: DestroyMove) -> tuple[str, tuple[int, ...]]:
    return str(move.operator_name), tuple(sorted(int(tid) for tid in move.task_ids))


def _weighted_choice_mapping(weights: Mapping[str, float], rng: random.Random) -> str:
    positive = [(str(name), max(0.0, float(weight))) for name, weight in weights.items() if float(weight) > 0.0]
    if not positive:
        raise ExecutorContractError("all effective destroy operator weights are zero")
    total = sum(weight for _name, weight in positive)
    threshold = rng.random() * total
    cumulative = 0.0
    for name, weight in positive:
        cumulative += weight
        if cumulative >= threshold:
            return name
    return positive[-1][0]




def _dedupe_ints(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            continue
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _search_focus_destroy_metadata(
    move: Any,
    compiled_search_focus: Dict[str, Any],
    candidate_count_before: int,
    candidate_count_matched: int,
) -> Dict[str, Any]:
    metadata = dict(getattr(move, "metadata", {}) or {})
    return {
        "focus_targeted": bool(metadata.get("focus_targeted", False)),
        "direct_focus_task_hits": [
            int(tid) for tid in metadata.get("direct_focus_task_hits", []) or []
        ][:20],
        "blocking_task_hits": [
            int(tid) for tid in metadata.get("blocking_task_hits", []) or []
        ][:20],
        "blocking_neighborhood_hit_count": int(
            metadata.get("blocking_neighborhood_hit_count", 0) or 0
        ),
        "focus_target_task_ids": [
            int(tid) for tid in metadata.get("focus_target_task_ids", []) or []
        ][:20],
        "focus_annotation_applied": bool(
            metadata.get("focus_annotation_applied", False)
        ),
        "focus_candidate_group": str(
            metadata.get("focus_candidate_group", "other")
        ),
        "focus_evaluated_destroy_move_count": int(candidate_count_before),
        "focus_matched_destroy_move_count": int(candidate_count_matched),
    }


def _with_search_focus_metadata(move: DestroyMove, **metadata: Any) -> DestroyMove:
    return DestroyMove(
        operator_name=move.operator_name,
        shape=move.shape,
        task_ids=move.task_ids,
        affected_routes=move.affected_routes,
        features=move.features,
        score=move.score,
        metadata={**dict(move.metadata), **metadata},
    )


def _destroy_feature_usage(
    moves: Sequence[DestroyMove],
    policy: DestroyPolicy,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    group: Dict[str, Dict[str, Any]] = {}
    evaluated_move_count = len(moves)
    for name, score in policy.feature_weights.items():
        if float(score or 0.0) <= 0.0:
            continue
        nonzero = sum(
            1
            for move in moves
            if abs(float(getattr(move.features, str(name), 0.0))) > 1e-9
        )
        group[str(name)] = {
            "compiled_weight_0_to_10": float(score),
            "feature_evaluated_move_count": int(evaluated_move_count),
            "feature_nonzero_move_count": int(nonzero),
            "used_in_score": bool(evaluated_move_count > 0),
        }
    return {"destroy_features": group}


def _init_destroy_operator_stats(
    operator_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    return {
        str(name): {
            "used": 0.0,
            "accepted": 0.0,
            "global_best_improved": 0.0,
            "total_score": 0.0,
            "removed_count_sum": 0.0,
            **{f"{feature}_sum": 0.0 for feature in DESTROY_FEATURE_NAMES},
        }
        for name in operator_names
    }


def _accumulate_destroy_operator_stats(
    summary: Dict[str, Dict[str, float]],
    *,
    move: DestroyMove,
    accepted: bool,
    best_improved: bool,
    reward: float,
) -> None:
    name = str(move.operator_name)
    if name not in summary:
        summary[name] = _init_destroy_operator_stats([name])[name]
    bucket = summary[name]
    bucket["used"] += 1.0
    bucket["accepted"] += 1.0 if accepted else 0.0
    bucket["global_best_improved"] += 1.0 if best_improved else 0.0
    bucket["total_score"] += float(reward)
    bucket["removed_count_sum"] += float(len(move.task_ids))
    for feature in DESTROY_FEATURE_NAMES:
        bucket[f"{feature}_sum"] += float(getattr(move.features, feature, 0.0))


def _finalize_destroy_operator_stats(
    summary: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name, bucket in summary.items():
        used = int(bucket.get("used", 0.0))
        denom = max(1.0, float(used))
        row: Dict[str, float] = {
            "used": used,
            "accepted": int(bucket.get("accepted", 0.0)),
            "global_best_improved": int(bucket.get("global_best_improved", 0.0)),
            "total_score": round(float(bucket.get("total_score", 0.0)), 6),
            "mean_removed_count": (
                round(float(bucket.get("removed_count_sum", 0.0)) / denom, 6)
                if used else 0.0
            ),
        }
        for feature in DESTROY_FEATURE_NAMES:
            row[f"mean_{feature}"] = (
                round(float(bucket.get(f"{feature}_sum", 0.0)) / denom, 6)
                if used else 0.0
            )
        out[str(name)] = row
    return out


def _init_insertion_operator_summary(
    operator_names: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    return {
        str(name): {
            "used": 0.0,
            "accepted": 0.0,
            "global_best_improved": 0.0,
            "reward_sum": 0.0,
            "inserted_sum": 0.0,
            "unassigned_before_sum": 0.0,
            "unassigned_after_sum": 0.0,
            "tasks_analyzed_sum": 0.0,
            "positions_checked_sum": 0.0,
            "time_ms_sum": 0.0,
        }
        for name in operator_names
    }


def _accumulate_insertion_operator_summary(
    summary: Dict[str, Dict[str, float]],
    *,
    operator_name: str,
    last_insertion: Optional[Dict[str, Any]],
    accepted: bool,
    best_improved: bool,
    reward: float,
) -> None:
    name = str(operator_name)
    if name not in summary:
        summary[name] = _init_insertion_operator_summary([name])[name]
    bucket = summary[name]
    insertion = dict(last_insertion or {})
    bucket["used"] += 1.0
    bucket["accepted"] += 1.0 if accepted else 0.0
    bucket["global_best_improved"] += 1.0 if best_improved else 0.0
    bucket["reward_sum"] += float(reward)
    bucket["inserted_sum"] += float(insertion.get("inserted_count", 0) or 0)
    bucket["unassigned_before_sum"] += float(insertion.get("unassigned_before", 0) or 0)
    bucket["unassigned_after_sum"] += float(insertion.get("unassigned_after", 0) or 0)
    bucket["tasks_analyzed_sum"] += float(insertion.get("tasks_analyzed", 0) or 0)
    bucket["positions_checked_sum"] += float(
        insertion.get("positions_strict_checked", 0) or 0
    )
    bucket["time_ms_sum"] += float(insertion.get("time_ms", 0.0) or 0.0)


def _finalize_insertion_operator_summary(
    summary: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, bucket in summary.items():
        out[str(name)] = {
            "used": int(bucket.get("used", 0.0)),
            "accepted": int(bucket.get("accepted", 0.0)),
            "global_best_improved": int(bucket.get("global_best_improved", 0.0)),
            "reward_sum": round(float(bucket.get("reward_sum", 0.0)), 6),
            "inserted_sum": int(bucket.get("inserted_sum", 0.0)),
            "unassigned_before_sum": int(bucket.get("unassigned_before_sum", 0.0)),
            "unassigned_after_sum": int(bucket.get("unassigned_after_sum", 0.0)),
            "tasks_analyzed_sum": int(bucket.get("tasks_analyzed_sum", 0.0)),
            "positions_checked_sum": int(bucket.get("positions_checked_sum", 0.0)),
            "time_ms_sum": round(float(bucket.get("time_ms_sum", 0.0)), 4),
        }
    return out


def _remove_tasks(sol: AssignmentSolution, tids: Sequence[int]) -> None:
    if not tids:
        return
    removed = set(int(tid) for tid in tids)
    for aid, route in list(sol.routes.items()):
        sol.routes[int(aid)] = [int(tid) for tid in route if int(tid) not in removed]
    for tid in removed:
        sol.unassigned.add(int(tid))
    sol.eval = None


def _build_solver_diagnostics(
    *,
    policy: CompiledALNSPolicy,
    total_iters: int,
    actual_time_used_sec: float,
    best_update_iters: List[int],
    best_update_objective_keys: List[Dict[str, Any]],
    state_update_selected_source: str,
    initial_solution_feasible: bool,
    returned_solution_feasible: bool,
    operator_weights: Dict[str, Any],
    last_acceptance_decision: Optional[Dict[str, Any]],
    last_destroy_move: Optional[Dict[str, Any]],
    destroy_operator_summary: Dict[str, Any],
    insertion_operator_summary: Dict[str, Any],
    iteration_trace: List[Dict[str, Any]],
    last_insertion: Optional[Dict[str, Any]],
    final_constraint_report: Any,
    feasibility_rejection_reasons: Dict[str, int],
    execution_trace: Dict[str, Any],
    stop_reason: str,
) -> Dict[str, Any]:
    last_best_iter = best_update_iters[-1] if best_update_iters else None
    diagnostics = {
        "algorithm": "weighted_alns",
        "policy": _compiled_policy_trace(policy),
        "total_iters": int(total_iters),
        "actual_iters_used": int(total_iters),
        "actual_time_used_sec": max(0.0, float(actual_time_used_sec)),
        "stop_reason": str(stop_reason),
        "best_update_count": len(best_update_iters),
        "best_update_iters": [int(x) for x in best_update_iters],
        "best_update_objective_keys": list(best_update_objective_keys),
        "first_best_iter": int(best_update_iters[0]) if best_update_iters else None,
        "last_best_iter": int(last_best_iter) if last_best_iter is not None else None,
        "plateau_iters_after_last_update": (
            int(total_iters - last_best_iter)
            if last_best_iter is not None
            else int(total_iters)
        ),
        "initial_solution_feasible": bool(initial_solution_feasible),
        "state_update_selected_source": state_update_selected_source,
        "returned_solution_feasible": bool(returned_solution_feasible),
        "last_acceptance_decision": dict(last_acceptance_decision or {}),
        "last_destroy_move": dict(last_destroy_move or {}),
        "last_insertion": _compact_insertion_scope(last_insertion),
        "iteration_trace": list(iteration_trace),
        "destroy_operator_summary": _numericize_weight_tree(destroy_operator_summary),
        "insertion_operator_summary": _numericize_weight_tree(
            insertion_operator_summary
        ),
        "operator_weights": _numericize_weight_tree(_public_weight_tree(operator_weights)),
        "violation_ratios": _violation_ratio_diagnostics(final_constraint_report),
        "feasibility_rejection_reasons": {
            str(reason): int(count)
            for reason, count in feasibility_rejection_reasons.items()
        },
        "execution_trace": dict(execution_trace),
    }
    return diagnostics


def _build_execution_trace(
    *,
    trace_id: str,
    total_iters: int,
    destroy_operator_summary: Dict[str, Any],
    insertion_operator_summary: Dict[str, Any],
    trial_flow: Dict[str, int],
    rejection_reasons: Dict[str, int],
    insertion_failure_reasons: Dict[str, int],
    removed_task_count_sum: float,
    operator_prior_trace: Dict[str, Any],
    compiled_search_focus: Dict[str, Any],
    feasibility_rejection_reasons: Dict[str, int],
    final_constraint_report: Any,
    actual_time_used_sec: float,
    state_update_selected_source: str,
    stop_reason: str,
    iteration_trace: List[Dict[str, Any]],
    last_insertion: Optional[Dict[str, Any]],
    before_solution: AssignmentSolution,
    after_solution: AssignmentSolution,
) -> Dict[str, Any]:
    selected_counts = {
        str(name): int(values.get("used", 0) if isinstance(values, dict) else values)
        for name, values in destroy_operator_summary.items()
    }
    candidate_trials = max(1, int(trial_flow.get("candidate_trials", 0) or 0))
    tasks_reinserted = sum(
        int(values.get("inserted_sum", 0) or 0)
        for values in insertion_operator_summary.values()
        if isinstance(values, dict)
    )
    tasks_left = sum(
        int(values.get("unassigned_after_sum", 0) or 0)
        for values in insertion_operator_summary.values()
        if isinstance(values, dict)
    )
    dominant_insertion = max(
        insertion_failure_reasons.items(),
        key=lambda item: int(item[1]),
        default=("none", 0),
    )[0]
    search_focus_engagement = _build_executor_search_focus_engagement(
        compiled_search_focus=compiled_search_focus,
        iteration_trace=iteration_trace,
        last_insertion=last_insertion,
    )
    search_focus_progress = _search_focus_progress(
        before_solution=before_solution,
        after_solution=after_solution,
        compiled_search_focus=compiled_search_focus,
    )
    feature_usage = _merge_feature_usage(iteration_trace, last_insertion)
    return {
        "trace_id": trace_id,
        "kind": "alns",
        "iters": int(total_iters),
        "actual_time_used_sec": float(actual_time_used_sec),
        "stop_reason": str(stop_reason),
        "state_update_selected_source": str(state_update_selected_source),
        "compiled_search_focus": dict(compiled_search_focus),
        "executor_search_focus_engagement": search_focus_engagement,
        "executor_search_focus_progress": search_focus_progress,
        "executor_feature_usage": feature_usage,
        "operator_prior_trace": {
            **dict(operator_prior_trace),
            "actual_usage": selected_counts,
            "accepted_usage": {
                str(name): int(values.get("accepted", 0) or 0)
                for name, values in destroy_operator_summary.items()
                if isinstance(values, dict)
            },
            "reward": {
                str(name): float(values.get("total_score", 0.0) or 0.0)
                for name, values in destroy_operator_summary.items()
                if isinstance(values, dict)
            },
        },
        "operator_usage_summary": {
            "destroy": _operator_usage_destroy(destroy_operator_summary),
            "insertion": _operator_usage_insertion(insertion_operator_summary),
        },
        "operator_pair_summary": _operator_pair_summary(iteration_trace),
        "destroy": {
            "selected_operator_counts": selected_counts,
            "removed_task_count_avg": round(
                float(removed_task_count_sum) / candidate_trials, 6
            ),
        },
        "insertion": {
            "candidate_tasks_total": int(tasks_reinserted + tasks_left),
            "tasks_reinserted": int(tasks_reinserted),
            "tasks_left_unassigned": int(tasks_left),
            "dominant_insertion_failure": str(dominant_insertion),
            "insertion_failure_reasons": {
                str(k): int(v) for k, v in insertion_failure_reasons.items()
            },
            "recent_failed_insertion_task_ids": _recent_failed_insertion_task_ids(
                iteration_trace, last_insertion, limit=20
            ),
            "top_failed_insertion_tasks": _top_failed_insertion_task_rows(
                iteration_trace, last_insertion, limit=5
            ),
        },
        "trial_flow": _public_trial_flow(trial_flow),
        "acceptance_summary": _acceptance_summary(iteration_trace),
        "execution_progress_summary": _execution_progress_summary(iteration_trace),
        "best_progress": [
            {
                "trial": int(row.get("iteration", index) or index),
                "objective_terms": list(
                    dict(row.get("global_best_objective_keys", {}) or {}).get(
                        "terms", []
                    )
                    or []
                ),
                "objective_key": [
                    float(value)
                    for value in dict(
                        row.get("global_best_objective_keys", {}) or {}
                    ).get("key", [])
                    or []
                ],
            }
            for index, row in enumerate(iteration_trace, start=1)
            if bool(row.get("global_best_improved", False))
        ],
        "rejection_reasons": {str(k): int(v) for k, v in rejection_reasons.items()},
        "strict_feasibility_diagnostics": {
            "rejection_reasons": {
                str(k): int(v) for k, v in feasibility_rejection_reasons.items()
            },
            "returned_violation_ratios": _violation_ratio_diagnostics(final_constraint_report),
            "returned_check": _returned_feasibility_check(
                final_constraint_report,
            ),
        },
    }


def _acceptance_summary(iteration_trace: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return factual acceptance counts for the complete trial sequence."""
    mode_counts: Dict[str, int] = {}
    by_metric: Dict[str, Dict[str, float]] = {}
    accepted_improving = 0
    accepted_equal = 0
    accepted_worsening = 0
    rejected = 0
    accepted_worse_indices: list[int] = []
    later_improvement_indices: list[int] = []
    rejection_reason_counts: Dict[str, int] = {}

    for index, row in enumerate(iteration_trace, start=1):
        decision = dict(row.get("acceptance_decision", {}) or {})
        mode = str(decision.get("accept_mode", "unknown"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        accepted = bool(decision.get("accepted", row.get("accepted", False)))
        compare_result = int(decision.get("compare_result", 0) or 0)
        if accepted:
            if compare_result < 0:
                accepted_improving += 1
                later_improvement_indices.append(index)
            elif compare_result == 0:
                accepted_equal += 1
            else:
                accepted_worsening += 1
                accepted_worse_indices.append(index)
        else:
            rejected += 1
            reason = str(
                decision.get("rejection_reason")
                or row.get("rejection_reason")
                or "acceptance_rejected"
            )
            rejection_reason_counts[reason] = rejection_reason_counts.get(reason, 0) + 1
        if bool(row.get("global_best_improved", False)) and index not in later_improvement_indices:
            later_improvement_indices.append(index)
        if compare_result <= 0:
            continue
        metric = str(decision.get("first_worsened_metric", "unknown"))
        normalized = float(decision.get("normalized_worsening", 0.0) or 0.0)
        bucket = by_metric.setdefault(
            metric,
            {
                "trial_count": 0.0,
                "accepted_count": 0.0,
                "normalized_worsening_sum": 0.0,
                "normalized_worsening_max": 0.0,
            },
        )
        bucket["trial_count"] += 1.0
        bucket["accepted_count"] += 1.0 if accepted else 0.0
        bucket["normalized_worsening_sum"] += normalized
        bucket["normalized_worsening_max"] = max(
            float(bucket["normalized_worsening_max"]), normalized
        )

    accepted_worse_followed = sum(
        1
        for index in accepted_worse_indices
        if any(later > index for later in later_improvement_indices)
    )
    public_by_metric: Dict[str, Dict[str, Any]] = {}
    for metric, bucket in by_metric.items():
        count = int(bucket["trial_count"])
        public_by_metric[metric] = {
            "trial_count": count,
            "accepted_count": int(bucket["accepted_count"]),
            "normalized_worsening_sum": round(
                float(bucket["normalized_worsening_sum"]), 9
            ),
            "normalized_worsening_avg": round(
                float(bucket["normalized_worsening_sum"]) / max(1, count), 6
            ),
            "normalized_worsening_max": round(
                float(bucket["normalized_worsening_max"]), 6
            ),
        }
    return {
        "candidate_trials": len(iteration_trace),
        "mode_counts": dict(sorted(mode_counts.items())),
        "accepted_improving": int(accepted_improving),
        "accepted_equal": int(accepted_equal),
        "accepted_worsening": int(accepted_worsening),
        "rejected": int(rejected),
        "accepted_worsening_followed_by_later_improvement": int(accepted_worse_followed),
        "rejection_reasons": dict(sorted(rejection_reason_counts.items())),
        "worsening_by_metric": public_by_metric,
    }


def _execution_progress_summary(
    iteration_trace: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    improving: list[int] = []
    best: list[int] = []
    for index, row in enumerate(iteration_trace, start=1):
        decision = dict(row.get("acceptance_decision", {}) or {})
        if bool(decision.get("accepted", row.get("accepted", False))) and int(
            decision.get("compare_result", 0) or 0
        ) < 0:
            improving.append(index)
        if bool(row.get("global_best_improved", False)):
            best.append(index)
    return {
        "used_trials": len(iteration_trace),
        "accepted_improving_count": len(improving),
        "global_best_update_count": len(best),
        "first_improvement_trial": improving[0] if improving else None,
        "last_improvement_trial": improving[-1] if improving else None,
        "trailing_non_improving_trials": (
            len(iteration_trace) - improving[-1] if improving else len(iteration_trace)
        ),
        "first_best_update_trial": best[0] if best else None,
        "last_best_update_trial": best[-1] if best else None,
        "best_update_trials": [int(v) for v in best],
    }

def _build_executor_search_focus_engagement(
    *,
    compiled_search_focus: Dict[str, Any],
    iteration_trace: List[Dict[str, Any]],
    last_insertion: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    destroy_matched_move_count = 0
    destroy_selected_count = 0
    destroy_annotation_application_count = 0
    focus_matched_task_selection_count = 0
    focus_changed_selection_count = 0
    focus_feasible_task_count = 0
    attempted: List[int] = []
    inserted: List[int] = []
    selected_target_hit_counts: List[int] = []
    for item in iteration_trace:
        metadata = dict(item.get("destroy_search_focus_metadata", {}) or {})
        destroy_matched_move_count += int(
            metadata.get("focus_matched_destroy_move_count", 0) or 0
        )
        if metadata.get("focus_targeted"):
            destroy_selected_count += 1
            selected_target_hit_counts.append(
                len(metadata.get("direct_focus_task_hits", []) or [])
            )
        if metadata.get("focus_annotation_applied"):
            destroy_annotation_application_count += 1
        insertion = dict(item.get("search_focus_insertion", {}) or {})
        attempted.extend(
            int(tid) for tid in insertion.get("focus_tasks_attempted", []) or []
        )
        inserted.extend(
            int(tid) for tid in insertion.get("focus_tasks_inserted", []) or []
        )
        focus_feasible_task_count += int(
            insertion.get("focus_feasible_task_count", 0) or 0
        )
        focus_matched_task_selection_count += int(
            insertion.get("focus_matched_task_selection_count", 0) or 0
        )
        focus_changed_selection_count += int(
            insertion.get("focus_changed_selection_count", 0) or 0
        )
        focus_changed_selection_count += int(bool(metadata.get("focus_changed_selection", False)))
    if last_insertion:
        attempted.extend(
            int(tid)
            for tid in last_insertion.get(
                "focus_tasks_attempted",
                last_insertion.get("search_focus_tasks_attempted", []),
            )
            or []
        )
        inserted.extend(
            int(tid)
            for tid in last_insertion.get(
                "focus_tasks_inserted",
                last_insertion.get("search_focus_tasks_inserted", []),
            )
            or []
        )
        focus_feasible_task_count += int(
            last_insertion.get("focus_feasible_task_count", 0) or 0
        )
        focus_matched_task_selection_count += int(
            last_insertion.get("focus_matched_task_selection_count", 0) or 0
        )
        focus_changed_selection_count += int(
            last_insertion.get("focus_changed_selection_count", 0) or 0
        )
    requested_tasks = _unique_ints(
        compiled_search_focus.get("focus_task_ids", []) or []
    )
    requested_task_set = set(requested_tasks)
    engaged_tasks = inserted or attempted
    return {
        "requested_task_ids": requested_tasks,
        "engaged_task_ids": [
            int(tid)
            for tid in _unique_ints(engaged_tasks)
            if not requested_task_set or int(tid) in requested_task_set
        ],
        "focus_feasible_task_count": int(focus_feasible_task_count),
        "focus_matched_task_selection_count": int(focus_matched_task_selection_count),
        "focus_changed_selection_count": int(focus_changed_selection_count),
        "focus_selected_task_count": len(_unique_ints(inserted)),
        "focus_targeted_destroy_move_count": int(destroy_matched_move_count),
        "selected_focus_targeted_destroy_move_count": int(destroy_selected_count),
        "destroy_focus_annotation_application_count": int(
            destroy_annotation_application_count
        ),
        "selected_focus_target_hit_count_min": (
            min(selected_target_hit_counts)
            if selected_target_hit_counts
            else 0
        ),
        "selected_focus_target_hit_count_max": (
            max(selected_target_hit_counts)
            if selected_target_hit_counts
            else 0
        ),
        "selected_focus_target_hit_count_mean": round(
            sum(selected_target_hit_counts)
            / max(1, len(selected_target_hit_counts)),
            6,
        ),
    }


def _recent_failed_insertion_task_ids(
    iteration_trace: List[Dict[str, Any]],
    last_insertion: Optional[Dict[str, Any]],
    *,
    limit: int,
) -> List[int]:
    out: List[int] = []
    seen = set()

    def add(values: Any) -> None:
        for value in values or []:
            if isinstance(value, bool):
                continue
            tid = int(value.get("task_id") if isinstance(value, Mapping) else value)
            if tid in seen:
                continue
            seen.add(tid)
            out.append(tid)
            if len(out) >= int(limit):
                return

    if last_insertion:
        add(last_insertion.get("focus_tasks_failed", last_insertion.get("search_focus_tasks_failed", [])) or [])
        add(last_insertion.get("top_failed_tasks", []) or [])
    for item in reversed(iteration_trace):
        search_focus_insertion = dict(item.get("search_focus_insertion", {}) or {})
        add(search_focus_insertion.get("focus_tasks_failed", []) or [])
        add(search_focus_insertion.get("top_failed_tasks", []) or [])
        if len(out) >= int(limit):
            break
    return out[: int(limit)]


def _top_failed_insertion_task_rows(
    iteration_trace: List[Dict[str, Any]],
    last_insertion: Optional[Dict[str, Any]],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()

    def add(raw_rows: Any) -> None:
        for raw in raw_rows or []:
            if not isinstance(raw, Mapping):
                continue
            tid = raw.get("task_id")
            if tid is None or isinstance(tid, bool):
                continue
            task_id = int(tid)
            if task_id in seen:
                continue
            seen.add(task_id)
            rows.append(dict(raw))
            if len(rows) >= int(limit):
                return

    if last_insertion:
        add(last_insertion.get("top_failed_tasks", []) or [])
    for item in reversed(iteration_trace):
        add((dict(item.get("search_focus_insertion", {}) or {})).get("top_failed_tasks", []) or [])
        if len(rows) >= int(limit):
            break
    return rows[: int(limit)]



def _merge_feature_usage(
    iteration_trace: List[Dict[str, Any]],
    last_insertion: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    merged: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for item in iteration_trace:
        for usage in (item.get("destroy_feature_usage", {}), item.get("insertion_feature_usage", {})):
            _merge_feature_usage_tree(merged, usage)
    if last_insertion:
        _merge_feature_usage_tree(merged, dict(last_insertion.get("feature_usage", {}) or {}))
    return merged


def _merge_feature_usage_tree(
    merged: Dict[str, Dict[str, Dict[str, Any]]],
    usage: Any,
) -> None:
    if not isinstance(usage, Mapping):
        return
    for group, rows in usage.items():
        if not isinstance(rows, Mapping):
            continue
        group_out = merged.setdefault(str(group), {})
        for name, raw in rows.items():
            if not isinstance(raw, Mapping):
                continue
            row = group_out.setdefault(
                str(name),
                {
                    "compiled_weight_0_to_10": float(raw.get("compiled_weight_0_to_10", 0.0) or 0.0),
                    "feature_evaluated_move_count": 0,
                    "feature_nonzero_move_count": 0,
                    "used_in_score": False,
                },
            )
            row["compiled_weight_0_to_10"] = max(float(row.get("compiled_weight_0_to_10", 0.0) or 0.0), float(raw.get("compiled_weight_0_to_10", 0.0) or 0.0))
            row["feature_evaluated_move_count"] = int(row.get("feature_evaluated_move_count", 0) or 0) + int(raw.get("feature_evaluated_move_count", raw.get("candidate_opportunity_count", 0)) or 0)
            row["feature_nonzero_move_count"] = int(row.get("feature_nonzero_move_count", 0) or 0) + int(raw.get("feature_nonzero_move_count", raw.get("nonzero_feature_count", 0)) or 0)
            row["used_in_score"] = bool(row.get("used_in_score", False) or raw.get("used_in_score", False))


def _operator_usage_destroy(
    summary: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(name): {
            "used": int(values.get("used", 0) or 0),
            "accepted": int(values.get("accepted", 0) or 0),
            "global_best_improved": int(values.get("global_best_improved", 0) or 0),
            "mean_removed_count": float(values.get("mean_removed_count", 0.0) or 0.0),
            "total_reward": float(values.get("total_score", 0.0) or 0.0),
        }
        for name, values in summary.items()
        if isinstance(values, dict)
    }


def _operator_usage_insertion(
    summary: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    return {
        str(name): {
            "used": int(values.get("used", 0) or 0),
            "accepted": int(values.get("accepted", 0) or 0),
            "global_best_improved": int(
                values.get("global_best_improved", 0) or 0
            ),
            "total_reward": float(values.get("reward_sum", 0.0) or 0.0),
            "inserted_sum": int(values.get("inserted_sum", 0) or 0),
            "positions_checked_sum": int(
                values.get("positions_checked_sum", 0) or 0
            ),
            "failed_insertions": int(values.get("unassigned_after_sum", 0) or 0),
        }
        for name, values in summary.items()
        if isinstance(values, dict)
    }


def _compact_insertion_scope(last_insertion: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    insertion = dict(last_insertion or {})
    return {
        "selected_operator": insertion.get("selected_operator"),
        "operator_selection_scope": insertion.get("operator_selection_scope"),
        "task_selector_mode": insertion.get("task_selector_mode"),
        "task_selection_order": _unique_ints(insertion.get("task_selection_order", []) or []),
        "ejection_count": int(insertion.get("ejection_count", 0) or 0),
        "ejected_task_ids": _unique_ints(insertion.get("ejected_task_ids", []) or []),
        "operator_trial_use": dict(insertion.get("operator_trial_use", {}) or {}),
        "operator_task_selection_count": dict(
            insertion.get("operator_task_selection_count", {}) or {}
        ),
        "inserted_count": int(insertion.get("inserted_count", 0) or 0),
        "unassigned_before": int(insertion.get("unassigned_before", 0) or 0),
        "unassigned_after": int(insertion.get("unassigned_after", 0) or 0),
        "focus_task_ids": _unique_ints(insertion.get("focus_task_ids", []) or []),
        "task_selection_scope": str(insertion.get("task_selection_scope", "common_pending_pool") or "common_pending_pool"),
        "focus_changed_selection_count": int(insertion.get("focus_changed_selection_count", 0) or 0),
        "focus_matched_task_selection_count": int(insertion.get("focus_matched_task_selection_count", 0) or 0),
        "focus_tasks_attempted": _unique_ints(
            insertion.get("search_focus_tasks_attempted", []) or []
        ),
        "focus_tasks_inserted": _unique_ints(
            insertion.get("search_focus_tasks_inserted", []) or []
        ),
        "focus_tasks_failed": _unique_ints(
            insertion.get("search_focus_tasks_failed", []) or []
        ),
        "focus_feasible_task_count": int(
            insertion.get("focus_feasible_task_count", 0) or 0
        ),
    }


def _operator_pair_summary(
    iteration_trace: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate outcomes by the complete destroy-repair pair."""
    summary: Dict[str, Dict[str, Any]] = {}
    for row in iteration_trace:
        destroy_name = str(row.get("destroy_operator", "") or "")
        repair_name = str(row.get("insertion_operator", "") or "")
        if not destroy_name or not repair_name:
            continue
        key = f"{destroy_name}::{repair_name}"
        bucket = summary.setdefault(
            key,
            {
                "destroy_operator": destroy_name,
                "repair_operator": repair_name,
                "used": 0,
                "accepted": 0,
                "improved": 0,
                "best_updates": 0,
                "structurally_changed": 0,
                "accepted_structurally_changed": 0,
                "destroy_repair_noop": 0,
                "total_reward": 0.0,
            },
        )
        decision = dict(row.get("acceptance_decision", {}) or {})
        accepted = bool(decision.get("accepted", row.get("accepted", False)))
        compare_result = int(decision.get("compare_result", 0) or 0)
        bucket["used"] += 1
        bucket["accepted"] += 1 if accepted else 0
        bucket["improved"] += 1 if accepted and compare_result < 0 else 0
        bucket["best_updates"] += 1 if bool(row.get("global_best_improved", False)) else 0
        changed = bool(row.get("trial_structure_changed", False))
        bucket["structurally_changed"] += 1 if changed else 0
        bucket["accepted_structurally_changed"] += 1 if accepted and changed else 0
        bucket["destroy_repair_noop"] += 0 if changed else 1
        bucket["total_reward"] += float(row.get("reward", 0.0) or 0.0)
    for bucket in summary.values():
        used = max(1, int(bucket.get("used", 0) or 0))
        changed = int(bucket.get("structurally_changed", 0) or 0)
        bucket["structure_change_rate"] = round(changed / used, 6)
        bucket["noop_rate"] = round(int(bucket.get("destroy_repair_noop", 0) or 0) / used, 6)
        bucket["structural_passage_rate"] = round(
            int(bucket.get("accepted_structurally_changed", 0) or 0) / max(1, changed), 6
        )
        bucket["total_reward"] = round(float(bucket["total_reward"]), 6)
    return summary

def _search_focus_progress(
    *,
    before_solution: AssignmentSolution,
    after_solution: AssignmentSolution,
    compiled_search_focus: Dict[str, Any],
) -> Dict[str, Any]:
    return _search_focus_progress_from_snapshot(
        before_unassigned={int(tid) for tid in before_solution.unassigned},
        after_unassigned={int(tid) for tid in after_solution.unassigned},
        compiled_search_focus=compiled_search_focus,
    )


def _search_focus_progress_from_snapshot(
    *,
    before_unassigned: set[int],
    after_unassigned: set[int],
    compiled_search_focus: Dict[str, Any],
) -> Dict[str, Any]:
    focus_task_ids = _unique_ints(compiled_search_focus.get("focus_task_ids", []) or [])
    inserted = [
        tid
        for tid in focus_task_ids
        if int(tid) in before_unassigned and int(tid) not in after_unassigned
    ]
    return {
        "focus_task_count": len(focus_task_ids),
        "focus_tasks_inserted": inserted[:20],
        "focus_tasks_still_unassigned": [
            tid for tid in focus_task_ids if int(tid) in after_unassigned
        ][:20],
    }

def _unique_ints(values: Any, limit: int = 20) -> List[int]:
    out: List[int] = []
    seen = set()
    for value in values or []:
        if isinstance(value, bool):
            continue
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _public_trial_flow(trial_flow: Mapping[str, Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key, value in trial_flow.items():
        out[str(key)] = int(value)
    return out


def _violation_ratio_diagnostics(report: Any) -> Dict[str, Any]:
    ratios = dict(getattr(report, "violation_ratio_by_type", {}) or {})
    details = dict(getattr(report, "violation_details_by_type", {}) or {})
    names = set(ratios) | set(details)
    return {
        name: {
            "total_ratio": float(ratios.get(name, 0.0)),
            "max_individual_ratio": max(
                (float(item.get("ratio", 0.0)) for item in details.get(name, [])),
                default=0.0,
            ),
        }
        for name in sorted(names)
    }


def _returned_feasibility_check(final_report: Any) -> Dict[str, Any]:
    passed = bool(getattr(final_report, "is_feasible", False))
    reason = "feasible" if passed else "returned_working_is_infeasible"
    return {"passed": passed, "reason": reason}


@dataclass(frozen=True, slots=True)
class _AcceptanceBarrier:
    metric: str
    raw_worsening: float
    metric_scale: float
    normalized_worsening: float


@dataclass(slots=True)
class _AcceptanceDecision:
    compare_result: int
    accepted: bool
    accept_mode: str
    feasibility_admissible: bool = True
    accept_scope: str = "working_and_best_candidate"
    feasibility_reason: str = ""
    first_worsened_metric: Optional[str] = None
    raw_worsening: Optional[float] = None
    metric_scale: Optional[float] = None
    normalized_worsening: Optional[float] = None
    temperature: Optional[float] = None
    threshold: Optional[float] = None
    acceptance_probability: Optional[float] = None
    random_draw: Optional[float] = None
    rejection_reason: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "compare_result": int(self.compare_result),
            "accepted": bool(self.accepted),
            "accept_mode": str(self.accept_mode),
            "feasibility_admissible": bool(self.feasibility_admissible),
            "accept_scope": str(self.accept_scope),
            "feasibility_reason": str(self.feasibility_reason),
        }
        if self.first_worsened_metric is not None:
            data["first_worsened_metric"] = str(self.first_worsened_metric)
        if self.raw_worsening is not None:
            data["raw_worsening"] = float(self.raw_worsening)
        if self.metric_scale is not None:
            data["metric_scale"] = float(self.metric_scale)
        if self.normalized_worsening is not None:
            data["normalized_worsening"] = float(self.normalized_worsening)
        if self.temperature is not None:
            data["temperature"] = float(self.temperature)
        if self.threshold is not None:
            data["threshold"] = float(self.threshold)
        if self.acceptance_probability is not None:
            data["acceptance_probability"] = float(self.acceptance_probability)
        if self.random_draw is not None:
            data["random_draw"] = float(self.random_draw)
        if self.rejection_reason is not None:
            data["rejection_reason"] = str(self.rejection_reason)
        return data


def _alns_accept(
    cur_ev: EvalResult,
    trial_ev: EvalResult,
    objective_terms: List[Dict[str, Any]],
    mode: str,
    rng: random.Random,
    temperature: float,
    worsening_tolerance: float,
    feasibility_admissible: bool,
    accept_scope: str,
    feasibility_reason: str,
    acceptance_scales: Optional[Mapping[str, float]] = None,
) -> _AcceptanceDecision:
    """Apply ALNS acceptance without weakening the global lexicographic order.

    Strict comparison decides whether a trial is better/equal/worse.  For a worse
    trial, only the first worsened objective metric is priced; improvements in
    lower-priority metrics can never compensate for it.  The price uses a scale
    fixed for the whole ALNS action, so identical absolute changes have identical
    meaning regardless of the current incumbent value.
    """
    cmp = compare_quality(trial_ev, cur_ev, objective_terms)
    if cmp <= 0:
        return _AcceptanceDecision(
            compare_result=cmp,
            accepted=True,
            accept_mode=mode,
            feasibility_admissible=feasibility_admissible,
            accept_scope=accept_scope,
            feasibility_reason=feasibility_reason,
        )

    barrier = _acceptance_barrier(
        cur_ev,
        trial_ev,
        objective_terms,
        acceptance_scales=acceptance_scales,
    )
    if mode == "greedy":
        return _AcceptanceDecision(
            compare_result=cmp,
            accepted=False,
            accept_mode=mode,
            feasibility_admissible=feasibility_admissible,
            accept_scope=accept_scope,
            feasibility_reason=feasibility_reason,
            first_worsened_metric=barrier.metric,
            raw_worsening=barrier.raw_worsening,
            metric_scale=barrier.metric_scale,
            normalized_worsening=barrier.normalized_worsening,
            rejection_reason="greedy_worse_candidate",
        )

    common = {
        "compare_result": cmp,
        "accept_mode": mode,
        "feasibility_admissible": feasibility_admissible,
        "accept_scope": accept_scope,
        "feasibility_reason": feasibility_reason,
        "first_worsened_metric": barrier.metric,
        "raw_worsening": barrier.raw_worsening,
        "metric_scale": barrier.metric_scale,
        "normalized_worsening": barrier.normalized_worsening,
    }

    # Exploration may cross preference-cost barriers, but it must never undo
    # service recovery. This makes bounded acceptance safe even when unresolved
    # tasks remain in the incumbent.
    if barrier.metric in set(SERVICE_METRICS):
        return _AcceptanceDecision(
            accepted=False,
            rejection_reason="service_metric_worsening_protected",
            **common,
        )

    if mode == "threshold":
        threshold = max(0.0, float(worsening_tolerance))
        return _AcceptanceDecision(
            accepted=barrier.normalized_worsening <= threshold + 1e-12,
            threshold=threshold,
            **common,
        )

    if mode == "sa":
        effective_temperature = max(0.0, float(temperature))
        if effective_temperature <= 0.0:
            probability = 0.0
            draw = None
            accepted = False
        else:
            probability = math.exp(
                -barrier.normalized_worsening / max(1e-12, effective_temperature)
            )
            draw = rng.random()
            accepted = draw < probability
        return _AcceptanceDecision(
            accepted=accepted,
            temperature=effective_temperature,
            acceptance_probability=probability,
            random_draw=draw,
            **common,
        )

    raise ExecutorContractError(f"unknown acceptance mode: {mode}")


def _acceptance_barrier(
    cur_ev: EvalResult,
    trial_ev: EvalResult,
    objective_terms: Sequence[Mapping[str, Any]],
    *,
    acceptance_scales: Optional[Mapping[str, float]],
) -> _AcceptanceBarrier:
    eps = 1e-9
    scales = {str(k): float(v) for k, v in dict(acceptance_scales or {}).items()}
    for layer in objective_terms:
        metric = str(layer.get("metric", ""))
        direction = str(layer.get("direction", "min"))
        cur_value = _oriented_metric_value(cur_ev, metric, direction)
        trial_value = _oriented_metric_value(trial_ev, metric, direction)
        raw_delta = trial_value - cur_value
        if raw_delta == 0.0:
            continue
        if raw_delta < 0.0:
            raise ExecutorContractError(
                "acceptance barrier disagrees with lexicographic comparison"
            )
        scale = max(eps, float(scales.get(metric, 1.0)))
        return _AcceptanceBarrier(
            metric=metric,
            raw_worsening=float(raw_delta),
            metric_scale=float(scale),
            normalized_worsening=float(raw_delta / scale),
        )
    raise ExecutorContractError("worse trial has no first worsened objective metric")


def _build_acceptance_scales(
    instance: Instance,
    action_start_eval: EvalResult,
    objective_terms: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    return build_acceptance_scales(instance, action_start_eval, objective_terms)



def _sa_temperature_schedule(
    *,
    worsening_tolerance: float,
    max_iters: int,
) -> tuple[float, float]:
    """Return a budget-normalized SA schedule.

    Zero tolerance is exactly greedy.  For non-zero tolerance, the initial
    temperature equals the same normalized worsening unit used by threshold
    acceptance, and cooling reaches a fixed fraction at the action boundary.
    """
    initial = max(0.0, float(worsening_tolerance))
    if initial <= 0.0:
        return 0.0, 1.0
    steps = max(1, int(max_iters))
    cooling = math.exp(math.log(SA_FINAL_TEMPERATURE_RATIO) / float(steps))
    return initial, float(cooling)


def _oriented_metric_value(ev: EvalResult, metric: str, direction: str) -> float:
    value = float(ev.get_quality_metric(metric))
    return -value if str(direction).lower() == "max" else value


def _numericize_weight_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {str(key): _numericize_weight_tree(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_numericize_weight_tree(value) for value in node]
    if isinstance(node, (int, float)):
        return float(node)
    return node


def _public_weight_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {str(key): _public_weight_tree(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_public_weight_tree(value) for value in node]
    return node


def _compiled_policy_trace(policy: CompiledALNSPolicy) -> Dict[str, Any]:
    raw = policy.as_dict()
    return {
        "destroy_policy": _public_weight_tree(raw.get("destroy_policy", {})),
        "insertion_policy": _public_weight_tree(raw.get("insertion_policy", {})),
        "acceptance_policy": _public_weight_tree(raw.get("acceptance_policy", {})),
        "reaction_factor": raw.get("reaction_factor"),
    }



def _dedupe_events(events: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for event in events:
        value = str(event)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out



def _adaptive_operator_reward(
    before: EvalResult,
    after: EvalResult,
    *,
    accepted: bool,
    global_best_improved: bool,
    structure_changed: bool,
    objective_terms: Sequence[Mapping[str, Any]],
) -> float:
    if not structure_changed:
        return 0.0
    objective_improved = compare_quality(after, before, objective_terms) < 0
    if global_best_improved:
        return 8.0
    if objective_improved:
        return 4.0
    if accepted:
        return 0.5
    return 0.0


def _normalize_adaptive_weights(weights: Dict[str, float]) -> None:
    if not weights:
        return
    mean = sum(float(value) for value in weights.values()) / float(len(weights))
    if mean <= 1e-12:
        mean = 1.0
    for name in list(weights):
        weights[name] = max(0.10, min(10.0, float(weights[name]) / mean))

def _update_weights(
    weights: Dict[str, float],
    scores: Dict[str, float],
    used: Dict[str, int],
    reaction: float,
) -> None:
    for name in list(weights.keys()):
        if used.get(name, 0) <= 0:
            continue
        avg = float(scores.get(name, 0.0)) / max(1, int(used.get(name, 0)))
        weights[name] = (1.0 - float(reaction)) * float(weights[name]) + float(
            reaction
        ) * max(0.0, avg)
        weights[name] = max(0.10, min(10.0, float(weights[name])))
    _normalize_adaptive_weights(weights)


def _reset_segment_scores(scores: Dict[str, float], used: Dict[str, int]) -> None:
    for name in list(scores.keys()):
        scores[name] = 0.0
        used[name] = 0


def _clamp_int(value: int, lo: int, hi: int) -> int:
    return max(int(lo), min(int(hi), int(value)))
