"""Non-LLM external comparison baselines on the common ALNS core."""
from __future__ import annotations
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from experiments._objective import SEARCH_OBJECTIVE, lex_compare, objective_dict
from experiments._shared import quality_dict
from sar_alloc.config import Budget, Config
from sar_alloc.evaluator import evaluate
from sar_alloc.operators import (
    ALNS_REPAIR_OPERATOR_NAMES, DESTROY_OPERATOR_NAMES,
    AcceptancePolicy, CompiledALNSPolicy, DestroyPolicy, InsertionPolicy,
)
from sar_alloc.solution import AssignmentSolution
from sar_alloc.tools.assign_solvers import solve_assignment

BLOCK_TRIALS = 100
BASIC_REMOVE_RATIO = 0.30


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Frozen low-level/controller parameters selected before formal testing."""

    remove_ratio: float = BASIC_REMOVE_RATIO
    reaction_factor: float = 0.20
    block_trials: int = BLOCK_TRIALS
    remove_ratio_by_n_tasks: tuple[tuple[int, float], ...] = ()

    def __post_init__(self) -> None:
        if not (0.0 < float(self.remove_ratio) < 1.0):
            raise ValueError("remove_ratio must be in (0,1)")
        if not (0.0 <= float(self.reaction_factor) <= 1.0):
            raise ValueError("reaction_factor must be in [0,1]")
        if int(self.block_trials) <= 0:
            raise ValueError("block_trials must be positive")
        for n_tasks, ratio in self.remove_ratio_by_n_tasks:
            if int(n_tasks) <= 0 or not (0.0 < float(ratio) < 1.0):
                raise ValueError("scale-specific remove ratios must map positive sizes to (0,1)")
        sizes = [int(size) for size, _ in self.remove_ratio_by_n_tasks]
        if len(sizes) != len(set(sizes)):
            raise ValueError("scale-specific remove-ratio sizes must be unique")

    def effective_remove_ratio(self, n_tasks: int) -> float:
        mapping = {int(size): float(ratio) for size, ratio in self.remove_ratio_by_n_tasks}
        return float(mapping.get(int(n_tasks), self.remove_ratio))

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "remove_ratio": float(self.remove_ratio),
            "reaction_factor": float(self.reaction_factor),
            "acceptance": "greedy",
            "block_trials": int(self.block_trials),
            "remove_ratio_by_n_tasks": {
                str(int(size)): float(ratio)
                for size, ratio in sorted(self.remove_ratio_by_n_tasks)
            },
        }


def _plain_policy(
    *, destroy: str | None = None, repair: str | None = None,
    config: BaselineConfig | None = None,
) -> CompiledALNSPolicy:
    selected = config or BaselineConfig()
    remove_ratio = float(selected.remove_ratio)
    d_weights = {name: (10 if destroy is None or name == destroy else 0) for name in DESTROY_OPERATOR_NAMES}
    r_weights = {name: (10 if repair is None or name == repair else 0) for name in ALNS_REPAIR_OPERATOR_NAMES}
    return CompiledALNSPolicy(
        destroy_policy=DestroyPolicy(
            operator_weights=d_weights,
            feature_weights={"route_cost_release": 0, "target_feasibility_gain": 0, "bottleneck_relief": 0},
            remove_ratio=remove_ratio,
            selector_mode="random",
        ),
        insertion_policy=InsertionPolicy(
            operator_weights=r_weights,
            task_feature_weights={"cheapest_score": 0, "regret_score": 0, "scarcity_score": 0},
            task_selector_mode="random",
        ),
        acceptance_policy=AcceptancePolicy("greedy", 0.0),
        reaction_factor=float(selected.reaction_factor),
    )


@dataclass(slots=True)
class BaselineResult:
    solution: AssignmentSolution
    progress: list[dict[str, Any]]
    actions: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    solver_time_sec: float
    controller_time_sec: float


def _run_blocks(
    *, instance: Any, initial_solution: AssignmentSolution, trials: int, rng_seed: int,
    controller: str, config: BaselineConfig | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> BaselineResult:
    selected = config or BaselineConfig()
    effective_ratio = selected.effective_remove_ratio(len(instance.tasks))
    effective_config = BaselineConfig(
        remove_ratio=effective_ratio,
        reaction_factor=selected.reaction_factor,
        block_trials=selected.block_trials,
    )
    block_trials = int(selected.block_trials)
    if trials <= 0 or trials % block_trials != 0:
        raise ValueError(
            "comparison trial budget must be a positive multiple of "
            f"the selected block size {block_trials}"
        )
    if controller != "basic":
        raise ValueError(f"unknown baseline controller: {controller}")

    cfg = Config(rng_seed=int(rng_seed))
    initial = initial_solution.clone_search_state(); initial.normalize(instance)
    initial_eval = evaluate(initial, instance, cfg, update_solution_schedule=True)
    if not initial_eval.is_feasible:
        raise ValueError("stored comparison start is not hard-feasible")
    working = initial.clone_search_state()
    global_best = initial.clone_search_state()
    rng_stream = random.Random(int(rng_seed))
    d_w: dict[str, float] = {}; r_w: dict[str, float] = {}
    d_score: dict[str, float] = {}; d_used: dict[str, int] = {}
    r_score: dict[str, float] = {}; r_used: dict[str, int] = {}
    adaptation_elapsed = 0
    progress = [{"trial": 0, **objective_dict(initial_eval.quality_metrics)}]
    actions: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    solver_total = 0.0; controller_total = 0.0

    for block_index, start_trial in enumerate(range(0, trials, block_trials), start=1):
        before_q = quality_dict(global_best, instance, cfg)
        policy = _plain_policy(config=effective_config)

        t0 = time.perf_counter()
        result = solve_assignment(
            instance, working, cfg, Budget(max_iters=block_trials), policy,
            objective_terms=list(SEARCH_OBJECTIVE), rng_seed=int(rng_seed),
            compiled_search_focus={"focus_id": "global", "focus_task_ids": []},
            adaptive_destroy_weights=d_w, adaptive_repair_weights=r_w,
            adaptive_destroy_scores=d_score, adaptive_destroy_uses=d_used,
            adaptive_repair_scores=r_score, adaptive_repair_uses=r_used,
            adaptation_trials_elapsed=adaptation_elapsed,
            global_best_solution=global_best, rng_stream=rng_stream,
            adaptation_horizon_iters=int(trials),
        )
        elapsed = time.perf_counter() - t0; solver_total += elapsed
        working = result.working_solution.clone_search_state()
        if result.global_best_feasible is not None:
            global_best = result.global_best_feasible.clone_search_state()
        d_w = dict(result.adaptive_destroy_weights); r_w = dict(result.adaptive_repair_weights)
        d_score = dict(result.adaptive_destroy_scores); d_used = dict(result.adaptive_destroy_uses)
        r_score = dict(result.adaptive_repair_scores); r_used = dict(result.adaptive_repair_uses)
        adaptation_elapsed = int(result.adaptation_trials_elapsed)
        after_q = quality_dict(global_best, instance, cfg)
        improved = lex_compare(after_q, before_q) < 0
        end_trial = min(trials, start_trial + block_trials)
        progress.append({"trial": end_trial, **objective_dict(after_q)})
        if progress_callback is not None and (
            end_trial == trials or end_trial % 100 == 0
        ):
            progress_callback(end_trial, 0)
        actions.append({
            "action_index": block_index,
            "global_trial_start": start_trial,
            "global_trial_end": end_trial,
            "iters_used": block_trials,
            "controller": controller,
            "arm": None,
            "remove_ratio": float(effective_ratio),
            "reaction_factor": float(selected.reaction_factor),
            "block_trials": block_trials,
            "focus_id": "global",
            "semantic_features": "disabled",
            "acceptance": "greedy",
            "global_best_improved": bool(improved),
            "objective_before": objective_dict(before_q),
            "objective_after": objective_dict(after_q),
            "solver_time_sec": float(elapsed),
            "executor_stop_reason": str(result.diagnostics.get("stop_reason", "")),
        })

    final = global_best.clone_search_state(); final.normalize(instance)
    return BaselineResult(final, progress, actions, decisions, float(solver_total), float(controller_total))


def run_basic(*, instance: Any, initial_solution: AssignmentSolution, trials: int, rng_seed: int,
              progress_callback: Callable[[int, int], None] | None = None,
              config: BaselineConfig | None = None) -> BaselineResult:
    return _run_blocks(instance=instance, initial_solution=initial_solution, trials=trials, rng_seed=rng_seed,
                       controller="basic", progress_callback=progress_callback, config=config)
