from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EvaluationConfig:
    """Physical/evaluation settings only.

    Solution quality ordering is deliberately absent.  The run-global objective
    is fixed to two service terms followed by transfer energy and passed explicitly to every comparison.
    """

    min_time_window_ref: float = 1e-6
    min_energy_ref: float = 1e-6


@dataclass(slots=True)
class Budget:
    """Iteration-only optimization budget.

    Wall-clock time is measured after execution for diagnostics and reporting,
    but it is never an optimization stopping condition or an agent-controlled
    resource.
    """

    max_iters: int


@dataclass(slots=True)
class Config:
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    rng_seed: int = 0
    experiment_profile: str = "conference"
    # Frozen low-level ALNS parameter used by the single Agent.  It is not an
    # Agent control; formal experiment runners record the selected value.
    alns_reaction_factor: float = 0.20

    def __post_init__(self) -> None:
        if not (0.0 < float(self.alns_reaction_factor) <= 1.0):
            raise ValueError("alns_reaction_factor must be in (0,1]")
