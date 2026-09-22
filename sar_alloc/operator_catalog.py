'Primitive operator semantics and acceptance rules exposed to the Agent.'

from __future__ import annotations

from typing import Any, Dict, Iterable

from .domain import PRIORITY_RANK_CAPACITY, ORDERED_PRIORITY_WEIGHTS
from .operators.types import (
    DESTROY_FEATURE_NAMES,
    DESTROY_OPERATOR_NAMES,
    ALNS_REPAIR_OPERATOR_NAMES,
    INSERTION_TASK_FEATURE_NAMES,
    SELECTOR_MODES,
)


def build_destroy_operator_cards(operator_ids: Iterable[str] | None = None) -> list[Dict[str, Any]]:
    ids = list(DESTROY_OPERATOR_NAMES if operator_ids is None else operator_ids)
    cards: Dict[str, Dict[str, Any]] = {
        "random_removal": {
            "operator_id": "random_removal",
            "behavior": "generate independent uniform random task sets",
            "candidate_shape": "random task set",
        },
        "worst_cost_removal": {
            "operator_id": "worst_cost_removal",
            "behavior": "generate sets from tasks with the largest fixed local route-cost release",
            "candidate_shape": "high local-cost tasks",
        },
        "spatial_related_removal": {
            "operator_id": "spatial_related_removal",
            "behavior": "choose a seed and generate its nearest tasks in physical space",
            "candidate_shape": "spatial neighbourhood",
        },
        "time_related_removal": {
            "operator_id": "time_related_removal",
            "behavior": "choose a seed and generate tasks with the closest time-window centres",
            "candidate_shape": "temporal neighbourhood",
        },
    }
    unknown = [name for name in ids if name not in cards]
    if unknown:
        raise ValueError(f"unknown destroy operator ids: {unknown}")
    return [dict(cards[name]) for name in ids]


def build_repair_operator_cards(operator_ids: Iterable[str] | None = None) -> list[Dict[str, Any]]:
    ids = list(ALNS_REPAIR_OPERATOR_NAMES if operator_ids is None else operator_ids)
    cards: Dict[str, Dict[str, Any]] = {
        "best_insertion": {
            "operator_id": "best_insertion",
            "task_choice": "external task selector",
            "position_choice": "minimum incremental-energy strict-feasible position, with platform ID then position as deterministic ties",
            "role": "primitive greedy repair",
        },
        "random_insertion": {
            "operator_id": "random_insertion",
            "task_choice": "external task selector",
            "position_choice": "uniform random strict-feasible position",
            "role": "primitive diversification repair",
        },
    }
    unknown = [name for name in ids if name not in cards]
    if unknown:
        raise ValueError(f"unknown repair operator ids: {unknown}")
    return [dict(cards[name]) for name in ids]








def build_acceptance_cards(modes: Iterable[str] | None = None) -> list[Dict[str, Any]]:
    """Return cards using the public Agent-contract mode names.

    The compiler maps ``tolerant`` to the internal threshold rule and
    ``simulated_annealing`` to the internal SA rule.  Exposing internal names
    here previously left both exploratory modes undocumented in the prompt.
    """

    allowed = list(
        ("greedy", "tolerant", "simulated_annealing") if modes is None else modes
    )
    cards = {
        "greedy": {
            "mode": "greedy",
            "rule": "accept lexicographically improving or equal candidates only",
            "exploration_role": "none",
        },
        "tolerant": {
            "mode": "tolerant",
            "compiled_rule": "threshold",
            "rule": "protect both service objectives; at equal service accept energy worsening up to worsening_tolerance times max(1, action-start energy); improving/equal full vectors are always accepted",
            "exploration_role": "explore energy barriers without sacrificing service; retain the best full objective",
        },
        "simulated_annealing": {
            "mode": "simulated_annealing",
            "compiled_rule": "sa",
            "rule": "protect both service objectives; at equal service accept energy worsening probabilistically with exp(-normalized_energy_increase/temperature); temperature begins at worsening_tolerance and cools within the action",
            "exploration_role": "explore energy barriers without sacrificing service; retain the best full objective",
        },
    }
    unknown = [name for name in allowed if name not in cards]
    if unknown:
        raise ValueError(f"unknown acceptance mode ids: {unknown}")
    return [dict(cards[name]) for name in allowed]
