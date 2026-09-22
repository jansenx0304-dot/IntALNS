"""Shared final-choice mechanism for destroy candidates and repair tasks."""

from __future__ import annotations

import random
from typing import Callable, Sequence, TypeVar

from .types import SELECTOR_MODES

T = TypeVar("T")


def select_item(
    items: Sequence[T],
    *,
    mode: str,
    score_getter: Callable[[T], float],
    rng: random.Random,
    tie_key: Callable[[T], object] | None = None,
) -> T:
    """Uniform selection in the active pool; feature scores are disabled."""

    if not items:
        raise ValueError("cannot select from an empty item sequence")
    normalized = str(mode)
    if normalized not in SELECTOR_MODES:
        raise ValueError(f"unknown selector mode: {normalized!r}")
    return rng.choice(list(items))


__all__ = ["select_item"]
