'Fixed lexicographic objective: missed priority, unassigned count, transfer energy.'
from __future__ import annotations

from typing import Any, Mapping, Sequence

from sar_alloc.domain import QUALITY_METRICS, terms_from_order

SEARCH_OBJECTIVE = terms_from_order(QUALITY_METRICS)
SEARCH_OBJECTIVE_METRICS = tuple(QUALITY_METRICS)
LEX_EPS = 0.0


def objective_vector(q: Mapping[str, Any]) -> tuple[float, ...]:
    return tuple(float(q.get(name, 0.0)) for name in SEARCH_OBJECTIVE_METRICS)


def lex_compare(a: Mapping[str, Any] | Sequence[float], b: Mapping[str, Any] | Sequence[float], *, eps: float = LEX_EPS) -> int:
    av = objective_vector(a) if isinstance(a, Mapping) else tuple(float(v) for v in a)
    bv = objective_vector(b) if isinstance(b, Mapping) else tuple(float(v) for v in b)
    if len(av) != len(SEARCH_OBJECTIVE_METRICS) or len(bv) != len(SEARCH_OBJECTIVE_METRICS):
        raise ValueError("lexicographic comparison requires the complete three-dimensional objective")
    for x, y in zip(av, bv):
        if x < y - eps:
            return -1
        if x > y + eps:
            return 1
    return 0


def lex_outcome(hialns: Mapping[str, Any], basic: Mapping[str, Any]) -> str:
    cmp = lex_compare(hialns, basic)
    return "W" if cmp < 0 else ("L" if cmp > 0 else "T")


def decisive_level(a: Mapping[str, Any] | Sequence[float], b: Mapping[str, Any] | Sequence[float], *, eps: float = LEX_EPS) -> str | None:
    av = objective_vector(a) if isinstance(a, Mapping) else tuple(float(v) for v in a)
    bv = objective_vector(b) if isinstance(b, Mapping) else tuple(float(v) for v in b)
    for index, (x, y) in enumerate(zip(av, bv), start=1):
        if abs(x - y) > eps:
            return f"L{index}:{SEARCH_OBJECTIVE_METRICS[index-1]}"
    return None


def objective_dict(q: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(q.get(name, 0.0)) for name in SEARCH_OBJECTIVE_METRICS}


def first_decisive_delta(after: Mapping[str, Any], before: Mapping[str, Any], *, eps: float = LEX_EPS) -> dict[str, Any]:
    """Describe one lexicographic change without collapsing dimensions."""
    av = objective_vector(after)
    bv = objective_vector(before)
    level = decisive_level(after, before, eps=eps)
    cmp = lex_compare(after, before, eps=eps)
    index = None if level is None else int(level.split(":", 1)[0][1:]) - 1
    return {
        "comparison": "improved" if cmp < 0 else ("worsened" if cmp > 0 else "equal"),
        "decisive_level": level,
        "decisive_delta": None if index is None else float(av[index] - bv[index]),
        "before": list(bv),
        "after": list(av),
    }


__all__ = [
    "SEARCH_OBJECTIVE",
    "SEARCH_OBJECTIVE_METRICS",
    "LEX_EPS",
    "objective_vector",
    "objective_dict",
    "lex_compare",
    "lex_outcome",
    "decisive_level",
    "first_decisive_delta",
]
