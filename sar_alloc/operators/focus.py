"""Soft semantic focus helpers.

Focus never creates candidates, partitions a candidate pool, changes feasibility,
changes remove cardinality, or reserves trials.  It only marks candidates/tasks
inside the ordinary pool so the random selector can apply a
bounded preference.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple, TypeVar

from ..config import Config
from ..models import Instance
from ..solution import AssignmentSolution

T = TypeVar("T")


def _dedupe_ints(values: Iterable[Any]) -> Tuple[int, ...]:
    out: list[int] = []; seen: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            item = int(value)
        except (TypeError, ValueError):
            continue
        if item not in seen:
            seen.add(item); out.append(item)
    return tuple(out)


@dataclass(frozen=True, slots=True)
class FocusContext:
    focus_id: str = "global"
    target_task_ids: Tuple[int, ...] = ()
    minimum_target_hits: int = 1
    soft_greedy_bonus: float = 1.0
    soft_random_weight: float = 1.5

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "FocusContext":
        data = dict(raw or {})
        return cls(
            focus_id=str(data.get("focus_id", "global") or "global"),
            target_task_ids=_dedupe_ints(data.get("focus_task_ids", []) or []),
            minimum_target_hits=max(1, int(data.get("minimum_target_hits", 1) or 1)),
            soft_greedy_bonus=max(0.0, float(data.get("soft_greedy_bonus", 1.0) or 1.0)),
            soft_random_weight=max(1.0, float(data.get("soft_random_weight", 1.5) or 1.5)),
        )

    @property
    def active(self) -> bool:
        return self.focus_id != "global" and bool(self.target_task_ids)


def destroy_move_block_match(task_ids: Sequence[int], focus: FocusContext) -> tuple[bool, Dict[str, Any]]:
    removed = {int(tid) for tid in task_ids}
    hits = sorted(removed & set(focus.target_task_ids)) if focus.active else []
    matched = len(hits) >= int(focus.minimum_target_hits)
    return matched, {
        "direct_focus_task_hits": hits,
        "minimum_target_hits": int(focus.minimum_target_hits),
        "focus_target_task_ids": list(focus.target_task_ids),
    }


def annotate_destroy_moves_by_focus(moves: Sequence[T], focus: FocusContext, *, task_ids_getter) -> tuple[Dict[int, Dict[str, Any]], int]:
    metadata: Dict[int, Dict[str, Any]] = {}; targeted_count = 0
    for index, move in enumerate(moves):
        matched, details = destroy_move_block_match(task_ids_getter(move), focus)
        metadata[index] = {"focus_targeted": bool(matched), **details}
        targeted_count += int(matched)
    return metadata, targeted_count


def partition_repair_tasks(task_ids: Iterable[int], focus_task_ids: Iterable[int]) -> tuple[list[int], list[int]]:
    """Compatibility helper only; current selection never uses this partition."""
    focus_set = set(_dedupe_ints(focus_task_ids)); focused: list[int] = []; other: list[int] = []
    for tid in _dedupe_ints(task_ids):
        (focused if tid in focus_set else other).append(int(tid))
    return focused, other




__all__ = ["FocusContext", "destroy_move_block_match", "annotate_destroy_moves_by_focus", "partition_repair_tasks"]
