'Action-local focus targets. Solved targets lose their bias; the next action chooses focus anew.'
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping
from .solution import AssignmentSolution


def dedupe_ints(values: Iterable[Any]) -> List[int]:
    out: List[int] = []; seen: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            continue
        try: item = int(value)
        except (TypeError, ValueError): continue
        if item not in seen:
            seen.add(item); out.append(item)
    return out


def restrict_runtime_focus(runtime_focus: Mapping[str, Any], solution: AssignmentSolution) -> Dict[str, Any]:
    raw = dict(runtime_focus or {})
    focus_id = str(raw.get("focus_id", "global") or "global")
    unresolved = {int(tid) for tid in solution.unassigned}
    raw["focus_task_ids"] = [] if focus_id == "global" else [tid for tid in dedupe_ints(raw.get("focus_task_ids", []) or []) if tid in unresolved]
    return raw


@dataclass
class RuntimeFocusState:
    base: Dict[str, Any]
    focus_task_ids: List[int]
    initial_focus_task_ids: List[int]

    @classmethod
    def from_compiled(cls, compiled: Mapping[str, Any], solution: AssignmentSolution) -> "RuntimeFocusState":
        base = dict(compiled or {})
        active = restrict_runtime_focus(base, solution)
        targets = dedupe_ints(active.get("focus_task_ids", []) or [])
        return cls(base=base, focus_task_ids=targets, initial_focus_task_ids=list(targets))

    @property
    def focus_id(self) -> str:
        return str(self.base.get("focus_id", "global") or "global")

    def as_mapping(self) -> Dict[str, Any]:
        out = dict(self.base); out["focus_task_ids"] = list(self.focus_task_ids); return out

    def restrict_to(self, solution: AssignmentSolution) -> Dict[str, Any]:
        active = restrict_runtime_focus(self.as_mapping(), solution)
        self.focus_task_ids = dedupe_ints(active.get("focus_task_ids", []) or [])
        return active

    def record_best_update_after_refresh(self, **kwargs: Any) -> None:
        return None

    def diagnostics(self, *, enabled: bool, actual_time_used_sec: float) -> Dict[str, Any]:
        del actual_time_used_sec
        return {
            "enabled": False,
            "focus_id": self.focus_id,
            "initial_focus_task_ids": list(self.initial_focus_task_ids),
            "final_focus_task_ids": list(self.focus_task_ids),
            "refresh_count": 0,
            "reason": "Focus is fixed within this action; runtime only drops solved targets.",
        }




__all__ = ["RuntimeFocusState", "dedupe_ints", "restrict_runtime_focus"]
