'Factual focus feedback for the single-agent controller; observational, not causal.'
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

FOCUS_FEEDBACK_KIND = "three_level"
REVIEW_INSTRUCTION = """
Focus review for this single-agent variant:
- Having unresolved tasks makes service_bottleneck available; it does not establish that the bias is useful. global still attempts all unresolved tasks and retains every candidate. It does not abandon service recovery.
- Focus only biases which pending task is tried; it cannot make an infeasible insertion feasible or target the assigned blockers for removal. Repeated zero-insertability is not by itself evidence to strengthen or retain focus.
- A target with one or two feasible positions is a plausible reason to TEST focus, not a reason to reject it: earlier sampling may preserve a scarce insertion opportunity before other tasks occupy it. Conversely, zero positions are measured before the next destruction; destruction may open a position, so zero is not proof that focus can never help. Neither observation guarantees a benefit. Evaluate the whole window and distinguish opportunity protection from blocker removal.
- Objective progress is lexicographic improvement of (missed_priority, unassigned_count, energy_total). Distinguish service recovery from lower energy at equal service; never trade service for energy. Target insertions and selection changes alone do not prove service recovery.
- Compare continuing focus with a one-window global probe using recent outcomes. If a focus remains unproductive across windows, global is a legitimate alternative before service completion. Keep the other controls stable for a focus-only comparison where practical; do not prescribe a fixed alternation schedule.
- Productive service recovery can justify continuing focus. A poor global probe can justify returning to it. An untried alternative is unknown, not known to be worse. Do not invent counterfactual performance from observational history.
- Explain your focus choice against the available alternative in the rationale, citing live facts. Existing cases whose greedy task-selection premise is illegal do not establish benefits for this random-only variant.
"""
def build_focus_window(result: Mapping[str, Any], *, action_index: int) -> dict[str, Any]:
    """Read the actual executor contract strictly; never hide missing metrics as 0."""
    feedback = result["decision_feedback"]
    objective = feedback["objective_progress"]
    if list(objective["metric_order"]) != ["missed_priority", "unassigned_count", "energy_total"]:
        raise ValueError("focus feedback requires the fixed three-level experiment objective")
    before = list(objective["global_best_before"])
    after = list(objective["global_best_after"])
    if len(before) != 3 or len(after) != 3:
        raise ValueError("focus feedback requires complete before/after objectives")
    flow = feedback["search_flow"]
    focus = feedback["focus"]
    selected, observed = focus["selected"], focus["observed"]
    return {
        "action_index": int(action_index),
        "focus_id": str(selected["focus_id"]),
        "target_task_ids": list(selected["focus_task_ids"]),
        "best_objective_before": before,
        "best_objective_after": after,
        "service_prefix_improved": tuple(after[:2]) < tuple(before[:2]),
        "objective_improved": tuple(after) < tuple(before),
        "energy_improved_at_equal_service": before[:2] == after[:2] and after[2] < before[2],
        "best_update_count": int(flow["global_best_update_count"]),
        "focus_active_trials": int(flow["focus_active_trials"]),
        "best_updates_while_focus_active": int(flow["focus_active_best_updates"]),
        "target_task_draws": int(observed["focus_matched_task_selections"]),
        "target_ids_inserted_in_candidates": list(observed["tasks_inserted"]),
        "target_ids_still_unassigned": list(observed["tasks_still_unassigned"]),
        "selection_difference_diagnostic": int(result["search_focus_result"]["focus_changed_selection_count"]),
    }


def build_focus_review(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    windows = [deepcopy(row["focus_review_window"]) for row in history]
    last_focus = windows[-1]["focus_id"] if windows else None
    streak = 0
    for window in reversed(windows):
        if window["focus_id"] != last_focus or window["service_prefix_improved"]:
            break
        streak += 1
    groups = []
    for focus in ("global", "service_bottleneck"):
        group = [row for row in windows if row["focus_id"] == focus]
        groups.append({
            "focus_id": focus, "observed_windows": len(group), "untried": not group,
            "service_improving_windows": sum(row["service_prefix_improved"] for row in group) if group else None,
        })
    return {
        "kind": FOCUS_FEEDBACK_KIND,
        "recent_windows": windows[-3:], "outcomes_by_focus": groups,
        "same_focus_service_stagnation_windows": streak,
        "interpretation": (
            "Observational outcomes, not causal credit. Target insertions can occur in rejected intermediate candidates. "
            "Best updates while focus is active can come from any operator or task. Selection differences also reflect "
            "different random-number consumption and are not an optimization benefit. Untried alternatives are unknown."
        ),
    }
