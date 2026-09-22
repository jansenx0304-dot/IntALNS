"""Factual history for three-level search; no automatic strategy selection."""
from copy import deepcopy
from .domain import REMOVE_RATIO_OPTIONS

_CONTROL_FIELDS = ('focus_id', 'destroy_priority', 'repair_priority', 'destroy_selector',
                   'task_selector', 'remove_ratio', 'acceptance_mode', 'worsening_tolerance')

def _window(row):
    before, after = list(row['objective_before']), list(row['objective_after'])
    service = tuple(after[:2]) < tuple(before[:2])
    energy = after[:2] == before[:2] and after[2] < before[2]
    return dict(action_index=row['action_index'],
        phase_before='energy_refinement' if before[:2] == [0, 0] else 'service_recovery',
        control={k:deepcopy(row[k]) for k in _CONTROL_FIELDS},
        best_before=before, best_after=after, service_improved=service,
        energy_improved_at_equal_service=energy, objective_improved=service or energy,
        energy_gain_at_equal_service=before[2]-after[2] if after[:2]==before[:2] else None,
        acceptance_evidence=deepcopy(row.get('acceptance_evidence',{})),
        noop_rate=row['noop_rate'], structure_change_rate=row['structure_change_rate'],
        structural_passage_rate=row['structural_passage_rate'])

def evidence_feedback(feedback, memory, *, current_phase):
    windows=[_window(row) for row in memory.recent_controls(limit=12)]
    evidence={'current_phase':current_phase,'observed_action_count':len(windows),
        'recent_windows':windows[-2:],'control_history':windows,
        'operator_pair_observations':deepcopy(memory.operator_pair_history),
        'adaptive_destroy_weights':deepcopy(memory.adaptive_destroy_weights),
        'interpretation':'Actual complete controls and outcomes, not causal credit. Energy gains count only at equal service. An old service success is not evidence of energy refinement. Retain failed controls as well as successes.'}
    if windows:
        current=windows[-1]['best_after'];at_gap=[w for w in windows if w['best_before']==current]
        service_gap=[w for w in windows if w['best_before'][:2]==current[:2]]
        objective_stall=service_stall=0
        for w in reversed(windows):
            if w['objective_improved']:break
            objective_stall+=1
        for w in reversed(windows):
            if w['service_improved']:break
            service_stall+=1
        evidence.update(current_gap=current,current_gap_action_indices=[w['action_index'] for w in at_gap],
            current_service_gap_action_indices=[w['action_index'] for w in service_gap],
            consecutive_windows_without_objective_progress=objective_stall,
            consecutive_windows_without_service_progress=service_stall,
            current_gap_remove_ratio_trials=[{'remove_ratio':ratio,'observed_windows':sum(w['control']['remove_ratio']==ratio for w in at_gap)} for ratio in REMOVE_RATIO_OPTIONS],
            current_gap_interpretation='Exact-vector counts reset on energy progress; service-gap counts do not. Energy improvement does not mean residual tasks were recovered. A reset is not a reason to abandon productive control. Equal objective vectors need not have equal routes.')
        evidence['service_gap_neighborhood_trials'] = [
            {'action_index': w['action_index'], 'control': deepcopy(w['control']),
             'best_after': w['best_after'], 'acceptance_evidence': deepcopy(w['acceptance_evidence'])}
            for w in service_gap]
        evidence['service_gap_remove_ratio_trials'] = [
            {'remove_ratio': ratio, 'observed_windows': sum(w['control']['remove_ratio'] == ratio for w in service_gap)}
            for ratio in REMOVE_RATIO_OPTIONS]
        evidence['last_control_changes'] = ([k for k in _CONTROL_FIELDS
            if windows[-1]['control'][k] != windows[-2]['control'][k]] if len(windows) > 1 else [])
    return {'history_available':bool(windows),'focus_review':deepcopy(feedback['focus_review']),'search_evidence':evidence}
