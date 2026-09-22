"""Frozen, factual strategy examples; selection never executes an ALNS control."""
from functools import lru_cache
import json
from .paths import PROJECT_ROOT

@lru_cache(maxsize=1)
def bank():
    p=json.loads((PROJECT_ROOT/'configs/protocol.json').read_text(encoding='utf-8'))
    name=p.get('agent_example_file')
    return json.loads((PROJECT_ROOT/name).read_text(encoding='utf-8')) if name else None

def large_examples(observation):
    b=bank()
    if not b:return []
    blocks={x['block_id']:x['signals'] for x in observation.get('observation_blocks',[])}
    state=blocks.get('STEP01_state',{})
    evidence=blocks.get('STEP03_feedback',{}).get('search_evidence',{})
    recent=evidence.get('recent_windows',[])
    phase=('energy' if state.get('phase')=='energy_refinement' else 'initial' if not recent
           else 'progress' if recent[-1]['service_improved'] else 'stalled')
    # Retrieve a relevant reasoning case; the Agent still selects every control.
    if phase=='initial' and state.get('unassigned_task_count',0)/max(1,state.get('task_count',1))<=.15:
        phase='sparse_initial'
    if (phase == 'stalled' and state.get('task_count', 0) >= 200
            and evidence.get('consecutive_windows_without_service_progress', 0) >= 2
            and 'large_stalled' in b['phases']):
        phase = 'large_stalled'
    if (phase == 'initial' and state.get('task_count', 0) >= 200
            and 'large_initial' in b['phases']):
        phase = 'large_initial'
    if (phase == 'large_stalled' and evidence.get('consecutive_windows_without_service_progress', 0) >= 3
            and any(r['remove_ratio'] >= .4 and r['observed_windows'] for r in evidence.get('service_gap_remove_ratio_trials', []))
            and 'large_rebuild' in b['phases']):
        phase = 'large_rebuild'
    return [{'role':'step','example_id':'state_'+phase,'provenance':b['provenance'],
             'scope':b['scope'],**b['phases'][phase]}]


@lru_cache(maxsize=1)
def _compact_initial_bank():
    return json.loads((PROJECT_ROOT/'configs/compact_initial_example.json').read_text(encoding='utf-8'))


def compact_initial_example(observation):
    blocks = {b['block_id']: b['signals'] for b in observation.get('observation_blocks', [])}
    state = blocks.get('STEP01_state', {})
    history = blocks.get('STEP03_feedback', {}).get('search_evidence', {}).get('control_history', [])
    if 0 < state.get('task_count', 0) <= 50 and not history:
        return _compact_initial_bank()
    return None
