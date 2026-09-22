"""Retrieve development examples for the start of a large service backlog.

This module supplies information only; the Agent still chooses every control.
"""
from copy import deepcopy
from functools import lru_cache
import json

from .paths import PROJECT_ROOT


def information_phase(observation):
    blocks = {x['block_id']: x['signals'] for x in observation.get('observation_blocks', [])}
    state = blocks.get('STEP01_state', {})
    history = blocks.get('STEP03_feedback', {}).get('search_evidence', {}).get('control_history', [])
    n = state.get('task_count', 0)
    if n < 200:
        return None
    if not history and state.get('unassigned_task_count', 0) / max(1, n) > .15:
        return 'initial'
    if state.get('phase') == 'service_recovery' and len(history) == 1 and history[0]['best_before'][1] / n > .15:
        c = history[0]['control']
        if (c['focus_id'] == 'service_bottleneck' and c['remove_ratio'] == .3 and c['acceptance_mode'] == 'greedy'
                and c['destroy_priority'] == ['spatial_related_removal', 'worst_cost_removal', 'random_removal']
                and c['repair_priority'] == ['best_insertion', 'random_insertion']):
            return 'consolidate'
    return None


@lru_cache(maxsize=1)
def _bank():
    return json.loads((PROJECT_ROOT / 'configs/large_initial_examples.json').read_text(encoding='utf-8'))


def example(observation):
    phase = information_phase(observation)
    if phase is None:
        return None
    bank = _bank()
    return {'role': 'step', 'example_id': bank['revision'] + '_' + phase,
            'provenance': bank['provenance'], 'scope': bank['scope'], **deepcopy(bank['phases'][phase])}
