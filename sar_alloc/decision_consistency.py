"""Expose executed controls clearly and order the response before control emission."""
from copy import deepcopy


INSTRUCTION = """Decision consistency check:
- In step_decision write rationale first, then action, then control. Decide the next complete control from the observed state and executed history, then emit control values implementing that decision. All three keys belong inside step_decision.
- The last_actual_control is the executed strategy, not a previous rationale promise. If the rationale says to change an operator, focus, ratio or acceptance, the corresponding JSON field must actually differ. If retaining a strategy, say continuation. Before returning, compare every claimed change against the final control; a prose-only change has no effect.
- Repeated service stagnation and repeated energy stagnation are different. Use the true current phase and recent before/after vectors to decide continuation, an alternative neighborhood, or consolidation. A historical early service gain does not justify indefinitely repeating a residual control.
"""


def prepare(observation, schema, examples):
    observation = deepcopy(observation)
    for block in observation.get('observation_blocks', []):
        if block['block_id'] != 'STEP03_feedback':
            continue
        evidence = block['signals'].get('search_evidence', {})
        history = evidence.get('control_history', [])
        identical = 0
        if history:
            for row in reversed(history):
                if row['control'] != history[-1]['control']:
                    break
                identical += 1
        evidence['decision_consistency'] = {
            'last_actual_control': deepcopy(history[-1]['control']) if history else None,
            'consecutive_identical_executed_controls': identical,
            'last_actual_changes': deepcopy(evidence.get('last_control_changes', [])),
            'windows_without_service_progress': evidence.get('consecutive_windows_without_service_progress', 0),
            'windows_without_best_progress': evidence.get('consecutive_windows_without_objective_progress', 0),
            'interpretation': 'Executed values only. Natural-language plans that were not present in control were not tested.'}
    def order(node):
        if isinstance(node, list):
            return [order(value) for value in node]
        if not isinstance(node, dict):
            return node
        result = {key: order(value) for key, value in node.items()}
        if all(key in result for key in ('rationale', 'action', 'control')):
            result = {key: result[key] for key in ('rationale', 'action', 'control')} | {key: value for key, value in result.items() if key not in ('rationale', 'action', 'control')}
        return result
    # Dict ordering changes rendered information only, never schema constraints.
    rendered_schema = order(schema)
    assert rendered_schema == schema
    return observation, rendered_schema, order(deepcopy(examples))
