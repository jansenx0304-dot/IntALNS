"""Information for compact complete-service search; no control is executed here."""
from copy import deepcopy
from functools import lru_cache
import json
from .paths import PROJECT_ROOT


@lru_cache(maxsize=1)
def bank():
    return json.loads((PROJECT_ROOT/'configs/energy_examples.json').read_text(encoding='utf-8'))


def applies(observation):
    state=next((b['signals'] for b in observation.get('observation_blocks',[]) if b['block_id']=='STEP01_state'),{})
    return 0 < state.get('task_count',0) <= 50 and state.get('phase')=='energy_refinement'


def exposure(memory):
    weights=deepcopy(memory.adaptive_repair_weights)
    counts={name:sum(int(row.get('used',0)) for pair,row in memory.operator_pair_history.items() if pair.endswith('::'+name))
        for name in ('best_insertion','random_insertion')}
    return {'adaptive_repair_weights':weights,'cumulative_repair_trials':counts,
        'interpretation':'Actual adaptive weights multiply the chosen operator-priority weights. Mixed repair is not necessarily an equal split. A random-only POSITION window enables only that repair operator. Trial counts include earlier service phases and are not causal or current-plateau credit.'}


def example(observation):
    blocks={b['block_id']:b['signals'] for b in observation.get('observation_blocks',[])}
    evidence=blocks.get('STEP03_feedback',{}).get('search_evidence',{})
    history=evidence.get('control_history',[])
    energy=[w for w in history if w['phase_before']=='energy_refinement']
    if not energy:
        phase='settle'
    elif energy[-1]['control']['acceptance_mode']!='greedy':
        phase='consolidate'
    elif evidence.get('consecutive_windows_without_objective_progress',0)>=2:
        phase='explore'
    else:
        phase='progress'
    b=bank(); case=deepcopy(b['phases'][phase])
    def live_control(control):
        acceptance={'mode':control['acceptance_mode']}
        if acceptance['mode']!='greedy':acceptance['worsening_tolerance']=control['worsening_tolerance']
        return {'focus_id':'global','destroy_priority':deepcopy(control['destroy_priority']),
            'repair_priority':deepcopy(control['repair_priority']),'remove_ratio':control['remove_ratio'],
            'search_bias':{'destroy_selector':'random','task_selector':'random'},'acceptance':acceptance}
    def response(control,reason):
        return {'step_decision':{'rationale':{'reason':reason,'refs':[]},'action':'run_alns','control':control}}
    if phase=='progress':
        case['illustrative_valid_response']=response(live_control(energy[-1]['control']),
            'Continue the actual current energy neighborhood for another measured window; do not revert to an old service-recovery control while claiming continuation.')
    elif phase=='consolidate':
        control=live_control(energy[-1]['control']);control.update(repair_priority=['best_insertion'],remove_ratio=.2,acceptance={'mode':'greedy'})
        case['illustrative_valid_response']=response(control,
            'Test greedy best-position consolidation of the actual preceding exploration, keeping its destruction family and retaining the best solution.')
    elif phase=='explore':
        profiles=deepcopy(b['exploration_profiles'])
        def matches(window,control):
            past=window['control']
            return (past['destroy_priority'][:1]==control['destroy_priority'][:1]
                and past['repair_priority']==control['repair_priority']
                and past['remove_ratio']==control['remove_ratio']
                and past['acceptance_mode']==control['acceptance']['mode'])
        counts=[sum(matches(w,p['control']) for w in energy) for p in profiles]
        index=min(range(len(profiles)),key=lambda i:counts[i]);profile=profiles[index]
        case['retrieval_basis']={'profile_exposure_counts':{p['name']:n for p,n in zip(profiles,counts)},
            'selected_information_profile':profile['name'],'evidence':profile['basis'],
            'interpretation':'Counts are matching-family/repair/ratio/mode energy windows; prior tolerance, routes and adaptive weights can differ. Retrieval supplies an example, never an executable decision.'}
        case['illustrative_valid_response']=response(profile['control'],
            'Compare this less-exposed energy neighborhood with the failed controls in live history, and judge the subsequent best-energy outcome at complete service.')
    return {'role':'step','example_id':b['revision']+'_'+phase,'provenance':b['provenance'],
            'scope':b['scope'],**case}


INSTRUCTION="""Complete-service energy reasoning for compact instances:
- Service is already complete and must remain protected. First test global focus, moderate time-related destruction and greedy best-position repair as the retrieved compact descent hypothesis. One enabled Destroy operator is legal. Continue controls that lower retained best energy; accepted movement alone is not improvement.
- After repeated best-energy stagnation, the retrieved two-window example demonstrates one position-diversifying exploration window followed by greedy best-position consolidation. Random TASK selection is unchanged; random_insertion changes POSITION choice. A random-only repair probe is legal, and differs from adding it with a lower prior to best repair. Consult actual adaptive_repair_weights rather than assuming equal usage.
- Retrieved exploration profiles include time-only greedy neighborhoods, spatial/random-position .30 with tolerant .20, and SA/mixed alternatives. Read the ACTUAL selected profile and its evidence instead of always copying a spatial recipe. A position-diversifying profile was paired with same-family .20 greedy best-position consolidation; a time-only greedy profile tests a different neighborhood directly. These are counterfactual development observations, not guaranteed current-run winners or a forced cycle. A mixed-position policy is not the tested random-only policy.
- A tolerant window can change working routes without losing the retained best. Then test whether greedy best-position repair consolidates that exploration. If the complete pair fails, retain the negative evidence and compare a different family or SA/mixed-position alternative; do not indefinitely repeat the same failed pair. No extra trials or new control variables are available.
- The continuation example is constructed from the last ACTUAL energy control, not an early service control. The consolidation example retains the preceding exploration family. A plateau example reports actual profile exposure; prefer testing another justified hypothesis over copying an already failed pair. If only one window remains, consider greedy consolidation rather than beginning an unfinished exploration pair.
"""
