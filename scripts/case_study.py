"""Review every saved window and render one explicitly selected descriptive case."""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scripts.figure_style import METHODS, STYLE, configure, export

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('missed_priority', 'unassigned_count', 'energy_total')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)




def case_evidence(records, out):
    selection = json.loads((ROOT/'configs/paper_case.json').read_text(encoding='utf-8'))
    key = (selection['case_id'], selection['maturity'], selection['decision_index'])
    pair = {r['method']:r for r in records if (r['case_id'],r['maturity'])==key[:2]}
    agent=pair['agent']; index=key[2]-1; decision=agent['decisions'][index]; actual=agent['actions'][index]
    control=decision['decision']['control']; previous=agent['actions'][index-1]['decision']['control'] if index else {}
    changes={k:{'before':previous.get(k),'after':v} for k,v in control.items() if previous.get(k)!=v}
    start,end=actual['global_trial_start'],actual['global_trial_end']
    assert decision['observation_summary']['state']['global_trial']==start and end==start+100
    assert control['focus_id']==actual['focus_id'] and control['remove_ratio']==actual['remove_ratio']
    for k in ('destroy_priority','repair_priority'):
        assert control[k]==actual['operator_priority'][k]
    before,after = (actual['objective_progress'][k] for k in ('global_best_before','global_best_after'))
    basic = next(p for p in pair['basic']['checkpoints'] if p['trial']==end)
    notes = dict(CaseSize=agent['n_tasks'],CaseSeed=agent['instance_seed'],CaseMaturity=agent['maturity'],
        CaseStart=start,CaseFirstTrial=start+1,CaseEnd=end,
        CaseStall=decision['observation_summary']['feedback']['search_evidence']['consecutive_windows_without_service_progress'],
        CaseBeforePriority=f'{before[0]:g}',CaseBeforeCount=f'{before[1]:g}',
        CaseAfterPriority=f'{after[0]:g}',CaseAfterCount=f'{after[1]:g}',
        CaseBasicPriority=f'{basic[METRICS[0]]:g}',CaseBasicCount=f'{basic[METRICS[1]]:g}',
        CaseFinalCount=f"{agent['final_quality'][METRICS[1]]:g}",CaseBeforeRatio=previous['remove_ratio'],CaseAfterRatio=control['remove_ratio'],
        CaseServiceRejections=decision['observation_summary']['feedback']['search_evidence']['recent_windows'][-1]['acceptance_evidence']['rejection_reasons'].get('service_metric_worsening_protected',0))
    table_facts = compact_case_table(pair, index, out)
    notes.update(CaseUninsertable=table_facts['zero_feasible_task_count'],
                 CasePreviousTrials=table_facts['previous_candidate_trials'],
                 CasePlotFirst=table_facts['plot_first_checkpoint'],
                 CasePlotLast=table_facts['plot_last_checkpoint'])
    (out/'tables/case_notes.tex').write_text('% Generated from the selected saved case.\n'+
        '\n'.join('\\newcommand{\\'+k+'}{'+str(v)+'}' for k,v in notes.items())+'\n',encoding='utf-8',newline='\n')
    draw_case(pair,start,end,out)


def compact_case_table(pair, index, out):
    """Validate the transition and excerpt original values and rationale."""
    agent = pair['agent']
    action = agent['actions'][index]
    decision = agent['decisions'][index]
    previous = agent['actions'][index-1]['decision']['control']
    control = decision['decision']['control']
    observation = decision['observation_summary']
    search = observation['feedback']['search_evidence']
    before, after = (action['objective_progress'][k] for k in ('global_best_before', 'global_best_after'))
    start, end = action['global_trial_start'], action['global_trial_end']
    stall = int(search['consecutive_windows_without_service_progress'])
    preceding = agent['actions'][index-stall:index]
    if len(preceding) != stall or not preceding:
        raise ValueError('The recorded preceding service plateau is incomplete')
    if any(a['objective_progress'][k][:2] != before[:2]
           for a in preceding for k in ('global_best_before', 'global_best_after')):
        raise ValueError('The recorded service plateau disagrees with saved actions')
    if preceding[-1]['global_trial_end'] != start:
        raise ValueError('The plateau does not end at the selected decision')
    changed = {k for k in control if control[k] != previous.get(k)}
    if changed != {'remove_ratio'}:
        raise ValueError('The selected case no longer changes only removal strength')
    bottleneck = observation['bottleneck']
    no_position = int(bottleneck['zero_feasible_task_count'])
    if no_position != bottleneck['unassigned_task_count'] or no_position != before[1]:
        raise ValueError('The selected case does not have all residual tasks blocked from direct insertion')
    acceptance = search['recent_windows'][-1]['acceptance_evidence']
    previous_trials = int(acceptance['candidate_trials'])
    rejections = int(acceptance['rejection_reasons']['service_metric_worsening_protected'])
    if previous_trials != preceding[-1]['global_trial_end'] - preceding[-1]['global_trial_start']:
        raise ValueError('The previous-window candidate count disagrees with saved actions')
    if not 0 <= rejections <= previous_trials:
        raise ValueError('Invalid rejection count')
    basic = next(p for p in pair['basic']['checkpoints'] if p['trial'] == end)
    for method, vector, checkpoint in [('agent', before, start), ('agent', after, end)]:
        point = next(p for p in pair[method]['checkpoints'] if p['trial'] == checkpoint)
        if [point[k] for k in METRICS] != vector:
            raise ValueError('Case table values disagree with recorded checkpoints')
    facts = dict(case_id=agent['case_id'], maturity=agent['maturity'], decision_index=index+1,
        decision_at=start, next_checkpoint=end, prior_service_stall_intervals=stall,
        service_before=before[:2], service_after=after[:2],
        basic_service_after=[basic[k] for k in METRICS[:2]],
        zero_feasible_task_count=no_position, previous_candidate_trials=previous_trials,
        previous_service_worsening_rejections=rejections,
        removal_ratio_before=previous['remove_ratio'], removal_ratio_after=control['remove_ratio'],
        retained_controls={k: v for k, v in control.items() if k != 'remove_ratio'},
        plot_first_checkpoint=start-2*(end-start), plot_last_checkpoint=end+(end-start),
        interpretation='Recorded observation, control change and subsequent outcome; no causal attribution.')
    write_json(out/'data/case_table.json', facts)
    reason = decision['decision']['rationale']['reason']
    cut = reason.index(' while keeping')
    excerpt = reason[:cut]
    assert excerpt.endswith('coupled assigned blockers') and reason.startswith(excerpt)
    observation_fields = [
        ('observation_summary.bottleneck.zero_feasible_task_count', no_position),
        ('observation_summary.feedback.search_evidence.consecutive_windows_without_service_progress', stall),
        ('observation_summary.feedback.search_evidence.recent_windows[-1].energy_improved_at_equal_service',
         search['recent_windows'][-1]['energy_improved_at_equal_service']),
    ]
    control_fields = [('decision.control.remove_ratio', control['remove_ratio']),
                      ('decision.control.focus_id', control['focus_id'])]
    write_json(out/'data/case_raw_excerpt.json', dict(
        source_record=f"results/test/seed_0/records/{agent['case_id']}__{agent['maturity']}__agent.json",
        source_decision_index_zero_based=index, decision_at=start,
        observation_fields=[dict(path=path, value=value) for path,value in observation_fields],
        control_fields=[dict(path=path, value=value) for path,value in control_fields],
        rationale_path='decision.rationale.reason', rationale_full=reason,
        rationale_excerpt=excerpt, substring_start=0, substring_end=cut,
        note='Original field names and values; exact contiguous rationale excerpt. Line wrapping and JSON whitespace are presentation only.'))
    def field(path, value):
        name = path.rsplit('.',1)[-1]
        value_text = json.dumps(value, ensure_ascii=False)
        return r'\texttt{"'+name.replace('_',r'\_\allowbreak{}')+r'":\nobreakspace '+value_text.replace('_',r'\_\allowbreak{}')+'}'
    lines = [r'% Generated verbatim field/value and rationale excerpts; provenance in case_raw_excerpt.json.',
        r'\begingroup\small\raggedright',
        r'\begin{tabular}{@{}>{\raggedright\arraybackslash}p{\linewidth}@{}}', r'\toprule',
        r'\textit{Observed allocation and feedback} \\']
    lines += [field(path,value)+r' \\' for path,value in observation_fields]
    lines += [r'\midrule', r'\textit{Agent rationale} \\',
        chr(96)*2+excerpt+chr(39)*2+r' \\',
        r'\midrule', r'\textit{Executed control} \\']
    lines += [field(path,value)+r' \\' for path,value in control_fields]
    lines += [r'\bottomrule', r'\end{tabular}', r'\endgroup', '']
    (out/'tables/case_decision.tex').write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    return facts




def draw_case(pair, start, end, out):
    configure()
    width = end-start
    first, last = start-2*width, end+width
    points = {method: [p for p in pair[method]['checkpoints'] if first <= p['trial'] <= last]
              for method in METHODS}
    if any([p['trial'] for p in rows] != list(range(first, last+1, width)) for rows in points.values()):
        raise ValueError('The local plot requires the complete saved checkpoint sequence')
    write_csv(out/'data/decision_case_plot.csv', [dict(method=m, **p) for m in METHODS for p in points[m]])
    fig, ax = plt.subplots(figsize=(3.24, 2.30))
    fig.subplots_adjust(left=.17, right=.96, bottom=.23, top=.89)
    rows = points['agent']
    ax.plot([p['trial'] for p in rows], [p['missed_priority'] for p in rows],
            **STYLE['agent'], linewidth=1.3, markersize=4,
            markerfacecolor='white', markeredgewidth=.9, zorder=3)
    lo, hi = min(p['missed_priority'] for p in rows), max(p['missed_priority'] for p in rows)
    ax.set(xlim=(first-.15*width,last+.15*width), ylim=(lo-.7,hi+.7),
           xlabel='iters', ylabel='Priority loss $L_1$')
    ax.set_xticks([p['trial'] for p in rows])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax.tick_params(length=2.5, width=.6, labelsize=8, pad=2)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y', color='#e4e4e4', linewidth=.45)
    ax.axvspan(start, end, color='#eeeeee', zorder=0)
    ax.axvline(start, color='#777777', linewidth=.8, linestyle=(0,(3,3)), zorder=1)
    ax.annotate('Decision', xy=(start,hi+.1), xytext=(start,hi+.48),
                ha='center', va='bottom', fontsize=9,
                arrowprops=dict(arrowstyle='-',color='#777777',linewidth=.6))
    for trial, offset in [(start,(-24,-18)),(end,(7,-18))]:
        point=next(p for p in rows if p['trial']==trial)
        ax.annotate(f"({point['missed_priority']:g}, {point['unassigned_count']:g})",
                    xy=(trial,point['missed_priority']),xytext=offset,
                    textcoords='offset points',fontsize=9,ha='left',va='top')
    export(fig, out/'figures/decision_case')
