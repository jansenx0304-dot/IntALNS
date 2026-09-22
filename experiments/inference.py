"""The existing exact instance-clustered sign-flip test, counted by integer DP."""
from collections import Counter,defaultdict
from math import lcm
from statistics import mean

def clustered_sign_flip(pairs):
    clusters=defaultdict(list)
    for row in pairs:
        clusters[row['case_id']].append({'W':1,'T':0,'L':-1}[row['outcome']])
    if not clusters:raise ValueError('No paired instances')
    denominator=lcm(*(len(v) for v in clusters.values()))
    integer_scores={key:sum(v)*(denominator//len(v)) for key,v in clusters.items()}
    distribution=Counter({0:1})
    for value in integer_scores.values():
        updated=Counter()
        for total,count in distribution.items():
            updated[total+value]+=count
            updated[total-value]+=count
        distribution=updated
    observed_integer=sum(integer_scores.values())
    permutations=2**len(clusters)
    assert sum(distribution.values())==permutations
    extreme=sum(count for total,count in distribution.items() if abs(total)>=abs(observed_integer))
    p=extreme/permutations
    return {'method':'Exact two-sided paired sign-flip on instance mean win-minus-loss; all nested starts flip together',
        'instance_scores':{key:mean(v) for key,v in clusters.items()},'independent_instances':len(clusters),
        'observed_sum':observed_integer/denominator,'permutations':permutations,'p_two_sided':p,
        'agent_superiority_supported':observed_integer>0 and p<.05,
        'calculation':'Integer dynamic counting of the same exact sign assignments; no Monte Carlo or alternative test',
        'limitation':'One algorithm seed and one model realization per condition. Nested starts are dependent. Retrospective cohort reuse and sequential method selection limit confirmatory interpretation.'}
