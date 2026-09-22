"""Resumable Agent/Basic experiments with explicit data-use separation."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
from datetime import datetime, timezone

from sar_alloc.paths import PROJECT_ROOT
from sar_alloc.instance_io import load_instance_from_json
from sar_alloc.solution import AssignmentSolution
from experiments.agent import build_client, run_agent, sample_progress
from experiments.baselines import BaselineConfig, run_basic
from experiments._objective import SEARCH_OBJECTIVE_METRICS
from experiments.validate import recheck
from experiments.protocol import SPLITS, data_path, result_path


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')

def baseline_config(method,size):
    raw=dict(read(PROJECT_ROOT/'configs/baselines.json')[method][str(size)])
    raw.pop('acceptance',None)
    for key in ['remove_ratio_by_n_tasks']:
        if key in raw: raw[key]=tuple((int(k),v) for k,v in raw[key].items())
    return BaselineConfig(**raw)

def run_one(row,start,method,protocol,output):
    instance=load_instance_from_json(PROJECT_ROOT/row['instance_path'])
    initial=AssignmentSolution.from_dict(start['solution'])
    before=recheck(instance,start['solution'])
    seed=protocol['algorithm_seed'];trials=protocol['trials']
    common={'case_id':row['case_id'],'n_tasks':row['n_tasks'],'instance_seed':row['instance_seed'],
        'maturity':start['maturity'],'fraction':start['fraction'],'start_seed':start['start_seed'],
        'algorithm_seed':seed,'method':method,'trials':trials,'objective_order':protocol['objective'],
        'start_quality':{k:before[k] for k in SEARCH_OBJECTIVE_METRICS}}
    client=None
    for attempt in range(1,protocol['max_run_attempts']+1):
        attempt_path=output/'attempts'/f"{row['case_id']}__{start['maturity']}__{method}__{attempt}.json"
        if attempt_path.exists(): continue
        started=time.perf_counter()
        attempt_record={**common,'attempt':attempt,'started_utc':datetime.now(timezone.utc).isoformat()}
        try:
            if method=='agent':
                client=build_client('real',model=protocol['model'],reasoning=protocol['reasoning'],base_url=None,api_key=None)
                run,points,actions,decisions=run_agent(row,start,algorithm_seed=seed,trials=trials,client=client)
                method_config={'model':protocol['model'],'reasoning':protocol['reasoning'],
                    'temperature':protocol['temperature'],'max_tokens':protocol['max_tokens'],
                    'selectors':'random','feature_weights':0,'action_trials':100}
            else:
                config=baseline_config(method,row['n_tasks'])
                result=run_basic(instance=instance,initial_solution=initial,
                    trials=trials,rng_seed=seed,config=config)
                points,actions,decisions=result.progress,result.actions,result.decisions
                solution={'routes':result.solution.routes,'unassigned':sorted(result.solution.unassigned)}
                check=recheck(instance,solution)
                run={'final_solution':solution,'final_quality':{k:check[k] for k in SEARCH_OBJECTIVE_METRICS},
                    'hard_feasible':True,'trials_used':trials,'solver_time_sec':result.solver_time_sec,
                    'llm_time_sec':0.0,'llm_usage':{},'total_time_sec':time.perf_counter()-started}
                method_config=config.to_dict()
            check=recheck(instance,run['final_solution'])
            assert all(math.isclose(check[k],run['final_quality'][k],rel_tol=1e-12,abs_tol=1e-8) if k=='energy_total'
                       else check[k]==run['final_quality'][k] for k in SEARCH_OBJECTIVE_METRICS)
            assert run['trials_used']==trials
            checkpoints=sample_progress(points,tuple(protocol['checkpoints']),final_trial=trials)
            vectors=[tuple(p[k] for k in SEARCH_OBJECTIVE_METRICS) for p in checkpoints]
            assert all(a>=b for a,b in zip(vectors,vectors[1:]))
            assert all(math.isclose(vectors[-1][i],check[k],rel_tol=1e-12,abs_tol=1e-8) if k=='energy_total'
                       else vectors[-1][i]==check[k] for i,k in enumerate(SEARCH_OBJECTIVE_METRICS))
            record={**common,**run,'config':method_config,'attempt':attempt,'status':'complete',
                'checkpoints':checkpoints,'actions':actions,'decisions':decisions,'validation':check}
            write(output/'records'/f"{row['case_id']}__{start['maturity']}__{method}.json",record)
            write(attempt_path,{**attempt_record,'status':'complete','elapsed_sec':time.perf_counter()-started})
            print(row['case_id'],start['maturity'],method,run['final_quality'],flush=True)
            return True
        except Exception as exc:
            # Do not log request headers, credentials, endpoint URLs or SDK exception text.
            write(attempt_path,{**attempt_record,'status':'failed','error_type':type(exc).__name__,
                'cause_type':type(exc.__cause__).__name__ if exc.__cause__ else None,
                'http_status':getattr(exc.__cause__, 'status_code', None),
                'elapsed_sec':time.perf_counter()-started,'llm_usage':client.usage_summary() if client else {}})
            print('TECHNICAL FAILURE',row['case_id'],start['maturity'],method,attempt,type(exc).__name__,flush=True)
    return False

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=SPLITS,default='test')
    parser.add_argument('--methods',nargs='+',choices=['agent','basic'],default=['agent','basic'])
    parser.add_argument('--output',type=Path)
    parser.add_argument('--maturities',nargs='+')
    parser.add_argument('--sizes',nargs='+',type=int)
    parser.add_argument('--env-file',type=Path)
    args=parser.parse_args()
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file,override=False)
    protocol=read(PROJECT_ROOT/'configs/protocol.json')
    if args.split=='test' and protocol['status']!='frozen':
        raise RuntimeError('Formal experiments require the development protocol to be frozen')
    data=data_path(args.split,protocol);output=(args.output or result_path(args.split)).resolve()
    protected=PROJECT_ROOT/'results'
    if output==protected or protected in output.parents:
        raise ValueError('归档结果只读；请将重新运行的输出放到 outputs 或其他目录。')
    rows=read(data/'manifest.json');starts=read(data/'starts.json')
    failed=[]
    for row in rows:
        if args.sizes and row['n_tasks'] not in args.sizes: continue
        for start in [s for s in starts if s['case_id']==row['case_id']]:
            if args.maturities and start['maturity'] not in args.maturities: continue
            for method in args.methods:
                dest=output/'records'/f"{row['case_id']}__{start['maturity']}__{method}.json"
                if dest.exists():
                    assert read(dest)['status']=='complete'
                    continue
                if not run_one(row,start,method,protocol,output):failed.append(dest.name)
    print('Incomplete:',failed,flush=True)
    if failed: raise SystemExit(1)

if __name__=='__main__': main()
