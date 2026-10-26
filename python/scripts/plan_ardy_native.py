"""Fixed paths and native history/correction factorial, with separate clocks."""
import argparse
import copy
import json
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
base=ROOT/'develop_logs/test_runs/20261024_native_pilots'
plan=json.loads((base/'ardy_plan_cfg_corrected.json').read_text());jobs=[];tasks=[];reused=[]
if not args.pilot:
    cache_check=json.loads((ROOT/'develop_logs/test_runs/20261024_ardy_cache_check/equivalence.json').read_text())
    if cache_check['status']!='passed':raise ValueError('Exact prompt-cache equivalence required before batch optimization')
for before,after in [(.4,-.2),(.2,-.1),(.6,-.3)]:
    for direction in [[0.,1.],[1.,0.],[0.,-1.],[-1.,0.]]:
        tasks.append({'direction_xz':direction,'speed_before_m_s':before,'speed_after_m_s':after})
for task_id,task in enumerate(tasks[:1] if args.pilot else tasks):
    for seed in ([20] if args.pilot else range(20,25)):
        block=[]
        for clock in (['realtime'] if args.pilot else ['paused','realtime']):
            for condition,old in enumerate(plan['jobs']):
                if not args.pilot and task_id==0 and seed==20:
                    queue=base/'ardy_queue_cfg_corrected_fair' if clock=='paused' else ROOT/'develop_logs/test_runs/20261024_ardy_realtime_pilot/queue'
                    state=json.loads((queue/'state.json').read_text())
                    key=old['id'] if clock=='paused' else f'ardy_task0_seed20_realtime_condition{condition}'
                    row=state['jobs'][key]
                    if row['status']!='success':raise ValueError(f'Complete paired pilot before batch: {key}')
                    reused.append({'task':0,'seed':20,'clock':clock,'condition':condition,'run':row['run_dir']});continue
                job=copy.deepcopy(old);job.update(id=f'ardy_task{task_id}_seed{seed}_{clock}_condition{condition}',seed=seed,task_id=task_id,estimated_seconds=240)
                job['scene'].update(native_root_path=task,clock=clock)
                if not args.pilot:job['config']['_frozen_prompt_cache']=True;job['estimated_seconds']=60
                if clock=='realtime':job['required_outputs']+=['{run_dir}/clock_trace.json']
                block.append(job)
        random.Random(task_id*1000+seed).shuffle(block);jobs.extend(block)
destination=ROOT/'develop_logs/test_runs'/('20261024_ardy_realtime_pilot' if args.pilot else '20261024_ardy_matrix')/'plan.json'
if destination.exists():raise ValueError('Refusing to replace fixed plan')
plan.update(jobs=jobs,tasks=tasks,reused=reused,devices=[0] if args.pilot else [0,1],report_reserve_seconds=14400,
    task_scope='Predeclared geometric root paths plus forward/backward text; not a new unknown root goal at 5s. History is model output, not physical execution.',
    independence='12 path scenarios; paired seeds and clocks. Identical clock trajectories are not independent observations.')
json_write(destination,plan);print(json.dumps({'path':str(destination),'jobs':len(jobs)}))
