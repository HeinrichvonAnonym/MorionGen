"""Fixed native initial-history sample, not selected by generated outcomes."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write

p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--clock',choices=['paused','realtime'],default='paused');args=p.parse_args()
base=ROOT/'develop_logs/test_runs/20261024_native_pilots'
template=json.loads((base/'primal_plan_exact600.json').read_text())
tasks=[{'family':0,'initial_seed':9000}]
tasks += [{'family':1,'initial_seed':9000+i} for i in range(5)]
tasks += [{'family':2,'initial_seed':9000+i} for i in range(6)]
jobs=[];reused=[]
pilot_sources={0:('20261024_primal_instrumented_pilot/queue','primal_initial0_seed20'),
               1:('20261024_native_pilots/primal_queue_exact600','primal_native_initial1_seed20'),
               6:('20261024_native_pilots/primal_queue_exact600','primal_native_initial2_seed20')}
if args.clock=='realtime':pilot_sources={0:('20261024_primal_realtime_pilot/queue','primal_initial0_seed20_realtime')}
for task_id,task in enumerate(tasks[:1] if args.pilot else tasks):
    for seed in ([20] if args.pilot else range(20,25)):
        if not args.pilot and seed==20 and task_id in pilot_sources:
            queue,key=pilot_sources[task_id]
            receipt=json.loads((ROOT/'develop_logs/test_runs'/queue/'state.json').read_text())['jobs'][key]
            if receipt['status']!='success':raise ValueError('Finish exact-length paired pilot before expansion')
            reused.append({'task':task_id,'seed':seed,'run':receipt['run_dir']});continue
        job=copy.deepcopy(template['jobs'][0])
        job.update(id=f'primal_initial{task_id}_seed{seed}',seed=seed,task_id=task_id,estimated_seconds=150)
        if args.clock=='realtime':job['id']+='_realtime';job['estimated_seconds']=240
        job['scene'].update(task,frames=600,initial_history=task['family'],clock=args.clock,
                            clock_scope='Unconditioned native continuation; no external update interface. Realtime paces native primitive denoising, without a controller; duplicate trajectories are not independent samples.')
        command=job['command']
        command[command.index('--initial-history')+1]=str(task['family'])
        command[command.index('--seed')+1]=str(seed)
        command += ['--initial-seed',str(task['initial_seed']),'--clock',args.clock]
        for ground in range(2):
            for reproj in range(2):
                for inertia in range(2):
                    stem=f'ground{ground}_reproject{reproj}_inertialize{inertia}'
                    job['required_outputs'] += ['{run_dir}/factorial/'+stem+suffix for suffix in ['_canonical.npz','_metrics.json']]
        jobs.append(job)
destination=ROOT/'develop_logs/test_runs'/('20261024_primal_instrumented_pilot' if args.pilot else '20261024_primal_matrix')/'plan.json'
if args.clock=='realtime':destination=ROOT/'develop_logs/test_runs'/('20261024_primal_realtime_pilot' if args.pilot else '20261024_primal_realtime_matrix')/'plan.json'
if destination.exists():raise ValueError('Refusing to replace registered plan')
template.update(jobs=jobs,devices=[0,1],tasks=tasks,reused=reused,report_reserve_seconds=14400,
    sampling_scope='One adapted stand, five short and six long model-generated histories; not 12 independent real motions')
json_write(destination,template);print(destination)
