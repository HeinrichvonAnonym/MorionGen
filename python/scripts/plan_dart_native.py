"""Twelve fixed goal geometries, paired action noise and both execution clocks."""
import copy
import json
import math
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.runtime import json_write
data=Path(load_config()['data_root']);base=ROOT/'develop_logs/test_runs/20261024_native_pilots'
plan=json.loads((base/'dart_plan.json').read_text());tasks=[];jobs=[];reused=[]
pilot_runs={('paused',0):'20261024_134600_dart_demo_8c3dc8b7',('paused',1):'20261024_134609_dart_demo_aa50bb74',
            ('realtime',0):'20261024_143554_dart_demo_ba5bd1b7',('realtime',1):'20261024_143757_dart_demo_4a32d87b'}
for distance in [1.,.75,1.5]:
    for degrees in [0,90,180,270]:
        angle=math.radians(degrees);c=math.cos(angle);s=math.sin(angle)
        tasks.append({'goal_start':[round(distance*c,8),round(distance*s,8)],
                      'goal_next':[round(-distance*c-.5*s,8),round(-distance*s+.5*c,8)]})
for task_id,task in enumerate(tasks):
    for seed in range(20,25):
        block=[]
        for clock in ['paused','realtime']:
            for condition,old in enumerate(plan['jobs']):
                if task_id==0 and seed==20:
                    run=ROOT/'develop_logs/test_runs'/pilot_runs[(clock,condition)]
                    if not (run/'motion.npz').is_file():raise ValueError('Missing accepted pilot')
                    reused.append({'task':task_id,'seed':seed,'clock':clock,'condition':condition,'run':str(run)})
                    continue
                job=copy.deepcopy(old)
                job.update(id=f'dart_task{task_id}_seed{seed}_{clock}_goal{condition}',seed=seed,task_id=task_id,estimated_seconds=60)
                job['scene'].update(task,clock=clock)
                command=job['command'];command[command.index('--seed')+1]=str(seed)
                command+=['--clock',clock,'--goal-start',*map(str,task['goal_start']),'--goal-next',*map(str,task['goal_next'])]
                job['required_outputs']+=['{run_dir}/trace.json']
                block.append(job)
        random.Random(task_id*1000+seed).shuffle(block);jobs.extend(block)
assets=data/'outputs/assets/dart-official'
identity=[str(p) for p in assets.rglob('*') if p.is_file() and p.suffix in ['.pth','.pt','.ckpt','.yaml']]
plan.update(jobs=jobs,tasks=tasks,reused=reused,identity_files=identity,report_reserve_seconds=14400,
    independence='Scene-level paired inference. Clocks are execution modes, never additional independent samples when trajectories coincide.')
destination=ROOT/'develop_logs/test_runs/20261024_dart_matrix/plan.json'
if destination.exists():raise ValueError('Refusing to replace fixed matrix')
json_write(destination,plan);print(json.dumps({'path':str(destination),'jobs':len(jobs),'reused':len(reused)}))
