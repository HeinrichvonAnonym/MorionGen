"""Native human PhysX context-source contrast, twelve fixed reach geometries."""
import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
base=ROOT/'develop_logs/test_runs'
plan=json.loads((base/'20261024_closd_pilot/plan.json').read_text());jobs=[];tasks=[];reused=[]
for distance in [2.,1.5,2.5]:
    for degrees in [0,90,180,270]:
        angle=math.radians(degrees);c=math.cos(angle);s=math.sin(angle)
        tasks.append({'goal_start':[round(distance*c,8),round(distance*s,8)],
                      'goal_next':[round(-.75*distance*c-s,8),round(-.75*distance*s+c,8)]})
for task_id,task in enumerate(tasks[:1] if args.pilot else tasks):
    for seed in ([20] if args.pilot else range(20,25)):
        block=[]
        for clock in (['realtime'] if args.pilot else ['paused','realtime']):
            for old in plan['jobs']:
                source=old['command'][old['command'].index('--context-source')+1]
                if not args.pilot and task_id==0 and seed==20:
                    queue='20261024_closd_pilot/queue_loader_v3' if clock=='paused' else '20261024_closd_realtime_pilot/queue'
                    key=old['id'] if clock=='paused' else f'closd_task0_seed20_realtime_{source}'
                    row=json.loads((base/queue/'state.json').read_text())['jobs'][key]
                    if row['status']!='success':raise ValueError('Complete paired native pilot before bulk admission')
                    reused.append({'task':0,'seed':20,'clock':clock,'condition':source,'run':row['run_dir']});continue
                job=copy.deepcopy(old);job.update(id=f'closd_task{task_id}_seed{seed}_{clock}_{source}',seed=seed,task_id=task_id,estimated_seconds=180)
                job['scene'].update(task,clock=clock)
                command=job['command'];command[command.index('--seed')+1]=str(seed)
                command+=['--clock',clock,'--goal-start',*map(str,task['goal_start']),'--goal-next',*map(str,task['goal_next'])]
                job['required_outputs']+=['{run_dir}/trace.json','{run_dir}/resolved_config.json'];block.append(job)
        random.Random(task_id*1000+seed).shuffle(block);jobs.extend(block)
destination=base/('20261024_closd_realtime_pilot' if args.pilot else '20261024_closd_matrix')/'plan.json'
if destination.exists():raise ValueError('Refusing to replace registered matrix')
plan.update(jobs=jobs,tasks=tasks,reused=reused,
    scope='Official pretrained human DiP + PHC + PhysX; fixed absolute root goals; only DiP prefix source switches. Native tracker execution feedback remains enabled.')
json_write(destination,plan);print(destination)
