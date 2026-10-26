"""Preregister 12 spatial tasks x 5 seeds, retaining complete native factorial blocks."""
import copy
import argparse
import json
import math
import random
import sys
from datetime import datetime,timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config,environment_root
from motiogen.runtime import json_write,checksum
from motiogen.assets import find_asset

cfg=load_config()
p=argparse.ArgumentParser()
p.add_argument('--trajectory',choices=['pilot','batch'])
args=p.parse_args()
if args.trajectory:
    specification_path=ROOT/'config/native_trajectory_experiments.json'
    spec=json.loads(specification_path.read_text())
    folder=ROOT/'develop_logs/test_runs/20261024_omnicontrol_trajectory'
    original=json.loads((ROOT/'develop_logs/test_runs/20261024_control_matrix/plan.json').read_text())
    tasks=[];jobs=[];reused=[]
    for side in spec['sides']:
        sign=1 if side=='right' else -1
        for family in spec['paths']:
            for amplitude in spec['amplitudes_m']:
                constraints=[]
                for frame_index,frame in enumerate(spec['anchor_frames']):
                    u=frame_index/(len(spec['anchor_frames'])-1);angle=(u-.5)*math.pi
                    center=spec['center_m'];x=sign*center[0];y=center[1];z=center[2]
                    if family=='line':y+=amplitude*(2*u-1)
                    elif family=='horizontal_arc':x+=sign*amplitude*math.sin(angle);y+=amplitude*(math.cos(angle)-1)
                    elif family=='vertical_arc':y+=amplitude*(math.cos(angle)-1);z+=amplitude*math.sin(angle)
                    constraints.append({'time_seconds':frame/spec['fps'],'joint':side+'_wrist',
                                        'position_m':[round(x,8),round(y,8),round(z,8)]})
                scene={'name':'native_wrist_trajectory','kind':'position','duration_seconds':spec['frames']/spec['fps'],
                    'prompt':f'A person slowly moves their {side} hand in front of their body.',
                    'constraints':constraints,'trajectory_family':family,'trajectory_amplitude_m':amplitude,
                    'description':spec['metric_scope']}
                task=len(tasks);tasks.append({'id':task,'scene':scene})
                if args.trajectory=='pilot' and task!=0:continue
                for seed in spec['seeds']:
                    if args.trajectory=='pilot' and seed!=20:continue
                    block=[]
                    for condition in spec['conditions']:
                        job_id=f'omnicontrol_trajectory_task{task}_seed{seed}_'+''.join(str(v) for v in condition.values())
                        if args.trajectory=='batch' and task==0 and seed==20:
                            state=json.loads((folder/'pilot_queue/state.json').read_text())['jobs'][job_id]
                            audit=json.loads((folder/'analysis.json').read_text())
                            if state['status']!='success' or not audit['pilot_passed']:
                                raise ValueError('Complete and scientifically audit all four trajectory pilot conditions before batch admission')
                            reused.append({'task':task,'seed':seed,'method':'omnicontrol','condition':condition,
                                'run':state['run_dir'],'motion_sha256':checksum(Path(state['run_dir'])/'motion.npz')})
                            continue
                        block.append({'id':job_id,'method':'omnicontrol','scene':scene,'seed':seed,'task_id':task,
                            'config':{'_intervention':condition},
                            'command':[str(environment_root(cfg,'omnicontrol')/'bin/python'),str(ROOT/'python/scripts/baseline_worker.py'),'--run-dir','{run_dir}'],
                            'required_outputs':['{run_dir}/motion.npz','{run_dir}/metrics.json','{run_dir}/intervention.json'],
                            'estimated_seconds':100,'timeout_seconds':900})
                    random.Random(task*1000+seed).shuffle(block);jobs.extend(block)
    destination=folder/(args.trajectory+'_plan.json')
    if destination.exists():raise ValueError('Refusing to overwrite registered trajectory plan')
    result={k:original[k] for k in ['devices','deadline','report_reserve_seconds','identity_files']}
    result['identity_files']=result['identity_files']+[str(specification_path)]
    result.update(jobs=jobs,tasks=tasks,reused=reused,registered_at=datetime.now(timezone.utc).isoformat(),
        domain='multi-anchor wrist trajectory',specification=spec,specification_sha256=checksum(specification_path),
        scope='Coverage amendment; original single-point matrix remains unchanged. Native 196-frame OmniControl, nine timestamped hints. No added trajectory controller.')
    json_write(destination,result)
    print(json.dumps({'plan':str(destination),'jobs':len(jobs),'reused':len(reused),'tasks':len(tasks)}))
    raise SystemExit(0)
base=ROOT/'develop_logs/test_runs/20261024_native_pilots'
pilot=json.loads((base/'audited_pilots.json').read_text())
scene0=json.loads((ROOT/'config/scenarios/reach.json').read_text())
jobs=[]; reused=[]; tasks=[]
for side in ['right','left']:
    for forward in [.6,.4]:
        for height in [1.2,.9,1.5]:
            scene=copy.deepcopy(scene0)
            task=len(tasks)
            scene['prompt']=f'A person reaches forward with the {side} hand.'
            scene['constraints'][0].update(joint=f'{side}_wrist',position_m=[.35 if side=='right' else -.35,forward,height])
            tasks.append({'id':task,'scene':scene})
            for seed in range(20,25):
                block=[]
                for method,conditions in [('motionlcm',[{'control_residual':v} for v in [0,1]]),
                    ('omnicontrol',[{'realism':r,'spatial':s} for r in [0,1] for s in [0,1]])]:
                    for condition in conditions:
                        matched=[r for r in pilot['rows'] if r['method']==method and r['seed']==seed
                                 and r['scene']==scene and r['condition']==condition and r['finite']]
                        if matched:
                            reused.append({'task':task,'seed':seed,'method':method,'condition':condition,
                                           'run':matched[0]['run'],'motion_sha256':matched[0]['motion_sha256']})
                            continue
                        block.append({'id':f'{method}_task{task}_seed{seed}_'+''.join(str(v) for v in condition.values()),
                            'method':method,'scene':scene,'seed':seed,'task_id':task,
                            'config':{'_intervention':condition},
                            'command':[str(environment_root(cfg,method)/'bin/python'),str(ROOT/'python/scripts/baseline_worker.py'),
                                       '--run-dir','{run_dir}'],'required_outputs':['{run_dir}/motion.npz','{run_dir}/metrics.json','{run_dir}/intervention.json'],
                            'estimated_seconds':150 if method=='omnicontrol' else 60,'timeout_seconds':900})
                random.Random(1000*task+seed).shuffle(block)
                jobs.extend(block)
evidence=json.loads((ROOT/'config/inference_evidence.json').read_text())
identity=[]
for method in ['motionlcm','omnicontrol']:
    entry=evidence[method]; entry=entry.get('profiles',{}).get('full',entry)
    identity.extend(str(find_asset(cfg,method,f['baseline_asset'])) for f in entry['files'] if 'baseline_asset' in f)
revision=json.loads((ROOT/'config/survey_revision.json').read_text())
plan={'devices':[0,1],'deadline':revision['experiment_deadline_at'],'report_reserve_seconds':14400,
      'jobs':jobs,'identity_files':identity,'reused':reused,'tasks':tasks,
      'sampling':'12 fixed world-space target tasks; 5 seeds each; order randomized within block; no selection by outcomes',
      'inference':'Scene-level paired intervals; no cross-method rank; infeasible targets retained as failures'}
destination=ROOT/'develop_logs/test_runs/20261024_control_matrix/plan.json'
if destination.exists(): raise ValueError('Preregistered plan already exists; use a new explicit revision')
json_write(destination,plan)
print(json.dumps({'plan':str(destination),'new_jobs':len(jobs),'reused':len(reused),'tasks':len(tasks)}))
