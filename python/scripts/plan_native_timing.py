"""Cold versus warm inference timing, separate from scientific sample counts."""
import copy
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
p=argparse.ArgumentParser();p.add_argument('--missing-omni',action='store_true');args=p.parse_args()
base=ROOT/'develop_logs/test_runs/20261024_native_pilots';jobs=[];identities=[]
for filename in (['plan.json'] if args.missing_omni else ['plan.json','chois_plan.json']):
    plan=json.loads((base/filename).read_text());identities+=plan.get('identity_files',[])
    for old in plan['jobs']:
        if old['method'] not in ['motionlcm','omnicontrol','chois']:continue
        if args.missing_omni and not(old['method']=='omnicontrol' and old['config']['_intervention']['realism']==0):continue
        job=copy.deepcopy(old);job['id']='timing_'+job['id'];job['timeout_seconds']=1800;job['estimated_seconds']=900
        job.setdefault('config',{})['timing']={'warmup':1,'repetitions':3}
        if job['method']=='chois':
            job['config']['_benchmark_native_sampler']=3
            job['required_outputs']+=['{run_dir}/native_sampler_timing.json']
        else:job['command']+=['--benchmark']
        jobs.append(job)
destination=ROOT/'develop_logs/test_runs/20261024_native_timing/plan.json'
if args.missing_omni:destination=destination.with_name('plan_missing_omni.json')
if destination.exists():raise ValueError('Timing plan already registered')
json_write(destination,{'devices':[1],'deadline':json.loads((ROOT/'config/survey_revision.json').read_text())['experiment_deadline_at'],
    'report_reserve_seconds':14400,'jobs':jobs,'identity_files':sorted(set(identities)),
    'scope':'Timing repetitions are not independent scientific motion samples; one warmup and three measured repeats per condition'})
print(json.dumps({'plan':str(destination),'jobs':len(jobs)}))
