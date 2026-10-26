"""Register pacing decomposition replays; not additional capability samples."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
base=ROOT/'develop_logs/test_runs'
source=json.loads((base/'20261024_primal_realtime_matrix/plan.json').read_text())
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
jobs=[]
for clock in ['paused','realtime']:
    job=copy.deepcopy(source['jobs'][0]);job['id']='primal_timing_'+clock;job['seed']=20
    job['scene']['clock']=clock
    job['command'][job['command'].index('--seed')+1]='20'
    job['command'][job['command'].index('--clock')+1]=clock
    job['reference_run']=next(r['run'] for r in analysis['rows']
        if r['method']=='primal' and r['task']==0 and r['seed']==20 and r['clock']==clock)
    jobs.append(job)
destination=base/'20261024_primal_timing/plan.json'
if destination.exists():raise ValueError('Timing plan already registered; do not overwrite identity')
json_write(destination,{'devices':[0,1],'deadline':source['deadline'],
    'report_reserve_seconds':14400,'identity_files':source['identity_files'],'jobs':jobs,
    'scope':'Standing seed20, eight conditions, both clocks. Direct pacing-time decomposition and output-equivalence replay only; excluded from scientific matrix counts.'})
print(json.dumps({'plan':str(destination),'jobs':len(jobs)}))
