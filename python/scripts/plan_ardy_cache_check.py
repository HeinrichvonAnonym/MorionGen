"""Two identical native runs: cache creation then exact embedding reuse."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
base=ROOT/'develop_logs/test_runs/20261024_native_pilots'
plan=json.loads((base/'ardy_plan_cfg_corrected.json').read_text());jobs=[]
for repeat in range(2):
    job=copy.deepcopy(plan['jobs'][0]);job['id']=f'ardy_exact_prompt_cache_repeat{repeat}'
    job['config']['_frozen_prompt_cache']=True;job['required_outputs']+=['{run_dir}/prompt_cache.json']
    job['estimated_seconds']=300;jobs.append(job)
plan.update(jobs=jobs,report_reserve_seconds=14400)
destination=ROOT/'develop_logs/test_runs/20261024_ardy_cache_check/plan.json'
if destination.exists():raise ValueError('Cache calibration already registered')
json_write(destination,plan);print(destination)
