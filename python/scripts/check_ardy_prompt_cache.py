"""Require cold/cache-hit and original-pilot motion equivalence before bulk reuse."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
base=ROOT/'develop_logs/test_runs/20261024_ardy_cache_check'
state=json.loads((base/'queue/state.json').read_text());runs=[];cache=[]
for i in range(2):
    job=state['jobs'][f'ardy_exact_prompt_cache_repeat{i}']
    if job['status']!='success':raise RuntimeError('Cache calibration still in progress: '+job['status'])
    path=Path(job['run_dir']);runs.append(path);cache.append(json.loads((path/'prompt_cache.json').read_text()))
if not cache[1]['hit'] or cache[0]['identity']!=cache[1]['identity'] or cache[0]['tensor_sha256']!=cache[1]['tensor_sha256']:
    raise ValueError('Cache hit or identity contract failed')
original=ROOT/'develop_logs/test_runs/20261024_143850_ardy_demo_c4774c8b'
reference=np.load(original/'motion.npz')['joints'];checks=[]
for run in runs:
    actual=np.load(run/'motion.npz')['joints']
    equal=actual.shape==reference.shape and np.array_equal(reference,actual)
    checks.append({'run':str(run),'identical_to_uncached_pilot':equal,
                   'max_absolute_difference_m':float(abs(reference-actual).max()) if actual.shape==reference.shape else None})
if not all(r['identical_to_uncached_pilot'] for r in checks):raise ValueError('Caching changed generated motion: '+str(checks))
json_write(base/'equivalence.json',{'status':'passed','reference':str(original),'checks':checks,'cache':cache,
    'scope':'Same fixed prompts, ARDY checkpoint, task, initial state and seed; exact stored-coordinate equality. Other prompts require separate identity-bound encodings.'})
print(json.dumps(checks))
