import sys,json
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts',str(Path(__file__).parent)]
import numpy as np
from evalkit_k39_prepare import BASE,atomic,interp
from motiogen.evalkit.k.trustworthy import guard,fidelity
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.temporal import temporal
out=BASE/'evalkit_k39_angular_probe_20261117';records=[]
for p in out.glob('*/*/*/probe.json'):
 key,method,geo=p.parts[-4:-1];b=np.load(BASE/f'evalkit_k39_ablation_20261117/{key}/{method}/g0s0/fit.npz');ref=np.load(BASE/f'evalkit_k39_20261117/inputs/{key}/{method}/common.npz');t=ref['times'];R=ref['R'];D=ref['D'];B=common(interp(b['times'],b['fitted'],t),list(b['names']))
 for name,result in json.loads(p.read_text()).items():
  c=np.load(p.parent/(name+'.npz'));F=common(interp(c['times'],c['fitted'],t),list(c['names']));_,g=guard(R,B,F,t);bad=[]
  for scope,source in [('input',D),('reference',R)]:
   ta=temporal(source,B,t);tb=temporal(source,F,t)
   for metric in ta:
    for field,budget in [('step_max_deg',5.),('velocity_error',10.)]:
     a,v=ta[metric][field],tb[metric][field]
     if isinstance(a,dict):a,v=a['p95_abs'],v['p95_abs']
     if a is not None and v is not None and v>a+budget:bad.append(scope+':'+metric+':'+field)
  records.append({'window':key,'method':method,'geometry':geo,'variant':name,'input_pass':result['accepted'],'reference_failures':g['failures'],'temporal_failures':bad,'original_reference_fidelity':fidelity(R,F,t)})
atomic(out/'acceptance.json',records)
for r in records:print(r['window'],r['geometry'],r['variant'],r['input_pass'],r['reference_failures'],r['temporal_failures'])
