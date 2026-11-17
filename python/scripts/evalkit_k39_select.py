"""Replay the frozen 18-pair ablation through source-only selection and reference scoring."""
import sys,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
import numpy as np
from evalkit_k39_prepare import BASE,OUT,read,atomic
from motiogen.evalkit.k.selection import selection_guard
from motiogen.evalkit.k.trustworthy import fidelity
from motiogen.evalkit.k.bones import common
from evalkit_k39_regression import score
cfg=read('config/evalkit_k39_20261117/runtime_config.json');destination=BASE/'evalkit_k39_policy_replay_20261117';results=[]
for row in read(OUT/'exports.json'):
 if not row['selected']:continue
 key,method=row['window_key'],row['method'];work=destination/key/method;work.mkdir(parents=True,exist_ok=True)
 src=BASE/f'evalkit_k39_ablation_20261117/{key}/{method}';rule=cfg['per_method'][method];variant=('g1' if rule['geometry'] else 'g0')+('s1' if rule['optimized'] else 's0')
 baseline=src/'g0s0';raw=src/variant;z=np.load(baseline/'fit.npz');D=common(z['source'],list(z['names']));B=common(z['fitted'],list(z['names']));t=z['times']
 options=[('baseline',baseline,baseline/'kinematic_model.osim'),('candidate',raw,raw/'kinematic_model.osim')]
 if rule['optimized']:
  for limbs in ['legs','legs_arms']:
   path=BASE/f'evalkit_k39_transport_ablation_20261117/{key}/{method}/{variant}_{limbs}'
   if path.exists():options.append(('transport_'+limbs,path,raw/'kinematic_model.osim'))
 eligible=[];checks={}
 for name,path,model in options:
  q=np.load(path/'fit.npz');F=common(q['fitted'],list(q['names']));ok,check=selection_guard(D,B,F,t)
  metrics=fidelity(D,F,t);objective=sum(v['p95_abs']/5 for v in metrics['angles'].values())+sum(v['p95_abs']/.05 for v in metrics['positions'].values())
  checks[name]=dict(check,accepted=ok)
  if name=='baseline' or (ok and not (name=='candidate' and rule.get('require_limb_transport',False))):eligible.append((objective,name,path,model))
  # Match runtime policy: transport is attempted only after raw rejection.
  if name=='candidate' and ok and not rule.get('require_limb_transport',False):break
 chosen=min(eligible);selection={'selected':chosen[1],'checks':checks,'scope':'replay unchanged frozen pose candidates with new whole-window selection; no ref in selection'}
 for name,path,model in [('baseline',baseline,baseline/'kinematic_model.osim'),('candidate',raw,raw/'kinematic_model.osim'),('selected',chosen[2],chosen[3])]:
  target=work/name;target.mkdir(exist_ok=True);shutil.copy2(path/'fit.npz',target/'fit.npz');shutil.copy2(model,target/'kinematic_model.osim')
  q=np.load(target/'fit.npz')
  if 'pre_root_fitted' not in q:
   # Diagnostic transport archives already contain root-aligned output; do not invent pre-root values.
   values={k:q[k] for k in q.files};values['pre_root_fitted']=np.full_like(q['fitted'],np.nan);np.savez_compressed(target/'fit.npz',**values)
 result={'row':{k:row[k] for k in ['id','method','group_id','subset','window_key','source_window_s']},'status':'computed','selection':selection,**score(row,work,selection)}
 atomic(work/'result.json',result);results.append(str(work/'result.json'));print(key,method,chosen[1],result['reference_regressions'],flush=True)
atomic(destination/'index.json',results)
