import sys,json
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts',str(Path(__file__).parent)]
import numpy as np
from evalkit_k39_prepare import BASE,OUT,read,atomic
from motiogen.evalkit.k.trustworthy import fit_variant,guard,fidelity
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.limb_transport import transport
from motiogen.evalkit.k.root_alignment import align_fitted_root
import opensim as osim
osim.Logger.setLevelString('error')
cfg=read('config/evalkit_k39_20261117/experiment_config.json');cfg['pose_optimizer']['direction_weight']=1/(2*np.sin(np.deg2rad(.5)));cfg['pose_optimizer']['max_nfev']=100
output=BASE/'evalkit_k39_angular_probe_replay_20261117';output.mkdir(exist_ok=True);atomic(output/'config.json',cfg)
rows=read(OUT/'exports.json')
for method,key in [('motionlcm','031'),('kimodo','074'),('kimodo','010')]:
 row=next(r for r in rows if r['method']==method and r['window_key']==key)
 old=BASE/f'evalkit_k39_ablation_20261117/{key}/{method}/g0s0';z=np.load(old/'fit.npz');t=z['times'];labels=list(z['labels']);names=list(z['names']);D=common(z['source'],names);B=common(z['fitted'],names)
 for geometry in [False,True]:
  p=output/key/method/('g1' if geometry else 'g0');fit_variant(row['source'],p,cfg,geometry,True,nominal_work=p.parent/'nominal')
  c=np.load(p/'fit.npz');F=c['fitted'];records={}
  for limbs in [(),('legs',),('legs','arms')]:
   if limbs:
    q,receipt=transport(old/'kinematic_model.osim',p/'kinematic_model.osim',labels,z['coordinates'],c['coordinates'],limbs)
    q,F,valid,shift,root=align_fitted_root(p/'kinematic_model.osim',t,labels,q,c['source'],names)
   else:q=c['coordinates']
   name='_'.join(limbs) or 'raw';np.savez_compressed(p/(name+'.npz'),times=t,coordinates=q,fitted=F,names=names,source=c['source'],labels=labels)
   ok,check=guard(D,B,common(F,names),t);records[name]={'accepted':ok,'checks':check,'fidelity':fidelity(D,common(F,names),t)}
   print(method,key,geometry,name,ok,check['failures'],flush=True)
  atomic(p/'probe.json',records)
