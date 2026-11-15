import sys,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
sys.path[:0]=['utils','utils/motiogen']
import opensim as osim,biomech
from motiogen.evalkit.k import load
root=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k_arms_20261116/ablation_v2/both')
rows=json.loads(Path('config/evalkit_k_validation_20261116/regression_summary.json').read_text());results={}
for row in rows:
 method=row['method']
 if method not in ('motionlcm','omnicontrol','kimodo'):continue
 motion=load(row['source']);p,names,fps=motion.legacy();work=root/method/'work';times,labels,data=biomech.read_storage(work/'ik/ik.mot')
 m=osim.Model(str(work/'model.osim'));state=m.initSystem();cs=m.getCoordinateSet();out={}
 for side,s in [('left','l'),('right','r')]:
  coords=[cs.get(n+'_'+s) for n in ['arm_flex','arm_add','arm_rot','elbow_flex']];inds=[labels.index(c.getName()) for c in coords];bounds=np.array([[c.getRangeMin(),c.getRangeMax()] for c in coords]).T
  target=p[:,[names.index(side+'_'+n) for n in ['shoulder','elbow','wrist']]];vec=np.diff(target,axis=1);vec/=np.linalg.norm(vec,axis=-1,keepdims=True);err=[];pos=[];directions=[]
  for i,t in enumerate(times):
   biomech._set(m,state,labels,data[i]);j=int(round(t*fps));x0=np.clip(np.deg2rad(data[i,inds]),bounds[0]+1e-6,bounds[1]-1e-6)
   def residual(x):
    for c,v in zip(coords,x):c.setValue(state,v,False)
    m.realizePosition(state);a=np.array([biomech._child_origin(m,state,n+'_'+s) for n in ['acromial','elbow','radius_hand']]);v=np.diff(a,axis=0);v/=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.r_[(v-vec[j]).ravel(),.001*(x-x0)]
   sol=least_squares(residual,x0,bounds=bounds,max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8)
   if np.linalg.norm(sol.fun[:6])>.05:
    for seed in ([0.,-.5,0.,1.],[2.,-.5,0.,1.],[-2.,-.5,0.,1.]):
     candidate=least_squares(residual,np.clip(seed,bounds[0]+1e-6,bounds[1]-1e-6),bounds=bounds,max_nfev=80)
     if np.linalg.norm(candidate.fun)<np.linalg.norm(sol.fun):sol=candidate
   residual(sol.x);a=np.array([biomech._child_origin(m,state,n+'_'+s) for n in ['acromial','elbow','radius_hand']]);v=np.diff(a,axis=0);v/=np.linalg.norm(v,axis=-1,keepdims=True)
   angle=lambda v:np.rad2deg(np.arccos(np.clip(np.dot(*v),-1,1)))
   err.append(angle(v)-angle(vec[j]));pos.append(np.linalg.norm(a-target[j],axis=-1));directions.append(np.rad2deg(np.arccos(np.clip((v*vec[j]).sum(-1),-1,1))))
  out[side]={'p95':float(np.percentile(abs(np.array(err)),95)),'position_p95':np.percentile(pos,95,axis=0).tolist(),'direction_p95':np.percentile(directions,95,axis=0).tolist()}
 results[method]=out
Path('develop_logs/test_runs/20261116_k_arms/directions_multistart.json').write_text(json.dumps(results,indent=2));print(results)
