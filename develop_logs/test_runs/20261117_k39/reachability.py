import sys,json
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts']
import numpy as np
import opensim as osim
import biomech
from evalkit_k35_legs import BASE,atomic
from motiogen.evalkit.k.geometry import unit
osim.Logger.setLevelString('error');out={}
for variant in ['g0s1','g1s1']:
 p=BASE/'evalkit_k39_joint_pilot_20261117/074/kimodo'/variant;z=np.load(p/'fit.npz');names=list(z['labels']);model=osim.Model(str(p/'kinematic_model.osim'));state=model.initSystem();result={}
 for side in ['l','r']:
  q=z['coordinates'][0].copy();values=np.linspace(0,175,1751);angles=[]
  for v in values:
   q[names.index('knee_angle_'+side)]=v;biomech._set(model,state,names,q);points=np.array([biomech._child_origin(model,state,j+'_'+side) for j in ['hip','walker_knee','ankle']]);u=unit(np.diff(points,axis=0));angles.append(float(np.degrees(np.arccos(np.clip(u[0]@u[1],-1,1)))))
  k=int(np.argmax(angles));result[side]={'range_deg':[0,175],'grid_deg':.1,'maximum_geometric_knee_deg':angles[k],'at_coordinate_deg':float(values[k]),'source':'copied model FK, fixed other coordinates; no reference enters sweep'}
 out[variant]=result
atomic(Path('develop_logs/test_runs/20261117_k39/reachability.json'),out);print(out)
