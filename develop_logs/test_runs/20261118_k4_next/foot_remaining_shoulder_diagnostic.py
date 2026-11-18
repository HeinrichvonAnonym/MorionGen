import sys,pathlib,json,numpy as np
r=pathlib.Path('/home/heinrich/MorionGen');sys.path[:0]=[str(r/'utils'),str(r/'utils/motiogen')]
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.degradation.metrics import points,angle
import biomech,opensim as osim
osim.Logger.setLevelString('error');b=pathlib.Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');fit=b/'closure_position_foot_20261118_v1/fits/fixed_endpoint/002/motionlcm';motion=load(b/'runtime_inputs/002/motionlcm/candidate/motion.npz','motionlcm',config=r/'config/evalkit_k4_20261117/input.json');d=points(motion);t,labels,q=biomech.read_storage(fit/'coordinates_refined_native.mot');p,names=model_fk(fit/'model.osim',labels,q,direct_couplers=True);print(type(p),getattr(p,'shape',None),str(names)[:500])
idx={n:i for i,n in enumerate(names)};fv=p[:,idx['right_shoulder']]-p[:,idx['left_shoulder']];dv=d['right_shoulder']-d['left_shoulder'];err=angle(fv,dv);model=osim.Model(str(fit/'model.osim'));state=model.initSystem();cs=model.getCoordinateSet();rows=[];ref=json.loads((fit/'refinement.json').read_text())
for i in np.argsort(err)[-8:][::-1]:
 limits=[]
 for j,n in enumerate(labels):
  c=cs.get(n)
  if c.getMotionType()!=1 or c.getDefaultLocked() or c.isDependent(state):continue
  lo,hi=np.rad2deg([c.getRangeMin(),c.getRangeMax()]);dist=min(q[i,j]-lo,hi-q[i,j])
  if dist<.1:limits.append(dict(coordinate=n,value_deg=float(q[i,j]),distance_to_bound_deg=float(dist)))
 rows.append(dict(frame=int(i),time=float(t[i]),shoulder_error_deg=float(err[i]),limits=limits,refinement=ref['frames'][int(i)]))
result=dict(D_shoulder_error_percentiles=np.percentile(err,[50,95,100]).tolist(),worst=rows,scope='Input-only native FK diagnosis; no R or scoring feedback in solver')
(b/'foot_remaining_shoulder_diagnostic_v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
