import sys,json,pathlib,numpy as np
r=pathlib.Path('/home/heinrich/MorionGen');sys.path[:0]=[str(r/'utils'),str(r/'utils/motiogen')]
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.trajectory import problem
b=pathlib.Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
results=[]
for win in ('073','077'):
 for geom in ('prior','chain'):
  fit=b/f'closure_trajectory_geometry_20261118_v1/fits/balanced_{geom}/{win}/motionlcm';receipt=json.loads((fit/'receipt.json').read_text());print(receipt['key']['inputs'])
  settings=receipt['key']['config']['trajectory_refinement']
  # Frozen input path is explicit in receipt provenance.
  inp=next(iter(receipt['key']['inputs']))
  print('inputkey',inp)
  t,labels,q=biomech.read_storage(fit/'coordinates_before_trajectory.mot')
  native=b/f'closure_trajectory_geometry_20261118_v1/fits/native_{geom}/{win}/motionlcm'
  _,ln,qn=biomech.read_storage(native/'coordinates_refined_native.mot')
  import opensim as osim
  m=osim.Model(str(fit/'model.osim'));s=m.initSystem();cs=m.getCoordinateSet();changes=[]
  for i,n in enumerate(labels):
   if cs.contains(n) and cs.get(n).getMotionType()==1 and not cs.get(n).getDefaultLocked() and not cs.get(n).isDependent(s):
    c=cs.get(n);lower,upper=np.rad2deg([c.getRangeMin(),c.getRangeMax()]);clip=np.clip(q[:,i],lower,upper);delta=abs(clip-q[:,i])
    if delta.max()>1e-5:changes.append(dict(name=n,max_clip_deg=float(delta.max()),frames=int((delta>1e-5).sum()),value_range=[float(q[:,i].min()),float(q[:,i].max())],bounds=[lower,upper]))
  results.append(dict(window=win,geometry=geom,seed_vs_native_max_deg=float(abs(q-qn).max()),clipped=changes))
print(json.dumps(results,indent=2));(b/'trajectory_seed_clipping_audit_v1.json').write_text(json.dumps(results,indent=2)+'\n')
