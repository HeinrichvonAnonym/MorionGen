import sys,json
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path.cwd()/'utils'),str(Path.cwd()/'utils/motiogen')]
import opensim as osim
import biomech
from motiogen.evalkit.k.scoring import angles
from motiogen.evalkit.k.geometry import unit
osim.Logger.setLevelString('error')
b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k34_20261117/fits/solver/010/kimodo/fit')
z=np.load(b/'kinematics.npz');labels=list(z['labels']);names=list(z['names']);model=osim.Model(str(b/'model.osim'));state=model.initSystem();cs=model.getCoordinateSet()
ix=[names.index('left_'+n) for n in ['hip','knee','ankle']]
p=z['source'][:,ix];theta=np.degrees(np.arccos(np.clip((unit(p[:,1]-p[:,0])*unit(p[:,2]-p[:,1])).sum(-1),-1,1)));i=int(theta.argmax());target=unit(np.diff(p[i],axis=0));row=z['coordinates'][i].copy();results=[]
# Signed sweeps are diagnostic only. Negative coordinates are outside the
# production asset domain and never enter selected fitting outputs.
for knee in [-30,-25,-20,-10,0,10,20,25,30]:
 row[labels.index('knee_angle_l')]=knee;biomech._set(model,state,labels,row)
 actual=np.array([biomech._child_origin(model,state,j) for j in ['hip_l','walker_knee_l','ankle_l']]);dirs=unit(np.diff(actual,axis=0))
 results.append({'requested_knee':knee,'actual_knee_deg':float(np.degrees(cs.get('knee_angle_l').getValue(state))),'direction_errors_deg':np.degrees(np.arccos(np.clip((dirs*target).sum(-1),-1,1))).tolist()})
report={'frame':i,'time':float(z['times'][i]),'input_geometric_knee_angle':float(theta[i]),'sweep':results,'scope':'fixed other coordinates; diagnostic response, not a fit or anatomical validation'}
Path('develop_logs/test_runs/20261117_k34/knee_direction.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
