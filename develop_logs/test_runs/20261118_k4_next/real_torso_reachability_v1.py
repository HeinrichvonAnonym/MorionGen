"""Input-only fixed-torso relaxation and conditional local solve; no R access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ap=argparse.ArgumentParser()
ap.add_argument('--code-root',type=Path,required=True)
ap.add_argument('--fit',type=Path,required=True)
ap.add_argument('--input',type=Path,required=True)
ap.add_argument('--method',required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args()
sys.path[:0]=[str(a.code_root/'utils'),str(a.code_root/'utils/motiogen')]
import numpy as np
import opensim as osim
from scipy.optimize import least_squares
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.constraints import couplers,synchronize
from motiogen.evalkit.k.refinement import bounded_jacobian
from motiogen.degradation.metrics import points,angle,stats

osim.Logger.setLevelString('error')
motion=load(a.input,a.method,config=a.code_root/'config/evalkit_k4_20261117/input.json')
D=points(motion)
times,labels,q=biomech.read_storage(a.fit/'coordinates_refined_native.mot')
np.testing.assert_allclose(times-times[0],motion.times-motion.times[0],atol=1e-7)
model=osim.Model(str(a.fit/'model.osim'));state=model.initSystem();bindings=couplers(model,state)
names=['pelvis_tilt','pelvis_list','pelvis_rotation',*biomech.LUMBAR]
handles=[model.getCoordinateSet().get(n) for n in names]
indices=[list(labels).index(n) for n in names]
lo=np.array([h.getRangeMin() for h in handles]);hi=np.array([h.getRangeMax() for h in handles])
joints=['hip_l','hip_r','acromial_l','acromial_r','back']
def fk(frame,x=None):
    biomech._set(model,state,labels,q[frame])
    if x is not None:
        for h,v in zip(handles,x):h.setValue(state,float(v),False)
    synchronize(state,bindings);model.realizePosition(state)
    p=np.array([biomech._child_origin(model,state,n) for n in joints])
    parent=model.getJointSet().get('back').getParentFrame().getPositionInGround(state)
    parent=np.array([parent.get(i) for i in range(3)])
    return p,parent
def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
def feature(p):
    hip=p[:2].mean(0);shoulder=p[2:4].mean(0)
    return np.array([shoulder-hip,p[3]-p[2],p[1]-p[0]])
G=np.stack([D[n] for n in ['left_hip','right_hip','left_shoulder','right_shoulder']],1)
native=[];lengths=[];pivots=[]
for i in range(len(q)):
    p,parent=fk(i);native.append(p)
    lengths.append([np.linalg.norm(parent-p[:2].mean(0)),np.linalg.norm(p[2:4].mean(0)-p[4]),np.linalg.norm(p[3]-p[2])])
    pivots.append(np.linalg.norm(parent-p[4]))
native=np.array(native);lengths=np.array(lengths)
assert np.max(np.ptp(lengths,axis=0))<1e-8,'Fixed-link relaxation invalid: moving translation'
assert max(pivots)<1e-8,'Lumbar parent/child pivot not coincident'
aa,bb,width=np.median(lengths,axis=0)
span=np.linalg.norm(G[:,2:4].mean(1)-G[:,:2].mean(1),axis=1)
radial=np.maximum.reduce([np.zeros(len(span)),abs(aa-bb)-span,span-aa-bb])
width_bound=abs(np.linalg.norm(G[:,3]-G[:,2],axis=1)-width)/2
native_error=np.linalg.norm((native[:,2:4]-native[:,:2].mean(1)[:,None])-(G[:,2:4]-G[:,:2].mean(1)[:,None]),axis=-1).max(1)
frames=sorted(set(np.linspace(0,len(q)-1,3,dtype=int).tolist()+np.argsort(native_error)[-3:].tolist()))
def measures(p,g):
    return dict(shoulder_cm=(100*np.linalg.norm((p[2:4]-p[:2].mean(0))-(g[2:4]-g[:2].mean(0)),axis=-1)).tolist(),
                directions_deg=angle(feature(p),feature(g)).tolist())
rows=[]
for frame in frames:
    g=G[frame];seed=np.clip(np.deg2rad(q[frame,indices]),lo+1e-8,hi-1e-8)
    def residual(x):
        p,_=fk(frame,x)
        return np.concatenate([((p[2:4]-p[:2].mean(0))-(g[2:4]-g[:2].mean(0))).ravel()/.05,
                               ((unit(feature(p))-unit(feature(g)))/np.deg2rad(5)).ravel()])
    attempts=[]
    for name,start in [('native',seed),('neutral',np.clip(np.zeros(len(seed)),lo+1e-8,hi-1e-8))]:
        r=least_squares(residual,start,bounds=(lo,hi),jac=lambda x:bounded_jacobian(residual,x,lo,hi,1e-5),max_nfev=200)
        p,_=fk(frame,r.x)
        attempts.append(dict(seed=name,cost=float(r.cost),success=bool(r.success),nfev=r.nfev,optimality=float(r.optimality),angles_deg=np.rad2deg(r.x).tolist(),metrics=measures(p,g)))
    rows.append(dict(frame=int(frame),time_s=float(motion.times[frame]),native=measures(native[frame],g),attempts=attempts))
out=dict(scope='Actual D-only fidelity diagnostic. No R; sampled local solves are not a global optimality proof or qualified temporal candidate.',
    frames=len(q),selection='Three uniform frames plus three largest D shoulder-relative errors',coordinate_names=names,
    fixed_geometry=dict(hip_to_lumbar_m=float(aa),lumbar_to_shoulder_m=float(bb),shoulder_width_m=float(width),max_link_variation_m=float(np.max(np.ptp(lengths,axis=0))),max_pivot_gap_m=float(max(pivots))),
    relaxed_lower_bounds=dict(scope='Free link orientations, fixed geometry and hip-midpoint root anchoring; radial shoulder-midpoint and per-frame maximum unilateral shoulder error bounds. Ignores coordinate constraints.',
        shoulder_mid_radial_cm=stats(100*radial),max_shoulder_width_cm=stats(100*width_bound)),rows=rows,
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.fit/'model.osim',a.fit/'coordinates_refined_native.mot',a.input,Path(__file__)]})
a.output.parent.mkdir(parents=True,exist_ok=True)
with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
print(json.dumps(out,indent=2))
