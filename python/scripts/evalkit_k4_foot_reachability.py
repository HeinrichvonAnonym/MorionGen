"""D-only sampled foot reachability: fixed proximal pose versus released leg.

This is a diagnostic, not a global impossibility proof or a runtime candidate.
It preserves both model assets and fits and never reads the original reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import opensim as osim
from scipy.optimize import least_squares
import biomech
from motiogen.evalkit.k.config import ROOT
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.constraints import couplers, synchronize
from motiogen.evalkit.k.refinement import bounded_jacobian
from motiogen.degradation.metrics import points, angle


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for n in ('fit','domain-fit','input','output'):
        ap.add_argument('--'+n, type=Path, required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--side', choices=['left','right'], required=True)
    a = ap.parse_args()
    osim.Logger.setLevelString('error')
    motion = load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    targets = points(motion)
    times, labels, q = biomech.read_storage(a.fit/'coordinates_refined_native.mot')
    assert len(times) == len(motion.times)
    np.testing.assert_allclose(times-times[0], motion.times-motion.times[0], atol=1e-7)
    side = a.side
    suffix = 'l' if side=='left' else 'r'
    goal = np.stack([targets[side+'_'+n] for n in ['hip','knee','ankle','foot']],1)
    finite = np.isfinite(goal).all(axis=(1,2)) & (np.linalg.norm(np.diff(goal,axis=1),axis=-1)>1e-8).all(axis=1)
    assert finite.any()
    coordinate_names = [x+'_'+suffix for x in ['hip_flexion','hip_adduction','hip_rotation','knee_angle','ankle_angle','subtalar_angle']]
    indices = [list(labels).index(n) for n in coordinate_names]
    model_paths = {'current':a.fit/'model.osim','expanded_domain':a.domain_fit/'model.osim'}
    rows = []
    frames = None
    for policy, path in model_paths.items():
        model = osim.Model(str(path));state = model.initSystem();bindings = couplers(model,state)
        handles = [model.updCoordinateSet().get(n) for n in coordinate_names]
        lower = np.array([h.getRangeMin() for h in handles]);upper = np.array([h.getRangeMax() for h in handles])
        joints = [n+'_'+suffix for n in ['hip','walker_knee','ankle','mtp']]
        def fk(frame,x):
            biomech._set(model,state,labels,q[frame])
            for h,value in zip(handles,x):h.setValue(state,float(value),False)
            synchronize(state,bindings);model.realizePosition(state)
            return np.array([biomech._child_origin(model,state,n) for n in joints])
        def metrics(p,g):
            return dict(directions_deg=angle(np.diff(p,axis=0),np.diff(g,axis=0)).tolist(),
                        relative_positions_cm=(100*np.linalg.norm((p-p[0])-(g-g[0]),axis=-1)).tolist())
        if frames is None:
            errors = np.full(len(q),-np.inf)
            for i in np.flatnonzero(finite):
                p=fk(i,np.deg2rad(q[i,indices]));errors[i]=angle(p[3]-p[2],goal[i,3]-goal[i,2])
            valid=np.flatnonzero(finite)
            frames=sorted(set(valid[np.linspace(0,len(valid)-1,min(3,len(valid))).astype(int)].tolist()+np.argsort(errors)[-min(3,len(valid)):].tolist()))
        for frame in frames:
            seed=np.clip(np.deg2rad(q[frame,indices]),lower+1e-9,upper-1e-9)
            g=goal[frame];target_direction=unit(g[3]-g[2])
            def fixed_residual(x):
                full=seed.copy();full[-2:]=x;p=fk(frame,full)
                return unit(p[3]-p[2])-target_direction
            grid=[np.linspace(lower[j],upper[j],41) for j in [-2,-1]]
            best=min((np.array([x,y]) for x in grid[0] for y in grid[1]),key=lambda x:float(np.dot(fixed_residual(x),fixed_residual(x))))
            grid_error=float(angle(unit(fixed_residual(best)+target_direction),target_direction))
            opt=least_squares(fixed_residual,np.clip(best,lower[-2:]+1e-9,upper[-2:]-1e-9),bounds=(lower[-2:],upper[-2:]),
                jac=lambda x:bounded_jacobian(fixed_residual,x,lower[-2:],upper[-2:],1e-5),max_nfev=100)
            fixed=seed.copy();fixed[-2:]=opt.x
            def released_residual(x):
                p=fk(frame,x)
                return np.concatenate(((unit(np.diff(p,axis=0))-unit(np.diff(g,axis=0))).ravel()/np.deg2rad(5),
                    ((p[1:]-p[0])-(g[1:]-g[0])).ravel()/.05))
            candidates=[]
            for start in (seed,fixed):
                result=least_squares(released_residual,start,bounds=(lower,upper),
                    jac=lambda x:bounded_jacobian(released_residual,x,lower,upper,1e-5),max_nfev=150)
                candidates.append(result)
            selected=min(candidates,key=lambda r:float(np.dot(r.fun,r.fun)))
            rows.append(dict(policy=policy,frame=int(frame),time_s=float(motion.times[frame]),
                current=metrics(fk(frame,seed),g),fixed_proximal=metrics(fk(frame,fixed),g),released_proximal=metrics(fk(frame,selected.x),g),
                grid_best_foot_angle_deg=grid_error,seed_deg=np.rad2deg(seed).tolist(),fixed_deg=np.rad2deg(fixed).tolist(),
                released_deg=np.rad2deg(selected.x).tolist(),bounds_deg=np.rad2deg([lower,upper]).tolist(),
                released_success=bool(selected.success),released_cost=float(selected.cost)))
    data=dict(scope='D-only sampled geometric reachability; current proximal pose retained in fixed scan; released fit changes only six ipsilateral independent leg coordinates. No global lower-bound claim.',
              selection='Three uniformly spaced valid frames plus three largest current D foot-direction residuals, same frames in both models',
              input_hashes=motion.provenance['inputs'],method=a.method,side=side,coordinate_names=coordinate_names,
              model_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in model_paths.items()},rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(data,f,indent=2,allow_nan=False)
    print(json.dumps({'frames':frames,'rows':len(rows),'output':str(a.output)}))


if __name__=='__main__':main()
