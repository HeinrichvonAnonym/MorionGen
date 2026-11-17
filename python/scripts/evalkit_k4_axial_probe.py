"""Input-only axial reachability witnesses on an unchanged model.

Fit the existing pelvis/back coordinates to observed hips and shoulders while
omitting limb objectives. A successful witness proves axial feasibility at the
tested frame; a failed local solve is not a proof of model unreachability.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from scipy.optimize import least_squares
import opensim as osim
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.timebase import native_clock
from motiogen.evalkit.k.refinement import bounded_jacobian
from motiogen.degradation.metrics import angle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','input','protocol','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--method',required=True)
    args=parser.parse_args();osim.Logger.setLevelString('error')
    protocol=json.loads(args.protocol.read_text());d=load(args.input,args.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    if json.loads((args.fit/'receipt.json').read_text())['key']['inputs']!=d.provenance['inputs']:
        raise ValueError('Frozen D identity changed')
    file=args.fit/'coordinates_refined_native.mot';t,labels,q=biomech.read_storage(file);native_clock(t,d.times)
    names=['left_hip','right_hip','left_shoulder','right_shoulder']
    joints=['hip_l','hip_r','acromial_l','acromial_r'];target=d.positions[:,[d.names.index(n) for n in names]]
    model=osim.Model(str(args.fit/'model.osim'));state=model.initSystem();cs=model.getCoordinateSet()
    active=['pelvis_tilt','pelvis_list','pelvis_rotation','lumbar_extension','lumbar_bending','lumbar_rotation']
    handles=[cs.get(n) for n in active];lo=np.array([c.getRangeMin() for c in handles]);hi=np.array([c.getRangeMax() for c in handles])
    if any(c.getDefaultLocked() or c.isDependent(state) for c in handles):raise ValueError('Axial probe requires these independent rotational coordinates')
    cols=[labels.index(n) for n in active];frames=[model.getJointSet().get(n).getChildFrame() for n in joints]
    def local(p):return p-p[:2].mean(0)
    def vectors(p):return np.stack([p[1]-p[0],p[3]-p[2],p[2:].mean(0)-p[:2].mean(0)])
    def unit(v):return v/np.linalg.norm(v,axis=-1,keepdims=True)
    args.output.mkdir(parents=True,exist_ok=False);rows=[]
    for i in protocol['frame_indices']:
        if not 0<=i<len(q) or not np.isfinite(target[i]).all():raise ValueError('Invalid predeclared frame')
        goal=target[i];gu=unit(vectors(goal));gp=local(goal);chord=2*np.sin(np.deg2rad(5)/2)
        def evaluate(x):
            biomech._set(model,state,labels,q[i])
            for c,v in zip(handles,x):c.setValue(state,float(v),False)
            model.realizePosition(state)
            return np.array([[f.getPositionInGround(state).get(k) for k in range(3)] for f in frames])
        def residual(x):
            p=evaluate(x)
            return np.r_[((unit(vectors(p))-gu)/chord).ravel(),((local(p)-gp)/.05).ravel()]
        for seed in protocol['seeds']:
            if seed=='native':start=np.deg2rad(q[i,cols])
            elif seed=='neutral':
                start=np.array([c.getDefaultValue() for c in handles]);start[2]=np.deg2rad(q[i,cols[2]])
            else:raise ValueError('Undeclared seed')
            fit=least_squares(residual,np.clip(start,lo+1e-9,hi-1e-9),bounds=(lo,hi),
                jac=lambda x:bounded_jacobian(residual,x,lo,hi,1e-5),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10)
            p=evaluate(fit.x);errors=angle(vectors(p),vectors(goal));positions=np.linalg.norm(local(p)-gp,axis=-1)*100
            rows.append(dict(frame=i,seed=seed,coordinates_deg=np.rad2deg(fit.x).tolist(),directions_deg=dict(zip(['pelvis','shoulders','trunk'],errors.tolist())),
                local_positions_cm=dict(zip(names,positions.tolist())),nfev=fit.nfev,optimality=fit.optimality,loss=float(fit.fun@fit.fun),
                feasible_witness=bool(max(errors)<5 and max(positions)<5)))
    result=dict(protocol=protocol,rows=rows,model_sha256=hashlib.sha256((args.fit/'model.osim').read_bytes()).hexdigest(),
                native_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),inputs=d.provenance['inputs'],active_coordinates=active,
                implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='D-only isolated axial witnesses on fixed geometry/ranges; limb goals omitted. Failure is not a global infeasibility proof; full-body and temporal validation remain separate.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows))


if __name__=='__main__':main()
