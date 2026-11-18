"""D-only diagnostics of a saved joint-trajectory objective and linearized steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr
from scipy.optimize import lsq_linear
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.trajectory import problem


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',required=True)
    p.add_argument('--coordinates',default='coordinates_trajectory_candidate.mot',choices=['coordinates_trajectory_candidate.mot','coordinates_before_trajectory.mot'])
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a new diagnostic output')
    saved=json.loads((a.fit/'receipt.json').read_text())
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    if saved['key']['inputs']!=motion.provenance['inputs']:raise ValueError('Input identity changed')
    settings=saved['key']['config']['trajectory_refinement']
    times,labels,q=biomech.read_storage(a.fit/a.coordinates)
    observed,names,_=motion.legacy();observed=np.array(observed,copy=True)
    for name in motion.provenance.get('degradation',{}).get('prior_joints',[]):
        name=name.removesuffix('_body')
        if name in names:observed[:,list(names).index(name)]=np.nan
    obj,x,_,cols,active=problem(a.fit/'model.osim',labels,q,observed,names,times,settings)
    residual=obj.residual(x);jac=obj.jacobian(x);gradient=np.asarray(jac.T@residual).ravel()
    lower=np.tile(obj.lower,len(q));upper=np.tile(obj.upper,len(q))
    norms=np.sqrt(np.asarray(jac.power(2).sum(0)).ravel())
    rows=[]
    for k in np.argsort(abs(gradient))[-12:][::-1]:
        rows.append(dict(frame=int(k//len(active)),coordinate=active[k%len(active)],value_deg=float(np.rad2deg(x[k])),
                         gradient=float(gradient[k]),column_norm=float(norms[k]),
                         at_outward_bound=bool((x[k]-lower[k]<1e-6 and gradient[k]>0) or (upper[k]-x[k]<1e-6 and gradient[k]<0))))
    controls=[]
    for label,scaling in [('unscaled',np.ones(len(x))),('column_scaled',1/np.maximum(norms,1)),('bounded_column_scaled',1/np.maximum(norms,1))]:
        system=jac@sparse.diags(scaling)
        if label=='bounded_column_scaled':
            solution=lsq_linear(system,-residual,bounds=((lower-x)/scaling,(upper-x)/scaling),
                                tol=1e-6,max_iter=50,lsmr_tol=1e-8,lsmr_maxiter=500)
            result=(solution.x,solution.status,solution.nit,np.linalg.norm(system@solution.x+residual),None,None,None)
        else:result=lsmr(system,-residual,atol=1e-8,btol=1e-8,maxiter=500)
        step=scaling*result[0];trials=[]
        for alpha in (1.,.5,.25,.125,.0625,.03125,.015625):
            candidate=np.clip(x+alpha*step,lower+1e-9,upper-1e-9)
            value=obj.residual(candidate)
            trials.append(dict(alpha=alpha,squared_cost=float(value@value),max_coordinate_step_deg=float(np.rad2deg(abs(candidate-x)).max())))
        controls.append(dict(name=label,iterations=int(result[2]),stop_code=int(result[1]),
                             linear_residual_norm=float(result[3]),estimated_condition=float(result[6]) if result[6] is not None else None,trials=trials))
    result=dict(source_fit=str(a.fit),source_coordinates=a.coordinates,initial_squared_cost=float(residual@residual),
                active_coordinates=active,gradient_max_abs=float(abs(gradient).max()),top_gradients=rows,
                column_norm_percentiles=np.percentile(norms,[0,25,50,75,100]).tolist(),linearized_controls=controls,
                source_sha256={n:hashlib.sha256((a.fit/n).read_bytes()).hexdigest() for n in ['model.osim',a.coordinates,'receipt.json']},
                implementation_sha256={str(n.relative_to(ROOT)):hashlib.sha256(n.read_bytes()).hexdigest() for n in [Path(__file__),ROOT/'utils/motiogen/evalkit/k/trajectory.py',ROOT/'utils/motiogen/evalkit/k/trajectory_objective.py']},
                scope='Local linearized convergence diagnostic; clipped trial steps are not fitted candidates or global reachability evidence. D-only, no R access, saved model/coordinates unchanged.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ['initial_squared_cost','gradient_max_abs','column_norm_percentiles','linearized_controls']},indent=2))


if __name__=='__main__':main()
