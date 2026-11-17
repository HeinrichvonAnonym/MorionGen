"""Fixed-model knee geometry reachability from declared coordinate range and D only."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.config import ROOT
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.constraints import couplers,synchronize
from motiogen.degradation.metrics import points,angle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    motion=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json');target=points(motion)
    model_path=a.fit/'model.osim';model=osim.Model(str(model_path));state=model.initSystem();bindings=couplers(model,state)
    result={'source_model':str(model_path),'source_model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),
            'input_hashes':motion.provenance['inputs'],'rule':'Dense full-range scan and bounded scalar extrema; sampled numerical reachability evidence, not formal interval proof',
            'scope':'Geometric angle hip-knee-ankle for this fixed model; no physiological-angle equivalence; D only; R comparison remains independent',
            'new_holdout_uses':0,'sides':{}}
    data={}
    for side,s in [('left','l'),('right','r')]:
        c=model.updCoordinateSet().get('knee_angle_'+s);lo,hi=c.getRangeMin(),c.getRangeMax()
        def flex(q):
            c.setValue(state,float(q),False);synchronize(state,bindings);model.realizePosition(state)
            hip=biomech._child_origin(model,state,'hip_'+s);knee=biomech._child_origin(model,state,'walker_knee_'+s);ankle=biomech._child_origin(model,state,'ankle_'+s)
            return float(180-angle(hip-knee,ankle-knee))
        grid=np.linspace(lo,hi,2001);values=np.array([flex(q) for q in grid]);candidates=[(float(v),float(q)) for v,q in zip(values,grid)]
        turns=np.flatnonzero(np.diff(np.sign(np.diff(values)))!=0)+1
        for i in turns:
            for sign in (-1,1):
                opt=minimize_scalar(lambda x:sign*flex(x),bounds=(grid[i-1],grid[i+1]),method='bounded')
                candidates.append((flex(opt.x),float(opt.x)))
        minimum=min(candidates);maximum=max(candidates)
        observed=180-angle(target[side+'_hip']-target[side+'_knee'],target[side+'_ankle']-target[side+'_knee'])
        lower_bound=np.maximum(np.maximum(minimum[0]-observed,observed-maximum[0]),0)
        # Rigid transforms upstream and ankle rotation cannot change this metric.
        rng=np.random.default_rng(914);invariance=[];qmid=(lo+hi)/2;expected=flex(qmid)
        for _ in range(20):
            for name in ('pelvis_tilt','pelvis_list','pelvis_rotation','hip_flexion_'+s,'hip_adduction_'+s,'hip_rotation_'+s,'ankle_angle_'+s):
                coord=model.updCoordinateSet().get(name);a0,b0=coord.getRangeMin(),coord.getRangeMax()
                if name=='pelvis_rotation':a0,b0=-np.pi,np.pi
                coord.setValue(state,float(rng.uniform(a0,b0)),False)
            invariance.append(abs(flex(qmid)-expected))
        result['sides'][side]={'coordinate_range_deg':np.rad2deg([lo,hi]).tolist(),'model_geometric_range_deg':[minimum[0],maximum[0]],
            'max_at_coordinate_deg':float(np.rad2deg(maximum[1])), 'max_adjacent_scan_change_deg':float(np.max(abs(np.diff(values)))),
            'observed_D_range_deg':[float(np.nanmin(observed)),float(np.nanmax(observed))],
            'unavoidable_error_sampled_deg':{'median':float(np.nanmedian(lower_bound)),'p95':float(np.nanpercentile(lower_bound,95))},
            'frames_over_5deg_lower_bound':int(np.sum(lower_bound>5)),'observed_frames':int(np.isfinite(observed).sum()),
            'upstream_ankle_invariance_max_deg':max(invariance),'scope':'Fixed geometry/domain only; changing geometry or knee domain requires a separate evaluation'}
        data[side+'_q_deg']=np.rad2deg(grid);data[side+'_model_flex_deg']=values;data[side+'_D_flex_deg']=observed;data[side+'_lower_bound_deg']=lower_bound
    np.savez_compressed(a.output/'reachability.npz',**data)
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['sides'],indent=2))


if __name__=='__main__':main()
