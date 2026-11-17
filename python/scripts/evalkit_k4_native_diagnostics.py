"""D-only native FK diagnostics before interpolation, filtering or reference scoring."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.motion import load,Motion
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.evalkit.k.root import compensate
from motiogen.evalkit.k.timebase import native_clock
from motiogen.evalkit.k.scoring import unit,summary,DIRECTIONS
from motiogen.evalkit.k.resampling import audit_model
from motiogen.degradation.metrics import points,angle
from evalkit_k4_score import serializable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('fit','input','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--method',required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Keep diagnostics immutable')
    d=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json');file=a.fit/'coordinates_refined_native.mot'
    if not file.exists():file=a.fit/'ik/ik.mot'
    time,labels,q=biomech.read_storage(file);time,clock=native_clock(time,d.times)
    model=a.fit/'model.osim';before,names,consistent=model_fk(model,labels,q,direct_couplers=True,return_coordinates=True)
    _,root=compensate(model,d.times,labels,consistent,d);f=Motion(root['after'],names,d.times,'native_model_fk')
    dp,fp=points(d),points(f);diagnostics={}
    for n,(start,end) in {**DIRECTIONS,'pelvis_line':('left_hip','right_hip')}.items():
        v,u=dp[end]-dp[start],fp[end]-fp[start];error=angle(v,u)
        rate=np.linalg.norm(np.diff(unit(u)-unit(v),axis=0)/np.diff(d.times)[:,None],axis=1)*180/np.pi
        diagnostics[n]={'error_deg':error.tolist(),'input_step_deg':angle(v[:-1],v[1:]).tolist(),'output_step_deg':angle(u[:-1],u[1:]).tolist(),
                        'added_direction_rate_deg_s':rate.tolist(),'summary':summary(error),'rate_summary':summary(rate)}
    audit=audit_model(model);bounds={}
    for row in audit['joints'].values():
        for n,c in row['coordinates'].items():
            if n not in labels or len(c['range_rad'])!=2:continue
            x=np.deg2rad(q[:,labels.index(n)]);low,high=c['range_rad']
            bounds[n]={'range_deg':np.rad2deg([low,high]).tolist(),'native_minmax_deg':[float(np.rad2deg(x.min())),float(np.rad2deg(x.max()))],
                       'within_0_1deg_of_limit_frames':np.flatnonzero(np.minimum(abs(x-low),abs(x-high))<np.deg2rad(.1)).tolist()}
    result={'source':str(a.fit),'native_file':str(file),'native_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
            'inputs':d.provenance['inputs'],'times':d.times.tolist(),'clock':clock,'directions':diagnostics,'coordinate_bounds':bounds,
            'root_shift_m':root['shift_m'].tolist(),'scope':'Native model FK vs D, including identical input root compensation. No R, reference lengths, interpolation or score-based fitting selection.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
