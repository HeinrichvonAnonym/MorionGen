"""Independent original-R validation of a D-only fixed-model range audit."""
import argparse,json,hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from motiogen.evalkit.k.config import ROOT
from motiogen.evalkit.k.motion import Motion,load
from motiogen.degradation.reference import sample
from motiogen.degradation.metrics import points,angle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('audit','reference','input','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--method',required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Preserve independent score snapshots')
    limits=json.loads((a.audit/'result.json').read_text());d=load(a.input,a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    with np.load(a.reference) as z:r=Motion(z['positions'],list(z['names']),z['times'],'R')
    rp=points(sample(r,d.times),True);dp=points(d)
    report={'source_reference':str(a.reference),'source_reference_sha256':hashlib.sha256(a.reference.read_bytes()).hexdigest(),
            'clock':'All D timestamps; original R sampled only in independent scorer','sides':{}}
    for side in ['left','right']:
        def flex(p):return 180-angle(p[side+'_hip']-p[side+'_knee'],p[side+'_ankle']-p[side+'_knee'])
        rf,df=flex(rp),flex(dp);lo,hi=limits['sides'][side]['model_geometric_range_deg'];bound=np.maximum(np.maximum(lo-rf,rf-hi),0)
        report['sides'][side]={'R_min_max_deg':[float(np.nanmin(rf)),float(np.nanmax(rf))],
                             'D-R_absolute_p95_deg':float(np.nanpercentile(abs(df-rf),95)),
                             'F-R_lower_bound_p95_deg':float(np.nanpercentile(bound,95)),
                             'R_frames_over_5deg_lower_bound':int((bound>5).sum()),'denominator':len(rf),
                             'observed_frames':int(np.isfinite(rf).sum())}
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['sides'],indent=2))


if __name__=='__main__':main()
