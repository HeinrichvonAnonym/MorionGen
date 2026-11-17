"""Independent R/D/F scoring plus a fixed common-baseline time intersection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import evalkit_k4_score as base
ROOT=Path(__file__).resolve().parents[2]
from motiogen.evalkit.k.motion import Motion,load
from motiogen.degradation.reference import sample
from motiogen.evalkit.k.scoring import score,summary,unit
from motiogen.degradation.metrics import points,angle
from motiogen.evalkit.k.timebase import supported,BOUNDARY_ROUNDOFF_S


def at(motion,times):
    valid=supported(times,motion.times[0],motion.times[-1])
    p=np.full((len(times),len(motion.names),3),np.nan)
    p[valid]=sample(motion,np.clip(times[valid],motion.times[0],motion.times[-1])).positions
    return Motion(p,motion.names,times,motion.method)


def jump_mask(times,d,events):
    mask=np.zeros(len(times),bool)
    for event in events:
        for left,right in zip(d.times[:-1],d.times[1:]):
            if max(left,event['start_s'])<min(right,event['end_s']):
                mask|=(times>=left-(right-left))&(times<=right+(right-left))
    return mask


def add_pelvis_diagnostic(result,r,d,f,mask):
    raw={'R':points(r,True),'D':points(d),'F':points(f)}
    vectors={k:p['right_hip']-p['left_hip'] for k,p in raw.items()}
    strata={'full':np.ones(len(r.times),bool),'continuous':~mask,'source_jump_tolerance':mask}
    for comparison,x,y in [('D-R','R','D'),('F-D','D','F'),('F-R','R','F')]:
        a,b=vectors[x],vectors[y];error=angle(a,b)
        sa,sb=angle(a[:-1],a[1:]),angle(b[:-1],b[1:])
        rate=(np.diff(unit(b),axis=0)-np.diff(unit(a),axis=0))/np.diff(r.times)[:,None]
        for stratum,m in strata.items():
            block=result['comparisons'][comparison][stratum];steps=m[1:]&m[:-1]
            block['directions_deg']['pelvis_line']=summary(error[m],[5,10,20])
            block['temporal']['pelvis_line']={'reference_step_deg':summary(sa[steps]),'output_step_deg':summary(sb[steps]),
                'signed_step_change_deg':summary((sb-sa)[steps],signed=(sb-sa)[steps]),
                'direction_rate_vector_error_deg_s':summary(np.linalg.norm(rate[steps],axis=-1)*180/np.pi)}
    result['added_diagnostic']='Left/right hip-line direction and continuity; versioned ablation metric, not an anatomical pelvis frame. Original baseline reports remain unchanged.'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fits','baseline','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--window',required=True);p.add_argument('--method',required=True);a=p.parse_args()
    run=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
    records=json.loads((run/'artifact_verification.json').read_text())['records']
    artifact={(r['window'],r['method']):r for r in records}[(a.window,a.method)]
    source=run/'inputs'/a.window/a.method/'candidate'
    d=load(source/'factorial/reference.npz' if a.method=='primal' else source,a.method)
    with np.load(run/'references'/f'{a.window}.npz') as z:r=Motion(z['positions'],list(z['names']),z['times'],'reference')
    with np.load(a.baseline/a.window/a.method/'fit.npz') as z:bt=z['times']
    times=d.times[supported(d.times,bt[0],bt[-1])]
    with np.load(a.fits/a.window/a.method/'fk.npz') as z:f=Motion(z['positions'],list(z['names']),z['times'],'model_fk')
    events=artifact['source_jump_events'];mask=jump_mask(times,d,events)
    full_r,full_f=at(r,d.times),at(f,d.times);full_mask=jump_mask(d.times,d,events)
    result=score(full_r,d,full_f,full_mask,normalization_reference=r)
    result.update({'window':a.window,'method':a.method,'fitted_artifact':str(a.fits/a.window/a.method/'fk.npz'),
        'original_reference_file':str(run/'references'/f'{a.window}.npz'),'fitted_frames':len(f.times),
        'fit_boundary_missing_frames':int((~supported(d.times,f.times[0],f.times[-1])).sum()),
        'native_rotation_readout':artifact['head_neck_relative_deg'],
        'native_rotation_scope':'Frozen D native frame transfer; separate from model F',
        'clock_boundary_roundoff_s':BOUNDARY_ROUNDOFF_S})
    raw_d,raw_f=at(d,r.times),at(f,r.times);raw_mask=jump_mask(r.times,d,events)
    result['original_clock']=score(r,raw_d,raw_f,raw_mask)
    result['original_clock']['input_boundary_missing_frames']=int((~supported(r.times,d.times[0],d.times[-1])).sum())
    result['original_clock']['fit_boundary_missing_frames']=int((~supported(r.times,f.times[0],f.times[-1])).sum())
    rt,dt,ft=sample(r,times),sample(d,times),at(f,times)
    result['paired_baseline_clock']=score(rt,dt,ft,mask,normalization_reference=r)
    add_pelvis_diagnostic(result['paired_baseline_clock'],rt,dt,ft,mask)
    add_pelvis_diagnostic(result,full_r,d,full_f,full_mask)
    add_pelvis_diagnostic(result['original_clock'],r,raw_d,raw_f,raw_mask)
    result['paired_baseline_clock']['definition']='D timestamps inside frozen common-baseline F span, fixed before candidates; candidate missing values remain in denominator; full D/original-clock results also retained'
    result['paired_baseline_clock']['excluded_D_boundary_frames']=len(d.times)-len(times)
    result['scoring_code_sha256']={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__),ROOT/'python/scripts/evalkit_k4_score.py',ROOT/'utils/motiogen/evalkit/k/scoring.py',ROOT/'utils/motiogen/evalkit/k/timebase.py']}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/f'{a.window}_{a.method}.json').write_text(json.dumps(base.serializable(result),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
