"""Score saved model FK against frozen R and D in a separate process."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion,load
from motiogen.evalkit.k.scoring import score
from motiogen.degradation.reference import sample


def serializable(value):
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,list):return [serializable(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def score_one(run, fits, key, method, artifact):
    with np.load(run/'references'/f'{key}.npz') as z:
        r=Motion(z['positions'],list(z['names']),z['times'],'reference')
    source=run/'inputs'/key/method/'candidate'
    d=load(source/'factorial/reference.npz' if method=='primal' else source,method)
    path=fits/key/method/artifact
    with np.load(path) as z:
        positions=z['positions_assembled'] if 'positions_assembled' in z else z['positions']
        f=Motion(positions,list(z['names']),z['times'],'fitted_model_fk')
    # Primary: actual output timestamps. Missing F endpoints retain NaNs and
    # remain in the denominator. No extrapolation or time warping.
    valid=(d.times>=f.times[0]-1e-9)&(d.times<=f.times[-1]+1e-9)
    fp=np.full((len(d.times),len(f.names),3),np.nan)
    fp[valid]=sample(f,np.clip(d.times[valid],f.times[0],f.times[-1])).positions
    fitted=Motion(fp,f.names,d.times,'fitted_model_fk')
    truth=sample(r,d.times)
    mask=np.zeros(len(d.times),bool)
    for event in artifact_index[(key,method)]['source_jump_events']:
        for left,right in zip(d.times[:-1],d.times[1:]):
            if max(left,event['start_s'])<min(right,event['end_s']):
                pad=right-left;mask|=(d.times>=left-pad)&(d.times<=right+pad)
    result=score(truth,d,fitted,mask,normalization_reference=r)
    # Original-clock check retains every R sample in the fixed window. The
    # input/output boundary samples outside a format's observed clock stay NaN.
    rp=np.full((len(r.times),len(d.names),3),np.nan)
    fp=np.full((len(r.times),len(f.names),3),np.nan)
    dm=(r.times>=d.times[0])&(r.times<=d.times[-1])
    fm=(r.times>=f.times[0])&(r.times<=f.times[-1])
    rp[dm]=sample(d,r.times[dm]).positions;fp[fm]=sample(f,r.times[fm]).positions
    rawmask=np.zeros(len(r.times),bool)
    for event in artifact_index[(key,method)]['source_jump_events']:
        for left,right in zip(d.times[:-1],d.times[1:]):
            if max(left,event['start_s'])<min(right,event['end_s']):
                rawmask|=(r.times>=left-(right-left))&(r.times<=right+(right-left))
    result['original_clock']=score(r,Motion(rp,d.names,r.times,method),
                                  Motion(fp,f.names,r.times,'fitted_model_fk'),rawmask)
    result['original_clock']['input_boundary_missing_frames']=int((~dm).sum())
    result['original_clock']['fit_boundary_missing_frames']=int((~fm).sum())
    result.update({'window':key,'method':method,'fitted_artifact':str(path),
                   'original_reference_file':str(run/'references'/f'{key}.npz'),
                   'fitted_frames':len(f.times),'fit_boundary_missing_frames':int((~valid).sum()),
                   'native_rotation_readout':artifact_index[(key,method)]['head_neck_relative_deg'],
                   'native_rotation_scope':'Frozen D native-frame transfer, separate from model F; not fitted head accuracy'})
    return result


def main():
    global artifact_index
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fits',type=Path,required=True);ap.add_argument('--artifact',default='fk.npz')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--window');ap.add_argument('--method')
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    run=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
    records=json.loads((run/'artifact_verification.json').read_text())['records']
    artifact_index={(r['window'],r['method']):r for r in records}
    rows=[]
    for row in records:
        key,method=row['window'],row['method']
        if args.window and key!=args.window or args.method and method!=args.method:continue
        if not (args.fits/key/method/args.artifact).exists():continue
        result=score_one(run,args.fits,key,method,args.artifact)
        (args.output/f'{key}_{method}.json').write_text(json.dumps(serializable(result),indent=2,allow_nan=False)+'\n')
        print(key,method,'scored',flush=True);rows.append([key,method])
    if not args.window and not args.method:
        (args.output/'manifest.json').write_text(json.dumps({'scored':rows,'expected_paths':656,'new_holdout_uses':0},indent=2)+'\n')


if __name__=='__main__':main()
