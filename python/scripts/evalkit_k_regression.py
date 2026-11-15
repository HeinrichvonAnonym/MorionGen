#!/usr/bin/env python3
"""Eight actual baseline outputs: full-body IK, downstream interfaces and geometric overlays."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]


def sample_path(value):
    p=Path(value);p=ROOT/p if not p.is_absolute() else p
    return Path(str(p).replace('9bbdd73e-4410-4279-9639-6161f0c95839/', '9bbdd73e-4410-4279-9639-6161f0c958391/'))


def check(args):
    method,source,out=args[:3];fit_only=len(args)>3 and args[3];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    import contextlib
    with (out/'python.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            import opensim as osim
            import biomech
            from motiogen.evalkit.run import evaluate
            from motiogen.evalkit.k import load,Motion
            from motiogen.evalkit.k.fitting import fit
            from motiogen.evalkit.k.body import evaluate_body
            from motiogen.evalkit.k.geometry import tiers
            from evalkit_k import process
            source=sample_path(source)
            motion=load(source,method)
            if fit_only:
                from motiogen.evalkit.k.posture import evaluate as k_evaluate, summary as k_summary
                f=fit(source,out/'work',75.,0.)
                result={'ik':{**f['receipt']['ik_marker_error'],'unmeasurable_share':float(f['unmeasurable'].mean())},'k':k_summary(k_evaluate(motion))}
            else:
                result=evaluate(source,out,0.,{s:{k:0. for k in ('heel','toe')} for s in ('l','r')},method=method,
                                dims=('residual','ergo','kinematic'))
            process(source,method,out/'k')
            f=fit(source,out/'work',75.,0.)  # Verifies cache reuse with the current input/content key.
            model=osim.Model(str(out/'work/model.osim'));state=model.initSystem()
            indices=np.arange(0,len(f['times']),10);times=f['times'][indices]
            names=[v[2] for v in biomech.MARKERS.values()]
            fitted=[]
            for row in f['coords'][indices]:
                biomech._set(model,state,f['labels'],row)
                fitted.append([biomech._child_origin(model,state,joint) for joint,_,_ in biomech.MARKERS.values()])
            fitted=np.asarray(fitted)
            reference=evaluate_body(motion)
            predicted=evaluate_body(Motion(fitted,names,times,'fitted_joint_centres'))
            errors={}
            for name,item in predicted.items():
                good=item['valid']
                truth=np.interp(times,motion.times-motion.times[0],reference[name]['values_deg'])
                difference=item['values_deg'][good]-truth[good]
                errors[name]={**tiers(np.abs(difference),len(times)),
                              'signed_bias_deg':float(difference.mean()) if len(difference) else None}
            source_points,source_names,_=motion.legacy()
            original=np.stack([np.stack([np.interp(times,motion.times-motion.times[0],source_points[:,source_names.index(n),j])
                                         for j in range(3)],-1) for n in names],1)
            np.savez_compressed(out/'overlay.npz',times=times,names=names,source=original,fitted=fitted)
            finite=None
            if not fit_only:
                with np.load(out/'frames.npz') as z:finite=bool(np.isfinite(z['residual_force']).all())
            receipt={'status':'computed','method':method,'source':str(source),'ik':result['ik'],
                     'calibration':f['receipt']['calibration'],'neck':result['k']['metrics']['neck'],
                     'geometric_fit_errors':errors,'residual_interface_finite':finite,
                     'scope':'source joint-centre geometry versus fitted model joint centres; not independent anatomical accuracy',
                     'ergo':result.get('ergo')}
            # Small downstream solver regression, separated from scientific muscle validation.
            if method=='motionlcm' and not fit_only:
                from motiogen.evalkit import muscle
                smoke=muscle.evaluate(out/'work',1.,1.2)
                receipt['muscle_interface']={'frames':len(smoke['times']),'finite':bool(np.isfinite(smoke['effort']).all())}
        except Exception as e:
            import traceback
            traceback.print_exc()
            receipt={'status':'failed','method':method,'reason':f'{type(e).__name__}: {e}'}
        (out/'regression.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
    return receipt


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    samples=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['baseline_samples']
    tasks=[(method,rows[0]['path'],str(a.out/method)) for method,rows in samples.items()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=list(pool.map(check,tasks))
    (a.out/'summary.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    for r in results:print(r['method'],r['status'],r.get('reason',''))


if __name__=='__main__':main()
