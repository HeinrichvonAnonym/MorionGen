#!/usr/bin/env python3
"""Frozen-K2 parity and explicit-K3 downstream interface regressions, development only."""
import argparse,json,sys,contextlib,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2]


def one(task):
    mode,row,out,core=task
    sys.path[:0]=[str(core/'utils'),str(core/'utils/motiogen')]
    method=row['method'];folder=out/mode/method;folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    with (folder/'run.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import opensim as osim
            osim.Logger.setLevelString('error')
            from motiogen.evalkit.k import load,evaluate
            from motiogen.evalkit.k.fitting import fit
            source=row['source']
            if mode=='parity':
                result=fit(source,folder/'work',75.,0.)
                with np.load(out/'fits'/row['window_key']/method/'legacy_raw/kinematics.npz') as z:
                    t=z['times'];new=z['coordinates'];labels=list(z['labels'])
                old=np.stack([np.interp(t,result['times'],result['coords'][:,result['labels'].index(n)]) for n in labels],1)
                k=evaluate(load(source,method))
                np.savez_compressed(folder/'native_k.npz',**{r+'_'+key:v[key] for r,v in k['metrics'].items() for key in ['rotations','valid']})
                error=float(np.max(np.abs(old-new)))
                rec={'status':'passed' if error<1e-8 else 'failed','coordinate_max_abs_difference':error,
                     'scope':'frozen K2 versus frozen K3 legacy_raw on identical corrected inputs and common times',
                     'comparison_frames':len(t),'core_checkout':str(core)}
            else:
                from motiogen.evalkit.run import evaluate as evaluate_run,add_dims
                config=json.loads((ROOT/'config/evalkit_k_v3.json').read_text())
                r=evaluate_run(source,folder,0.,{s:{k:0. for k in ['heel','toe']} for s in 'lr'},method=method,
                               dims=('residual','kinematic'),k_config=config)
                # Exercise adding a dimension from the stored explicit config and cached fit.
                r=add_dims(source,folder,('ergo',))
                with np.load(folder/'frames.npz') as z:
                    finite=bool(np.isfinite(z['residual_force']).all());masks=[k for k in z.files if k.startswith('k_region')]
                rec={'status':'passed' if finite and len(masks)==18 else 'failed','residual_finite':finite,
                     'region_mask_count':len(masks),'ergo':r['ergo'],'stored_k_version':r['receipt']['k_config']['version'],
                     'scope':'interface regression only; no force or muscle accuracy claim'}
                if method=='motionlcm':
                    from motiogen.evalkit import muscle
                    mu=muscle.evaluate(folder/'work',1.,1.2)
                    rec['lower_limb_muscle']={'frames':len(mu['times']),'finite':bool(np.isfinite(mu['effort']).all())}
                    if not rec['lower_limb_muscle']['finite']:rec['status']='failed'
        except Exception as e:
            import traceback;traceback.print_exc();rec={'status':'failed','reason':type(e).__name__+': '+str(e)}
    rec.update(method=method,seconds=time.monotonic()-started)
    (folder/'result.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n');return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['parity','downstream']);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--core-root',type=Path,default=ROOT);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    if a.mode=='parity':rows=json.loads((a.out/'exports.json').read_text())[:8]
    else:
        sys.path.insert(0,str(ROOT/'python/scripts'));from evalkit_k_regression import sample_path
        samples=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['baseline_samples']
        rows=[{'method':m,'source':str(sample_path(v[0]['path']))} for m,v in samples.items()]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=[]
        for r in pool.map(one,[(a.mode,r,a.out,a.core_root) for r in rows]):
            results.append(r);print(r['method'],r['status'],round(r['seconds'],1),r.get('reason',''),flush=True)
    (ROOT/'config/evalkit_k3_20261116'/f'{a.mode}_regression.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['status']!='passed' for r in results):raise SystemExit(1)

if __name__=='__main__':main()
