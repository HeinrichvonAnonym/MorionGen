#!/usr/bin/env python3
"""Actual generated outputs: source fidelity only, never labelled original-reference accuracy."""
import argparse,json,copy,sys,time,contextlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
from evalkit_k3_experiment import VARIANTS,interp
from evalkit_k_regression import sample_path


def one(task):
    method,path,out=task;work=out/'actual'/method;work.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    with (work/'run.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import opensim as osim
            osim.Logger.setLevelString('error')
            from motiogen.evalkit.k import load
            from motiogen.evalkit.k.fitting import fit
            from motiogen.evalkit.k.bones import common
            from motiogen.evalkit.k.scoring import compare
            from motiogen.evalkit.k.config import digest
            config=json.loads((ROOT/'config/evalkit_k3_20261116/experiment_config.json').read_text());source=sample_path(path);motion=load(source,method,config=config)
            data={};receipts={};failures={}
            for v in VARIANTS:
                c=copy.deepcopy(config);c['fitting'].update(strategy='direction_refinement' if v.startswith('optimized') else 'legacy',projection=v.endswith('projected'),
                    base_cache=str(work/('base_projected' if v.endswith('projected') else 'base_raw')))
                try:
                    receipts[v]=fit(source,work/v,75.,0.,config=c)
                    with np.load(work/v/'kinematics.npz') as z:data[v]={k:z[k] for k in z.files}
                except Exception as e:failures[v]=str(e)
            if not data:raise RuntimeError(str(failures))
            first=next(iter(data.values()));t=first['times'];mask=t<=motion.times[-1]-motion.times[0]+1e-8
            for a in data.values():mask &= (t>=a['times'][0])&(t<=a['times'][-1]+1e-8)
            t=t[mask];G=common(interp(first['times'],first['source'],t),list(first['names']))
            F={v:common(interp(a['times'],a['fitted'],t),list(a['names'])) for v,a in data.items()};P={v:common(interp(a['times'],a['projected'],t),list(a['names'])) for v,a in data.items()}
            masks={v:{k:np.interp(t,receipts[v]['times'],m.astype(float))>.99999 for k,m in receipts[v]['region_angle_valid'].items()} for v in data}
            scores,arrays=compare(G,G,P,F,t,masks,float(np.percentile(G[:,[5,9],1],5)))
            # Strip artificial R entries: this call has no original reference at all.
            for v,s in scores.items():
                for space in ['raw','normalized']:
                    for category in ['angles','positions','directions']:
                        for metric,terms in s[space][category].items():
                            s[space][category][metric]={'fit_fidelity':terms['fit_deviation'],'input_projection':terms['projection'],'fit_to_projected':terms['fit_after_projection']}
                s.pop('length_component',None);s['contact'].pop('R',None);s['contact']['G']=s['contact'].pop('D')
                s['calibration']=receipts[v]['receipt']['calibration']
            arrays.pop('R',None);arrays['G']=arrays.pop('D')
            arrays={k:v for k,v in arrays.items() if not k.endswith('_e2e')};arrays['source_lengths']=arrays.pop('reference_lengths')
            np.savez_compressed(work/'comparison.npz',**arrays)
            result={'method':method,'source':str(source),'status':'computed','scope':'G to F actual generated output fidelity; no original reference R; source-derived lengths for normalized comparison',
                    'scores':scores,'failures':failures,'input_hashes':motion.provenance['inputs'],'arrays':str(work/'comparison.npz'),'seconds':time.monotonic()-start}
        except Exception as e:
            import traceback;traceback.print_exc();result={'method':method,'status':'failed','reason':str(e),'seconds':time.monotonic()-start}
    (work/'result.json').write_text(json.dumps(result,allow_nan=False)+'\n');return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    samples=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['baseline_samples']
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        results=[]
        for r in pool.map(one,[(m,s[0]['path'],a.out) for m,s in samples.items()]):
            results.append(r);print(r['method'],r['status'],round(r['seconds'],1),flush=True)
    (a.out/'actual_summary.json').write_text(json.dumps(results,allow_nan=False)+'\n')
if __name__=='__main__':main()
