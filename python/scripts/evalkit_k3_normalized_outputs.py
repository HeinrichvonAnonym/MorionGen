#!/usr/bin/env python3
"""Save scoring-only normalized geometry and verify length/rotation invariants."""
import argparse,json,sys,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.bones import normalize,project_motion
from motiogen.evalkit.k.scoring import angles
from motiogen.evalkit.k import load
from motiogen.evalkit.k.config import digest


def one(row):
    path=Path(row['arrays']);dest=path.with_name('normalized_scoring.npz')
    with np.load(path) as z:a={k:z[k] for k in z.files}
    keys=['R','D']+[k for k in a if k.startswith(('P:','F:'))]
    normal={k:normalize(a[k],a['reference_lengths']) for k in keys};angle_delta=0.;closure=0.
    for key,value in normal.items():
        original=angles(a[key]);current=angles(value)
        for metric in original:
            good=np.isfinite(original[metric])&np.isfinite(current[metric])
            if good.any():angle_delta=max(angle_delta,float(np.max(abs(original[metric][good]-current[metric][good]))))
    components={}
    for key in [k for k in keys if k.startswith('F:')]:
        delta=(a['D']-a['R'])+(a[key]-a['D'])-(a[key]-a['R'])
        closure=max(closure,float(np.nanmax(abs(delta))))
        components['length_component:'+key]=(a[key]-a['R'])-(normal[key]-normal['R'])
    motion=load(row['source'],row['method'],config='config/evalkit_k3_20261116/experiment_config.json')
    before=motion.content_key();projected,_=project_motion(motion);again,_=project_motion(projected)
    projection_delta=float(np.nanmax(abs(projected.positions-again.positions)))
    assert motion.content_key()==before
    if motion.rotations is not None:np.testing.assert_array_equal(projected.rotations,motion.rotations)
    if angle_delta>1e-4 or closure>1e-10 or projection_delta>1e-8:
        raise ValueError(f'Invariant failure: {row["id"]} {row["method"]}: {angle_delta} {closure} {projection_delta}')
    np.savez_compressed(dest,times=a['times'],names=a['names'],reference_lengths=a['reference_lengths'],**normal,**components)
    return {'id':row['id'],'method':row['method'],'angle_invariance_max_deg':angle_delta,'vector_closure_max_m':closure,
        'projection_idempotence_max_m':projection_delta,'native_rotations_retained':True,'source_unchanged':True,
        'comparison':str(path),'comparison_sha256':digest(path),'normalized':str(dest),'normalized_sha256':digest(dest)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    if shutil.disk_usage(a.out).free<80*1024**3:raise ValueError('80 GiB reserve required')
    paths=list((a.out/'fits').glob('*/*/result.json'));rows=[json.loads(p.read_text()) for p in paths];rows=[r for r in rows if r['status']=='computed']
    with ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(one,rows))
    result={'scope':'scorer-only reference lengths; normalized artifacts do not enter fitting','records':results,
        'count':len(results),'maximum_angle_change_deg':max(r['angle_invariance_max_deg'] for r in results),
        'maximum_projection_idempotence_error_m':max(r['projection_idempotence_max_m'] for r in results)}
    target=ROOT/'config/evalkit_k3_20261116/normalization_verification.json';target.write_text(json.dumps(result,indent=2)+'\n')
    print('normalized records',len(results),'angle max',result['maximum_angle_change_deg'])

if __name__=='__main__':main()
