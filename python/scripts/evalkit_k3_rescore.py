#!/usr/bin/env python3
"""Use one original-window length vector for all eight formats, preserving frozen fits."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.bones import common,lengths
from motiogen.evalkit.k.export import ALIASES
from motiogen.evalkit.k.scoring import compare
from motiogen.evalkit.k.config import digest


def one(path):
    row=json.loads(path.read_text());dest=path.with_name('result_original_lengths.json')
    if row['status']!='computed':
        dest.write_text(json.dumps(row)+'\n');return {'id':row['id'],'method':row['method'],'status':row['status']}
    variants=list(row['scores']);identity={'initial_result_sha256':digest(path),'reference_sha256':digest(row['reference']),
        'arrays_sha256':digest(row['arrays']),'script_sha256':digest(__file__),
        'scoring_sha256':digest(ROOT/'utils/motiogen/evalkit/k/scoring.py'),
        'quality_sha256':{v:digest(path.parent/v/'region_quality.npz') for v in variants},
        'length_scope':'complete original native-sampling window; common to all eight formats; scoring only'}
    if dest.exists():
        old=json.loads(dest.read_text())
        if (old.get('scoring_identity')==identity and Path(old['arrays']).exists()
                and digest(old['arrays'])==old.get('scoring_arrays_sha256')):return old['length_scope_check']
    with np.load(row['arrays']) as z:old_arrays={k:z[k] for k in z.files}
    with np.load(row['reference']) as z:
        reference_lengths=lengths(common(z['points'],[ALIASES.get(n,n) for n in z['names']]))
    times=old_arrays['times'];supports={}
    for variant in variants:
        with np.load(path.parent/variant/'region_quality.npz') as z:
            supports[variant]={k.removesuffix('_angle_valid'):np.interp(times,z['times'],z[k].astype(float))>=.99999
                for k in z.files if k.endswith('_angle_valid')}
    scores,arrays=compare(old_arrays['R'],old_arrays['D'],{v:old_arrays['P:'+v] for v in variants},
        {v:old_arrays['F:'+v] for v in variants},times,supports,float(np.nanpercentile(old_arrays['D'][:,[5,9],1],5)),
        reference_lengths=reference_lengths)
    for v in variants:
        # The fitted poses, clocks, raw metrics and policy-selection evidence stay exact.
        if scores[v]['raw']!=row['scores'][v]['raw']:raise ValueError('Raw scoring changed: '+row['id']+' '+row['method'])
        if scores[v]['coverage']!=row['scores'][v]['coverage']:raise ValueError('Support mask changed')
        for key in ['position_coverage','registered_angle_coverage','calibration']:scores[v][key]=row['scores'][v][key]
    array_path=path.with_name('comparison_original_lengths.npz');np.savez_compressed(array_path,**arrays)
    check={'id':row['id'],'method':row['method'],'window_key':row['window_key'],'status':'rescored',
        'maximum_length_change_m':float(np.nanmax(abs(reference_lengths-old_arrays['reference_lengths']))),
        'raw_metrics_identical':True,'original_reference_lengths_m':[None if not np.isfinite(x) else float(x) for x in reference_lengths]}
    updated={**row,'scores':scores,'arrays':str(array_path),'initial_scoring_result':str(path),
        'identity_scope':'original frozen fitting and initial scoring; scoring_identity identifies the current length convention',
        'scoring_identity':identity,'scoring_arrays_sha256':digest(array_path),'length_scope_check':check}
    dest.write_text(json.dumps(updated,allow_nan=False)+'\n');return check


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    paths=list((a.out/'fits').glob('*/*/result.json'));expected=len(json.loads((a.out/'exports.json').read_text()))
    if len(paths)!=expected:raise ValueError('Complete the registered fitting batch before final rescoring')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:checks=list(pool.map(one,paths))
    success=[r for r in checks if r['status']=='rescored'];by_window={}
    for r in success:
        previous=by_window.setdefault(r['window_key'],r['original_reference_lengths_m'])
        if previous!=r['original_reference_lengths_m']:raise ValueError('Different reference lengths within a source window')
    result={'scope':'scoring-only correction; frozen fits and original result.json files retained; raw angular selection unchanged',
        'records':checks,'completed':len(success),'expected':expected,'windows':len(by_window),
        'all_formats_share_original_window_lengths':True,
        'maximum_length_change_m':max(r['maximum_length_change_m'] for r in success),
        'script_sha256':digest(__file__),'scoring_sha256':digest(ROOT/'utils/motiogen/evalkit/k/scoring.py')}
    (ROOT/'config/evalkit_k3_20261116/scoring_revision.json').write_text(json.dumps(result,indent=2)+'\n')
    print('original-window rescoring',len(success),'/',expected,'maximum length correction m',result['maximum_length_change_m'])


if __name__=='__main__':main()
