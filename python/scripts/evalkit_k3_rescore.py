#!/usr/bin/env python3
"""Use one original-window length vector for all eight formats, preserving frozen fits."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.bones import common,lengths
from motiogen.evalkit.k.export import ALIASES
from motiogen.evalkit.k.scoring import compare,head_chain,source_head_geometry
from motiogen.evalkit.k.config import digest
from motiogen.evalkit.k import load
from motiogen.evalkit.k.fitting3 import support


def one(path):
    row=json.loads(path.read_text());dest=path.with_name('result_original_lengths.json')
    if row['status']!='computed':
        dest.write_text(json.dumps(row)+'\n');return {'id':row['id'],'method':row['method'],'status':row['status']}
    variants=list(row['scores']);config=json.loads((ROOT/'config/evalkit_k3_20261116/experiment_config.json').read_text())
    motion=load(row['source'],row['method'],config=config);prior=motion.legacy_prior_joints()
    identity={'initial_result_sha256':digest(path),'reference_sha256':digest(row['reference']),
        'arrays_sha256':digest(row['arrays']),'script_sha256':digest(__file__),
        'scoring_sha256':digest(ROOT/'utils/motiogen/evalkit/k/scoring.py'),
        'support_sha256':digest(ROOT/'utils/motiogen/evalkit/k/fitting3.py'),
        'motion_sha256':digest(ROOT/'utils/motiogen/evalkit/k/motion.py'),'source_inputs':motion.provenance['inputs'],
        'quality_sha256':{v:digest(path.parent/v/'region_quality.npz') for v in variants},
        'length_scope':'complete original native-sampling window; common to all eight formats; scoring only'}
    if dest.exists():
        old=json.loads(dest.read_text())
        if (old.get('scoring_identity')==identity and Path(old['arrays']).exists()
                and digest(old['arrays'])==old.get('scoring_arrays_sha256')):return old['length_scope_check']
    with np.load(row['arrays']) as z:old_arrays={k:z[k] for k in z.files}
    with np.load(row['reference']) as z:
        reference_lengths=lengths(common(z['points'],[ALIASES.get(n,n) for n in z['names']]))
        native_head=head_chain(z['points'],list(z['names']));native_times=z['times']
    from evalkit_k3_experiment import interp
    head_query=old_arrays['times']+row['source_clock_offset_s']
    head_lengths=lengths(native_head,[-1,0,1,2])
    head_reference=interp(native_times,native_head,np.clip(head_query,native_times[0],native_times[-1]))
    head_readback=interp(motion.times,head_chain(motion.positions,motion.names),head_query)
    head_score=source_head_geometry(head_reference,head_readback,head_lengths,motion.provenance.get('degradation',{}).get('prior_joints',[]))
    head_path=path.with_name('head_position_geometry.npz')
    np.savez_compressed(head_path,times=head_query,reference=head_reference,readback=head_readback,reference_lengths=head_lengths,
        names=['pelvis_mid','shoulder_mid','neck','head'])
    times=old_arrays['times'];supports={};qualities={};full_times={}
    for variant in variants:
        receipt=json.loads((path.parent/variant/'k3_receipt.json').read_text());settings=receipt['evidence']['config']['fitting']
        with np.load(path.parent/variant/'kinematics.npz') as z:
            full_times[variant]=z['times']
            quality=support(z['source'],z['fitted'],list(z['names']),settings['quality_angle_deg'],settings['quality_position_m'],prior)
        qualities[variant]=quality
        supports[variant]={k:np.interp(times,full_times[variant],v['angle_valid'].astype(float))>=.99999 for k,v in quality.items()}
        np.savez_compressed(path.parent/variant/'region_quality_alias_fixed.npz',times=full_times[variant],
            **{region+'_'+key:value for region,q in quality.items() for key,value in q.items()})
    scores,arrays=compare(old_arrays['R'],old_arrays['D'],{v:old_arrays['P:'+v] for v in variants},
        {v:old_arrays['F:'+v] for v in variants},times,supports,float(np.nanpercentile(old_arrays['D'][:,[5,9],1],5)),
        reference_lengths=reference_lengths)
    support_changes={}
    for v in variants:
        # Unfiltered original-reference metrics and the policy-selection windows stay exact.
        for category,metrics in scores[v]['raw'].items():
            for metric,terms in metrics.items():
                for stage,value in terms.items():
                    if stage!='supported_end_to_end' and value!=row['scores'][v]['raw'][category][metric][stage]:
                        raise ValueError('Unfiltered raw scoring changed: '+row['id']+' '+row['method'])
        calibration=row['scores'][v]['calibration'];qualified=bool(calibration['qualified'])
        scores[v]['calibration']=calibration
        scores[v]['position_coverage']={k:float(np.mean(q['position_valid']&qualified)) for k,q in qualities[v].items()}
        scores[v]['registered_angle_coverage']={k:float(np.mean(q['angle_valid']&qualified)) for k,q in qualities[v].items()}
        changes={}
        for reg in scores[v]['coverage']:
            old=row['scores'][v];new=scores[v];values={}
            for kind,a,b in [('angle_fidelity',old['coverage'][reg]['share'],new['coverage'][reg]['share']),
                ('registered_angle',old['registered_angle_coverage'][reg],new['registered_angle_coverage'][reg]),
                ('registered_position',old['position_coverage'][reg],new['position_coverage'][reg])]:
                if a!=b:values[kind]={'before':a,'after':b}
            if values:changes[reg]=values
        if changes and (row['method']!='chois' or set(changes)-{'left_arm','right_arm'}):raise ValueError('Unexpected support change')
        support_changes[v]=changes
    array_path=path.with_name('comparison_original_lengths.npz');np.savez_compressed(array_path,**arrays)
    check={'id':row['id'],'method':row['method'],'window_key':row['window_key'],'status':'rescored',
        'maximum_length_change_m':float(np.nanmax(abs(reference_lengths-old_arrays['reference_lengths']))),
        'unfiltered_raw_metrics_identical':True,'support_changes':support_changes,
        'original_reference_lengths_m':[None if not np.isfinite(x) else float(x) for x in reference_lengths]}
    updated={**row,'scores':scores,'arrays':str(array_path),'initial_scoring_result':str(path),
        'identity_scope':'original frozen fitting and initial scoring; scoring_identity identifies the current length convention',
        'scoring_identity':identity,'scoring_arrays_sha256':digest(array_path),'length_scope_check':check,
        'source_head_geometry':head_score,'head_position_arrays':str(head_path),
        'observation_name_mapping':{'aliases':motion.legacy_aliases(),'prior_joints_in_fitting_view':prior,
            'scope':'body-wrist observation remains separate from an extra hand-point prior'}}
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
    result={'scope':'scoring and observation-name correction; frozen fits and original result.json files retained; unfiltered raw angular selection unchanged',
        'records':checks,'completed':len(success),'expected':expected,'windows':len(by_window),
        'all_formats_share_original_window_lengths':True,
        'maximum_length_change_m':max(r['maximum_length_change_m'] for r in success),
        'support_changed_records':sum(any(r['support_changes'].values()) for r in success),
        'script_sha256':digest(__file__),'scoring_sha256':digest(ROOT/'utils/motiogen/evalkit/k/scoring.py')}
    (ROOT/'config/evalkit_k3_20261116/scoring_revision.json').write_text(json.dumps(result,indent=2)+'\n')
    print('original-window rescoring',len(success),'/',expected,'maximum length correction m',result['maximum_length_change_m'])


if __name__=='__main__':main()
