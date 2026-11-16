#!/usr/bin/env python3
"""Check completed artifacts, frozen evidence and explicit development-only provenance."""
import argparse,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def check_actual_reuse(out,summary):
    import numpy as np
    sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
    from motiogen.evalkit.k.bones import common
    from motiogen.evalkit.k.scoring import angles
    rows=[]
    for method,entry in summary['methods'].items():
        first=out/'actual'/method/entry['selected']/'kinematics.npz';second=out/'downstream'/method/'work/kinematics.npz'
        with np.load(first) as x,np.load(second) as y:
            assert list(x['labels'])==list(y['labels']) and list(x['names'])==list(y['names'])
            clock=float(np.max(abs(x['times']-y['times'])));source=float(np.max(abs(x['source']-y['source'])))
            position=float(np.max(np.linalg.norm(x['fitted']-y['fitted'],axis=-1)))
            delta=abs(x['coordinates']-y['coordinates']);frame,column=np.unravel_index(np.argmax(delta),delta.shape);name=str(x['labels'][column])
            coordinates={'name':name,'difference':float(delta[frame,column]),'unit':'m' if name in ['pelvis_tx','pelvis_ty','pelvis_tz'] else 'deg'}
            scores=[]
            for z in [x,y]:
                fit=angles(common(z['fitted'],list(z['names'])));original=angles(common(z['source'],list(z['names'])))
                scores.append({m:float(np.nanpercentile(abs(fit[m]-original[m]),95)) for m in fit})
            precision=max(abs(scores[0][m]-scores[1][m]) for m in scores[0])
        # Comparison limits follow common position resolution and half of report rounding.
        assert clock<1e-12 and source<1e-12 and position<1e-4 and precision<.005,(method,clock,source,position,precision)
        rows.append({'method':method,'variant':entry['selected'],'frozen_actual':str(first),'current_runtime':str(second),
            'file_sha256':[digest(first),digest(second)],'maximum_time_difference_s':clock,'maximum_source_difference_m':source,
            'maximum_fitted_position_difference_m':position,'maximum_coordinate_difference':coordinates,
            'maximum_fidelity_p95_difference_deg':precision,'status':'consistent_at_report_precision'})
    result={'scope':'reused actual-output fidelity versus fresh selected-config runtime; not exact coordinate equality',
        'limits':{'time_s':1e-12,'source_m':1e-12,'fitted_position_m':1e-4,'reported_angle_p95_deg':.005},'records':rows}
    (ROOT/'config/evalkit_k3_20261116/actual_reuse_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    cfg=ROOT/'config/evalkit_k3_20261116';preservation=read(cfg/'preservation.json');freeze=read(cfg/'experiment_freeze.json')
    changed=[p for p,h in preservation['files'].items() if digest(ROOT/p)!=h]
    assert not changed,changed
    checks={}
    for key,path in [('config_sha256',cfg/'experiment_config.json'),('selection_sha256',cfg/'selection.json'),
                     ('exports_sha256',a.out/'exports.json'),('reference_contract_sha256',cfg/'reference_translation_correction.json')]:
        assert digest(path)==freeze[key],key
        checks[key]=freeze[key]
    summary=read(cfg/'results_summary.json');exports=read(a.out/'exports.json');selection=read(cfg/'selection.json')['records']
    assert not summary['partial'] and len(summary['result_paths'])==len(exports)==656
    assert len(selection)==82 and len({r['group_id'] for r in selection})==16
    assert sum(r['historical'] for r in selection)==22
    assert all(r['partition']=='development' for r in exports)
    rows=[read(p) for p in summary['result_paths']]
    assert len({(r['id'],r['method']) for r in rows})==len(rows)
    identity_failures=[];inputs={};cell_counts={}
    for row in rows:
        if row['status']!='computed':continue
        if digest(row['reference'])!=row['identity']['reference_sha256']:identity_failures.append(row['id']+':reference')
        for p,h in row['scoring_identity']['source_inputs'].items():
            if p in inputs:assert inputs[p]==h
            inputs[p]=h
        for variant in row['scores']:
            cell_counts[variant]=cell_counts.get(variant,0)+1
            receipt=read(a.out/'fits'/row['window_key']/row['method']/variant/'k3_receipt.json')
            assert receipt['evidence']['config']['fitting']['projection']==variant.endswith('projected')
            strategy='direction_refinement' if variant.startswith('optimized') else 'legacy'
            assert receipt['evidence']['config']['fitting']['strategy']==strategy
            for p,h in receipt['evidence']['inputs'].items():
                if p in inputs:assert inputs[p]==h
                inputs[p]=h
    identity_failures += [p for p,h in inputs.items() if digest(p)!=h]
    assert not identity_failures,identity_failures
    parity=read(cfg/'parity_regression.json');downstream=read(cfg/'downstream_regression.json')
    assert len(parity)==len(downstream)==8
    assert all(r['status']=='passed' for r in parity+downstream)
    muscle=[r['lower_limb_muscle'] for r in downstream if 'lower_limb_muscle' in r]
    assert len(muscle)==1 and muscle[0]['finite'] and muscle[0]['frames']==11
    normalization=read(cfg/'normalization_verification.json')
    assert normalization['count']==summary['completed']
    scoring=read(cfg/'scoring_revision.json')
    assert scoring['completed']==summary['completed'] and scoring['windows']==82
    assert scoring['all_formats_share_original_window_lengths']
    assert all(r.get('unfiltered_raw_metrics_identical') for r in scoring['records'] if r['status']=='rescored')
    actual=read(a.out/'actual_summary.json');assert len(actual)==8
    assert all(r['status']=='computed' and len(r['scores'])==4 for r in actual)
    actual_reuse=check_actual_reuse(a.out,summary)
    tails=read(cfg/'tail_diagnostics.json');assert all(len(v['records'])==82 for v in tails['methods'].values())
    chosen=read(ROOT/'config/evalkit_k_v3.json')
    assert chosen['selection']['summary_sha256']==digest(cfg/'results_summary.json')
    assert all(len(r['folds'])==10 for r in summary['methods'].values())
    tests=ROOT/'develop_logs/test_runs/20261116_k3/tests_final.log';log=tests.read_text()
    count=int(re.search(r'Ran (\d+) tests',log)[1]);assert count>=54 and log.rstrip().endswith('OK')
    assert all('source_head_geometry' in r and Path(r['head_position_arrays']).exists() for r in rows if r['status']=='computed')
    plans=subprocess.check_output(['git','diff',preservation['commit'],'--','plans'],cwd=ROOT,text=True)
    assert not plans
    visual=ROOT/'develop_logs/notes/20261116_k3_figures/visual_review.json'
    review=read(visual) if visual.exists() else {'status':'pending'}
    if review.get('status')=='reviewed':
        for p,h in review['files'].items():assert digest(p)==h
        figures=read(visual.parent/'index.json')
        expected={r['path'] for r in figures['figures']}
        expected.update(str(visual.parent/name) for name in ['overview.png','head_fast_turn.png','head_source_jump.png'])
        assert len(expected)==39 and expected.issubset(review['files']),expected-set(review['files'])
    free=shutil.disk_usage(a.out).free;assert free>=80*1024**3
    size=int(subprocess.check_output(['du','-sb',str(a.out)],text=True).split()[0])
    result={'status':'passed' if review.get('status')=='reviewed' else 'machine_checks_passed_visual_review_pending',
        'development_only':True,'new_holdout_runs':0,'protected_files_unchanged':len(preservation['files']),
        'plans_unchanged':True,'hyoid':'unchanged; protected upper-body implementation and model-resource configuration',
        'experimental_freeze':freeze['commit'],'frozen_manifest_checks':checks,'completed':summary['completed'],
        'failed_records':summary['failed'],'cell_counts':cell_counts,'fit_input_files_rehashed':len(inputs),
        'corrected_input_parity':{'formats':len(parity),'maximum_difference':max(r['coordinate_max_abs_difference'] for r in parity)},
        'downstream_formats':len(downstream),'lower_limb_muscle_smoke':muscle[0],'normalization_records':normalization['count'],
        'normalization_max_angle_change_deg':normalization['maximum_angle_change_deg'],
        'original_window_lengths':{'common_to_all_formats':True,'maximum_length_correction_m':scoring['maximum_length_change_m'],'unfiltered_raw_scoring_unchanged':True},
        'observation_name_correction':{'format':'chois','records_with_support_change':scoring['support_changed_records']},
        'actual_fidelity_formats':len(actual),'tests':{'passed':count,'log':str(tests),'sha256':digest(tests)},
        'actual_fidelity_reuse':{'formats':len(actual_reuse['records']),
            'maximum_fitted_position_difference_m':max(r['maximum_fitted_position_difference_m'] for r in actual_reuse['records']),
            'maximum_fidelity_p95_difference_deg':max(r['maximum_fidelity_p95_difference_deg'] for r in actual_reuse['records'])},
        'visual_review':review,'resource':{'max_cpu_workers':3,'numeric_threads_per_worker':1,'free_gib':free/1024**3,
            'corrected_output_gib':size/1024**3,'minimum_reserve_gib':80,
            'note':'short diagnostic jobs paused one owned worker; elapsed times are not throughput benchmarks'},
        'selected_config_sha256':digest(ROOT/'config/evalkit_k_v3.json')}
    (cfg/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    postprocess=a.out/'postprocess_status.json'
    if result['status']=='passed' and postprocess.exists():
        pipeline=read(postprocess)
        assert all(stage['returncode']==0 for stage in pipeline['completed'])
        pipeline.update(status='verified',verification=str(cfg/'verification.json'),
                        selected_config_sha256=result['selected_config_sha256'])
        postprocess.write_text(json.dumps(pipeline,indent=2)+'\n')
    print(result['status'],'protected',len(preservation['files']),'results',summary['completed'],'tests',count)


if __name__=='__main__':main()
