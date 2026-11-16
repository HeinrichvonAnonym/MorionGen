#!/usr/bin/env python3
"""Check completed artifacts, frozen evidence and explicit development-only provenance."""
import argparse,hashlib,json,re,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


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
    normalization=read(cfg/'normalization_verification.json')
    assert normalization['count']==summary['completed']
    scoring=read(cfg/'scoring_revision.json')
    assert scoring['completed']==summary['completed'] and scoring['windows']==82
    assert scoring['all_formats_share_original_window_lengths']
    assert all(r.get('raw_metrics_identical') for r in scoring['records'] if r['status']=='rescored')
    actual=read(a.out/'actual_summary.json');assert len(actual)==8
    assert all(r['status']=='computed' and len(r['scores'])==4 for r in actual)
    chosen=read(ROOT/'config/evalkit_k_v3.json')
    assert chosen['selection']['summary_sha256']==digest(cfg/'results_summary.json')
    assert all(len(r['folds'])==10 for r in summary['methods'].values())
    tests=ROOT/'develop_logs/test_runs/20261116_k3/tests_final.log';log=tests.read_text()
    count=int(re.search(r'Ran (\d+) tests',log)[1]);assert count>=52 and log.rstrip().endswith('OK')
    plans=subprocess.check_output(['git','diff',preservation['commit'],'--','plans'],cwd=ROOT,text=True)
    assert not plans
    visual=ROOT/'develop_logs/notes/20261116_k3_figures/visual_review.json'
    review=read(visual) if visual.exists() else {'status':'pending'}
    if review.get('status')=='reviewed':
        for p,h in review['files'].items():assert digest(p)==h
    free=shutil.disk_usage(a.out).free;assert free>=80*1024**3
    size=int(subprocess.check_output(['du','-sb',str(a.out)],text=True).split()[0])
    result={'status':'passed' if review.get('status')=='reviewed' else 'machine_checks_passed_visual_review_pending',
        'development_only':True,'new_holdout_runs':0,'protected_files_unchanged':len(preservation['files']),
        'plans_unchanged':True,'hyoid':'unchanged; protected upper-body implementation and model-resource configuration',
        'experimental_freeze':freeze['commit'],'frozen_manifest_checks':checks,'completed':summary['completed'],
        'failed_records':summary['failed'],'cell_counts':cell_counts,'fit_input_files_rehashed':len(inputs),
        'corrected_input_parity':{'formats':len(parity),'maximum_difference':max(r['coordinate_max_abs_difference'] for r in parity)},
        'downstream_formats':len(downstream),'normalization_records':normalization['count'],
        'normalization_max_angle_change_deg':normalization['maximum_angle_change_deg'],
        'original_window_lengths':{'common_to_all_formats':True,'maximum_length_correction_m':scoring['maximum_length_change_m'],'raw_scoring_unchanged':True},
        'actual_fidelity_formats':len(actual),'tests':{'passed':count,'log':str(tests),'sha256':digest(tests)},
        'visual_review':review,'resource':{'max_cpu_workers':3,'numeric_threads_per_worker':1,'free_gib':free/1024**3,
            'corrected_output_gib':size/1024**3,'minimum_reserve_gib':80,
            'note':'short diagnostic jobs paused one owned worker; elapsed times are not throughput benchmarks'},
        'selected_config_sha256':digest(ROOT/'config/evalkit_k_v3.json')}
    (cfg/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(result['status'],'protected',len(preservation['files']),'results',summary['completed'],'tests',count)


if __name__=='__main__':main()
