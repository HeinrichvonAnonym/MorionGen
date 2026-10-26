"""Summarize existing scientific/browser receipts; never gate GPU execution."""
import argparse
import base64
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import checksum,json_write

p=argparse.ArgumentParser()
p.add_argument('--require-complete',action='store_true')
args=p.parse_args()
base=ROOT/'develop_logs/test_runs'
def read(relative):return json.loads((base/relative).read_text())
analysis=read('20261024_native_matrix_analysis/analysis.json')
trajectory=read('20261024_omnicontrol_trajectory/analysis.json')
weights=read('20261024_native_weight_audit/audit.json')
preprocessing=read('20261024_native_preprocessing/audit.json')
metrics=read('20261024_native_metric_reconstruction/audit.json')
invariants=read('20261024_native_invariants/audit.json')
execution=read('20261024_native_execution/audit.json')
kimodo=read('20261024_kimodo_pair_audit/audit.json')
kimodo_weights=read('20261024_kimodo_weight_audit/audit.json')
browser=read('20261024_native_pilots/preview_checks.json')
report=ROOT/'develop_logs/notes/motiongen_survey_revision'
snapshot=json.loads((report/'src/data.json').read_text())
build=json.loads((report/'dist/data-app-build.json').read_text())
runs={r['run'] for r in analysis['rows']}
def normalized(path):
    path=Path(path)
    return str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
weight_runs={normalized(r['run']) for r in weights['rows'] if r['passed']}
preprocessing_runs={normalized(r['run']) for r in preprocessing['checks'] if r['passed']}
cells={(r['run'],r.get('variant')) for r in analysis['rows']}
reconstructed={(r['run'],r.get('variant')) for r in metrics['checks'] if r['passed']}
checks=[]
def check(name,passed,detail):checks.append({'check':name,'passed':bool(passed),'detail':detail})
cohorts=read('20261025_source_replays/audit.json')
text_profiles=snapshot['queries'].get('ardy_text_setup',{}).get('rows',[])
text_profile_checks=[]
for profile in text_profiles:
    selected=[]
    for r in analysis['rows']:
        if r['method']!='ardy' or r['clock']!=profile['clock']:continue
        cfg=json.loads((ROOT/r['run']/'config.json').read_text())
        native_profile='required_hit_cache' if cfg.get('_frozen_prompt_cache') else 'native_precompute'
        if native_profile!=profile['profile']:continue
        raw=json.loads((ROOT/r['run']/'worker_result.json').read_text())['timing']
        selected.append({'text_setup_seconds':sum(raw['text_model_load_and_encode']['samples_seconds']),
                         'model_load_seconds':sum(raw['model_load']['samples_seconds'])})
    passed=len(selected)==profile['runs']
    for field in ['text_setup_seconds','model_load_seconds']:
        values=[r[field] for r in selected]
        passed=passed and bool(values) and len(values)==profile[field+'_n']
        if values:
            passed=passed and np.isclose(np.median(values),profile[field+'_median'],rtol=0,atol=1e-9)
            passed=passed and np.isclose(np.quantile(values,.95),profile[field+'_p95'],rtol=0,atol=1e-9)
    text_profile_checks.append({'clock':profile['clock'],'profile':profile['profile'],'runs':len(selected),'passed':bool(passed)})
check('ardy_startup_profiles_recomputed_from_raw_timers',len(text_profiles)==4
      and sum(r['runs'] for r in text_profiles)==480 and all(r['passed'] for r in text_profile_checks),text_profile_checks)
source_cutoffs={'primary':analysis['updated_at'],'trajectory':trajectory['updated_at']}
check('same_frozen_inference_source_and_atomic_replays',cohorts['passed']
      and cohorts['analysis_cutoffs']==source_cutoffs
      and snapshot['queries'].get('source_replay_methods',{}).get('source',{}).get('analysisCutoffs')==source_cutoffs
      and snapshot['queries'].get('source_replay_methods',{}).get('rows')==cohorts['rows']
      and cohorts['complete_groups']==cohorts['expected_groups']
      and cohorts['completed_processes']==cohorts['expected_processes'],
      {'complete_groups':cohorts['complete_groups'],'expected_groups':cohorts['expected_groups'],
       'completed_processes':cohorts['completed_processes'],'passed':cohorts['passed']})
epochs=read('20261025_execution_epochs/audit.json')
epoch_cutoffs={'primary':analysis['updated_at'],'trajectory':trajectory['updated_at']}
check('reboot_timing_sensitivity_current_and_complete',
      epochs['analysis_cutoffs']==epoch_cutoffs
      and snapshot['queries'].get('execution_epoch_sensitivity',{}).get('source',{}).get('analysisCutoffs')==epoch_cutoffs
      and snapshot['queries'].get('execution_epoch_sensitivity',{}).get('rows')==epochs['effects']
      and len(epochs['effects'])==7
      and all(r['primary_paired_n']==60 and r['same_epoch_paired_n']==59
              and r['removed_cross_epoch_blocks']==1 for r in epochs['effects']),
      {'cutoffs':epochs['analysis_cutoffs'],'mixed_blocks':epochs['mixed_blocks']})
coverage=analysis['matrix_acceptance']
check('all_registered_condition_cells',len(coverage)==11 and all(
    r['accepted_cells']==r['expected_cells'] and r['complete_blocks']==60 and r['complete_tasks']==12
    and not r['cells_requiring_receipt_reconciliation'] for r in coverage),
    {'accepted':sum(r['accepted_cells'] for r in coverage),'expected':sum(r['expected_cells'] for r in coverage)})
check('no_rejected_scientific_blocks',not analysis['rejected'],analysis['rejected'])
check('supplementary_native_trajectory_domain_complete',trajectory['accepted_cells']==240
      and trajectory['complete_blocks']==60 and trajectory['complete_tasks']==12 and trajectory['pilot_passed'],
      {k:trajectory[k] for k in ['accepted_cells','expected_cells','complete_blocks','complete_tasks']})
check('trajectory_input_negative_controls_and_receipts',len(trajectory['packing'])==12
      and all(p['anchors']==9 and p['hint_exact'] for p in trajectory['packing'])
      and trajectory['negative_controls_passed'] and trajectory['negative_control_compared_groups']==10
      and all(len(r['tasks'])==6 for r in trajectory['negative_control_members'])
      and all(r['intervention_checks']['passed'] and r['finite'] for r in trajectory['rows']),
      {'packing_tasks':len(trajectory['packing']),'compared_negative_groups':trajectory['negative_control_compared_groups']})
clock_counts={method:sum(p['method']==method for p in analysis['clock_pairs'])
              for method in ['ardy','dart','primal','closd']}
check('all_clock_pairings_identity_and_shape',clock_counts=={'ardy':240,'dart':120,'primal':480,'closd':120}
      and all(p['same_assets'] and p['same_shape'] for p in analysis['clock_pairs']),clock_counts)
clock_audit=read('20261024_native_clock_observations/audit.json')
expected_clock_runs={r['run'] for r in analysis['rows'] if r['method'] in ['dart','closd'] and r['clock']=='realtime'}
check('native_clock_observation_timestamp_bounds',clock_audit['passed']
      and {r['run'] for r in clock_audit['checks']}==expected_clock_runs,
      {'checked_runs':len(clock_audit['checks']),'ambiguous_runs':sum(r['timing_ambiguity'] for r in clock_audit['checks'])})
conditional_checks=[]
method_factors={m:{f for c in analysis['contrasts'] if c['method']==m for f in c['factors']}
                for m in {c['method'] for c in analysis['contrasts']}}
for main in analysis['contrasts']:
    if len(main['factors'])!=1:continue
    simple=[r for r in analysis.get('simple_effects',[]) if r['method']==main['method']
            and r['clock']==main['clock'] and r['metric']==main['metric'] and r['factor']==main['factors'][0]]
    if not simple:
        if len(method_factors[main['method']])>1:
            conditional_checks.append({'method':main['method'],'clock':main['clock'],
                'metric':main['metric'],'factor':main['factors'][0],'passed':False,
                'reason':'Conditional comparisons missing for an estimated multi-factor main effect'})
        continue
    expected=2**len(simple[0]['fixed'])
    delta=sum(r['mean_equal_group_weight'] for r in simple)/len(simple)-main['mean_equal_group_weight']
    conditional_checks.append({'method':main['method'],'clock':main['clock'],
        'metric':main['metric'],'factor':main['factors'][0],
        'passed':len(simple)==expected and all(r['paired_n']==main['paired_n'] for r in simple)
            and math.isclose(delta,0.,abs_tol=1e-9),'difference':delta})
check('conditional_effects_reconcile_to_marginals',bool(conditional_checks)
      and all(r['passed'] for r in conditional_checks),
      {'comparisons':len(conditional_checks),'failures':[r for r in conditional_checks if not r['passed']]})
seed_checks=[]
for query,evidence in [('native_effects',analysis),('trajectory_effects',trajectory)]:
    if snapshot['queries'][query]['source'].get('analysisCutoff')!=evidence['updated_at']:
        seed_checks.append({'query':query,'passed':False,'reason':'Report snapshot predates current analysis; rebuild before numeric reconciliation'})
        continue
    for effect in snapshot['queries'][query]['rows']:
        task_values={}
        for block in evidence['blocks']:
            if not block['complete'] or block.get('method','omnicontrol')!=effect['method'] or block.get('clock','paused')!=effect['clock']:
                continue
            matched=[c for c in block['contrasts'] if c['factors']==effect['factors'] and c['metric']==effect['metric']]
            if matched:task_values.setdefault(block['task'],{})[block['seed']]=matched[0]['delta']
        balanced={task:values for task,values in task_values.items() if set(values)==set(range(20,25))}
        observed=effect.get('seed_sensitivity',{})
        passed=observed.get('balanced_tasks')==sorted(balanced) and observed.get('observed_tasks')==len(task_values)
        expected_omissions={seed:sum(sum(value for key,value in values.items() if key!=seed)/4
                            for values in balanced.values())/len(balanced) for seed in range(20,25)} if balanced else {}
        actual_omissions={r['omitted_seed']:r['equal_task_mean_delta'] for r in observed.get('leave_one_seed_out',[])}
        passed=passed and set(expected_omissions)==set(actual_omissions) and all(
            math.isclose(value,actual_omissions[seed],rel_tol=1e-10,abs_tol=1e-10) for seed,value in expected_omissions.items())
        seed_checks.append({'query':query,'method':effect['method'],'metric':effect['metric'],
                            'factors':effect['factors'],'passed':passed})
check('balanced_seed_omissions_recomputed',bool(seed_checks) and all(r['passed'] for r in seed_checks),
      {'comparisons':len(seed_checks),'failures':[r for r in seed_checks if not r['passed']]})
curve_groups={}
for curve in snapshot['queries']['motion_curves']['rows']:
    curve_groups.setdefault(curve['run'],[]).append(curve)
curve_checks=[]
for row in analysis['rows']+trajectory['rows']:
    visual_id=row['run']+('#'+row['variant'] if row.get('variant') else '')
    if visual_id not in curve_groups:continue
    motion_file=ROOT/row['run']/('factorial/'+row['variant']+'_canonical.npz' if row.get('variant') else 'motion.npz')
    with np.load(motion_file) as raw_motion:
        raw_xyz=np.asarray(raw_motion['joints'],dtype=np.float64)
        raw_fps=float(raw_motion['fps'])
    raw_jerk=np.diff(raw_xyz,n=3,axis=0)*raw_fps**3
    full_precision_metric=float(np.sqrt(np.mean(raw_jerk**2)))
    curve=curve_groups[visual_id]
    # A run can appear in both a pilot and a post-hoc diagnostic case.
    cases={r['case'] for r in curve}
    for case in cases:
        local=sorted((r for r in curve if r['case']==case),key=lambda r:r['frame'])
        values=[r.get('jerk_component_rms_m_s3') for r in local[3:]]
        valid=bool(values) and all(v is not None and math.isfinite(v) for v in values)
        curve_rms=math.sqrt(sum(v*v for v in values)/len(values)) if valid else None
        expected=row['values']['jerk_rms']
        curve_checks.append({'run':visual_id,'case':case,'passed':valid
            and all(r.get('jerk_component_rms_m_s3') is None for r in local[:3])
            and math.isclose(curve_rms,full_precision_metric,rel_tol=1e-12,abs_tol=1e-12)
            # Native DART/CLoSD saved reductions ran in float32. Use the
            # existing core reconstruction contract for that comparison;
            # the curve-to-float64-source check above remains much tighter.
            and math.isclose(full_precision_metric,expected,rel_tol=1e-6,abs_tol=1e-8),
            'reconstructed':curve_rms,'raw_float64_metric':full_precision_metric,'saved_metric':expected})
accepted_visual_ids={r['run']+('#'+r['variant'] if r.get('variant') else '') for r in analysis['rows']+trajectory['rows']}
stale_visuals=[r['run'] for r in snapshot['queries']['motion_cases']['rows'] if r['method']!='kimodo' and r['run'] not in accepted_visual_ids]
check('plotted_jerk_reconciles_to_raw_metrics',bool(curve_checks) and all(r['passed'] for r in curve_checks) and not stale_visuals,
      {'new_matrix_motion_cases':len(curve_checks),'failures':[r for r in curve_checks if not r['passed']],
       'motion_cases_outside_current_accepted_evidence':stale_visuals,
       'scope':'Curve squared mean matches float64 recomputation from saved coordinates (1e-12); saved native float32 reductions use existing metric reconstruction tolerances (rtol 1e-6, atol 1e-8). Historical Kimodo mapping is not asserted here'})
latency_checks=[]
for row in snapshot['queries']['native_runs']['rows']:
    if row['method'] not in ('ardy','dart','closd'):continue
    source=ROOT/row['run']/('worker_result.json' if row['method']=='ardy' else 'trace.json')
    original=json.loads(source.read_text())
    samples=(original['timing']['replan']['samples_seconds'] if row['method']=='ardy'
             else [step['seconds'] for step in original['steps']])
    displayed=row.get('request_latency') or {}
    passed=displayed.get('samples_seconds')==samples and displayed.get('source')==str(source.relative_to(ROOT))
    passed=passed and all(math.isfinite(s) and s>=0 for s in samples) and len(samples)=={'ardy':50,'dart':75,'closd':600}[row['method']]
    latency_checks.append({'run':row['run'],'passed':passed,'requests':len(samples)})
check('request_latency_curves_preserve_all_recorded_calls',bool(latency_checks) and all(r['passed'] for r in latency_checks),
      {'runs':len(latency_checks),'failures':[r for r in latency_checks if not r['passed']],
       'scope':'No startup or slow request removed; distinct replan, policy+step and env_step timing scopes retained'})
closd_observations=[]
for row in analysis['rows']:
    if row['method']!='closd':continue
    native_config=json.loads((ROOT/row['run']/'resolved_config.json').read_text())
    actual={key:native_config['env'].get(key) for key in ['obs_v','self_obs_v','fut_tracks','control_mode','controlFrequencyInv']}
    actual['sim_substeps']=native_config['sim']['substeps']
    closd_observations.append({'run':row['run'],'actual':actual,
        'passed':actual=={'obs_v':7,'self_obs_v':1,'fut_tracks':False,'control_mode':'isaac_pd','controlFrequencyInv':2,'sim_substeps':2}})
check('closd_loaded_observation_contract',bool(closd_observations) and all(r['passed'] for r in closd_observations),
      {'runs':len(closd_observations),'failures':[r for r in closd_observations if not r['passed']],
       'scope':'self_obs_v1 has body states, not v3 force sensors; task demo calls imitation obs v7; isaac_pd sends DOF position targets; 600 env_step records are control steps, each normally advances two simulate calls with two internal substeps'})
check('all_run_weights',runs<=weight_runs,{'missing':sorted(runs-weight_runs)})
check('all_run_preprocessing',runs<=preprocessing_runs,{'missing':sorted(runs-preprocessing_runs)})
check('all_core_metrics_reconstructed',cells==reconstructed and metrics['passed']
      and {r['run'] for r in metrics.get('reset_checks',[]) if r['passed']}=={r['run'] for r in analysis['rows'] if r['method']=='closd'},
      {'analysis_cells':len(cells),'reconstructed_cells':len(reconstructed),'reset_aware_runs':len(metrics.get('reset_checks',[]))})
check('output_negative_controls_current',invariants['status']=='passed' and invariants['analysis_cutoff']==analysis['updated_at'],
      {'audit_cutoff':invariants['analysis_cutoff'],'analysis_cutoff':analysis['updated_at']})
check('no_observed_gpu_overlap',execution['passed_observed_checks'],
      {'overlaps':execution['overlaps'],'missing_terminal_intervals':execution['missing_terminal_intervals']})
test_path=base/'20261024_native_execution/unit_tests.xml'
test_suites=list(ET.parse(test_path).getroot().iter('testsuite')) if test_path.exists() else []
test_counts={field:sum(int(s.get(field,0)) for s in test_suites)
             for field in ['tests','failures','errors','skipped']}
test_names={c.get('name') for s in test_suites for c in s.findall('testcase')}
required_runtime_tests={'test_actual_timeout_terminates_spawned_process_group',
    'test_child_receives_requested_physical_device_and_missing_protocol_fails',
    'test_device_leases_are_exclusive_but_independent',
    'test_failure_isolation_bounded_retry_and_resume_identity'}
check('unit_and_real_process_supervision_tests',test_counts['tests']>0
      and test_counts['failures']==test_counts['errors']==0 and required_runtime_tests<=test_names,
      {**test_counts,'scope':'CPU process/lease and queue tests; actual GPU routing and scientific experiment acceptance are separate checks'})
check('kimodo_416_pairs',kimodo['accepted_pairs']==kimodo['expected_pairs']==416 and not kimodo['failures'],
      {'accepted':kimodo['accepted_pairs'],'failures':kimodo['failures']})
check('kimodo_weight_identity',kimodo_weights['passed'],kimodo_weights['scope'])
replay_job=read('20261024_kimodo_recorrection/queue/state.json')['jobs']['kimodo_official_correction_replay']
replay=json.loads((Path(replay_job['run_dir'])/'replay.json').read_text()) if replay_job['status']=='success' else {}
check('kimodo_raw_to_corrected_numeric_identity',replay.get('checked_pairs')==416 and replay.get('passed')
      and all(f['max_abs_difference']==0 for c in replay.get('checks',[]) for f in c['fields']),
      {'checked_pairs':replay.get('checked_pairs'),'device':replay.get('device')})
timing=read('20261024_native_timing/analysis.json')
primal_timing=timing.get('primal_clock_decomposition',[])
check('primal_pacing_decomposition_and_identical_replay',len(primal_timing)==16
      and all(r['passed'] for r in primal_timing),{'checked_conditions':len(primal_timing)})
primary_timing=timing.get('primal_primary_decomposition',[])
check('primal_primary_nonwait_timings_complete_and_recomputed',
      timing.get('primary_analysis_cutoff')==analysis['updated_at'] and len(primary_timing)==960
      and all(r['passed'] for r in primary_timing)
      and all(sum(r['clock']==clock for r in primary_timing)==480 for clock in ['paused','realtime']),
      {'checked_conditions':len(primary_timing),'summary':timing.get('primal_primary_summary',[])})
check('report_current_analysis',snapshot['queries']['matrix_acceptance']['source'].get('analysisCutoff')==analysis['updated_at'],
      {'report_cutoff':snapshot['queries']['matrix_acceptance']['source'].get('analysisCutoff'),'analysis_cutoff':analysis['updated_at']})
check('report_final_completion_state',snapshot.get('buildStatus')=='complete',
      {'build_status':snapshot.get('buildStatus'),'scope':'Preview snapshots deliberately do not pass final delivery acceptance'})
check('report_current_trajectory_analysis',snapshot['queries'].get('trajectory_coverage',{}).get('source',{}).get('analysisCutoff')==trajectory['updated_at'],
      {'trajectory_cutoff':trajectory['updated_at']})
check('browser_checked_current_html',browser.get('status')=='passed' and browser.get('build',{}).get('htmlSha256')==checksum(report/'dist/index.html'),
      {'status':browser.get('status'),'checked_html':browser.get('build',{}).get('htmlSha256')})
check('built_data_matches_reviewed_snapshot',build['sourceSnapshotSha256']==checksum(report/'src/data.json')
      and browser.get('build',{}).get('snapshotSha256')==build['sourceSnapshotSha256'],
      {'built_snapshot':build['sourceSnapshotSha256'],'reviewed_snapshot':checksum(report/'src/data.json')})
check('eight_methods_sixteen_figures',len(browser.get('methodPages',[]))==8 and all(r['diagrams']==2 for r in browser.get('methodPages',[])),
      {'methods':len(browser.get('methodPages',[])),'diagrams':browser.get('diagrams',{}).get('count')})
embedded=[asset for report_row in snapshot['queries']['historical_reports']['rows']
          for asset in report_row.get('video_assets',{}).values()]
check('original_video_bytes_embedded',len(embedded)==20 and all(
    hashlib.sha256(base64.b64decode(a['base64'],validate=True)).hexdigest()==a['sha256']==checksum(Path(a['source_path']))
    for a in embedded),{'original_mp4_files':len(embedded),'bytes':sum(a['bytes'] for a in embedded)})
embedded_images=[asset for report_row in snapshot['queries']['historical_reports']['rows']
                 for asset in report_row.get('image_assets',{}).values()]
check('original_case_and_gt_images_embedded',len(embedded_images)==24 and all(
    hashlib.sha256(base64.b64decode(a['base64'],validate=True)).hexdigest()==a['sha256']==checksum(Path(a['source_path']))
    for a in embedded_images),{'original_png_files':len(embedded_images)})
offline_path=base/'20261024_native_pilots/offline/preview_checks.json'
offline=json.loads(offline_path.read_text()) if offline_path.exists() else {}
offline_file=Path(offline.get('build',{}).get('offline',{}).get('path','/nonexistent-motiongen-export'))
check('single_html_file_protocol_playback',offline.get('status')=='passed' and offline_file.is_file()
      and offline.get('build',{}).get('offline',{}).get('sha256')==checksum(offline_file)
      and offline.get('build',{}).get('snapshotSha256')==build['sourceSnapshotSha256']
      and offline.get('build',{}).get('htmlSha256')==build['html']['sha256'],
      {'status':offline.get('status'),'file':str(offline_file)})
coordinates=read('20261024_spatial_coordinates/audit.json')
chois_metrics=read('20261024_chois_native_metrics/audit.json')
ardy_history=read('20261024_ardy_history_audit/audit.json')
check('ardy_common_recent_history_equal_denominators',ardy_history['passed']
      and {r['run'] for r in ardy_history['checks']}=={r['run'] for r in analysis['rows'] if r['method']=='ardy'},
      ardy_history['summary'])
chois_contact=read('20261024_chois_contact_audit/audit.json')
check('chois_full_mesh_contact_confusion_reconstructed',chois_contact['passed']
      and {r['run'] for r in chois_contact['checks']}=={r['run'] for r in analysis['rows'] if r['method']=='chois'},
      {'checked_runs':len(chois_contact['checks']),'metrics':len(chois_contact['summary'])})
chois_load=read('20261024_chois_checkpoint_load/audit.json')
check('chois_actual_inference_tensors_fully_loaded',chois_load['passed'] and not chois_load['missing_keys']
      and chois_load['all_loaded_values_exact'] and set(chois_load['runs'])=={r['run'] for r in analysis['rows'] if r['method']=='chois'},
      {'loaded_tensors':chois_load['loaded_state_tensors'],'unused_published_keys':chois_load['unexpected_keys'],
       'full_checkpoint_strict_probe':chois_load['full_checkpoint_strict_probe']['passed']})
check('chois_saved_object_metrics_independently_reconstructed',chois_metrics['passed']
      and {r['run'] for r in chois_metrics['checks']}=={r['run'] for r in analysis['rows'] if r['method']=='chois'}
      and {r['run'] for r in chois_metrics.get('persistence_checks',[])}=={r['run'] for r in analysis['rows'] if r['method']=='chois'}
      and all(r['passed'] for r in chois_metrics.get('persistence_checks',[])),
      {'checked_runs':len(chois_metrics['checks']),'metrics':len(chois_metrics['summary']),
       'contact_persistence_runs':len(chois_metrics.get('persistence_checks',[]))})
expected_coordinate_runs={r['run'] for r in analysis['rows'] if r['method']=='motionlcm'}
checked_coordinate_runs={r['run'] for r in coordinates['runs']}
check('motionlcm_input_and_native_coordinate_audit',coordinates['passed']
      and all(r['official_sparse_constructor_exact'] for r in coordinates['packing'])
      and len(coordinates['packing'])==12 and expected_coordinate_runs==checked_coordinate_runs,
      {'tasks':len(coordinates['packing']), 'checked_runs':len(checked_coordinate_runs),
       'missing_runs':sorted(expected_coordinate_runs-checked_coordinate_runs)})
marker=read('20261024_closd_marker_probe/analysis.json')
marker_rows=marker['rows'];marker_runs={r['run'] for r in marker_rows}
same_planner_cases={r['case'] for r in marker['cross_clock_cases'] if r['same_planner_goal_frame']}
marker_complete=(len(marker_rows)==24 and len(marker['cases'])==6
    and {c['case'] for c in marker['cases']}==same_planner_cases
    and not marker_runs.intersection(runs)
    and all(r['finite'] and r['frames']==600 and r['record_sha256']==checksum(ROOT/r['run']/'marker_probe.json') for r in marker_rows))
check('closd_marker_diagnostic_complete_and_separate',marker_complete,
    {'condition_cells':len(marker_rows),'complete_cases':len(marker['cases']),
     'overlap_with_primary':sorted(marker_runs.intersection(runs)),
     'scope':'Acceptance requires complete traceable diagnostic data, not a preferred hypothesis outcome. Native replay equivalence and isolated-schedule differences remain explicit results.'})
check('report_current_marker_diagnostic',snapshot['queries'].get('closd_marker_cases',{}).get('source',{}).get('analysisCutoff')==marker['updated_at'],
    {'diagnostic_cutoff':marker['updated_at']})
receipt={'analysis_cutoff':analysis['updated_at'],'passed':all(c['passed'] for c in checks),
         'checks':checks,'scope':'Consolidates existing acceptance evidence; not a substitute for source review, narrative conclusions, or visual inspection. Incomplete coverage is expected while queues run and does not stop them.'}
json_write(base/'20261024_native_acceptance/audit.json',receipt)
print(json.dumps({'passed':receipt['passed'],'checks':len(checks),
                  'remaining':[c for c in checks if not c['passed']]},ensure_ascii=False))
if args.require_complete and not receipt['passed']:raise SystemExit(1)
