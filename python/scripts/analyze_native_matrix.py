"""Evidence-backed factorial deltas; incomplete cells never become zero scores."""
import hashlib
import json
import sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.metrics import angle_degrees
from motiogen.native_analysis import factorial_block,clustered_interval,foot_sensitivity,root_dwell,wall_observation_recovery,physics_contact_sensitivity,boundary_kinematics
from motiogen.runtime import json_write,checksum
from motiogen.feedback import score_root_goals
from motiogen.native_receipt_checks import intervention_checks
from motiogen.native_analysis import task_sensitivity,process_incidents,common_history_consistency
from motiogen.request_latency import matched_ardy_steady_metrics
from motiogen.native_analysis import geometric_contact_persistence

ANALYSIS_SOURCES={str(p.relative_to(ROOT)):checksum(p) for p in [
    Path(__file__).resolve(),ROOT/'utils/motiogen/native_analysis.py',
    ROOT/'utils/motiogen/metrics.py',ROOT/'utils/motiogen/feedback.py',
    ROOT/'utils/motiogen/motion.py',ROOT/'utils/motiogen/native_receipt_checks.py',
    ROOT/'config/native_metric_contract.json',ROOT/'utils/motiogen/request_latency.py']}

BASE=ROOT/'develop_logs/test_runs'
SOURCES=[('20261024_control_matrix/plan.json','20261024_control_matrix/queue/state.json'),
 ('20261024_control_matrix/continuation_plan.json','20261024_control_matrix/queue_fair/state.json'),
 ('20261024_native_followups/plan.json','20261024_native_followups/queue_fair/state.json'),
 ('20261024_dart_matrix/plan.json','20261024_dart_matrix/queue/state.json'),
 ('20261024_native_pilots/ardy_plan_cfg_corrected.json','20261024_native_pilots/ardy_queue_cfg_corrected_fair/state.json'),
 ('20261024_ardy_realtime_pilot/plan.json','20261024_ardy_realtime_pilot/queue/state.json'),
 ('20261024_ardy_matrix/plan.json','20261024_ardy_matrix/queue/state.json'),
 ('20261024_ardy_resource_recovery/plan.json','20261024_ardy_resource_recovery/queue/state.json'),
 ('20261024_primal_matrix/plan.json','20261024_primal_matrix/queue/state.json'),
 ('20261024_primal_receipt_recovery/plan.json','20261024_primal_receipt_recovery/queue/state.json'),
 ('20261024_primal_realtime_pilot/plan.json','20261024_primal_realtime_pilot/queue/state.json'),
 ('20261024_primal_realtime_matrix/plan.json','20261024_primal_realtime_matrix/queue/state.json'),
 ('20261024_closd_pilot/plan.json','20261024_closd_pilot/queue_loader_v3/state.json'),
 ('20261024_closd_realtime_pilot/plan.json','20261024_closd_realtime_pilot/queue/state.json'),
 ('20261024_closd_matrix/plan.json','20261024_closd_matrix/queue/state.json'),
 ('20261024_control_matrix/recovery_plan.json','20261024_control_matrix/recovery_queue/state.json'),
 ('20261024_ardy_matrix/recovery_plan.json','20261024_ardy_matrix/recovery_queue/state.json'),
 ('20261025_source_replays/plan.json','20261025_source_replays/queue/state.json')]
FACTORS={'motionlcm':['control_residual'],'omnicontrol':['realism','spatial'],
         'chois':['guidance'],'dart':['goal_update'],'ardy':['history_long','correction'],
         'primal':['ground_snap','reprojection','inertialization'],'closd':['execution_context']}
rows=[];seen=set();coverage={};rejected=[];operational_timing={}
incidents={}
contract=json.loads((ROOT/'config/native_metric_contract.json').read_text())
weight_audit_path=BASE/'20261024_native_weight_audit/audit.json'
weight_audit={r['run']:r for r in json.loads(weight_audit_path.read_text())['rows']} if weight_audit_path.exists() else {}
superseded=set();replacement_checks=[]
deferred_source_jobs=set();source_replay_replacements=[]
def load_replacements():
    """Superseded runs and replacement receipts of the matrix plans (module state read by read_run)."""
    for plan_file,state_file in SOURCES:
        if not (BASE/plan_file).exists() or not (BASE/state_file).exists():continue
        p=json.loads((BASE/plan_file).read_text());s=json.loads((BASE/state_file).read_text())
        for group in p.get('source_replay_groups',[]):
            if all(s['jobs'].get(job,{}).get('status')=='success' for job in group['jobs']):
                superseded.update(str((ROOT/run).resolve()) for run in group['prior_runs'])
                source_replay_replacements.append({**group,'replacement_runs':[str(Path(s['jobs'][job]['run_dir']).relative_to(ROOT)) for job in group['jobs']]})
            else:deferred_source_jobs.update(group['jobs'])
        for replacement in p.get('replacements',[]):
            if s['jobs'][replacement['job']]['status']=='success':
                old=Path(replacement['prior_run'])/'factorial';new=Path(s['jobs'][replacement['job']]['run_dir'])/'factorial'
                checks=[]
                for file in sorted(old.glob('*_canonical.npz')):
                    a=load_motion(file)['joints'];b=load_motion(new/file.name)['joints']
                    checks.append({'variant':file.stem,'exact_same_coordinates':bool(np.array_equal(a,b)),
                        'max_abs_delta_m':float(np.max(np.abs(a-b))) if a.shape==b.shape else None})
                if len(checks)!=8 or not all(c['exact_same_coordinates'] for c in checks):
                    raise ValueError('Instrumented PRIMAL replay changed original motion: '+replacement['job'])
                replacement_checks.append({'job':replacement['job'],'old_run':replacement['prior_run'],'replacement_run':str(new.parent),'checks':checks,
                    'scope':'Instrumentation replay, not an additional independent sample; original raw data retained'})
            superseded.add(str(Path(replacement['prior_run']).resolve()))

def read_primal(path,task,manifest):
    folder=path/'factorial'
    records=json.loads((folder/'timing.json').read_text())
    if len(records)!=8:raise ValueError('Incomplete PRIMAL factorial process marked successful')
    for record in records:
        motion_path=folder/(record['variant']+'_canonical.npz');motion=load_motion(motion_path)
        metrics=json.loads((folder/(record['variant']+'_metrics.json')).read_text())['metrics']
        values={k:v['value'] for k,v in metrics.items() if isinstance(v.get('value'),(int,float)) and np.isfinite(v['value'])}
        values.update(boundary_kinematics(motion['joints'],motion['fps'],np.arange(14,len(motion['joints']),14)))
        if record.get('gpu_peak_bytes') is not None:values['peak_torch_allocated_gib']=record['gpu_peak_bytes']/2**30
        setup_path=folder/'setup_timing.json'
        setup=json.loads(setup_path.read_text()) if setup_path.exists() else None
        if setup:values.update({k:v for k,v in setup.items() if isinstance(v,(float,int)) and k.endswith('_seconds')})
        values.update(native_generation_wall_seconds=record['seconds'],native_generation_wall_rtf=record['seconds']/(record['frames']/record['fps']))
        if 'nonwait_generation_seconds' in record:
            wait=record['paced_wait_seconds'];nonwait=record['nonwait_generation_seconds']
            if not (np.isfinite([wait,nonwait]).all() and min(wait,nonwait)>=0
                    and np.isclose(wait+nonwait,record['seconds'],rtol=0,atol=1e-9)):
                raise ValueError('PRIMAL recorded wait/nonwait decomposition mismatch')
            values.update(paced_wait_seconds=wait,nonwait_generation_seconds=nonwait,
                          nonwait_generation_rtf=nonwait/(record['frames']/record['fps']))
        if record.get('clock','paused')=='paused':
            values.update(model_step_total_seconds=record['seconds'],model_step_rtf=record['seconds']/(record['frames']/record['fps']))
        feet=foot_sensitivity(motion,contract)
        if feet['status']=='measured':
            values['foot_floor_penetration_fraction']=feet['floor_penetration_fraction']
            f=next(r for r in feet['rows'] if r['height_m']==.05 and r['speed_m_s']==.1)
            values['foot_candidate_contact_fraction_5cm']=f['candidate_fraction']
            if f['sliding_fraction_given_candidate'] is not None:values['foot_sliding_given_5cm_contact_10cms']=f['sliding_fraction_given_candidate']
        row={'method':'primal','task':task,'seed':manifest['seed'],'clock':record.get('clock','paused'),
             'condition':{k:int(record[k]) for k in FACTORS['primal']},'values':values,
             'units':{k:v.get('unit','') for k,v in metrics.items()},'run':str(path.relative_to(ROOT)),
             'variant':record['variant'],'motion_sha256':checksum(motion_path),'frames':len(motion['joints']),'fps':float(motion['fps']),
             'finite':bool(np.isfinite(motion['joints']).all()),'input_signature':checksum(folder/'initial_history.npz'),
             'asset_receipts_sha256':checksum(path/'asset_receipts.json'),
             'trajectory_sha256':hashlib.sha256(np.asarray(motion['joints'],dtype='<f8').tobytes()).hexdigest(),
             'calls':record.get('native_branch_calls'),'foot_sensitivity':feet}
        family=int(record.get('initial_history',manifest['scenario'].get('initial_history',0)))
        initial_seed=int(record.get('initial_seed',manifest['scenario'].get('initial_seed',9000)))
        row['resampling_group']='stand' if family==0 else 'warm_rng_'+str(initial_seed+family)
        row['initial_state_scope']='Model-generated warm terminal history; short/long histories with the same effective RNG seed share a bootstrap parent, not independent real source motions.'
        row['boundary_scope']='Native generator concatenates 14 committed frames per primitive; recomputed at frames 14,28,... from canonical output. Position step is not a discontinuity residual against a reference velocity.'
        row['setup_timing']=setup
        row['task_spec']={**manifest['scenario'],'initial_history':family,'initial_seed':initial_seed}
        if not row['finite']:rejected.append({'run':row['run'],'variant':record['variant'],'reason':'non-finite motion'})
        else:rows.append(row)

def read_run(path,task,clock='paused'):
    path=Path(path);path=path if path.is_absolute() else ROOT/path
    if str(path.resolve()) in superseded:return
    if str(path) in seen:return
    seen.add(str(path))
    manifest=json.loads((path/'manifest.json').read_text());method=manifest['baseline']
    if method not in FACTORS:return
    if method=='primal':return read_primal(path,task,manifest)
    motion=load_motion(path/'motion.npz');intervention=json.loads((path/'intervention.json').read_text())
    condition=dict(intervention['condition'])
    if method=='ardy':condition['history_long']=int(condition.pop('history_frames')==160)
    if method=='closd':condition['execution_context']=int(condition.pop('context_source')=='simulation')
    metrics=json.loads((path/'metrics.json').read_text())['metrics']
    values={key:value['value'] for key,value in metrics.items() if isinstance(value.get('value'),(int,float)) and np.isfinite(value['value'])}
    units={key:value.get('unit','') for key,value in metrics.items()}
    feet=foot_sensitivity(motion,contract)
    physics=None
    if method=='closd':
        with np.load(path/'native_physics.npz',allow_pickle=False) as native:
            physics=physics_contact_sensitivity(native['joints'],native['contact_forces'],native['body_names'],motion['fps'],contract)
        main_contact=next(r for r in physics['rows'] if r['force_threshold_n']==5.)
        values['native_force_foot_contact_fraction_5n']=main_contact['contact_fraction']
        values['native_nonfoot_contact_frame_fraction_5n']=main_contact['nonfoot_contact_frame_fraction']
        if main_contact['foot_body_translation_over_10cms_given_contact'] is not None:
            values['native_force_contact_foot_body_motion_fraction_5n_10cms']=main_contact['foot_body_translation_over_10cms_given_contact']
        units.update({k:'fraction' for k in values if k.startswith('native_force_') or k.startswith('native_nonfoot_')})
    if feet['status']=='measured':
        values['foot_floor_penetration_fraction']=feet['floor_penetration_fraction']
        for foot in feet['rows']:
            if foot['height_m']==.05 and foot['speed_m_s']==.1:
                values['foot_candidate_contact_fraction_5cm']=foot['candidate_fraction']
                if foot['sliding_fraction_given_candidate'] is not None:
                    values['foot_sliding_given_5cm_contact_10cms']=foot['sliding_fraction_given_candidate']
    if method=='ardy':
        common=common_history_consistency(motion['overlap_errors_m'],
            intervention['condition']['history_frames'],8,len(motion['joint_names']),len(motion['joints'])//8)
        values['history_consistency_common32_m']=common['mean_m']
        values['history_consistency_common32_samples']=common['samples']
        units['history_consistency_common32_m']='m'
        units['history_consistency_common32_samples']='joint-frame observations (not independent samples)'
        path_spec=intervention['task']
        if path_spec=='Root2D waypoints: native X=0, Z=.4*min(t,5)-.2*max(t-5,0), metres':
            path_spec={'speed_before_m_s':.4,'speed_after_m_s':-.2,'root_direction_xz':[0.,1.]}
        if not isinstance(path_spec,dict):raise ValueError('Unknown ARDY task receipt')
        frames=np.arange(7,len(motion['joints']),8)
        seconds=frames/float(motion['fps'])
        distance=path_spec['speed_before_m_s']*np.minimum(seconds,5.)+path_spec['speed_after_m_s']*np.maximum(seconds-5.,0.)
        direction=np.asarray(path_spec['root_direction_xz'])*np.array([1.,-1.])
        errors=np.linalg.norm(motion['joints'][frames,0,:2]-distance[:,None]*direction,axis=-1)
        values.update(root_waypoint_mean_error_m=float(errors.mean()),root_waypoint_success_5cm=float((errors<=.05).mean()))
        all_seconds=np.arange(len(motion['joints']))/float(motion['fps'])
        all_distance=path_spec['speed_before_m_s']*np.minimum(all_seconds,5.)+path_spec['speed_after_m_s']*np.maximum(all_seconds-5.,0.)
        all_error=np.linalg.norm(motion['joints'][:,0,:2]-all_distance[:,None]*direction,axis=-1)
        values['root_reference_path_mean_deviation_m']=float(all_error.mean())
        values['root_reference_path_last5s_mean_deviation_m']=float(all_error[-round(5*motion['fps']):].mean())
        event_rows=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        updates=[r for r in event_rows if r['event']=='prompt_update']
        if updates:
            values['text_update_motion_delay_seconds']=updates[0]['applied_seconds']-updates[0]['requested_seconds']
        clock_trace=path/'clock_trace.json'
        if clock_trace.exists() and updates:
            real_steps=json.loads(clock_trace.read_text())['steps']
            first=next(r for r in real_steps if r['request_wall_seconds']>=5.)
            values['wall_input_observation_delay_seconds']=first['request_wall_seconds']-5.
            values['wall_first_updated_chunk_completion_seconds']=first['request_wall_seconds']+first['seconds']-5.
            values['max_schedule_lag_seconds']=max(r['schedule_lag_seconds'] for r in real_steps)
    if method=='chois':
        with np.load(path/'raw/native_prediction.npz') as raw:
            persistence=geometric_contact_persistence(raw['hand_object_distances_m'],motion['fps'])
        values['geometric_hand_contact_eligible_transitions']=persistence['eligible_hand_transitions']
        units['geometric_hand_contact_eligible_transitions']='hand transitions (not independent samples)'
        values['geometric_hand_contact_longest_span_seconds']=persistence['longest_observed_span_seconds']
        if persistence['retention_fraction'] is not None:
            values['geometric_hand_contact_retention_fraction']=persistence['retention_fraction']
        # Official CHOIS appends middle-finger anchors (body joints 28/43) as
        # joints 22/23. The historical transport named these *_wrist for
        # contact evaluation, while retaining anatomical wrists as *_wrist_body.
        # Preserve that legacy proxy, but do not call it anatomical elbow ROM.
        joint_index={str(name):i for i,name in enumerate(motion['joint_names'])}
        for side in ['left','right']:
            key=side+'_elbow_rom'
            if key in values:
                values['legacy_hand_anchor_'+key]=values.pop(key)
                units['legacy_hand_anchor_'+key]='deg'
            shoulder=motion['joints'][:,joint_index[side+'_shoulder']]
            elbow=motion['joints'][:,joint_index[side+'_elbow']]
            wrist=motion['joints'][:,joint_index[side+'_wrist_body']]
            angles,valid=angle_degrees(shoulder-elbow,wrist-elbow)
            if valid.any():values[key]=float(np.ptp(180-angles[valid]));units[key]='deg'
        rotations=motion['object_rotations']
        values['object_raw_rotation_orthogonality_mean']=float(np.linalg.norm(rotations@rotations.transpose(0,2,1)-np.eye(3),axis=(-2,-1)).mean())
        units['object_raw_rotation_orthogonality_mean']='Frobenius (dimensionless)'
        native=list((path/'raw/chois_results/evaluation_metrics_json').rglob('evaluation_metrics_for_all_test_data.json'))
        if len(native)!=1:raise ValueError(f'Expected one native object receipt: {path}')
        for key,value in json.loads(native[0].read_text()).items():
            if isinstance(value,(float,int)) and np.isfinite(value):values['native_object_'+key]=value
    task_results=None;reset_semantics=None
    trace_path=path/'trace.json'
    if trace_path.exists():
        trace=json.loads(trace_path.read_text());clock=trace.get('clock',clock)
        if 'cold_start_seconds' in trace:values['cold_start_seconds']=trace['cold_start_seconds'];units['cold_start_seconds']='s'
        steps=trace.get('steps',[])
        if steps:
            times=[r['seconds'] for r in steps]
            values.update(request_p50_seconds=float(np.quantile(times,.5)),request_p95_seconds=float(np.quantile(times,.95)),
                          model_step_total_seconds=float(sum(times)),model_step_rtf=float(sum(times)/(len(motion['joints'])/motion['fps'])))
        if 'goal_error_xy_m' in trace:
            values['native_reset_count']=sum(r['terminated'] or r['truncated'] for r in steps)
            units['native_reset_count']='count'
            error=np.asarray(trace['goal_error_xy_m']);values['new_goal_final_error_m']=float(error[-1]);units['new_goal_final_error_m']='m'
            goals=trace.get('requested_goals')
            if goals is None and task==0:goals=[{'frame':0,'position':[1,0,0]},{'frame':150,'position':[-1,.5,0]}]
            if goals is None:raise ValueError('DART goal metadata absent for nonpilot task')
            if clock=='realtime':
                observed=next(r for r in trace['steps'] if r['request_wall_seconds']>=5.)
                goals=[goals[0],{**goals[1],'frame':observed['frame']}]
                values['wall_input_observation_delay_seconds']=observed['request_wall_seconds']-5.
            task_results=score_root_goals(motion['joints'],goals,fps=motion['fps'],radius=contract['root_goal_radius_m'],hold_seconds=contract['root_goal_hold_seconds'])
            values['new_goal_final_error_m']=task_results[1]['final_error_m']
            values['new_goal_success_30cm_hold_05s']=int(task_results[1]['success'])
            if task_results[1]['recovery_seconds'] is not None:values['successful_new_goal_recovery_seconds']=task_results[1]['recovery_seconds']
        if method=='closd':
            boundaries=sorted({p['frame'] for p in trace['planner'] if p['frame']>0})
            values.update(boundary_kinematics(motion['joints'],motion['fps'],boundaries))
            goals=trace.get('requested_goals')
            if goals is None and task==0:goals=[{'frame':0,'position':[2,0,0]},{'frame':150,'position':[-1.5,1,0]}]
            if goals is None:raise ValueError('CLoSD task goals absent')
            if clock=='realtime':
                observed=next(r for r in trace['steps'] if r['request_wall_seconds']>=5.)
                goals=[goals[0],{**goals[1],'frame':observed['frame']}]
                values['wall_input_observation_delay_seconds']=observed['request_wall_seconds']-5.
            task_results=score_root_goals(motion['joints'],goals,fps=motion['fps'],radius=contract['root_goal_radius_m'],hold_seconds=contract['root_goal_hold_seconds'])
            values['new_goal_success_30cm_hold_05s']=int(task_results[1]['success'])
            values['new_goal_final_error_m']=task_results[1]['final_error_m']
            values['native_reset_count']=sum(r['done'] for r in trace['steps'])
            from motiogen.native_analysis import reset_aware_arrival
            reset_semantics=reset_aware_arrival(motion['joints'],motion['fps'],goals[-1]['position'],
                goals[-1]['frame'],[r['frame'] for r in trace['steps'] if r['done']],
                radius=contract['root_goal_radius_m'],hold_seconds=contract['root_goal_hold_seconds'])
            for metric in ['new_goal_success_before_next_reset','new_goal_success_reset_free_episode']:
                values[metric]=reset_semantics[metric];units[metric]='fraction'
            if task_results[1]['recovery_seconds'] is not None:values['successful_new_goal_recovery_seconds']=task_results[1]['recovery_seconds']
            values['env_step_total_seconds']=values.pop('model_step_total_seconds')
            values['env_step_rtf']=values.pop('model_step_rtf')
            if 'execution_wall_seconds' in trace:values['execution_wall_seconds']=trace['execution_wall_seconds']
            consumed=next((r for r in trace['planner'] if r.get('target') is not None and np.allclose(r['target'],goals[-1]['position'])),None)
            if consumed:
                values['planner_goal_consumption_motion_delay_seconds']=(consumed['frame']-goals[-1]['frame'])/float(motion['fps'])
                if clock=='realtime':
                    step=next(r for r in steps if r['frame']==consumed['frame'])
                    values['wall_planner_consumption_delay_lower_seconds']=step['request_wall_seconds']-5.
                    values['wall_planner_consumption_delay_upper_seconds']=step['request_wall_seconds']+step['seconds']-5.
                    if consumed.get('context_wall_seconds') is not None:values['wall_planner_consumption_delay_seconds']=consumed['context_wall_seconds']-5.
            samples=[r['seconds'] for r in trace.get('planner_sampling',[]) if r['during_execution']]
            if samples:
                values['dip_sampler_total_seconds']=float(sum(samples))
                values['dip_sampler_p95_seconds']=float(np.quantile(samples,.95))
                values['env_step_excluding_dip_seconds']=values['env_step_total_seconds']-sum(samples)
            actions=[r['phc_action_seconds'] for r in steps if r.get('phc_action_seconds') is not None]
            if actions:values['phc_action_total_seconds']=float(sum(actions))
        if task_results is not None:
            values['initial_goal_success_30cm_hold_05s']=int(task_results[0]['success'])
            values['initial_goal_final_error_m']=task_results[0]['final_error_m']
            if task_results[0]['recovery_seconds'] is not None:values['initial_goal_recovery_seconds']=task_results[0]['recovery_seconds']
            values.update(root_dwell(motion['joints'],motion['fps'],task_results[-1]['goal']['position'],
                radius=contract['root_goal_radius_m'],speed_limit=contract['root_stationary_speed_m_s'],tail_seconds=contract['root_dwell_tail_seconds']))
            if clock=='realtime':
                # DART's last eight-frame block can exceed the saved 600-frame
                # window. Do not attach its actual return time to an earlier,
                # clipped pose that was not the returned endpoint.
                observed_steps=[r for r in trace['steps']
                    if r['frame']+(7 if method=='dart' else 0)<len(motion['joints'])]
                completion=[r.get('completion_wall_seconds',r['request_wall_seconds']+r['seconds']) for r in observed_steps]
                indices=[r['frame']+(7 if method=='dart' else 0) for r in observed_steps]
                observed=wall_observation_recovery(motion['joints'][indices,0],completion,task_results[-1]['goal']['position'],
                    radius=contract['root_goal_radius_m'],hold_seconds=contract['root_goal_hold_seconds'])
                values['wall_native_observed_goal_success']=int(observed['success'])
                if observed['success']:values['wall_native_observed_recovery_seconds']=observed['first_entry_after_request_seconds']
    worker=path/'worker_result.json'
    if worker.exists():
        timing=json.loads(worker.read_text())
        if isinstance(timing.get('real_time_factor'),(int,float)):values['model_rtf']=timing['real_time_factor']
        if method=='chois' and 'model_rtf' in values:
            # cond_sample_res also reconstructs meshes and evaluates the native
            # metrics. Only the dedicated sampler timing excludes those stages.
            values['native_generation_evaluation_rtf']=values.pop('model_rtf')
        if timing.get('peak_cuda_allocated_bytes') is not None:values['peak_torch_allocated_gib']=timing['peak_cuda_allocated_bytes']/2**30
        stages=timing.get('timing',{})
        for stage,key in [('model_load','model_load_seconds'),('text_model_load_and_encode','text_setup_seconds')]:
            if stage in stages:values[key]=sum(stages[stage]['samples_seconds'])
        request=stages.get('replan',stages.get('generation'))
        if request:
            samples=request['samples_seconds'];values['request_count']=len(samples)
            values['first_request_seconds']=samples[0]
            if len(samples)>1:
                values['request_p50_seconds']=float(np.quantile(samples,.5));values['request_p95_seconds']=float(np.quantile(samples,.95))
            if method=='ardy':
                values.update(matched_ardy_steady_metrics(samples))
                steady=samples[(160 if condition['history_long'] else 32)//8:]
                if steady:
                    values.update(full_history_request_count=len(steady),full_history_request_mean_seconds=float(np.mean(steady)),
                        full_history_request_p50_seconds=float(np.quantile(steady,.5)),full_history_request_p95_seconds=float(np.quantile(steady,.95)),
                        full_history_request_rtf=float(np.mean(steady)/.4))
    scene=json.loads((path/'scenario.json').read_text())
    asset=path/'asset_receipts.json'
    row={'method':method,'task':task,'seed':manifest['seed'],'clock':clock,'condition':condition,'values':values,'units':units,
         'run':str(path.relative_to(ROOT)),'motion_sha256':checksum(path/'motion.npz'),
         'frames':len(motion['joints']),'fps':float(motion['fps']),'finite':bool(np.isfinite(motion['joints']).all()),
         'input_signature':hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest(),
         'asset_receipts_sha256':checksum(asset) if asset.exists() else None,'calls':intervention['calls'],'foot_sensitivity':feet,
         'task_results':task_results,'trajectory_sha256':hashlib.sha256(np.asarray(motion['joints'],dtype='<f8').tobytes()).hexdigest()}
    row['physics_contact_sensitivity']=physics
    if reset_semantics is not None:row['reset_semantics']=reset_semantics
    if method=='chois':row['geometric_contact_persistence']=persistence
    if method=='ardy':
        runtime_config=json.loads((path/'config.json').read_text())
        row['text_execution_profile']='required_hit_cache' if runtime_config.get('_frozen_prompt_cache') else 'native_precompute'
    row['task_spec']=scene
    if method=='chois':
        native_sequences=list((path/'raw/chois_results/res_npz_files').rglob('*.npz'))
        if len(native_sequences)==1:
            with np.load(native_sequences[0],allow_pickle=False) as native_sequence:
                row['task_spec']={**scene,'native_sequence':str(native_sequence['seq_name'])}
    if task_results is not None:
        row['task_clock_scope']='Realtime goal event mapped to first native step observing wall time >=5s; paused event at motion frame150. Recovery is in motion seconds, not wall seconds; wall delay reported separately.'
    if not row['finite']:rejected.append({'run':row['run'],'reason':'non-finite motion'});return
    rows.append(row)

def main(destination=BASE/'20261024_native_matrix_analysis/analysis.json'):
    load_replacements()
    for plan_file,state_file in SOURCES:
        if not (BASE/plan_file).exists() or not (BASE/state_file).exists():continue
        plan=json.loads((BASE/plan_file).read_text());state=json.loads((BASE/state_file).read_text())
        for job in plan['jobs']:
            if job['method'] not in FACTORS:continue
            receipt=state['jobs'].get(job['id'],{'status':'absent'})
            for incident in process_incidents(receipt):
                key=(job['id'],incident['run_dir'],incident['status'])
                incidents[key]={'job':job['id'],'method':job['method'],**incident,
                                'queue_evidence':str((BASE/state_file).relative_to(ROOT))}
            # Later continuation state supersedes the migrated parent's pending rows.
            coverage[job['id']]={'method':job['method'],'status':receipt['status'],'reason':receipt.get('reason')}
            if receipt['status']=='success' and job['id'] not in deferred_source_jobs:
                run_path=Path(receipt['run_dir'])
                relative=str(run_path.relative_to(ROOT)) if run_path.is_absolute() else str(run_path)
                started=receipt.get('started_at');registered=state.get('process_create_time')
                operational_timing[relative]={'queue':state_file,
                    'batch_queue_wait_seconds':max(0.,datetime.fromisoformat(started).timestamp()-registered) if started and registered else None,
                    'worker_seconds':receipt.get('elapsed_seconds'),
                    'scope':'Time from this batch queue process creation to task admission; not native online input response or model compute. Reused pilot retains its original admission.'}
                read_run(receipt['run_dir'],job.get('task_id',0),job.get('scene',{}).get('clock','paused'))
        for reused in plan.get('reused',[]):read_run(reused['run'],reused['task'],reused.get('clock','paused'))

    groups=defaultdict(list)
    derived_units={'root_waypoint_success_5cm':'fraction','new_goal_success_30cm_hold_05s':'fraction',
     'initial_goal_success_30cm_hold_05s':'fraction',
     'boundary_position_jump':'m','boundary_velocity_jump':'m/s','boundary_jerk_rms':'m/s^3',
     'wall_native_observed_goal_success':'fraction','foot_candidate_contact_fraction_5cm':'fraction',
     'foot_sliding_given_5cm_contact_10cms':'fraction','native_reset_count':'count',
     'native_object_mean_obj_com_pos_err':'mm','native_object_mean_start_obj_com_pos_err':'mm',
     'native_object_mean_end_obj_com_pos_err':'mm','native_object_mean_penetration_score':'m',
     'native_object_mean_hand_penetration_score':'m','native_object_mean_contact_f1_score':'fraction'}
    for row in rows:
        row['intervention_checks']=intervention_checks(row,ROOT)
        row['operational_timing']=operational_timing.get(row['run'])
        if row['operational_timing'] is not None:
            row['operational_timing']['eager_mp4_created']=(ROOT/row['run']/'motion.mp4').is_file()
        binding_path=ROOT/row['run']/'weight_binding.json'
        row['weight_identity_evidence']=str(binding_path.relative_to(ROOT)) if binding_path.exists() else str(weight_audit_path.relative_to(ROOT))
        if binding_path.exists():
            binding=json.loads(binding_path.read_text())
            if 'after_run_unchanged' in binding:
                row['weight_identity']=binding['identity'] if binding.get('after_run_unchanged') else None
                row['weight_identity_scope']='Bound before load and file change-times checked after native inference'
            else:
                retrospective=weight_audit.get(row['run'],{})
                row['weight_identity']=retrospective.get('identity') if retrospective.get('passed') else None
                row['weight_identity_scope']=retrospective.get('scope','Initial binding lacks after-run receipt; reconciliation required')
        else:
            binding=weight_audit.get(row['run'],{})
            row['weight_identity']=binding.get('identity') if binding.get('passed') else None
            row['weight_identity_scope']=binding.get('scope','Dedicated weight identity receipt pending reconciliation; empty asset lists do not prove equality')
        for metric in row['values']:
            if row['units'].get(metric):continue
            unit=derived_units.get(metric)
            if metric.startswith('native_object_'):
                unit=contract['chois_native_units'].get(metric.removeprefix('native_object_'),unit)
            if unit is None:
                if metric.endswith('_seconds'):unit='s'
                elif metric.endswith('_m_s'):unit='m/s'
                elif metric.endswith('_m'):unit='m'
                elif metric.endswith('_fraction'):unit='fraction'
                elif metric.endswith('_rtf'):unit='s/s'
                elif metric.endswith('_gib'):unit='GiB'
                elif metric.endswith('_count'):unit='count'
            if unit is not None:row['units'][metric]=unit
        groups[(row['method'],row['task'],row['seed'],row['clock'])].append(row)
    blocks=[];deltas=defaultdict(list);simple_deltas=defaultdict(list)
    for (method,task,seed,clock),group in groups.items():
        same_inputs=len({r['input_signature'] for r in group})==1
        same_assets=all(r['weight_identity'] is not None for r in group) and len({r['weight_identity'] for r in group})==1
        same_shape=len({(r['frames'],r['fps']) for r in group})==1
        valid_interventions=all(r['intervention_checks']['passed'] for r in group)
        result=factorial_block(group,FACTORS[method])
        result.update(method=method,task=task,seed=seed,clock=clock,same_inputs=same_inputs,same_assets=same_assets,same_shape=same_shape,valid_interventions=valid_interventions)
        if not(same_inputs and same_assets and same_shape and valid_interventions):
            result['complete']=False;result['rejection']='Paired input/assets/frame-rate or executed-intervention receipt mismatch';result['contrasts']=[]
        for contrast in result['contrasts']:
            key=(method,clock,tuple(contrast['factors']),contrast['metric'])
            deltas[key].append((task,contrast['delta']))
        simple=result.pop('simple_contrasts',[])
        if result['complete']:
            for contrast in simple:
                key=(method,clock,contrast['factor'],tuple(sorted(contrast['fixed'].items())),contrast['metric'])
                simple_deltas[key].append((task,contrast['delta']))
        blocks.append(result)
    def summarize_effect_values(method,metric,values,include_sensitivity=True):
        parent_by_task={r['task']:r.get('resampling_group',str(r['task'])) for r in rows if r['method']==method}
        summary=clustered_interval([v for _,v in values],[task for task,_ in values],resample_groups=[parent_by_task[task] for task,_ in values])
        if include_sensitivity:
            summary['sensitivity']=task_sensitivity([v for _,v in values],[task for task,_ in values],[parent_by_task[task] for task,_ in values])
        # A signed delta has no universal better direction, especially for interactions.
        summary.pop('improved_fraction',None)
        units={r['units'].get(metric) for r in rows if r['method']==method and metric in r['values'] and r['units'].get(metric)}
        if len(units)>1:raise ValueError(f'Mixed metric units: {method}/{metric}/{units}')
        return {'unit':next(iter(units),''),**summary}
    summaries=[{'method':method,'clock':clock,'factors':list(factors),'metric':metric,
                **summarize_effect_values(method,metric,values)}
               for (method,clock,factors,metric),values in deltas.items()]
    simple_summaries=[{'method':method,'clock':clock,'factor':factor,'fixed':dict(fixed),'metric':metric,
                       **summarize_effect_values(method,metric,values,False)}
                      for (method,clock,factor,fixed,metric),values in simple_deltas.items()]
    clock_groups=defaultdict(dict)
    for row in rows:
        key=(row['method'],row['task'],row['seed'],json.dumps(row['condition'],sort_keys=True))
        clock_groups[key][row['clock']]=row
    clock_pairs=[]
    for (method,task,seed,condition),pair in clock_groups.items():
        if set(pair)!=set(['paused','realtime']):continue
        a,b=pair['paused'],pair['realtime']
        clock_pairs.append({'method':method,'task':task,'seed':seed,'condition':condition,
            'same_assets':a['weight_identity'] is not None and a['weight_identity']==b['weight_identity'],
            'same_shape':(a['frames'],a['fps'])==(b['frames'],b['fps']),
            'exact_same_coordinates':a['trajectory_sha256']==b['trajectory_sha256'],
            'paused_run':a['run'],'realtime_run':b['run'],
            'scope':'Separate clock outcomes; matching coordinates never counted as two independent motion samples'})
    capability=[]
    success_metrics=['position_success_rate','root_waypoint_success_5cm','initial_goal_success_30cm_hold_05s','new_goal_success_30cm_hold_05s','new_goal_success_before_next_reset','new_goal_success_reset_free_episode','wall_native_observed_goal_success']
    for method in FACTORS:
        method_rows=[r for r in rows if r['method']==method]
        for clock in sorted({r['clock'] for r in method_rows}):
            for condition in sorted({json.dumps(r['condition'],sort_keys=True) for r in method_rows if r['clock']==clock}):
                selected=[r for r in method_rows if r['clock']==clock and json.dumps(r['condition'],sort_keys=True)==condition]
                for metric in success_metrics:
                    measured=[r for r in selected if metric in r['values'] and r['intervention_checks']['passed'] and r['weight_identity'] is not None]
                    if not measured:continue
                    stats=clustered_interval([r['values'][metric] for r in measured],[r['task'] for r in measured])
                    stats.pop('improved_fraction',None)
                    capability.append({'method':method,'clock':clock,'condition':condition,'metric':metric,
                        'completed_runs':len(measured),'runs_with_any_failure':sum(r['values'][metric]<1 for r in measured),
                        'run_mean_success_fraction':float(np.mean([r['values'][metric] for r in measured])),
                        'scope':'Position/waypoint rates are within-run constraint fractions; goal rates are binary per run. Process failures excluded and reported separately.',**stats})
    acceptance=[]
    for method in FACTORS:
        for clock in (['paused','realtime'] if method in ['ardy','dart','primal','closd'] else ['paused']):
            selected=[r for r in rows if r['method']==method and r['clock']==clock]
            valid=[r for r in selected if r['intervention_checks']['passed'] and r['weight_identity'] is not None]
            keys={(r['task'],r['seed'],tuple(r['condition'][k] for k in FACTORS[method])) for r in valid}
            if len(keys)!=len(valid):raise ValueError('Duplicate scientific cell: '+method+'/'+clock)
            if any(task not in range(12) or seed not in range(20,25) for task,seed,_ in keys):raise ValueError('Unexpected task/seed outside registered matrix')
            per_task=5*2**len(FACTORS[method]);expected=12*per_task
            task_rows=[{'task':task,'accepted_cells':sum(key[0]==task for key in keys),'expected_cells':per_task} for task in range(12)]
            acceptance.append({'method':method,'clock':clock,'accepted_cells':len(keys),'expected_cells':expected,
                'cells_requiring_receipt_reconciliation':len(selected)-len(valid),
                'complete_blocks':sum(b['complete'] for b in blocks if b['method']==method and b['clock']==clock),
                'expected_blocks':60,'covered_tasks':sum(t['accepted_cells']>0 for t in task_rows),
                'complete_tasks':sum(t['accepted_cells']==per_task for t in task_rows),'expected_tasks':12,'task_rows':task_rows})
    for incident in incidents.values():
        if incident.get('run_dir'):
            run=Path(incident['run_dir']);run=run if run.is_absolute() else ROOT/run
            incident['run']=str(run.relative_to(ROOT))
            stderr=run/'stderr.log'
            if stderr.exists():
                errors=[line for line in stderr.read_text(errors='replace').splitlines() if 'Error:' in line or 'Exception:' in line]
                if errors:incident['underlying_error']=errors[-1]
        latest=coverage.get(incident['job'],{})
        incident['later_same_job_success']=latest.get('status')=='success'
        incident['resolution']='后续同一任务已成功；保留此次执行失败' if incident['later_same_job_success'] else '原始失败保留，不进入能力失败率'
        if incident['method']=='primal' and 'Native primitive output too short' in incident.get('underlying_error',''):
            replacement=next((r for r in rows if r['method']=='primal' and r['task']==0 and r['seed']==20 and r['clock']=='paused' and r['frames']==600),None)
            if replacement:
                incident['replacement_run']=replacement['run']
                incident['resolution']='修正原生块数上界后重新生成并截取600帧；新运行已纳入完整配对块，旧588帧运行不用于推断'
    result={'updated_at':datetime.now(timezone.utc).isoformat(),'rows':rows,'blocks':blocks,'contrasts':summaries,'rejected':rejected,
            'analysis_source_sha256':ANALYSIS_SOURCES,
            'simple_effects':simple_summaries,
            'matrix_acceptance':acceptance,
            'replacement_checks':replacement_checks,
            'source_replay_replacements':source_replay_replacements,
            'capability_rates':capability,
            'clock_pairs':clock_pairs,
            'coverage':{m:dict(Counter(v['status'] for v in coverage.values() if v['method']==m)) for m in FACTORS},
            'failures':list(incidents.values()),
            'interpretation':'Within fixed pretrained systems; each clock analyzed separately, not pooled as independent samples. Marginal contrasts average other factors; interactions are differences of differences. Equal task weight; task-cluster bootstrap except PRIMAL groups short/long initial histories sharing the effective warm RNG seed. Intervals are conditional on selected tasks, not a population architecture ranking.'}
    if any(checksum(ROOT/path)!=digest for path,digest in ANALYSIS_SOURCES.items()):
        raise RuntimeError('Analysis implementation changed during this calculation; retain previous coherent snapshot')
    json_write(destination,result)
    print(json.dumps({'output':str(destination),'runs':len(rows),'complete_blocks':sum(r['complete'] for r in blocks),'coverage':result['coverage'],'rejected':rejected}))


if __name__=='__main__':
    main(Path(sys.argv[1]) if len(sys.argv)>1 else BASE/'20261024_native_matrix_analysis/analysis.json')
