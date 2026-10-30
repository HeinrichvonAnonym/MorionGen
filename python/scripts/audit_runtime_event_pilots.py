"""Audit version consumption and paired negative controls; keep every failure."""
import collections
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'utils'))
from motiogen.runtime import json_write

BASE = ROOT / 'develop_logs/test_runs/20261028_supplement'
sources = [
    (BASE/'dart_matrix/plan.json', BASE/'dart_matrix/queue/state.json'),
    (BASE/'dart_load_pilot.json', BASE/'dart_load_queue/state.json'),
    (ROOT/'config/ardy_runtime_pilot_20261028.json', BASE/'ardy_runtime_queue/state.json'),
    (BASE/'ardy_full_block_plan.json', BASE/'ardy_full_block_queue/state.json'),
    (BASE/'ardy_matrix/plan.json', BASE/'ardy_matrix/queue/state.json'),
    (BASE/'closd_runtime/plan.json', BASE/'closd_runtime/queue/state.json'),
    (BASE/'closd_runtime/full_block_plan.json', BASE/'closd_runtime/full_block_queue/state.json'),
    (BASE/'closd_matrix/plan.json', BASE/'closd_matrix/queue/state.json'),
    (BASE/'ardy_reserve_retry/plan.json', BASE/'ardy_reserve_retry/queue/state.json')]
rows, groups = [], collections.defaultdict(list)
for planpath, statepath in sources:
    if not statepath.exists():
        continue
    plan, state = json.loads(planpath.read_text()), json.loads(statepath.read_text())
    for job in plan['jobs']:
        status = state['jobs'][job['id']]
        cmd = job['command']
        eventfile = Path(cmd[cmd.index('--events')+1])
        payload = json.loads(eventfile.read_text())['events']
        clock = cmd[cmd.index('--clock')+1]
        row = {'job':job['id'], 'method':job['method'], 'seed':job['seed'],
               'clock':clock, 'load':'--load' in cmd, 'ignore':'--ignore' in cmd,
               'status':status['status'], 'run_dir':status.get('run_dir'),
               'event_payload':payload, 'queue_state':str(statepath),
               'elapsed_seconds':status.get('elapsed_seconds'),
               'resource_profile':job.get('config',{}).get('resources',{}).get('gpu_reserve_gib',3)}
        rows.append(row)
        row['cohort_role']=('original_resource_profile_attempt_retained'
            if row['method']=='ardy' and row['job'].startswith('task2_seed0_') and row['resource_profile']==3
            else 'primary_planned_block')
        key = (row['method'], row['seed'], clock, row['load'], json.dumps(payload,sort_keys=True),row['resource_profile'])
        groups[key].append(row)
        if row['status'] != 'success':
            row['failure_reason'] = status.get('reason')
            continue
        run = Path(row['run_dir'])
        trace = json.loads((run/'runtime_trace.json').read_text())
        if row['method'] == 'closd':
            events = trace['events']
            consume = trace['steps']
            compute = trace['planner']
            stampkey = 'action_submit_ns'
            row['clock_semantics'] = 'native_simulation' if clock=='realtime' else 'paused_simulation_diagnostic'
        else:
            events = [json.loads(line) for line in (run/'input_events.jsonl').read_text().splitlines()]
            events = [e for e in events if e['event']=='read']
            consumption = json.loads((run/'consumption.json').read_text())
            consume, compute, stampkey = consumption['rows'], trace['steps'], 'consume_ns'
            row['clock_semantics'] = consumption['clock']
            row['held_frames'] = sum(x['held'] for x in consume)
            row['held_fraction'] = row['held_frames']/len(consume)
            row['peak_buffer_frames'] = max(x['buffer_frames'] for x in consume)
        changed = [x for x in compute if x['condition_version']==1]
        affected = [x for x in consume if x['condition_version']==1]
        row['consumed_frames'] = len(consume)
        row['event_count'] = len(events)
        row['condition_consumed'] = bool(changed)
        row['affected_action_consumed'] = bool(affected)
        row['update_response_failed_within_observed_execution'] = not bool(affected) if not row['ignore'] else None
        row['execution_acceptance'] = ('negative_control_held_old_condition' if not changed and not affected else 'negative_control_failed') if row['ignore'] else ('updated_action_consumed' if affected else 'no_updated_action_consumed')
        row['negative_control_pass'] = (not changed and not affected) if row['ignore'] else None
        row['first_condition_frame'] = changed[0]['frame'] if changed else None
        row['first_affected_frame'] = affected[0]['frame'] if affected else None
        row['event_chain_order_pass'] = None
        if events and changed and affected:
            e, c, a = events[0], changed[0], affected[0]
            timestamps = [e['send_ns'],e['arrival_ns'],e['read_ns'],c['condition_consumed_ns'],
                          c['compute_end_ns'],c.get('reference_commit_ns',c.get('commit_ns')),a[stampkey]]
            row['event_chain_ns'] = timestamps
            row['event_chain_order_pass'] = timestamps==sorted(timestamps)
            row['arrival_to_first_action_seconds'] = (a[stampkey]-e['arrival_ns'])/1e9
            row['arrival_to_read_seconds'] = (e['read_ns']-e['arrival_ns'])/1e9
        else:
            row['arrival_to_first_action_seconds'] = None
        fps = 30 if row['method'] in ['dart','closd'] else 20
        late = [x.get('deadline_lateness_seconds') for x in consume]
        row['deadline_miss_frames'] = sum(v is not None and v>1/fps for v in late) if clock=='realtime' else None
        row['deadline_miss_definition'] = 'Consumption/step submission more than one native frame interval later than its absolute schedule.'
        resources = [json.loads(x) for x in (run/'resources.jsonl').read_text().splitlines()]
        row['gpu_peak_used_bytes_sampled'] = max((x.get('gpu',{}).get('used_bytes',0) for x in resources),default=0)
        if row['load']:
            load=json.loads((run/'compute_load.json').read_text())
            row['controlled_load_matmuls']=sum(x['matmuls'] for x in load['cycles'])
            row['controlled_load_active_wall_seconds']=sum((x['end_ns']-x['start_ns'])/1e9 for x in load['cycles'])
            row['controlled_load_allocated_bytes']=load['allocated_bytes']
            row['load_measurement_scope']='Recorded synchronized matmul operations, active wall intervals and allocated bytes. Historical resources.jsonl does not contain SM utilization; the compute_load.json generic utilization sentence is not evidence of a utilization measurement.'

pairs=[]
for key, members in groups.items():
    update=[x for x in members if not x['ignore']]
    ignore=[x for x in members if x['ignore']]
    pair={'method':key[0],'seed':key[1],'clock':key[2],'load':key[3],'resource_profile':key[5],
          'jobs':[x['job'] for x in members], 'paired_identity_unique':len(update)==len(ignore)==1}
    pairs.append(pair)
    if not pair['paired_identity_unique'] or any(x['status']!='success' for x in members):
        pair['status']='incomplete_or_duplicate';continue
    u,n=update[0],ignore[0]
    a=np.load(Path(u['run_dir'])/'motion.npz')['joints']
    b=np.load(Path(n['run_dir'])/'motion.npz')['joints']
    boundary=u['first_condition_frame']
    if boundary is None or boundary<=0:
        pair['status']='update_not_consumed';continue
    common_length=min(len(a),len(b))
    prefix_end=min(boundary,common_length)
    pair['generated_frames_update']=len(a)
    pair['generated_frames_ignore']=len(b)
    pair['prefix_frames']=prefix_end
    pair['prefix_covers_preupdate']=prefix_end==boundary
    pair['prefix_max_abs_difference_m']=float(np.max(np.abs(a[:prefix_end]-b[:prefix_end])))
    pair['prefix_exact']=bool(np.array_equal(a[:prefix_end],b[:prefix_end]))
    pair['post_update_common_frames']=max(0,common_length-boundary)
    pair['post_update_max_abs_difference_m']=float(np.max(np.abs(a[boundary:common_length]-b[boundary:common_length]))) if common_length>boundary else None
    pair['held_frame_difference']=u.get('held_frames',0)-n.get('held_frames',0) if key[0]!='closd' else None
    pair['status']='checked'
json_write(BASE/'runtime_pilot_audit.json', {'rows':rows,'pairs':pairs,
    'scope':'Queue completion is not scientific acceptance. Failed/pending jobs retained. Per-frame metrics are not independent samples. Paused response is diagnostic only; simulation and playback clocks remain separate.'})
print(json.dumps({'rows':len(rows),'states':dict(collections.Counter(x['status'] for x in rows)),
    'pairs':len(pairs),'chain_failures':[x['job'] for x in rows if x.get('event_chain_order_pass') is False],
    'prefix_failures':[x['jobs'] for x in pairs if x.get('prefix_exact') is False]}))
