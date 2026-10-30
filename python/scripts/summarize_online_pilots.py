"""Task-clustered runtime diagnostics, failures retained separately from latency."""
import json,sys,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import json_write,checksum
B=ROOT/'develop_logs/test_runs/20261028_supplement';source=B/'runtime_pilot_audit.json'
allrows=json.loads(source.read_text())['rows'];rows=[]
primary=[r for r in allrows if r['cohort_role']=='primary_planned_block']
assert len(primary)==192
for method in ['ardy','dart','closd']:
 for clock in ['realtime','paused']:
  for load in [False,True]:
   group=[r for r in primary if r['method']==method and r['clock']==clock and r['load']==load and not r['ignore']]
   controls=[r for r in primary if r['method']==method and r['clock']==clock and r['load']==load and r['ignore']]
   values=[r['arrival_to_first_action_seconds'] for r in group if r.get('arrival_to_first_action_seconds') is not None]
   rows.append(dict(method=method,clock=clock,load=load,independent_contexts=4,update_conditions=len(group),
    response_failure_count=sum(not r.get('affected_action_consumed',False) for r in group),negative_control_pass_count=sum(r.get('negative_control_pass',False) for r in controls),
    successful_response_p50_seconds=float(np.median(values)) if values else None,successful_response_p95_seconds=float(np.percentile(values,95)) if values else None,
    held_frames_total=sum(r['held_frames'] for r in group if r.get('held_frames') is not None) if any(r.get('held_frames') is not None for r in group) else None,deadline_miss_frames_total=sum(r['deadline_miss_frames'] for r in group if r.get('deadline_miss_frames') is not None) if any(r.get('deadline_miss_frames') is not None for r in group) else None,
    latency_scope='Conditional on response occurring; missing/failed responses are counted, not assigned zero latency. Paused is diagnostic only.'))
execution=json.loads((B/'online_motion_response.json').read_text())['rows'];by={(r['method'],r['job']):r for r in execution};pairs=[]
for update in primary:
 if update['ignore'] or update['method']=='ardy':continue
 candidates=[c for c in primary if c['method']==update['method'] and c['ignore'] and c['seed']==update['seed'] and c['clock']==update['clock'] and c['load']==update['load'] and c['event_payload']==update['event_payload']]
 assert len(candidates)==1
 ignore=candidates[0];on=by[(update['method'],update['job'])];off=by[(ignore['method'],ignore['job'])]
 row=dict(method=update['method'],clock=update['clock'],load=update['load'],seed=update['seed'],task=json.dumps(update['event_payload'],sort_keys=True),update_job=update['job'],ignore_job=ignore['job'])
 for metric in ['revealed_goal_distance_at_wall_20s_m','revealed_goal_final_distance_m','consumed_motion_jerk_rms_mps3']:
  a,b=on.get(metric),off.get(metric);row[metric]=a-b if a is not None and b is not None else None
 pairs.append(row)
effects=[]
for method in ['dart','closd']:
 for clock in ['realtime','paused']:
  for load in [False,True]:
   group=[p for p in pairs if p['method']==method and p['clock']==clock and p['load']==load]
   for metric in ['revealed_goal_distance_at_wall_20s_m','revealed_goal_final_distance_m','consumed_motion_jerk_rms_mps3']:
    clusters=collections.defaultdict(list)
    for p in group:
     if p[metric] is not None:clusters[p['task']].append(p[metric])
    values=np.array([np.mean(v) for v in clusters.values()]);rng=np.random.default_rng(20261029)
    ci=np.quantile(values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1),[.025,.975])
    effects.append(dict(method=method,clock=clock,load=load,metric=metric,contexts=len(values),mean_update_minus_ignore=float(values.mean()),task_bootstrap95=ci.tolist()))
json_write(B/'online_pilot_summary.json',dict(rows=rows,paired_goal_effects=effects,pairs=pairs,original_resource_failures=[r for r in allrows if r['status']!='success'],source_sha256=checksum(source),scope='Four independent contexts per method; two seeds averaged within context for paired effects. Descriptive bootstrap intervals, not sufficient formal task coverage. Real-time and paused are separate. Load conditions have host/warmup/order variation; do not conclude load improves response from raw latency differences.'))
print(json.dumps(rows,ensure_ascii=False))
