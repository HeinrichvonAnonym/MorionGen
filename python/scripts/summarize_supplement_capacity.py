"""Measured finite-pilot capacity; proposed expansion is not an active budget."""
import json,sys,collections
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
B=ROOT/'develop_logs/test_runs/20261028_supplement'
source=B/'runtime_pilot_audit.json';runtime=json.loads(source.read_text())['rows'];rows=[]
for method in ['ardy','dart','closd']:
 a=[r for r in runtime if r['method']==method];primary=[r for r in a if r['cohort_role']=='primary_planned_block'];times=[r['elapsed_seconds'] for r in primary if r.get('elapsed_seconds') is not None]
 rows.append(dict(route=method,planned_primary_conditions=len(primary),recorded_attempt_conditions=len(a),primary_process_success=sum(r['status']=='success' for r in primary),
  measured_attempt_lease_hours=sum(r.get('elapsed_seconds') or 0 for r in a)/3600,
  median_job_seconds=float(np.median(times)),p95_job_seconds=float(np.percentile(times,95)),
  proposed_conditions=16*2*2*2*2,proposed_independent_contexts=16,
  proposed_lease_hours_at_p95=float(np.percentile(times,95)*256/3600),
  scope='16 new independent contexts × 2 seeds × realtime/paused × normal/load × update/ignore; not yet launched. Same GPU-lease child load included, not counted as another GPU.'))
for method,name in [('motiongpt','20261028_103000_motiongpt_native_generation_367d3c2a'),('motionlcm','20261028_103145_motionlcm_t2m_native_generation_8abd974e')]:
 p=ROOT/'develop_logs/test_runs'/name/'generation.json';d=json.loads(p.read_text());times=[r['seconds'] for r in d['rows']]
 rows.append(dict(route=method,completed=16,cold_seconds=d['cold_seconds'],median_inference_seconds=float(np.median(times)),p95_inference_seconds=float(np.percentile(times,95)),
  proposed_independent_source_texts=128,proposed_conditions=256,proposed_model_hours_at_p95=float(np.percentile(times,95)*256/3600),
  scope='128 deduplicated sources × 2 seeds, compatibility/length-stratified before viewing new scores. Excludes evaluator, loading, IO and preparation; no small-sample FID ranking.'))
tracking=json.loads((B/'common_tracking_analysis.json').read_text());uhc=[r for r in tracking['rows'] if r['method']=='UHC'];phc=json.loads((B/'phc_timing_replay_analysis.json').read_text())['rows']
rows.append(dict(route='uhc',completed_conditions=16,independent_sources=8,measured_cpu_inference_seconds=sum(r['total_inference_seconds'] for r in uhc),proposed_new_sources=2,
 proposed_cpu_seconds_at_p95=float(np.percentile([r['total_inference_seconds'] for r in uhc],95)*4),scope='Only two additional official sample motions presently available. Ten-source coverage remains limited; do not promise raw-AMASS expansion from processed HumanML joints.'))
rows.append(dict(route='phc',completed_conditions=8,timing_replay_conditions=8,independent_sources=8,measured_policy_environment_seconds=sum(r['combined_seconds'] for r in phc),proposed_new_sources=2,
 scope='Two remaining official UHC sources require geometry conversion first. Eight timing replays are cost, not new independent sources.'))
repos=json.loads((B/'source_audit.json').read_text())['repositories'];prep=[]
for r in repos:
 prep.append(dict(name=r['name'],clone_elapsed_seconds=(datetime.fromisoformat(r['finished_at'])-datetime.fromisoformat(r['started_at'])).total_seconds(),commit=r['commit']))
lease=sum(r.get('proposed_lease_hours_at_p95',0)+r.get('proposed_model_hours_at_p95',0) for r in rows)
result=dict(rows=rows,repository_clone_costs=prep,proposal_compute_hours=lease,ideal_two_gpu_lower_bound_hours=lease/2,
 proposed_natural_time_hours=20,proposal_status='Proposed second-stage reservation only; requires scope/budget decision. Does not reset or extend previous 36-hour deadline.',
 budget_allocation={'experiment_and_recovery_wall_hours':16,'statistics_report_browser_wall_hours':4,'two_gpu_max_reserved_hours':32},
 scope=['Recorded job elapsed time is GPU lease occupancy, not CUDA kernel activity. Failed and replacement attempts retained.',
 'p95-based scheduling projection is not a probabilistic completion guarantee. Host contention and resource guard failures occurred.',
 '16 experiment wall hours on two cards is at most 32 GPU·h; 20 total wall hours includes four report hours, not 40 required GPU·h.',
 'Full environment installation timing was not recorded; cloning and available download timers cannot substitute for full preparation cost.',
 'No new authorization/download required for these proposed existing-asset expansions. Raw AMASS expansion needs separate source availability.'])
json_write(B/'phase1_capacity_and_proposal.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
