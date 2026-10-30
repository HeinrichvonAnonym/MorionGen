"""Validate paired timing-only replay against frozen native tracking trajectories."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
B=ROOT/'develop_logs/test_runs/20261028_supplement'
original=json.loads((B/'common_tracking_analysis.json').read_text())
lookup={r['source']:r for r in original['rows'] if r['method']=='PHC'}
state=B/'phc_common_tracking/timing_replay/queue/state.json';rows=[]
for status in json.loads(state.read_text())['jobs'].values():
 assert status['status']=='success'
 run=Path(status['run_dir']);record=json.loads((run/'tracking.json').read_text());prior=lookup[record['source']]
 a=np.load(run/'execution_trace.npz')['joints'];b=np.load(Path(prior['evidence'])/'execution_trace.npz')['joints']
 assert a.shape==b.shape and np.array_equal(a,b)
 policy=np.array([r['policy_seconds'] for r in record['steps']]);env=np.array([r['seconds'] for r in record['steps']]);combined=policy+env
 row=dict(source=record['source'],steps=len(a),identical_trajectory=True,max_position_difference_m=0.,
  policy_seconds=float(policy.sum()),environment_seconds=float(env.sum()),combined_seconds=float(combined.sum()),
  combined_rtf=float(combined.sum()/(len(a)/30)),combined_step_p50_seconds=float(np.median(combined)),
  combined_step_p95_seconds=float(np.percentile(combined,95)),policy_step_p50_seconds=float(np.median(policy)),
  run=str(run),trace_sha256=checksum(run/'execution_trace.npz'),source_trace_sha256=checksum(Path(prior['evidence'])/'execution_trace.npz'))
 rows.append(row)
json_write(B/'phc_timing_replay_analysis.json',dict(rows=rows,scope='Timing-only replays of the same eight references; all executed joints exactly identical. CUDA synchronization brackets both policy and environment calls. Combined cost excludes startup, queue waiting and external trace serialization; these are not new independent samples.'))
print(json.dumps({'identical_replays':len(rows),'combined_seconds':sum(r['combined_seconds'] for r in rows)}))
