"""Audit native observation clocks without inventing intermediate execution."""
import json
import argparse
import re
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.clock_audit import observed_hold_bounds
from motiogen.motion import load_motion
from motiogen.runtime import json_write,checksum

parser=argparse.ArgumentParser()
parser.add_argument('--reference-snapshot',type=Path)
args=parser.parse_args()

base=ROOT/'develop_logs/test_runs'
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
contract=json.loads((ROOT/'config/native_metric_contract.json').read_text())
checks=[]
for row in analysis['rows']:
    if row['method'] not in ['dart','closd'] or row['clock']!='realtime':continue
    folder=ROOT/row['run']
    trace=json.loads((folder/'trace.json').read_text())
    motion=load_motion(folder/'motion.npz')
    steps=[s for s in trace['steps'] if s['frame']+(7 if row['method']=='dart' else 0)<len(motion['joints'])]
    excluded=len(trace['steps'])-len(steps)
    request=np.asarray([s['request_wall_seconds'] for s in steps],float)
    elapsed=np.asarray([s['seconds'] for s in steps],float)
    direct=all('completion_wall_seconds' in s for s in steps)
    if direct:
        lower=upper=np.asarray([s['completion_wall_seconds'] for s in steps],float)
        scope='Direct native step completion timestamps; DART exposes the returned eight-frame block endpoint, not a per-frame execution clock.'
    else:
        # The saved request timestamp precedes target-reset/bookkeeping and
        # synchronized step timing. Their sum is a completion LOWER bound.
        # The next request (or recorded final execution end) is an upper bound.
        lower=request+elapsed
        upper=np.r_[request[1:],trace['execution_wall_seconds']]
        scope='CLoSD completion interval: request + measured env_step <= completion <= next request; last upper bound is the recorded end after terminal weight verification and motion/metric export, not an exact execution-end timestamp. No exact completion timestamp was stored.'
    indices=[s['frame']+(7 if row['method']=='dart' else 0) for s in steps]
    if 'requested_goals' in trace:
        target=np.asarray(trace['requested_goals'][-1]['position'])[:2]
        target_source=row['run']+'/trace.json:requested_goals'
    else:
        # Two reused DART pilot traces predate requested_goals metadata. Recover
        # the requested goal from their actual goal-update counterpart, not a
        # fabricated value or the frozen condition's unchanged active goal.
        peers=[r for r in analysis['rows'] if r['method']=='dart' and r['task']==row['task']
               and r['seed']==row['seed'] and r['clock']==row['clock'] and r['condition']['goal_update']==1]
        if len(peers)!=1:raise ValueError('Legacy target requires one complete update counterpart')
        target_source=peers[0]['run']+'/trace.json:steps[-1].goal'
        peer=json.loads((ROOT/peers[0]['run']/'trace.json').read_text())
        target=np.asarray(peer['steps'][-1]['goal'])[:2]
        if np.array_equal(target,np.asarray(peer['steps'][0]['goal'])[:2]):
            raise ValueError('Goal-update counterpart did not actually change target')
    inside=np.linalg.norm(motion['joints'][indices,0,:2]-target,axis=-1)<=contract['root_goal_radius_m']
    bounds=observed_hold_bounds(inside,lower,upper,hold_seconds=contract['root_goal_hold_seconds'])
    published=int(row['values']['wall_native_observed_goal_success'])
    consistent=int(bounds['definite'])<=published<=int(bounds['possible'])
    valid=bool(np.all(np.isfinite(request)) and np.all(elapsed>=0)
               and np.all(np.diff(request)>=0) and np.all(lower>=request)
               and np.all(lower<=upper) and consistent)
    checks.append({'method':row['method'],'run':row['run'],'direct_completion':direct,
        'observations':len(steps),'excluded_truncated_endpoints':excluded,'frame_stride':8 if row['method']=='dart' else 1,
        'point_success':published,'definite_success':bounds['definite'],'possible_success':bounds['possible'],
        'timing_ambiguity':bounds['definite']!=bounds['possible'],
        'max_completion_interval_seconds':float((upper-lower).max()),
        'median_observation_gap_seconds':float(np.median(np.diff(lower))),
        'scope':scope,'target_source':target_source,'passed':valid})
summary=[]
for method in ['dart','closd']:
    rows=[r for r in checks if r['method']==method]
    if not rows:continue
    summary.append({'method':method,'checked_runs':len(rows),'passed_runs':sum(r['passed'] for r in rows),
        'direct_completion_runs':sum(r['direct_completion'] for r in rows),
        'definite_success_runs':sum(r['definite_success'] for r in rows),
        'possible_success_runs':sum(r['possible_success'] for r in rows),
        'ambiguous_runs':sum(r['timing_ambiguity'] for r in rows),
        'frame_stride':rows[0]['frame_stride'],'scope':rows[0]['scope']})
receipt={'analysis_cutoff':analysis['updated_at'],'passed':bool(checks) and all(r['passed'] for r in checks),
    'checks':checks,'summary':summary,
    'scope':'Consecutive returned-state observations only, not proof of continuous contact/goal occupancy between samples. Timestamp bounds are deterministic logging bounds, not confidence intervals. Counts summarize this audit and do not replace scene-weighted paired effects.'}
json_write(base/'20261024_native_clock_observations/audit.json',receipt)
if args.reference_snapshot:
    reference_text=args.reference_snapshot.read_text()
    if args.reference_snapshot.suffix=='.html':
        chunks=re.findall(r'<script type="application/json" (?:id="data-app-reviewed-snapshot"|data-app-snapshot-chunk)>(.*?)</script>',reference_text,re.S)
        if not chunks:raise ValueError('Reference HTML contains no reviewed data payload')
        previous=json.loads(''.join(chunks))
    else:previous=json.loads(reference_text)
    prior={r['run']:r for r in previous['queries']['native_runs']['rows']
           if r['method']=='dart' and r['clock']=='realtime'}
    compared=0;differences=[]
    for row in analysis['rows']:
        if row['run'] not in prior:continue
        compared+=1
        for name in ['wall_native_observed_goal_success','wall_native_observed_recovery_seconds']:
            before=prior[row['run']]['values'].get(name);after=row['values'].get(name)
            if before!=after:differences.append({'run':row['run'],'metric':name,'before':before,'after':after})
    json_write(base/'20261024_native_clock_observations/terminal_clipping_correction.json',{
        'reference_snapshot':str(args.reference_snapshot),'reference_sha256':checksum(args.reference_snapshot),
        'analysis_cutoff':analysis['updated_at'],'compared_runs':compared,'differences':differences,
        'correction':'Exclude the final native eight-frame return whose true endpoint is outside the saved 600-frame window; do not pair its completion time with a clipped earlier frame.',
        'scope':'Audit of matching pre-existing report rows only; no motion regeneration or changes to dense trajectory metrics.'})
print(json.dumps({'passed':receipt['passed'],'summary':summary,'failures':[r for r in checks if not r['passed']]},ensure_ascii=False))
if not receipt['passed']:raise SystemExit(1)
