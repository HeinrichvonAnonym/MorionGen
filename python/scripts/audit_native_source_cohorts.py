"""Check frozen local inference modules, upstream patches and replay outcomes."""
import hashlib
import json
import sys
from collections import defaultdict,Counter
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.source_cohorts import SOURCES
from motiogen.runtime import checksum,json_write
base=ROOT/'develop_logs/test_runs';folder=base/'20261025_source_replays'
plan=json.loads((folder/'plan.json').read_text());state=json.loads((folder/'queue/state.json').read_text())
frozen=plan['frozen_sources'];summaries=[];checks=[];upstream=defaultdict(set)
archive_matches=all((folder/'frozen_sources'/f).is_file() and checksum(folder/'frozen_sources'/f)==h for f,h in frozen.items())
original=json.loads((folder/'original_analysis.json').read_text())
old_weights={r['run']:r.get('weight_identity') for r in original['rows']}
cutoffs={}
for domain,relative in [('primary','20261024_native_matrix_analysis/analysis.json'),('trajectory','20261024_omnicontrol_trajectory/analysis.json')]:
    analysis=json.loads((base/relative).read_text());cutoffs[domain]=analysis['updated_at'];seen=set()
    for r in analysis['rows']:
        if r['run'] in seen:continue
        seen.add(r['run']);path=ROOT/r['run'];inventory=json.loads((path/'implementation.json').read_text())
        mismatch=[f for f in SOURCES[r['method']] if inventory.get(f)!=frozen[f]]
        git=json.loads((path/'manifest.json').read_text())['git']
        upstream[r['method']].add(hashlib.sha256(json.dumps({k:git[k] for k in ['commit','patch']},sort_keys=True).encode()).hexdigest())
        checks.append({'domain':domain,'method':r['method'],'run':r['run'],'passed':not mismatch,'mismatched_modules':mismatch})
for method in sorted({r['method'] for r in checks}):
    subset=[r for r in checks if r['method']==method]
    summaries.append({'method':method,'checked_runs':len(subset),'same_frozen_source_runs':sum(r['passed'] for r in subset),
                      'upstream_commit_patch_versions':len(upstream[method]),'source_files':SOURCES[method]})
replays=[]
for job in plan['jobs']:
    record=state['jobs'][job['id']]
    if record['status']!='success':continue
    old=ROOT/job['prior_run'];new=Path(record['run_dir'])
    before=json.loads((old/'manifest.json').read_text());after=json.loads((new/'manifest.json').read_text())
    binding=json.loads((new/'weight_binding.json').read_text())
    same_inputs=before['scenario']==after['scenario']==job['scene'] and before['seed']==after['seed']==job['seed']
    old_cfg=json.loads((old/'config.json').read_text());new_cfg=json.loads((new/'config.json').read_text())
    same_inputs=same_inputs and all(old_cfg.get(k)==new_cfg.get(k) for k in ['_intervention','_frozen_prompt_cache','timing'])
    same_inputs=same_inputs and [v.replace(str(old),'{run_dir}') for v in before['command']]==[v.replace(str(new),'{run_dir}') for v in after['command']]
    if job['method']=='primal':
        with np.load(old/'factorial/initial_history.npz') as a,np.load(new/'factorial/initial_history.npz') as b:
            same_inputs=same_inputs and set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
    same_weights=binding['identity']==old_weights[job['prior_run']] and binding['after_run_unchanged']
    files=sorted((old/'factorial').glob('*_canonical.npz')) if job['method']=='primal' else [old/'motion.npz']
    comparisons=[]
    for file in files:
        target=new/file.relative_to(old)
        with np.load(file) as old_motion,np.load(target) as new_motion:
            a=old_motion['joints'];b=new_motion['joints']
            same_fps=float(old_motion['fps'])==float(new_motion['fps'])
        changed_frames=np.flatnonzero(np.any(a!=b,axis=tuple(range(1,a.ndim)))) if a.shape==b.shape else []
        comparisons.append({'file':str(file.relative_to(old)),'same_shape':a.shape==b.shape,
            'same_fps':same_fps,
            'first_coordinate_difference_frame':int(changed_frames[0]) if len(changed_frames) else None,
            'finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),
            'exact_same_coordinates':bool(np.array_equal(a,b)),
            'max_abs_delta_m':float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.shape==b.shape else None})
    timing={}
    if job['method']=='closd':
        goal=np.asarray(json.loads((new/'trace.json').read_text())['requested_goals'][-1]['position'])
        for label,path in [('prior',old),('replay',new)]:
            trace=json.loads((path/'trace.json').read_text())
            consumed=[p['frame'] for p in trace['planner'] if np.array_equal(np.asarray(p['target']),goal)]
            timing[label+'_goal_applied_frame']=trace['goal_applied_frame']
            timing[label+'_planner_goal_frame']=min(consumed) if consumed else None
    replays.append({'method':job['method'],'clock':job['scene'].get('clock','paused'),'task':job['task_id'],'seed':job['seed'],'group':job['replay_group'],'job':job['id'],'prior_run':job['prior_run'],
                    'run':str(new.relative_to(ROOT)),'same_inputs':same_inputs,'same_weights':same_weights,'comparisons':comparisons,**timing})
complete_groups=sum(all(state['jobs'][j]['status']=='success' for j in g['jobs']) for g in plan['source_replay_groups'])
receipt={'updated_at':datetime.now(timezone.utc).isoformat(),'analysis_cutoffs':cutoffs,'rows':summaries,'checks':checks,'replays':replays,
    'complete_groups':complete_groups,'expected_groups':len(plan['source_replay_groups']),
    'completed_processes':len(replays),'expected_processes':len(plan['jobs']),
    'frozen_sources_still_current':all(checksum(ROOT/f)==h for f,h in frozen.items()),
    'frozen_source_archive_matches':archive_matches,
    'frozen_source_file_count':len(frozen),
    'passed':all(r['passed'] for r in checks) and all(len(v)==1 for v in upstream.values()) and complete_groups==len(plan['source_replay_groups'])
       and archive_matches and all(checksum(ROOT/f)==h for f,h in frozen.items())
       and all(r['same_inputs'] and r['same_weights'] and all(c['same_shape'] and c['same_fps'] and c['finite'] for c in r['comparisons']) for r in replays),
    'source_sha256':checksum(Path(__file__)),
    'source_contract_sha256':checksum(ROOT/'utils/motiogen/source_cohorts.py'),
    'scope':'Local inference entry/modules are compared against the frozen replay contract; upstream commit and tracked patch are identical per method. Report/analysis and inactive other-method modules are outside this scoped contract. Whole-file mismatch is conservative and not proof that output changed. All selected task/seed groups retain both clocks and all interventions. Replayed coordinates are compared, but acceptance does not require the preferred result of exact equality. Old evidence remains; replacements are not extra independent samples.'}
json_write(folder/'audit.json',receipt)
print(json.dumps({'passed':receipt['passed'],'complete_groups':complete_groups,'expected_groups':len(plan['source_replay_groups']),
                  'completed_processes':len(replays),'mismatched_current_runs':sum(not r['passed'] for r in checks)}))
