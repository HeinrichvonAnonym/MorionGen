"""Causal, fixed-frame marker diagnostic; never pool into primary inference."""
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from itertools import combinations
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'utils'))
from motiogen.runtime import json_write, checksum

base = ROOT / 'develop_logs/test_runs/20261024_closd_marker_probe'
original_matrix=ROOT/'develop_logs/test_runs/20261025_source_replays/original_analysis.json'
primary = json.loads((original_matrix if original_matrix.exists() else ROOT / 'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json').read_text())
cross_clock_cases = []
for pair in primary['clock_pairs']:
    if pair['method'] != 'closd' or pair['exact_same_coordinates']:
        continue
    reference_trace = json.loads((ROOT / pair['paused_run'] / 'trace.json').read_text())
    target = reference_trace['requested_goals'][1]['position']
    clocks = {}
    for clock in ['paused', 'realtime']:
        run = ROOT / pair[clock + '_run']
        trace = json.loads((run / 'trace.json').read_text())
        consumed = [p['frame'] for p in trace['planner'] if p['frame'] >= trace['goal_applied_frame']
                    and np.allclose(p['target'], target, atol=1e-6, rtol=0)]
        assert consumed
        clocks[clock] = {'goal_applied_frame': trace['goal_applied_frame'], 'first_planner_goal_frame': min(consumed)}
    x = np.load(ROOT / pair['paused_run'] / 'motion.npz')['joints']
    y = np.load(ROOT / pair['realtime_run'] / 'motion.npz')['joints']
    delta = np.max(np.abs(x.astype(float) - y.astype(float)), axis=(1, 2))
    changed = np.flatnonzero(delta > 1e-6)
    assert len(changed)
    cross_clock_cases.append({'method': 'closd', 'case': pair['paused_run'], 'task': pair['task'],
        'seed': pair['seed'], 'condition': pair['condition'], 'paused_run': pair['paused_run'],
        'realtime_run': pair['realtime_run'], 'clocks': clocks,
        'same_planner_goal_frame': clocks['paused']['first_planner_goal_frame'] == clocks['realtime']['first_planner_goal_frame'],
        'first_difference_gt_1um_frame': int(changed[0]), 'max_delta_m': float(delta.max())})
assert len(cross_clock_cases) == 8 and sum(c['same_planner_goal_frame'] for c in cross_clock_cases) == 6
rows = []
for plan_name, queue_name in [('plan.json', 'queue'), ('extension_plan.json', 'extension_queue')]:
    if not (base / plan_name).exists():
        continue
    plan = json.loads((base / plan_name).read_text())
    state = json.loads((base / queue_name / 'state.json').read_text())
    for job in plan['jobs']:
        record = state['jobs'][job['id']]
        if record['status'] != 'success':
            continue
        run = Path(record['run_dir'])
        probe = json.loads((run / 'marker_probe.json').read_text())
        manifest = json.loads((run / 'manifest.json').read_text())
        binding = json.loads((run / 'weight_binding.json').read_text())
        assert manifest['scenario'] == job['scene'] and manifest['seed'] == job['seed']
        assert probe['complete'] and binding['after_run_unchanged']
        assert len(probe['records']) == 600
        for overlay in probe['overlays']:
            assert hashlib.sha256(overlay['executed_source'].encode()).hexdigest() == overlay['overlay_sha256']
        motion = np.load(run / 'motion.npz')['joints']
        assert motion.shape == (600, 22, 3) and np.isfinite(motion).all()
        force = np.asarray([r['marker_contact_force_n'] for r in probe['records']], dtype=float)
        assert force.shape == (600, 3) and np.isfinite(force).all()
        magnitudes = np.linalg.norm(force, axis=1)
        scene = job['scene']
        reference = scene['reference_paused'] if probe['goal_update_frame'] == 150 else scene['reference_realtime']
        original = np.load(ROOT / reference / 'motion.npz')['joints']
        original_binding = json.loads((ROOT / reference / 'weight_binding.json').read_text())
        assert binding['identity'] == original_binding['identity']
        delta = np.abs(motion.astype(float) - original.astype(float))
        row = {'method': 'closd', 'case': scene['reference_paused'], 'seed': job['seed'],
               'goal_update_frame': probe['goal_update_frame'], 'marker_mode': probe['marker_mode'],
               'run': str(run.relative_to(ROOT)), 'reference_run': reference,
               'reference_exact': bool(np.array_equal(motion, original)),
               'reference_max_delta_m': float(delta.max()),
               'marker_max_force_n': float(magnitudes.max()),
               'marker_nonzero_force_frames': np.flatnonzero(magnitudes > 1e-8).tolist(),
               'force_norm_by_frame_n': magnitudes.tolist(),
               'weight_identity': binding['identity'],
               'record_sha256': checksum(run / 'marker_probe.json'),
               'frames': len(motion), 'finite': True}
        if row['marker_mode'] == 'isolated':
            assert row['marker_max_force_n'] == 0, 'Isolation did not exclude target contacts'
        rows.append(row)

comparisons = []
for a, b in combinations(rows, 2):
    if a['case'] != b['case'] or a['marker_mode'] != b['marker_mode']:
        continue
    assert a['goal_update_frame'] != b['goal_update_frame']
    x = np.load(ROOT / a['run'] / 'motion.npz')['joints']
    y = np.load(ROOT / b['run'] / 'motion.npz')['joints']
    delta = np.max(np.abs(x.astype(float) - y.astype(float)), axis=(1, 2))
    nonzero = np.flatnonzero(delta > 1e-6)
    comparisons.append({'method': 'closd', 'case': a['case'], 'marker_mode': a['marker_mode'],
                        'frames': sorted([a['goal_update_frame'], b['goal_update_frame']]),
                        'a_run': a['run'], 'b_run': b['run'],
                        'exact_same_coordinates': bool(np.array_equal(x, y)),
                        'first_difference_gt_1um_frame': int(nonzero[0]) if len(nonzero) else None,
                        'max_delta_m': float(delta.max()), 'max_delta_by_frame_m': delta.tolist()})
cases = []
for case in sorted({r['case'] for r in rows}):
    group = [r for r in rows if r['case'] == case]
    comparisons_case = [c for c in comparisons if c['case'] == case]
    if len(group) != 4 or len(comparisons_case) != 2:
        continue
    native = next(c for c in comparisons_case if c['marker_mode'] == 'native')
    isolated = next(c for c in comparisons_case if c['marker_mode'] == 'isolated')
    original_case = next(c for c in cross_clock_cases if c['case'] == case)
    cases.append({'method': 'closd', 'case': case, 'task': original_case['task'],
                  'seed': original_case['seed'], 'condition': json.loads(original_case['condition']),
                  'native_replays_both_exact': all(r['reference_exact'] for r in group if r['marker_mode'] == 'native'),
                  'native_schedule_delta_m': native['max_delta_m'],
                  'isolated_schedule_delta_m': isolated['max_delta_m'],
                  'isolated_schedules_exact': isolated['exact_same_coordinates'],
                  'native_contacts_present': all(r['marker_max_force_n'] > 0 for r in group if r['marker_mode'] == 'native'),
                  'isolation_contacts_zero': all(r['marker_max_force_n'] == 0 for r in group if r['marker_mode'] == 'isolated')})
output = {'updated_at': datetime.now(timezone.utc).isoformat(), 'rows': rows, 'comparisons': comparisons,
          'cases': cases, 'complete_cases': len(cases), 'accepted_cells': len(rows),
          'cross_clock_cases': cross_clock_cases, 'primary_analysis_cutoff': primary['updated_at'],
          'scope': 'Post-hoc deterministic mechanism diagnosis of six cross-clock discrepant cases with the same planner update window. Fixed-frame replay x native/isolated marker collision group. Not a population estimate, not primary capability results, and not H timing evidence.',
          'source_sha256': {str(p.relative_to(ROOT)): checksum(p) for p in [Path(__file__).resolve(), ROOT / 'python/scripts/closd_marker_probe.py']}}
json_write(base / 'analysis.json', output)
print(json.dumps({k: output[k] for k in ['accepted_cells', 'complete_cases', 'cases']}))
