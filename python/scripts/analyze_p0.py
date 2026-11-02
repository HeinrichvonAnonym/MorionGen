#!/usr/bin/env python
"""Rows for P0 goal-lookahead runs: goal accuracy plus the M2 kinematic metrics.

--folder: a P0 folder holding <name>_plan.json and <name>_queue/state.json (or smoke_plan.json and queue/).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_next_round as nr


def goal_row(job, run):
    motion = nr.load_motion(run/'motion.npz'); joints = motion['joints']; fps = float(motion['fps'])
    task = json.loads((run/'intervention.json').read_text())['task']
    goals = task['goals']
    errors = np.array([g['error_m'] for g in goals])
    values = {'goal_error_m': float(errors.mean()), 'goal_error_max_m': float(errors.max()),
              'goal_success_5cm': float((errors <= .05).mean()), 'goal_failure_50cm': float((errors > .5).mean()),
              'goal_fraction': float(np.mean([g['fraction_toward_target'] for g in goals])),
              'correction_root_displacement_m': float(np.mean([g['correction_root_displacement_max_m'] for g in goals])),
              'boundary_jerk_0p1s': nr.common_step_boundary_jerk(joints, fps, np.asarray(motion['chunk_boundaries'], int)),
              'jerk_0p1s': nr.common_step_jerk(joints, fps)}
    metrics = json.loads((run/'metrics.json').read_text())['metrics']
    for key in ('trunk_inclination', 'right_upper_arm_elevation'):
        if isinstance(metrics.get(key, {}).get('value'), (int, float)):
            values[key] = metrics[key]['value']
    feet = nr.foot_sensitivity(motion, nr.CONTRACT)
    if feet['status'] == 'measured':
        f = next(r for r in feet['rows'] if r['height_m'] == .05 and r['speed_m_s'] == .1)
        if f['sliding_fraction_given_candidate'] is not None:
            values['foot_sliding_5cm_10cms'] = f['sliding_fraction_given_candidate']
    return values


def rows(folder):
    out, seen = [], set()
    plans = sorted(folder.glob('*_plan.json'))
    if (folder/'batch_plan.json').exists():
        plans = [p for p in plans if p.name.startswith(('batch', 'continuation'))]
    for plan_path in plans:
        name = plan_path.name[:-len('_plan.json')]
        queue = folder/('queue' if name == 'smoke' else f'{name}_queue')/'state.json'
        if not queue.exists():
            continue
        state = json.loads(queue.read_text())['jobs']
        plan = json.loads(plan_path.read_text())
        for job in plan['jobs']+[r['job'] for r in plan.get('reused', [])]:
            row = state.get(job['id'], {})
            run = Path(row['run_dir']) if row.get('status') == 'success' else None
            if run is None:
                reused = next((r for r in plan.get('reused', []) if r['id'] == job['id']), None)
                run = Path(reused['run']) if reused else None
            if run is None or job['id'] in seen:
                continue
            seen.add(job['id'])
            out.append({'id': job['id'], **(job['p0'] if 'p0' in job else job['p1']), 'seed': job['seed'], 'run': str(run),
                        'values': goal_row(job, run)})
    return out


METRICS = ['goal_error_m', 'goal_error_max_m', 'goal_success_5cm', 'goal_failure_50cm', 'goal_fraction',
           'correction_root_displacement_m', 'boundary_jerk_0p1s', 'jerk_0p1s', 'foot_sliding_5cm_10cms',
           'trunk_inclination', 'right_upper_arm_elevation']
MARGIN = .05


def analyze(folder):
    from motiogen.native_analysis import holm
    from motiogen.runtime import json_write
    from plan_p0 import SPEC_PATH
    spec = json.loads(SPEC_PATH.read_text(encoding='utf-8'))
    arms, offsets = list(spec['arms']), spec['offsets_m']
    table = {(r['arm'], r['offset_m'], r['task'], r['seed']): r['values'] for r in rows(folder)}

    def paired(on, off, chosen, metric, shift=0.):
        deltas, clusters = [], []
        for (a, o, task, seed), values in sorted(table.items()):
            base = table.get((off, o, task, seed))
            if a == on and o in chosen and base is not None and metric in values and metric in base:
                deltas.append(values[metric]-base[metric]-shift); clusters.append(task)
        return deltas, clusters

    def cell(arm, offset, metric):
        by = {}
        for (a, o, task, _), values in table.items():
            if a == arm and o == offset and metric in values:
                by.setdefault(task, []).append(values[metric])
        return float(np.mean([np.mean(v) for v in by.values()])) if by else None

    result = {'runs': len(table), 'expected_runs': spec['runs']}
    result['cells'] = {f'{a}|d{o:g}': {m: cell(a, o, m) for m in METRICS} for a in arms for o in offsets}
    hypotheses = {'H1': nr.test(*paired('G4', 'G0', offsets, 'goal_error_m')),
                  'H2': nr.test(*paired('G4', 'W0g', offsets, 'jerk_0p1s')),
                  'H3': nr.test(*paired('G4', 'W0g', offsets, 'foot_sliding_5cm_10cms', MARGIN))}
    hypotheses['H3']['margin'] = MARGIN; hypotheses['H3']['mean_difference'] = hypotheses['H3']['mean']+MARGIN
    for key, p in zip(('H1', 'H2', 'H3'), holm([hypotheses[k]['sign_flip_p'] for k in ('H1', 'H2', 'H3')])):
        hypotheses[key]['holm_p'] = p
        hypotheses[key]['supported'] = bool(p < .05 and hypotheses[key]['ci95'][1] < 0)
    accurate = {str(o): result['cells'][f'G4|d{o:g}']['goal_error_m'] <= .05 for o in offsets}
    hypotheses['H1']['G4_accurate_per_offset'] = accurate
    hypotheses['H1']['supported'] = bool(hypotheses['H1']['supported'] and all(accurate.values()))
    result['primary'] = hypotheses
    secondary = {}
    for on, off in [('G1', 'G0'), ('G2', 'G0'), ('G4', 'G0'), ('G4', 'G2'), ('G2', 'G1'), ('G4W', 'G4'), ('W0g', 'G0'), ('G4', 'W0g')]:
        secondary[f'{on}-{off}'] = {'pooled': {m: nr.test(*paired(on, off, offsets, m)) for m in METRICS},
                                    'per_offset': {str(o): {m: nr.test(*paired(on, off, [o], m)) for m in METRICS} for o in offsets}}
    result['secondary'] = secondary
    json_write(folder/'analysis.json', result)
    print(json.dumps({'runs': result['runs'], 'primary': {k: {x: v.get(x) for x in ('mean', 'ci95', 'holm_p', 'supported')}
                                                          for k, v in hypotheses.items()}}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    if args.analyze:
        analyze(Path(args.folder))
        return
    for r in sorted(rows(Path(args.folder)), key=lambda r: (r['offset_m'], r['arm'], r['task'], r['seed'])):
        v = r['values']
        print(f"{r['arm']:4s} d{r['offset_m']:g} task{r['task']:<2d} seed{r['seed']} err {v['goal_error_m']:.3f} "
              f"max {v['goal_error_max_m']:.3f} frac {v['goal_fraction']:.2f} bjerk {v['boundary_jerk_0p1s']:.1f} "
              f"jerk {v['jerk_0p1s']:.1f} slide {v.get('foot_sliding_5cm_10cms', float('nan')):.3f} "
              f"trunk {v.get('trunk_inclination', float('nan')):.1f} corr {v['correction_root_displacement_m']:.3f}")


if __name__ == '__main__':
    main()
