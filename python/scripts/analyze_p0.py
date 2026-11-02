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
            out.append({'id': job['id'], **job['p0'], 'seed': job['seed'], 'values': goal_row(job, run)})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    args = parser.parse_args()
    for r in sorted(rows(Path(args.folder)), key=lambda r: (r['offset_m'], r['arm'], r['task'], r['seed'])):
        v = r['values']
        print(f"{r['arm']:4s} d{r['offset_m']:g} task{r['task']:<2d} seed{r['seed']} err {v['goal_error_m']:.3f} "
              f"max {v['goal_error_max_m']:.3f} frac {v['goal_fraction']:.2f} bjerk {v['boundary_jerk_0p1s']:.1f} "
              f"jerk {v['jerk_0p1s']:.1f} slide {v.get('foot_sliding_5cm_10cms', float('nan')):.3f} "
              f"trunk {v.get('trunk_inclination', float('nan')):.1f} corr {v['correction_root_displacement_m']:.3f}")


if __name__ == '__main__':
    main()
