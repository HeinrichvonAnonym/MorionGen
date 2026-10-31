#!/usr/bin/env python
"""Analysis of the 20261030 next round: M0 (ARDY protocol, G0) and V1 (reach geometry).

Registered endpoints, gates and stop rules: config/next_round_20261030.json.
Inputs are the pilot and batch queue receipts; the pilot runs reused by the
batch enter exactly once. Clusters are tasks; seeds are paired repetitions.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.native_analysis import (boundary_kinematics, bone_length_cv, cluster_t_interval, clustered_interval,
                                      common_history_consistency, common_step_boundary_jerk, common_step_jerk,
                                      facing_heading, foot_sensitivity, holm, sign_flip_p, threshold_success)
from motiogen.runtime import json_write

OUT = ROOT/'develop_logs/test_runs/20261030_next_round'
CONTRACT = json.loads((ROOT/'config/native_metric_contract.json').read_text())
SPEC = json.loads((ROOT/'config/next_round_20261030.json').read_text())


def runs(block):
    """Job id -> (job, run_dir) over pilot and batch, each job once."""
    found = {}
    for stage in ('pilot', 'batch', 'continuation'):
        plan_path = OUT/block/f'{stage}_plan.json'
        state_path = OUT/block/f'{stage}_queue/state.json'
        if not plan_path.exists() or not state_path.exists():
            continue
        plan = json.loads(plan_path.read_text()); state = json.loads(state_path.read_text())['jobs']
        for row in plan.get('superseded', []):
            found.pop(row['id'], None)
        for job in plan['jobs']:
            row = state.get(job['id'], {})
            if row.get('status') == 'success' and 'legacy_run_dir' not in job:
                found[job['id']] = (job, Path(row['run_dir']))
    return found


def test(deltas, clusters):
    t = cluster_t_interval(deltas, clusters)
    result = {'mean': t['mean'], 'ci95': t['ci'], 'clusters': t['clusters'], 'pairs': t['paired_n']}
    if t['clusters'] > 1:
        result['sign_flip_p'] = sign_flip_p(t['cluster_means'])['p']
        result['bootstrap_ci95'] = clustered_interval(deltas, clusters)['ci95']
        result['excludes_zero'] = bool(t['ci'][0] > 0 or t['ci'][1] < 0)
    return result


def ardy_row(job, run):
    motion = load_motion(run/'motion.npz'); joints = motion['joints']; fps = float(motion['fps'])
    intervention = json.loads((run/'intervention.json').read_text()); task = intervention['task']
    windows = task['windows']
    target = [np.asarray(w['targets_xz']) for w in windows]
    final = [np.asarray(w['final_at_targets_xz']) for w in windows]
    generated = [np.asarray(w['generated_at_targets_xz']) for w in windows]
    errors = np.concatenate([np.linalg.norm(f-t, axis=-1) for f, t in zip(final, target)])
    end_errors = np.array([np.linalg.norm(f[-1]-t[-1]) for f, t in zip(final, target)])
    required = np.array([np.linalg.norm(g[-1]-t[-1]) for g, t in zip(generated, target)])
    bounds = np.asarray(motion['chunk_boundaries'], int)
    values = {'waypoint_error_m': float(errors.mean()), 'end_waypoint_error_m': float(end_errors.mean()),
              'required_correction_m': float(required.mean()),
              'applied_root_displacement_m': float(np.mean([w['correction_root_displacement_max_m'] for w in windows])),
              'boundary_jerk_0p1s': common_step_boundary_jerk(joints, fps, bounds), 'jerk_0p1s': common_step_jerk(joints, fps),
              'boundary_jerk_native': boundary_kinematics(joints, fps, bounds)['boundary_jerk_rms'],
              'jerk_native': float(np.sqrt(np.mean((np.diff(joints, n=3, axis=0)*fps**3)**2)))}
    values.update({k: v for k, v in threshold_success(errors).items()})
    metrics = json.loads((run/'metrics.json').read_text())['metrics']
    for key in ('trunk_inclination', 'right_upper_arm_elevation'):
        if isinstance(metrics.get(key, {}).get('value'), (int, float)):
            values[key] = metrics[key]['value']
    feet = foot_sensitivity(motion, CONTRACT)
    if feet['status'] == 'measured':
        f = next(r for r in feet['rows'] if r['height_m'] == .05 and r['speed_m_s'] == .1)
        if f['sliding_fraction_given_candidate'] is not None:
            values['foot_sliding_5cm_10cms'] = f['sliding_fraction_given_candidate']
        values['foot_penetration_fraction'] = feet['floor_penetration_fraction']
    history_frames = int(job['config']['_intervention'].get('history_frames', 32))
    common = common_history_consistency(motion['overlap_errors_m'], history_frames, 8, len(motion['joint_names']), len(joints)//8)
    values['history_consistency_32_m'] = common['mean_m']
    _, values['frame0_heading_deg'] = facing_heading(joints, motion['joint_names'])
    return values


def cell_means(rows, key, metric):
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if metric in r['values']:
            by[key(r)][r['task']].append(r['values'][metric])
    return {k: float(np.mean([np.mean(v) for v in tasks.values()])) for k, tasks in sorted(by.items())}


def paired(rows, match, on, off, metric, reduce_keys=()):
    """Per task-seed difference on-off; extra keys in reduce_keys are averaged within the task-seed pair."""
    groups = collections.defaultdict(lambda: {'on': [], 'off': []})
    for r in rows:
        if not match(r) or metric not in r['values']:
            continue
        side = 'on' if on(r) else 'off' if off(r) else None
        if side:
            groups[(r['task'], r['seed'])][side].append(r['values'][metric])
    deltas, clusters = [], []
    for (task, seed), g in sorted(groups.items()):
        if g['on'] and len(g['on']) == len(g['off']):
            deltas.append(float(np.mean(g['on'])-np.mean(g['off']))); clusters.append(task)
    return deltas, clusters


def analyze_m0(folder='m0'):
    rows = []
    for job_id, (job, run) in runs(folder).items():
        c = job['config']['_intervention']
        rows.append({'id': job_id, 'block': job_id.split('_')[0], 'task': job['task_id'], 'seed': job['seed'],
                     'condition': c, 'values': ardy_row(job, run)})
    m0a = [r for r in rows if r['block'] == 'm0a']; m0c = [r for r in rows if r['block'] == 'm0c']
    result = {'runs': {'m0a': len(m0a), 'm0c': len(m0c)}}
    metrics = ['waypoint_error_m', 'end_waypoint_error_m', 'required_correction_m', 'applied_root_displacement_m',
               'boundary_jerk_0p1s', 'jerk_0p1s', 'boundary_jerk_native', 'jerk_native', 'success_2cm', 'success_5cm',
               'success_10cm', 'trunk_inclination', 'right_upper_arm_elevation', 'foot_sliding_5cm_10cms',
               'history_consistency_32_m']
    cell = lambda r: f"{r['condition']['text']}|{r['condition']['waypoints']}|c{r['condition']['correction']}"
    result['m0a_cells'] = {m: cell_means(m0a, cell, m) for m in metrics}
    cond = lambda r, **kw: all(r['condition'].get(k) == v for k, v in kw.items())
    contrasts = {}
    for correction in (0, 1):
        contrasts[f'aligned_minus_conflict_sparse_c{correction}'] = test(*paired(
            m0a, lambda r: cond(r, waypoints='sparse', correction=correction),
            lambda r: r['condition']['text'] == 'aligned', lambda r: r['condition']['text'] == 'conflict', 'waypoint_error_m'))
        contrasts[f'dense_minus_sparse_aligned_c{correction}'] = test(*paired(
            m0a, lambda r: cond(r, text='aligned', correction=correction),
            lambda r: r['condition']['waypoints'] == 'dense', lambda r: r['condition']['waypoints'] == 'sparse', 'waypoint_error_m'))
    for text, waypoints in [('conflict', 'sparse'), ('aligned', 'sparse'), ('aligned', 'dense')]:
        for metric in ('boundary_jerk_0p1s', 'trunk_inclination', 'applied_root_displacement_m'):
            contrasts[f'correction_{text}_{waypoints}_{metric}'] = test(*paired(
                m0a, lambda r: cond(r, text=text, waypoints=waypoints),
                lambda r: r['condition']['correction'] == 1, lambda r: r['condition']['correction'] == 0, metric))
    result['m0a_contrasts'] = contrasts
    offsets = sorted({r['condition']['offset_m'] for r in m0c})
    dose = {}
    for offset in offsets:
        dose[str(offset)] = {m: test(*paired(m0c, lambda r: r['condition']['offset_m'] == offset,
                                              lambda r: r['condition']['correction'] == 1, lambda r: r['condition']['correction'] == 0, m))
                             for m in ('boundary_jerk_0p1s', 'boundary_jerk_native', 'jerk_0p1s', 'trunk_inclination',
                                       'right_upper_arm_elevation', 'foot_sliding_5cm_10cms', 'history_consistency_32_m',
                                       'end_waypoint_error_m')}
        base = [r['values']['boundary_jerk_0p1s'] for r in m0c if r['condition']['offset_m'] == offset and r['condition']['correction'] == 0]
        dose[str(offset)]['c0_boundary_jerk_0p1s_mean'] = float(np.mean(base)) if base else None
        for c in (0, 1):
            req = [r['values']['required_correction_m'] for r in m0c if r['condition']['offset_m'] == offset and r['condition']['correction'] == c]
            dose[str(offset)][f'required_correction_m_c{c}'] = float(np.mean(req)) if req else None
    result['m0c_dose_response'] = dose
    g0 = [o for o in offsets if o <= .5]
    deltas, clusters = paired(m0c, lambda r: r['condition']['offset_m'] in g0, lambda r: r['condition']['correction'] == 1,
                              lambda r: r['condition']['correction'] == 0, 'boundary_jerk_0p1s')
    primary = test(deltas, clusters)
    base = cell_means([r for r in m0c if r['condition']['offset_m'] in g0 and r['condition']['correction'] == 0],
                      lambda r: 'c0', 'boundary_jerk_0p1s').get('c0')
    primary['c0_mean'] = base
    primary['relative_increase'] = primary['mean']/base if base else None
    primary['offsets'] = g0
    primary['passed'] = bool(primary.get('excludes_zero') and primary['ci95'][0] > 0 and primary['relative_increase'] >= .2)
    result['G0'] = primary
    return result


def reach_row(job, run):
    motion = load_motion(run/'motion.npz'); joints = motion['joints']; fps = float(motion['fps'])
    names = [str(n) for n in motion['joint_names']]; ix = {n: i for i, n in enumerate(names)}
    spec = job['scene']['constraints'][0]; meta = job['scene']['next_round_task']
    frame = int(round(spec['time_seconds']*fps)); target = np.asarray(spec['position_m'])
    side = spec['joint'].split('_')[0]; other = ('left' if side == 'right' else 'right')
    sh, el, wr = (joints[:, ix[f'{side}_{j}']] for j in ('shoulder', 'elbow', 'wrist'))
    upper = np.linalg.norm(el-sh, axis=-1); fore = np.linalg.norm(wr-el, axis=-1)
    unit = lambda v: v/np.linalg.norm(v)
    refit = sh[frame]+np.median(upper)*unit(el[frame]-sh[frame])+np.median(fore)*unit(wr[frame]-el[frame])
    forward, heading = facing_heading(joints, names)
    error = float(np.linalg.norm(wr[frame]-target))
    values = {'wrist_error_m': error, 'other_wrist_distance_m': float(np.linalg.norm(joints[frame, ix[other+'_wrist']]-target)),
              'refit_wrist_error_m': float(np.linalg.norm(refit-target)), 'frame0_heading_deg': heading,
              'shoulder_distance_ratio': float(np.linalg.norm(target-sh[0])/(upper[0]+fore[0])),
              'arm_bone_cv': float(np.mean([upper.std()/upper.mean(), fore.std()/fore.mean()])),
              'pelvis_displacement_m': float(np.linalg.norm(joints[frame, 0, :2]-joints[0, 0, :2])),
              'jerk_0p1s': common_step_jerk(joints, fps),
              'jerk_native': float(np.sqrt(np.mean((np.diff(joints, n=3, axis=0)*fps**3)**2)))}
    values['stretch_contribution_m'] = values['refit_wrist_error_m']-error
    values['other_hand_closer'] = float(values['other_wrist_distance_m'] < error)
    values.update(threshold_success([error]))
    return values, meta


def analyze_v1(folder='v1'):
    rows = []
    for job_id, (job, run) in runs(folder).items():
        values, meta = reach_row(job, run)
        rows.append({'id': job_id, 'block': job_id.split('_')[0], 'method': job['method'], 'task': job['task_id'], 'seed': job['seed'],
                     'condition': job['config']['_intervention'], 'tier': meta['tier'], 'values': values})
    v1 = [r for r in rows if r['block'] == 'v1']; v1b = [r for r in rows if r['block'] == 'v1b']
    label = lambda r: r['method']+'|'+','.join(f'{k}={v}' for k, v in r['condition'].items())+'|'+r['tier']
    metrics = ['wrist_error_m', 'refit_wrist_error_m', 'stretch_contribution_m', 'other_wrist_distance_m', 'other_hand_closer',
               'success_2cm', 'success_5cm', 'success_10cm', 'frame0_heading_deg', 'shoulder_distance_ratio', 'arm_bone_cv',
               'pelvis_displacement_m', 'jerk_0p1s', 'jerk_native']
    result = {'folder': folder, 'runs': {'v1': len(v1), 'v1b': len(v1b)}, 'cells': {m: cell_means(v1, label, m) for m in metrics}}
    lcm = [r for r in v1 if r['method'] == 'motionlcm' and r['tier'] == 'reachable']
    on = [r for r in lcm if r['condition']['control_residual'] == 1]
    per_task = collections.defaultdict(list)
    for r in on:
        per_task[r['task']].append(r['values']['wrist_error_m'])
    primary = test([r['values']['wrist_error_m'] for r in on], [r['task'] for r in on])
    residual = test(*paired(lcm, lambda r: True, lambda r: r['condition']['control_residual'] == 1,
                            lambda r: r['condition']['control_residual'] == 0, 'wrist_error_m'))
    mean = primary['mean']
    primary['stop_rule'] = ('rewrite_P1_feasibility' if mean <= .10 else 'restore_P1_new_number' if mean > .20 else 'decide_with_V1b')
    primary['residual_on_minus_off'] = residual
    result['V1_primary'] = primary
    contrasts = {}
    omni = [r for r in v1 if r['method'] == 'omnicontrol']
    for tier in ('reachable', 'stepping', 'legacy', 'mirror'):
        mine = [r for r in omni if r['tier'] == tier]
        for metric in ('wrist_error_m', 'refit_wrist_error_m', 'arm_bone_cv', 'jerk_0p1s'):
            contrasts[f'omnicontrol_spatial_{tier}_{metric}'] = test(*paired(mine, lambda r: True,
                lambda r: r['condition']['spatial'] == 1, lambda r: r['condition']['spatial'] == 0, metric))
        lc = [r for r in v1 if r['method'] == 'motionlcm' and r['tier'] == tier]
        contrasts[f'motionlcm_residual_{tier}_wrist_error_m'] = test(*paired(lc, lambda r: True,
            lambda r: r['condition']['control_residual'] == 1, lambda r: r['condition']['control_residual'] == 0, 'wrist_error_m'))
    result['contrasts'] = contrasts
    off = [r for r in v1 if (r['method'] == 'motionlcm' and r['condition']['control_residual'] == 0) or
           (r['method'] == 'omnicontrol' and r['condition'] == {'realism': 0, 'spatial': 0})]
    tiers = {}
    for method in ('motionlcm', 'omnicontrol'):
        for tier in ('reachable', 'stepping', 'legacy', 'mirror'):
            task_means = collections.defaultdict(list)
            for r in off:
                if r['method'] == method and r['tier'] == tier:
                    task_means[r['task']].append(r['values']['wrist_error_m'])
            tiers[f'{method}|{tier}'] = [float(np.mean(v)) for v in task_means.values()]
    result['guidance_off_tier_task_means'] = tiers
    mirror = [r for r in v1 if r['tier'] == 'mirror']
    result['chirality'] = {'mirror_runs': len(mirror), 'other_hand_closer_fraction': float(np.mean([r['values']['other_hand_closer'] for r in mirror])) if mirror else None}
    if v1b:
        pairs = collections.defaultdict(dict)
        for r in v1 + v1b:
            key = (r['method'], json.dumps(r['condition'], sort_keys=True), r['task'], r['seed'])
            pairs[key][r['block']] = r['values']['wrist_error_m']
        by_cond = collections.defaultdict(lambda: ([], []))
        for (method, condition, task, seed), p in pairs.items():
            if 'v1' in p and 'v1b' in p:
                by_cond[method+'|'+condition][0].append(p['v1b']-p['v1']); by_cond[method+'|'+condition][1].append(task)
        result['V1b_direction_text_minus_default'] = {k: test(*v) for k, v in sorted(by_cond.items())}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block', choices=['m0', 'v1', 'all'], default='all')
    parser.add_argument('--m0-folder', default='m0', help='m0 or a registered amendment folder such as m0_amend2')
    parser.add_argument('--v1-folder', default='v1', help='v1 or a registered amendment folder such as v1_amend1')
    args = parser.parse_args()
    out = OUT/'analysis'; out.mkdir(parents=True, exist_ok=True)
    results = {}
    if args.block in ('m0', 'all'):
        results['m0'] = analyze_m0(args.m0_folder); json_write(out/'m0.json', results['m0'])
    if args.block in ('v1', 'all'):
        results['v1'] = analyze_v1(args.v1_folder); json_write(out/'v1.json', results['v1'])
    if args.block == 'all':
        g0 = results['m0']['G0']; v1 = results['v1']['V1_primary']['residual_on_minus_off']
        p = [g0.get('sign_flip_p', 1.), v1.get('sign_flip_p', 1.)]
        json_write(out/'primary_holm.json', {'endpoints': ['G0 boundary jerk c1-c0 (offsets <= 0.5 m)', 'V1 MotionLCM residual on-off, reachable tier'],
                                            'sign_flip_p': p, 'holm_adjusted': holm(p)})
    print(json.dumps({k: {'G0': v.get('G0', {}).get('passed'), 'V1': v.get('V1_primary', {}).get('stop_rule')} for k, v in results.items()}))


if __name__ == '__main__':
    main()
