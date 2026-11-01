#!/usr/bin/env python
"""Pilot audit (--audit) and analysis of V2: CHOIS guidance decomposition (config/v2_chois_20261101.json)."""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_next_round import test
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.native_analysis import common_step_jerk, holm
from motiogen.runtime import json_write

OUT = ROOT/'develop_logs/test_runs/20261101_v2_chois'
LEGACY = ROOT/'develop_logs/test_runs/20261024_native_followups'
ARMS = ['N0', 'G', 'HO', 'FF', 'W49', 'S05', 'S2', 'RND']
FEET = ['left_ankle', 'right_ankle', 'left_foot', 'right_foot']
OFFICIAL = {'official_foot_sliding': 'mean_fsliding_jnts', 'official_floor_height': 'mean_floor_height',
            'official_contact_percent': 'mean_contact_percent', 'official_hand_penetration': 'mean_hand_penetration_score',
            'official_penetration': 'mean_penetration_score', 'official_waypoints_xy_error': 'mean_waypoints_xy_pos_err',
            'official_start_object_error': 'mean_start_obj_com_pos_err', 'official_end_object_error': 'mean_end_obj_com_pos_err'}
METRICS = ['foot_jerk_0p1s', 'jerk_0p1s', 'mean_joint_speed_m_s', 'contact_distance', *OFFICIAL, 'generation_seconds']


def official(run):
    files = [p for p in (run/'raw/chois_results/evaluation_metrics_json').glob('*/*.json')
             if p.name != 'evaluation_metrics_for_all_test_data.json']
    if len(files) != 1:
        raise ValueError(f'{run}: expected one official per-window metrics file, found {len(files)}')
    return json.loads(files[0].read_text(encoding='utf-8'))


def chois_row(run):
    motion = load_motion(run/'motion.npz'); joints = np.asarray(motion['joints'], float); fps = float(motion['fps'])
    names = [str(n) for n in motion['joint_names']]
    feet = joints[:, [names.index(n) for n in FEET]]
    values = {'foot_jerk_0p1s': common_step_jerk(feet, fps), 'jerk_0p1s': common_step_jerk(joints, fps),
              'mean_joint_speed_m_s': float((np.linalg.norm(np.diff(joints, axis=0), axis=-1)*fps).mean())}
    scores = official(run)
    values.update({k: float(scores[v]) for k, v in OFFICIAL.items()})
    values['contact_distance'] = json.loads((run/'metrics.json').read_text(encoding='utf-8'))['metrics']['contact_distance']['value']
    values['generation_seconds'] = json.loads((run/'timing.json').read_text(encoding='utf-8'))['generation']['mean_seconds']
    steps = json.loads((run/'intervention.json').read_text(encoding='utf-8')).get('guidance', {}).get('steps', [])
    if steps:
        values['applied_perturbation'] = float(sum(s['posterior_variance']*1e3*s['applied_grad_norm'] for s in steps))
        values['feet_floor_grad_norm_sum'] = float(sum(s['feet_floor_grad_norm'] for s in steps))
        values['hand_object_grad_norm_sum'] = float(sum(s['hand_object_grad_norm'] for s in steps))
    return values


def states():
    found = {}
    later = sorted(p.name.removesuffix('_plan.json') for p in OUT.glob('continuation*_plan.json'))
    for stage in ['pilot', 'batch', *later]:
        path = OUT/f'{stage}_queue/state.json'
        if path.exists():
            found[stage] = (json.loads((OUT/f'{stage}_plan.json').read_text(encoding='utf-8')),
                            json.loads(path.read_text(encoding='utf-8'))['jobs'])
    return found


def legacy_runs():
    state = json.loads((LEGACY/'queue_fair/state.json').read_text(encoding='utf-8'))['jobs']
    return {k: Path(v['run_dir']) for k, v in state.items() if v.get('status') == 'success'}


def audit():
    plan, state = states()['pilot']; legacy = legacy_runs(); checks = []; runs = {}

    def check(name, ok, **detail):
        checks.append({'check': name, 'passed': bool(ok), **detail})
    for job in plan['jobs']:
        row = state.get(job['id'], {}); arm = job['v2']['arm']
        if row.get('status') != 'success':
            check('job_success', False, job=job['id'], status=row.get('status'), reason=row.get('reason')); continue
        run = Path(row['run_dir']); runs[arm] = run
        joints = np.asarray(load_motion(run/'motion.npz')['joints'])
        try:
            values = chois_row(run)
        except (ValueError, KeyError, OSError) as error:
            check('metrics_complete', False, job=job['id'], error=str(error)); continue
        check('finite_outputs', np.isfinite(joints).all() and all(np.isfinite(v) for v in values.values()), job=job['id'])
        check('time_within_3x_estimate', row['elapsed_seconds'] <= 3*job['estimated_seconds'], job=job['id'], elapsed=row['elapsed_seconds'])
        evidence = json.loads((run/'intervention.json').read_text(encoding='utf-8'))
        condition = job['config']['_intervention']
        expected = condition.get('last_step', 10)-1 if condition['guidance'] else 0
        check('guidance_calls', evidence['calls'].get('geometry_guidance', 0) == expected, job=job['id'],
              calls=evidence['calls'].get('geometry_guidance', 0), expected=expected)
        if job['v2'].get('legacy_job'):
            reference = np.asarray(load_motion(legacy[job['v2']['legacy_job']]/'motion.npz')['joints'])
            same = reference.shape == joints.shape
            check('legacy_bit_identical', same and np.array_equal(reference, joints), job=job['id'], legacy=job['v2']['legacy_job'],
                  max_abs_diff=float(np.abs(reference-joints).max()) if same else None)
        if arm == 'RND':
            steps = evidence['guidance']['steps']
            check('random_equal_norm', all(abs(s['random_direction_norm']/s['applied_grad_norm']-1) < 1e-4 for s in steps
                                           if s['applied_grad_norm'] > 0), job=job['id'])

    def motion(arm):
        return np.asarray(load_motion(runs[arm]/'motion.npz')['joints']) if arm in runs else None
    development = {}
    if 'Z0' in runs and 'N0' in runs:
        z, n = motion('Z0'), motion('N0')
        development['zero_scale_vs_N0_max_abs_m'] = float(np.abs(z-n).max())
        development['zero_scale_bit_identical_to_N0'] = bool(np.array_equal(z, n))
    for arm in ('HO', 'FF'):
        if arm in runs and 'G' in runs:
            check('term_arm_differs_from_G', not np.array_equal(motion(arm), motion('G')), arm=arm)
    if 'G' in runs:
        first = json.loads((runs['G']/'intervention.json').read_text(encoding='utf-8'))['guidance']['steps'][0]
        for arm in ('HO', 'FF', 'S05', 'S2', 'RND', 'Z0'):
            if arm in runs:
                other = json.loads((runs[arm]/'intervention.json').read_text(encoding='utf-8'))['guidance']['steps'][0]
                check('first_step_losses_match_G', other['t'] == first['t'] == 9 and
                      all(other[k] == first[k] for k in ('hand_object_loss', 'feet_floor_loss')), arm=arm)
    passed = bool(checks) and all(c['passed'] for c in checks)
    json_write(OUT/'pilot_audit.json', {'passed': passed, 'failed_checks': [c for c in checks if not c['passed']], 'checks': len(checks),
                                        'development_check': development,
                                        'criteria': json.loads((ROOT/'config/v2_chois_20261101.json').read_text(encoding='utf-8'))['pilot']['admission']})
    print(json.dumps({'passed': passed, 'checks': len(checks), 'development': development,
                      'failed': [c for c in checks if not c['passed']][:10]}))


def load_table():
    found = {}
    for plan, state in states().values():
        for row in plan.get('superseded', []):
            found.pop(row['id'], None)
        for item in plan.get('reused', []):
            found[item['id']] = (item['job'], Path(item['run']))
        for job in plan['jobs']:
            if state.get(job['id'], {}).get('status') == 'success' and not job['v2']['development_check']:
                found[job['id']] = (job, Path(state[job['id']]['run_dir']))
    table, objects = {}, {}
    for job, run in found.values():
        info = job['v2']
        table[(info['arm'], info['task'], job['seed'])] = chois_row(run); objects[info['task']] = info['object']
    return table, objects


def paired(table, on, off, metric, cluster=None):
    deltas, clusters = [], []
    for (arm, task, seed), values in sorted(table.items()):
        base = table.get((off, task, seed))
        if arm == on and base is not None and metric in values and metric in base:
            deltas.append(values[metric]-base[metric]); clusters.append(task if cluster is None else cluster[task])
    return deltas, clusters


def interaction(table, metric):
    deltas, clusters = [], []
    for (arm, task, seed), values in sorted(table.items()):
        rows = [table.get((a, task, seed)) for a in ('HO', 'FF', 'N0')]
        if arm == 'G' and all(r is not None for r in rows):
            deltas.append(values[metric]-rows[0][metric]-rows[1][metric]+rows[2][metric]); clusters.append(task)
    return deltas, clusters


def cell(table, arm, metric):
    by = collections.defaultdict(list)
    for (a, task, _), values in table.items():
        if a == arm and metric in values:
            by[task].append(values[metric])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def reading(ff, ho):
    if ff.get('excludes_zero') and ho.get('excludes_zero'):
        return 'both_terms'
    if ff.get('excludes_zero'):
        return 'feet_floor_term'
    if ho.get('excludes_zero'):
        return 'hand_object_term'
    return 'downgrade'


def analyze():
    table, objects = load_table()
    result = {'runs': len(table), 'expected_runs': 624, 'tasks': len(objects)}
    result['cells'] = {a: {m: cell(table, a, m) for m in METRICS+['applied_perturbation', 'feet_floor_grad_norm_sum', 'hand_object_grad_norm_sum']}
                       for a in ARMS}
    ff = test(*paired(table, 'FF', 'N0', 'foot_jerk_0p1s')); ho = test(*paired(table, 'HO', 'N0', 'foot_jerk_0p1s'))
    primary = {'FF-N0': ff, 'HO-N0': ho, 'reading': reading(ff, ho),
               'FF-N0_object_clusters': test(*paired(table, 'FF', 'N0', 'foot_jerk_0p1s', objects)),
               'n0_mean': cell(table, 'N0', 'foot_jerk_0p1s')}
    if primary['reading'] == 'both_terms':
        primary['interaction_G-HO-FF+N0'] = test(*interaction(table, 'foot_jerk_0p1s'))
    result['primary'] = primary
    result['replication'] = {m: test(*paired(table, 'G', 'N0', m)) for m in ('jerk_0p1s', 'foot_jerk_0p1s')}
    result['low_pass'] = {f'RND-{b}': test(*paired(table, 'RND', b, 'foot_jerk_0p1s')) for b in ('N0', 'G')}
    result['content'] = {f'{a}-N0': test(*paired(table, a, 'N0', 'mean_joint_speed_m_s')) for a in ('HO', 'G')}
    result['dose'] = {f'{a}-{b}|{m}': test(*paired(table, a, b, m)) for a in ('S05', 'S2', 'W49') for b in ('N0', 'G')
                      for m in ('foot_jerk_0p1s', 'jerk_0p1s')}
    family = {f'{a}-N0|{m}': test(*paired(table, a, 'N0', m)) for a in ARMS[1:] for m in METRICS}
    keys = [k for k, v in family.items() if 'sign_flip_p' in v]
    for k, p in zip(keys, holm([family[k]['sign_flip_p'] for k in keys])):
        family[k]['holm_p'] = p
    result['vs_N0'] = family
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'runs': result['runs'], 'reading': primary['reading'],
                      'FF-N0': {k: ff.get(k) for k in ('mean', 'ci95', 'sign_flip_p')},
                      'HO-N0': {k: ho.get(k) for k in ('mean', 'ci95', 'sign_flip_p')},
                      'G-N0 jerk': {k: result['replication']['jerk_0p1s'].get(k) for k in ('mean', 'ci95')}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    audit() if parser.parse_args().audit else analyze()
