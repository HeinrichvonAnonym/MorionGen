#!/usr/bin/env python
"""X3 analysis (config/x3_cross_method_layers_20261104.json): run metrics, admission receipts, H1-H4 per method, precision rule.

--stage pilot|confirm|top_up selects the task-seed cells; writes develop_logs/test_runs/20261104_x3/analysis_<stage>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
sys.path.insert(0, str(ROOT/'python/scripts'))
import analyze_next_round as nr
from motiogen import x3_tasks as xt
from motiogen.native_analysis import common_step_boundary_jerk, common_step_jerk, holm, sign_flip_p
from motiogen.runtime import json_write

REPS, SEED = 10000, 20261104
E_PRIMARY, E_SECONDARY = 'residual_force_rms_bw', 'unsupported_fraction_5pct_bw'
KINEMATIC = ['goal_error_m', 'goal_success_5cm', 'jerk_0p1s', 'jerk_frame', 'boundary_jerk_0p1s', 'foot_sliding_5cm_10cms']


def frame_jerk(joints, fps):
    d = np.diff(np.asarray(joints, float), 3, axis=0)*fps**3
    return float(np.sqrt(np.mean(d**2)))


def kinematics(path, goal_rows=None):
    motion = nr.load_motion(path/'motion.npz')
    rec = json.loads((path/'receipt.json').read_text())
    joints, fps, names = motion['joints'], float(motion['fps']), list(motion['joint_names'])
    goals = xt.goal_receipt(joints, names, goal_rows) if goal_rows is not None else rec['goals']
    errors = np.array([g['error_m'] for g in goals])
    row = {'goal_error_m': float(errors.mean()), 'goal_success_5cm': float((errors <= .05).mean()),
           'jerk_0p1s': common_step_jerk(joints, fps), 'jerk_frame': frame_jerk(joints, fps)}
    if 'chunk_boundaries' in motion:
        row['boundary_jerk_0p1s'] = common_step_boundary_jerk(joints, fps, np.asarray(motion['chunk_boundaries'], int))
    feet = nr.foot_sensitivity(motion, nr.CONTRACT)
    if feet['status'] == 'measured':
        f = next(r for r in feet['rows'] if r['height_m'] == .05 and r['speed_m_s'] == .1)
        if f['sliding_fraction_given_candidate'] is not None:
            row['foot_sliding_5cm_10cms'] = f['sliding_fraction_given_candidate']
    return row, rec


def collect(stage):
    reg = xt.prereg()
    table, receipts, missing = {}, {}, []
    for method in reg['methods']:
        arms = ['C0', *xt.GENERATED[method], *xt.CORRECTED[method]]
        if method == 'dart':
            arms.append('W1')
        for t in xt.job_tasks(method, stage):
            for s in xt.seeds(stage):
                gfile = xt.base_dir(method, t['id'], s)/'goals.json'
                if not gfile.exists():
                    missing.append(f'{method} task {t["id"]} seed {s}: no C0'); continue
                goals = json.loads(gfile.read_text())
                for d in reg['offsets_m']:
                    for arm in arms:
                        path = xt.run_dir(method, t['id'], s, arm, None if arm == 'C0' else d)
                        if not (path/'receipt.json').exists():
                            if not (method == 'dart' and arm == 'W1'):
                                missing.append(str(path.relative_to(xt.OUT)))
                            continue
                        row, rec = kinematics(path, goals[f'{d:g}'] if arm == 'C0' else None)
                        table[(method, arm, d, t['id'], s)] = row
                        receipts[(method, arm, d, t['id'], s)] = rec
    e_runs = xt.OUT/'e'/'runs'
    for (method, arm, d, task, seed), row in table.items():
        if d != 1.0:
            continue
        m = e_runs/f'{method}_t{task:02d}_s{seed}_{arm}'/'metrics.json'
        if m.exists():
            e = json.loads(m.read_text())
            row['E_valid'] = bool(e['valid'])
            row['ik_rms_m'] = e['ik']['rms_mean_m']; row['ik_max_m'] = e['ik']['max_m']
            for key in (E_PRIMARY, E_SECONDARY):
                row[key] = e['metrics'].get(key)
    return table, receipts, missing


def interval(values, clusters):
    keys = sorted(set(clusters))
    means = np.array([np.mean([v for v, c in zip(values, clusters) if c == k]) for k in keys])
    rng = np.random.default_rng(SEED)
    boot = means[rng.integers(0, len(means), (REPS, len(means)))].mean(1)
    out = {'mean': float(means.mean()), 'ci95': np.percentile(boot, [2.5, 97.5]).tolist(), 'clusters': len(keys),
           'pairs': len(values), 'cluster_means': means.tolist()}
    if len(means) > 1:
        out['sign_flip_p'] = sign_flip_p(means)['p']
        out['sd_cluster_means'] = float(means.std(ddof=1))
    return out


def paired(table, method, on, off, metric, offsets, shift=0., valid=False):
    values, clusters = [], []
    for (m, a, d, task, seed), row in sorted(table.items()):
        if m != method or a != on or d not in offsets:
            continue
        base = table.get((m, off, d, task, seed))
        if base is None or row.get(metric) is None or base.get(metric) is None:
            continue
        if valid and not (row.get('E_valid') and base.get('E_valid')):
            continue
        values.append(row[metric]-base[metric]-shift); clusters.append(task)
    return (interval(values, clusters) if values else None)


def levels(table, method, arm, d, metric):
    by = {}
    for (m, a, o, task, _), row in table.items():
        if m == method and a == arm and o == d and row.get(metric) is not None:
            by.setdefault(task, []).append(row[metric])
    if not by:
        return None
    means = np.array([np.mean(v) for v in by.values()])
    return {'mean': float(means.mean()), 'median': float(np.median(means)), 'tasks': len(means)}


def hypotheses(table, method, reg):
    offsets = reg['offsets_m']
    T = reg['methods'][method]['T_arm']
    c0_jerk = np.mean([r['jerk_0p1s'] for (m, a, *_), r in table.items() if m == method and a == 'C0'])
    margin_jerk = .2*float(c0_jerk)
    tests = {
        'H1a': ('lt', paired(table, method, T, 'C0', 'goal_error_m', offsets)),
        'H1b': ('lt', paired(table, method, T, 'C0', 'jerk_0p1s', offsets, shift=margin_jerk)),
        'H2': ('gt', paired(table, method, 'O', T, E_PRIMARY, [1.0], valid=True)),
    }
    if method in ('dart', 'omnicontrol'):
        tests['H3'] = ('gt', paired(table, method, 'S', T, 'jerk_0p1s', offsets))
    if method == 'dart':
        if any(a == 'W1' for (m, a, *_) in table if m == 'dart'):
            tests['H4a'] = ('gt', paired(table, method, 'W1', 'W0', 'boundary_jerk_0p1s', offsets))
        tests['H4b'] = ('lt', paired(table, method, 'W0X', 'W0', 'boundary_jerk_0p1s', offsets))
        tests['H4c'] = ('lt', paired(table, method, 'W0X', 'W0', 'foot_sliding_5cm_10cms', offsets, shift=.05))
    names = [k for k, (_, r) in tests.items() if r is not None and 'sign_flip_p' in r]
    adjusted = dict(zip(names, holm([tests[k][1]['sign_flip_p'] for k in names]))) if names else {}
    out = {}
    for key, (direction, r) in tests.items():
        if r is None or key not in adjusted:
            out[key] = {'status': 'not_testable', 'direction': direction, 'result': r}; continue
        side = r['ci95'][1] < 0 if direction == 'lt' else r['ci95'][0] > 0
        out[key] = {**r, 'direction': direction, 'holm_p': adjusted[key], 'supported': bool(adjusted[key] < .05 and side)}
    out['H1b']['margin'] = margin_jerk
    if 'H4c' in out:
        out['H4c']['margin'] = .05
    unvalid = paired(table, method, 'O', T, E_PRIMARY, [1.0], valid=False)
    out['H2_all_pairs_sensitivity'] = unvalid
    return out


def admission(table, receipts, method):
    rows = {k: v for k, v in receipts.items() if k[0] == method}
    report = {'runs': len(rows), 'finite': True}
    fk = [r['wrapper_fk'] for r in rows.values() if r.get('wrapper_fk')]
    if fk:
        report['wrapper_fk_pass'] = f'{sum(f["pass"] for f in fk)}/{len(fk)}'
        report['wrapper_fk_rms_max_m'] = max(f['rms_m'] for f in fk)
    if method == 'dart':
        g0 = [len(r['policy_calls']) for k, r in rows.items() if k[1] == 'G0']
        g4 = [len(r['policy_calls']) for k, r in rows.items() if k[1] == 'G4']
        c0 = [len(r['policy_calls']) for k, r in rows.items() if k[1] == 'C0']
        s = [r['optimizer'] for k, r in rows.items() if k[1] == 'S']
        report.update({'C0_policy_calls': sorted(set(c0)), 'G0_policy_calls': sorted(set(g0)), 'G4_policy_calls': sorted(set(g4)),
                       'S_steps': sorted({o['steps'] for o in s}), 'S_loss_decreased': f'{sum(o["loss_last"] < o["loss_first"] for o in s)}/{len(s)}',
                       'S_autocast': sorted({o['autocast'] for o in s})})
        for name in ('replay', 'w1_equivariance'):
            for stage in ('pilot', 'confirm'):
                f = xt.OUT/'dart'/f'{stage}_{name}.json'
                if f.exists():
                    report[f'{stage}_{name}'] = json.loads(f.read_text())
    if method == 'kimodo':
        report['postprocess'] = {arm: sorted({(r.get('postprocess_constraints_received'), r.get('postprocess_constraints_applied')) for k, r in rows.items() if k[1] == arm})
                                 for arm in ('O', 'TO')}
        report['raw_root_matches_base_max_m'] = max((r['raw_root_matches_base_m'] for r in rows.values() if 'raw_root_matches_base_m' in r), default=None)
        report['cfg_weights'] = {arm: sorted({tuple(r['cfg_weight']) for k, r in rows.items() if k[1] == arm}) for arm in ('C0', 'T', 'T+', 'O', 'TO')}
    if method == 'omnicontrol':
        report['active_calls'] = {arm: sorted({(r['calls'].get('realism_active', 0) > 0, r['calls'].get('spatial_active', 0) > 0) for k, r in rows.items() if k[1] == arm})
                                  for arm in ('C0', 'T', 'S', 'TS')}
    e = [r for k, r in table.items() if k[0] == method and 'E_valid' in r]
    if e:
        report['E_valid'] = f'{sum(r["E_valid"] for r in e)}/{len(e)}'
        report['E_ik_ok'] = f'{sum(r["ik_rms_m"] <= .03 and r["ik_max_m"] <= .10 for r in e)}/{len(e)}'
    return report


def precision(table, method, reg, hyp):
    out = {}
    for key, margin in (('H1b', hyp['H1b'].get('margin')), ('H4c', .05)):
        r = hyp.get(key)
        if not r or 'sd_cluster_means' not in r:
            continue
        half = 1.96*r['sd_cluster_means']/np.sqrt(12)/np.sqrt(2)
        out[key] = {'projected_half_width': float(half), 'half_margin': margin/2, 'top_up': bool(half > margin/2)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', default='confirm', choices=['pilot', 'confirm', 'top_up'])
    args = p.parse_args()
    reg = xt.prereg()
    table, receipts, missing = collect(args.stage)
    result = {'stage': args.stage, 'protocol': reg['protocol'], 'runs': len(table), 'missing': missing, 'methods': {}}
    for method in reg['methods']:
        arms = sorted({k[1] for k in table if k[0] == method})
        metrics = KINEMATIC+[E_PRIMARY, E_SECONDARY]
        cells = {f'{a}|d{d:g}': {m: levels(table, method, a, d, m) for m in metrics} for a in arms for d in reg['offsets_m']}
        hyp = hypotheses(table, method, reg)
        T = reg['methods'][method]['T_arm']
        secondary = {f'{a}-{b}': {m: paired(table, method, a, b, m, reg['offsets_m'] if m not in (E_PRIMARY, E_SECONDARY) else [1.0],
                                           valid=m in (E_PRIMARY, E_SECONDARY)) for m in metrics}
                     for a in arms for b in ('C0', T) if a != b}
        result['methods'][method] = {'cells': cells, 'hypotheses': hyp, 'secondary': secondary,
                                     'admission': admission(table, receipts, method),
                                     'precision': precision(table, method, reg, hyp) if args.stage == 'pilot' else None}
    json_write(xt.OUT/f'analysis_{args.stage}.json', result)
    for method, r in result['methods'].items():
        print(method, json.dumps(r['admission'], default=str)[:600])
        for k, h in r['hypotheses'].items():
            if isinstance(h, dict) and 'mean' in h:
                print(' ', k, round(h['mean'], 4), [round(x, 4) for x in h['ci95']], h.get('holm_p'), h.get('supported'))
            else:
                print(' ', k, h if h is None else h.get('status'))
        if r['precision']:
            print('  precision', r['precision'])
    print('missing', len(missing), missing[:10])


if __name__ == '__main__':
    main()
