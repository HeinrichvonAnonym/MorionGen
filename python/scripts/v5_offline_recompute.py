#!/usr/bin/env python
"""V5 offline recomputation for the 20261030 protocol (no generation, no GPU).

1. Archived 20261024 paused-clock effects re-reported with the cluster t
   interval and an exact sign-flip p over independent groups, next to the
   existing cluster bootstrap, for archived metrics and for 0.1 s step jerk.
2. Every archived motion: 0.1 s step jerk, 0.1 s step boundary jerk, bone CV.
3. PRIMAL: paired displacement caused by each post-processing factor.
4. Kimodo stratified benchmark: raw / official post-processed pairs.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT, data_root, load_config
from motiogen.motion import SOMA77_NAMES, load_motion
from motiogen.native_analysis import (bone_length_cv, cluster_t_interval, clustered_interval, common_step_boundary_jerk,
                                      common_step_jerk, sign_flip_p)
from motiogen.runtime import json_write

OUT = ROOT/'develop_logs/test_runs/20261030_next_round/v5'
ANALYSIS = ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
FACT = {'motionlcm': ['control_residual'], 'omnicontrol': ['spatial', 'realism'], 'chois': ['guidance'], 'dart': ['goal_update'],
        'ardy': ['correction', 'history_long'], 'closd': ['execution_context'],
        'primal': ['inertialization', 'ground_snap', 'reprojection']}
METRICS = ['hand_position_error', 'jerk_rms', 'boundary_jerk_rms', 'contact_distance', 'root_waypoint_mean_error_m',
           'trunk_inclination', 'right_upper_arm_elevation', 'history_consistency_common32_m', 'foot_floor_penetration_fraction',
           'foot_sliding_given_5cm_contact_10cms', 'tail_stationary_at_goal_fraction', 'tail_goal_occupancy_fraction',
           'jerk_common_0p1s', 'boundary_jerk_common_0p1s']
BONES = [('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'), ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')]


def motion_path(row):
    run = ROOT/row['run']
    return run/'factorial'/(row['variant']+'_canonical.npz') if row['method'] == 'primal' else run/'motion.npz'


def boundaries(row, motion):
    if row['method'] == 'primal':
        return np.arange(14, len(motion['joints']), 14)
    return np.asarray(motion['chunk_boundaries'], int) if 'chunk_boundaries' in motion else None


def recompute(rows):
    per_run, cache = [], {}
    for row in rows:
        motion = load_motion(motion_path(row))
        joints, fps = motion['joints'], float(motion['fps'])
        names = [str(n) for n in motion['joint_names']]
        if 'right_wrist_body' in names:
            # CHOIS: *_wrist is a hand anchor, *_wrist_body the anatomical wrist.
            names = ['right_wrist_anchor' if n == 'right_wrist' else n for n in names]
            names = ['right_wrist' if n == 'right_wrist_body' else n for n in names]
        row['values']['jerk_common_0p1s'] = common_step_jerk(joints, fps)
        bounds = boundaries(row, motion)
        if bounds is not None and len(bounds):
            value = common_step_boundary_jerk(joints, fps, bounds)
            if value is not None:
                row['values']['boundary_jerk_common_0p1s'] = value
        bones = bone_length_cv(joints, names, BONES)
        per_run.append({'method': row['method'], 'run': row['run'], 'variant': row.get('variant'), 'condition': row['condition'],
                        'fps': fps, 'jerk_native': row['values'].get('jerk_rms'), 'jerk_common_0p1s': row['values']['jerk_common_0p1s'],
                        'boundary_jerk_native': row['values'].get('boundary_jerk_rms'),
                        'boundary_jerk_common_0p1s': row['values'].get('boundary_jerk_common_0p1s'),
                        'bone_cv': {k: v['cv'] for k, v in bones.items()}})
        if row['method'] == 'primal':
            cache[(row['task'], row['seed'], row['variant'])] = joints
    return per_run, cache


def paired_effects(rows):
    effects = []
    for method, factors in FACT.items():
        mine = [r for r in rows if r['method'] == method]
        groups = collections.defaultdict(list)
        for r in mine:
            groups[r['task'], r['seed']].append(r)
        for factor in factors:
            for metric in METRICS:
                deltas, parents = [], []
                for (task, seed), rs in sorted(groups.items()):
                    on = [r['values'].get(metric) for r in rs if r['condition'][factor] == 1]
                    off = [r['values'].get(metric) for r in rs if r['condition'][factor] == 0]
                    if not on or len(on) != len(off) or any(v is None or not np.isfinite(v) for v in on+off):
                        continue
                    deltas.append(float(np.mean(on)-np.mean(off)))
                    parents.append(str(rs[0].get('resampling_group', task)))
                if len(set(parents)) < 2:
                    continue
                off_values = [r['values'][metric] for r in mine if r['condition'][factor] == 0 and metric in r['values']]
                off_mean = float(np.mean(off_values))
                t = cluster_t_interval(deltas, parents)
                flip = sign_flip_p(t['cluster_means'])
                boot = clustered_interval(deltas, parents)
                effects.append({'method': method, 'factor': factor, 'metric': metric, 'pairs': len(deltas), 'groups': t['clusters'],
                                'mean_delta': t['mean'], 't_ci95': t['ci'], 'sign_flip_p': flip['p'], 'sign_flip_method': flip['method'],
                                'bootstrap_ci95': boot['ci95'], 'off_mean': off_mean,
                                'relative_change': t['mean']/off_mean if off_mean else None,
                                't_excludes_zero': bool(t['ci'] and (t['ci'][0] > 0 or t['ci'][1] < 0)),
                                'bootstrap_excludes_zero': bool(boot['ci95'] and (boot['ci95'][0] > 0 or boot['ci95'][1] < 0))})
    return effects


def primal_displacement(cache):
    """Variant names are ground{0,1}_reproject{0,1}_inertialize{0,1}."""
    result = {}
    for factor, position in [('ground_snap', 0), ('reprojection', 1), ('inertialization', 2)]:
        values = []
        for (task, seed, variant), joints in cache.items():
            parts = variant.split('_')
            if parts[position].endswith('1'):
                continue
            parts[position] = parts[position][:-1]+'1'
            partner = cache.get((task, seed, '_'.join(parts)))
            if partner is not None and partner.shape == joints.shape:
                d = np.linalg.norm(partner-joints, axis=-1)
                values.append([d.mean(), d.max(), np.linalg.norm(partner[:, 0]-joints[:, 0], axis=-1).mean()])
        v = np.asarray(values)
        result[factor] = ({'pairs': len(v), 'mean_joint_displacement_m': float(v[:, 0].mean()),
                           'p95_max_joint_displacement_m': float(np.percentile(v[:, 1], 95)),
                           'mean_root_displacement_m': float(v[:, 2].mean())} if len(v) else {'pairs': 0})
    return result


def native_jerk(x, fps):
    return float(np.sqrt(np.mean((np.diff(x, n=3, axis=0)*fps**3)**2)))


def kimodo_pairs(root):
    post_root = root/'generated_postprocess'
    rows = []
    for post in sorted(post_root.rglob('motion.npz')):
        raw = root/'generated_raw'/post.relative_to(post_root)
        if not raw.exists():
            continue
        a, b = np.load(raw), np.load(post)
        ja, jb = a['posed_joints'].astype(float), b['posed_joints'].astype(float)
        fps = len(ja)/float(json.loads((post.parent/'meta.json').read_text())['duration'])
        ma = json.loads((raw.parent/'metrics.json').read_text())['per_motion_mean_gen']
        mb = json.loads((post.parent/'metrics.json').read_text())['per_motion_mean_gen']
        rel = post.relative_to(post_root).parts
        disp = np.linalg.norm(jb-ja, axis=-1)
        rows.append({'case': '/'.join(rel[:-1]), 'group': '/'.join(rel[:-2]), 'fps': fps,
                     'mean_joint_displacement_m': float(disp.mean()), 'max_joint_displacement_m': float(disp.max()),
                     'mean_root_displacement_m': float(np.linalg.norm(b['root_positions']-a['root_positions'], axis=-1).mean()),
                     'jerk_native': [native_jerk(x, fps) for x in (ja, jb)],
                     'jerk_common_0p1s': [common_step_jerk(x, fps) for x in (ja, jb)],
                     'bone_cv': [float(np.mean([v['cv'] for v in bone_length_cv(x, SOMA77_NAMES, BONES).values()])) for x in (ja, jb)],
                     'metrics_raw': ma, 'metrics_post': mb})
    summary = {}
    keys = ['jerk_native', 'jerk_common_0p1s', 'bone_cv'] + sorted({k for r in rows for k in r['metrics_raw'] if k in r['metrics_post']})
    for key in keys:
        deltas, groups, base = [], [], []
        for r in rows:
            pair = r[key] if key in r else (r['metrics_raw'].get(key), r['metrics_post'].get(key))
            if pair[0] is None or pair[1] is None or not np.isfinite(pair).all():
                continue
            deltas.append(pair[1]-pair[0]); groups.append(r['group']); base.append(pair[0])
        if len(set(groups)) < 2:
            continue
        t = cluster_t_interval(deltas, groups)
        summary[key] = {'pairs': len(deltas), 'groups': t['clusters'], 'raw_mean': float(np.mean(base)),
                        'mean_delta_post_minus_raw': t['mean'], 't_ci95': t['ci'],
                        'sign_flip_p': sign_flip_p(t['cluster_means'])['p']}
    disp = np.array([[r['mean_joint_displacement_m'], r['max_joint_displacement_m'], r['mean_root_displacement_m']] for r in rows])
    return {'pairs': len(rows), 'groups': len({r['group'] for r in rows}), 'effects': summary,
            'correction_magnitude': {'mean_joint_displacement_m': float(disp[:, 0].mean()),
                                     'p95_max_joint_displacement_m': float(np.percentile(disp[:, 1], 95)),
                                     'mean_root_displacement_m': float(disp[:, 2].mean())},
            'scope': 'Kimodo official post-processing on the 20261008 stratified benchmark; groups are constraint-category folders; '
                     'native Y-up positions (jerk and bone lengths are rotation invariant); sign-flip p is Monte Carlo above 24 groups'}, rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads(ANALYSIS.read_text())['rows'] if r['clock'] == 'paused']
    per_run, cache = recompute(rows)
    effects = paired_effects(rows)
    kimodo, kimodo_rows = kimodo_pairs(data_root(load_config())/'benchmarks/kimodo-stratified-20261008')
    ratio = collections.defaultdict(list)
    bone = collections.defaultdict(list)
    for r in per_run:
        if r['jerk_native'] and r['jerk_common_0p1s']:
            ratio[r['method']].append(r['jerk_native']/r['jerk_common_0p1s'])
        if r['bone_cv']:
            bone[r['method']+' | '+','.join(f'{k}={v}' for k, v in sorted(r['condition'].items()))].append(np.mean(list(r['bone_cv'].values())))
    json_write(OUT/'v5.json', {
        'source': str(ANALYSIS), 'paused_rows': len(rows), 'effects': effects,
        'native_over_common_jerk_median': {m: float(np.median(v)) for m, v in ratio.items()},
        'bone_cv_mean_by_condition': {k: {'runs': len(v), 'mean_cv': float(np.mean(v))} for k, v in sorted(bone.items())},
        'primal_displacement': primal_displacement(cache), 'kimodo': kimodo,
        'scope': 'Offline recomputation of archived motions only. Clusters: task, or PRIMAL resampling parent. '
                 'Primary interval: cluster t (G-1 df); bootstrap kept for comparison.'})
    json_write(OUT/'per_run.json', per_run)
    json_write(OUT/'kimodo_pairs.json', kimodo_rows)
    flips = [e for e in effects if e['t_excludes_zero'] != e['bootstrap_excludes_zero']]
    print(json.dumps({'effects': len(effects), 't_vs_bootstrap_disagreements': len(flips), 'kimodo_pairs': kimodo['pairs']}))


if __name__ == '__main__':
    main()
