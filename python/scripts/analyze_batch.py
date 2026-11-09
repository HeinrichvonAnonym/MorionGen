#!/usr/bin/env python
"""Frozen analysis plan of the large batch (config/analysis_plan_20261108.json and its revisions_after_freeze).

  analyze_batch.py --batch DIR [--out PATH] [--structure-only] [--final-comparison]

Inputs: DIR/evalkit/manifest_development.json (python/scripts/build_evalkit_manifest.py; development rows and the reused
Kimodo pairs, split deck_recheck) and its evalkit outputs under DIR/evalkit/development/ (layout()): method floor
(method/), self floor (self_manifest.json, self/), lumbar subsample (lumbar_manifest.json, upper added to self/), rotation
neck reading (on self/). Held-out rows are refused unless --final-comparison (split_use.enforcement); with it the
held-out rows are analysed apart and read against the development estimates (split_use.fallback_reading), and the
opening is recorded in develop_logs/notes/held_out_openings.md with time and commit.

Source layer: per-run values by analyze_native_matrix.read_run (the native matrix analysis, unchanged), 0.1 s step jerk
and boundary jerk by v5_offline_recompute.recompute (ARDY chunk boundaries, PRIMAL 14-frame commits) plus the DART
primitive starts (trace steps) and the CLoSD planner frames (trace planner, as analyze_native_matrix), bone-length CV of
the v5 bones; Kimodo pairs by v5_offline_recompute.kimodo_pairs. evalkit layer: evalkit_batch.rows_summary of both
floors, the rotation neck reading, the threshold band (registered penetration depth -+0.5 cm, rescore only) and the
stage-1 foot offsets.

Statistics (statistics.*): paired on - off within (task, text, seed), mean over seeds, equal task weight; primary 95%
percentile bootstrap over task clusters (PRIMAL: the registered resampling groups; Kimodo: source motion), 10000
resamples, seed 20261108; text x task crossed interval for MotionLCM / OmniControl; Holm per question and primary metric
over the methods' primary contrasts in the primary setting, p from the exact sign-flip test over cluster means, with a
cluster t interval; settings robustness, measurement consistency and gate pass rates judged by code.
"""
import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
sys.path.insert(0, str(ROOT/'utils/motiogen'))
sys.path.insert(0, str(ROOT/'python/scripts'))
from motiogen.native_analysis import (clustered_interval, cluster_t_interval, common_step_boundary_jerk,  # noqa: E402
                                      holm, sign_flip_p)

PLAN = ROOT/'config/analysis_plan_20261108.json'
VALIDATION = ROOT/'develop_logs/test_runs/20261109_analysis_validation/synthetic.json'
VALIDATION_EXTRA = VALIDATION.with_name('synthetic_skew_null_extra.json')
CROSSED_ROLE = 'pending validation (b)'
ENVIRONMENT = {'python': '3.13.9', 'numpy': '2.3.5', 'scipy': '1.16.3'}
TASKS = json.loads((ROOT/'config/inference_tasks_20261109.json').read_text())
REPS, SEED = 10000, 20261108
ALPHA = .05
CONSISTENCY_SHARE, CONSISTENCY_OFFSET_M = .05, .005
BAND_M = .005
FACTORS = {'motionlcm': ['control_residual'], 'omnicontrol': ['realism', 'spatial'], 'chois': ['guidance'],
           'dart': ['goal_update'], 'ardy': ['history_long', 'correction'], 'closd': ['execution_context'],
           'primal': ['ground_snap', 'reprojection', 'inertialization'], 'kimodo': ['correction']}
TEXT_METHODS = ('motionlcm', 'omnicontrol')
TEXTS = ('consistent', 'neutral')
TARGET = {'motionlcm': 'hand_position_error', 'omnicontrol': 'hand_position_error',
          'chois': 'native_object_mean_waypoints_xy_pos_err', 'ardy': 'root_waypoint_mean_error_m',
          'dart': 'new_goal_final_error_m', 'closd': 'new_goal_final_error_m'}
KIMODO_TARGET = ('constraint_end_effector', 'constraint_fullbody_keyframe', 'constraint_root2d_err')
NECK_ROTATION_METHODS = ('ardy', 'chois', 'kimodo', 'primal')
SOURCE_KEYS = ('jerk_common_0p1s', 'boundary_jerk_common_0p1s', 'bone_cv_mean')


def simple(factor, fixed):
    return {'kind': 'simple', 'factor': factor, 'fixed': fixed}


def marginal(factor):
    return {'kind': 'factorial', 'factors': [factor]}


def interaction(*factors):
    return {'kind': 'factorial', 'factors': list(factors)}


def conflict(fixed):
    """(neutral - consistent) within the given condition (text_factor.estimands.conflict_cost)."""
    return {'kind': 'text', 'fixed': fixed}


# questions.*.primary_contrasts; MotionLCM / OmniControl in each text condition (consistent: guidance_cost, neutral:
# deck_estimand), both in the Holm family; conflict cost and the listed secondaries without correction
PRIMARY = {'Q1': [('motionlcm', simple('control_residual', {})), ('omnicontrol', simple('spatial', {'realism': 1})),
                  ('chois', simple('guidance', {})), ('kimodo', simple('correction', {})),
                  ('ardy', simple('correction', {'history_long': 0})), ('dart', simple('goal_update', {}))],
           'Q2': [('primal', marginal('inertialization')), ('ardy', simple('history_long', {'correction': 1})),
                  ('closd', simple('execution_context', {}))]}
SECONDARY = {'Q1': [('omnicontrol', interaction('realism', 'spatial')), ('ardy', interaction('history_long', 'correction')),
                    ('motionlcm', conflict({'control_residual': 1})), ('omnicontrol', conflict({'realism': 1, 'spatial': 1}))],
             'Q2': [('primal', marginal('ground_snap')), ('primal', marginal('reprojection'))]}


def jerk_key(method):
    return 'boundary_jerk_common_0p1s' if method in ('ardy', 'dart') else 'jerk_common_0p1s'


def primary_metrics(question, method):
    """(name, key) of questions.<Q>.primary_metrics; None where the method has no such quantity."""
    target = f'{TARGET[method]}@source' if method in TARGET else None
    if question == 'Q1':
        return [('target_error', target), ('jerk_0p1s', f'{jerk_key(method)}@source'), ('residual_force_rms_bw', 'residual_rms_bw@method')]
    if question == 'Q2':
        return [('boundary_jerk', 'boundary_jerk_common_0p1s@source'), ('target_error', target),
                ('residual_force_rms_bw', 'residual_rms_bw@method')]
    return [('rula_certain_risk', 'rula_certain_risk@self'), ('reba_certain_risk', 'reba_certain_risk@self'),
            ('leg_muscle_effort', 'muscle_primary_muscle_effort_sum_a2_mean@self')]


SECONDARY_METRICS = [('residual_positive_share', 'residual_positive_share@method'), ('penetration_share', 'penetration_share@method'),
                     ('L5_S1_compression_p95_bw', 'upper_lumbar_L5_S1_compression_p95_bw@self'),
                     ('net_moment_lumbar_nm_kg_mean', 'net_moment_lumbar_nm_kg_mean@self'),
                     ('net_moment_shoulder_nm_kg_mean', 'net_moment_shoulder_nm_kg_mean@self')]


def alternates(key, method):
    """settings_robustness: the other applicable settings of a value key."""
    name, setting = key.split('@')
    if setting == 'source':
        return []
    out = [f'{name}@{"self" if setting == "method" else "method"}']
    if name == 'rula_certain_risk' and method in NECK_ROTATION_METHODS:
        out.append('rula_certain_risk@self_neckrot')
    if name in ('residual_positive_share', 'penetration_share') and setting == 'method':
        out += [f'{name}@method_band-5', f'{name}@method_band+5']
    return out


def evaluator(key):
    """Gate of the evaluator that produced a value key (None: source layer, no evalkit gate)."""
    name, setting = key.split('@')
    if setting == 'source':
        return None
    floor = 'method' if setting.startswith('method') else 'self'
    if setting == 'self_neckrot':
        return 'neck_rotation@self'
    if name.startswith(('rula_', 'reba_')):
        return f'ergo@{floor}'
    if name.startswith(('muscle_', 'net_moment_')):
        return f'muscle@{floor}'
    if name.startswith('upper_'):
        return f'upper_lumbar@{floor}'
    return f'residual@{floor}'


def constrained(method, spec, name):
    """metric_classes.constrained_metrics.per_switch: reported, never a conclusion (kept out of the Holm family)."""
    factor = spec.get('factor') or (spec.get('factors') or [None])[0]
    if name == 'target_error' and (method, factor) in (('kimodo', 'correction'), ('ardy', 'correction')):
        return 'constraint error of the output-layer correction (constrained metric)'
    if name == 'penetration_share' and (method, factor) in (('chois', 'guidance'), ('primal', 'ground_snap')):
        return 'foot height / penetration is optimized by this switch (constrained metric)'
    return None


def descriptive_only(method, name):
    if method == 'chois' and name in ('leg_muscle_effort', 'L5_S1_compression_p95_bw'):
        return 'CHOIS loads: hand-held object load not modelled (descriptive_only)'
    if name.startswith('net_moment_'):
        return 'net joint moments are not loads (descriptive_only)'
    return None


def spec_name(method, spec):
    if spec['kind'] == 'simple':
        fixed = ','.join(f'{k}={v}' for k, v in sorted(spec['fixed'].items()))
        return f"{spec['factor']} 1-0" + (f' at {fixed}' if fixed else '')
    if spec['kind'] == 'text':
        return 'conflict (neutral - consistent) at ' + ','.join(f'{k}={v}' for k, v in sorted(spec['fixed'].items()))
    return ' x '.join(spec['factors']) + (' 1-0 (averaged over the other switches)' if len(spec['factors']) == 1 else ' interaction')


# ---------------------------------------------------------------- engine (pure functions on units)

def code_condition(method, condition):
    c = dict(condition)
    if method == 'ardy' and 'history_frames' in c:
        c['history_long'] = int(c.pop('history_frames') == 160)
    return {k: int(c[k]) for k in FACTORS[method]}


def cell(method, condition):
    return tuple(condition[k] for k in FACTORS[method])


def group_deltas(units, method, spec, key, gate=None, value=None):
    """Paired deltas of one value key: [{task, parent, text, seed, paraphrase, zone, delta, units}] over the groups
    (task, text, seed) whose required cells hold finite values (and pass gate when given)."""
    value = value or (lambda u: u['values'].get(key))
    factors = FACTORS[method]
    groups = defaultdict(dict)
    for u in units:
        if u['method'] != method:
            continue
        g = (u['task'], u['seed']) if spec['kind'] == 'text' else (u['task'], u['text'], u['seed'])
        k = (u['text'],) + cell(method, u['condition']) if spec['kind'] == 'text' else cell(method, u['condition'])
        groups[g][k] = u
    out = []
    for g, cells in sorted(groups.items(), key=lambda x: json.dumps(x[0], default=str)):
        if spec['kind'] == 'simple':
            base = {**spec['fixed']}
            need = [(cell(method, {**base, spec['factor']: 1}), 1.), (cell(method, {**base, spec['factor']: 0}), -1.)]
        elif spec['kind'] == 'text':
            c = cell(method, spec['fixed'])
            need = [(('neutral',) + c, 1.), (('consistent',) + c, -1.)]
        else:
            sel = [factors.index(f) for f in spec['factors']]
            scale = 2 ** (len(factors) - len(sel))
            need = [(c, float(np.prod([2*c[i]-1 for i in sel]))/scale) for c in itertools.product((0, 1), repeat=len(factors))]
        us = [cells.get(c) for c, _ in need]
        if any(u is None for u in us):
            continue
        if any(not u.get('block_ok', True) for u in us):
            continue
        vs = [value(u) for u in us]
        if any(v is None or not np.isfinite(v) for v in vs):
            continue
        if gate and any(u['gates'].get(gate) != 1 for u in us):
            continue
        u0 = us[0]
        delta = math.fsum(w*v for (_, w), v in zip(need, vs)) if spec['kind'] == 'factorial' else float(vs[0]-vs[1])
        out.append({'task': u0['task'], 'parent': u0['parent'], 'text': u0['text'] if spec['kind'] != 'text' else None,
                    'seed': u0['seed'], 'paraphrase': u0['paraphrase'], 'zone': u0['zone'], 'delta': float(delta),
                    'units': [u['id'] for u in us], 'arms': arms(spec, need, us)})
    return out


def arms(spec, need, us):
    """on / off units of a pair (measurement consistency, gate rates); interactions have no arms."""
    if spec['kind'] == 'factorial' and len(spec['factors']) > 1:
        return None
    return {'on': [u['id'] for (_, w), u in zip(need, us) if w > 0], 'off': [u['id'] for (_, w), u in zip(need, us) if w < 0]}


def crossed_interval(d, reps=REPS, seed=SEED):
    """statistics.interval.cross_paraphrase: tasks and paraphrases resampled independently; every (task, paraphrase)
    cell (mean of its deltas) weighted by the product of its task and paraphrase draw counts."""
    tasks = sorted({x['task'] for x in d}, key=str); paras = sorted({x['paraphrase'] for x in d}, key=str)
    if len(tasks) < 2 or len(paras) < 2:
        return {'ci95': None, 'reason': f'not computable: {len(tasks)} task(s), {len(paras)} paraphrase(s); needs at least 2 of each'}
    c = np.zeros((len(tasks), len(paras))); m = np.zeros_like(c); acc = defaultdict(list)
    for x in d:
        acc[tasks.index(x['task']), paras.index(x['paraphrase'])].append(x['delta'])
    for (i, j), v in acc.items():
        c[i, j] = np.mean(v); m[i, j] = 1.
    rng = np.random.default_rng(seed)
    wt = np.zeros((reps, len(tasks))); wp = np.zeros((reps, len(paras)))
    np.add.at(wt, (np.repeat(np.arange(reps), len(tasks)), rng.integers(0, len(tasks), (reps, len(tasks))).ravel()), 1)
    np.add.at(wp, (np.repeat(np.arange(reps), len(paras)), rng.integers(0, len(paras), (reps, len(paras))).ravel()), 1)
    num = np.einsum('rt,tp,rp->r', wt, c*m, wp); den = np.einsum('rt,tp,rp->r', wt, m, wp)
    ok = den > 0
    return {'mean_cells': float((c*m).sum()/m.sum()), 'cells': int(m.sum()), 'tasks': len(tasks), 'paraphrases': len(paras),
            'ci95': np.quantile(num[ok]/den[ok], [.025, .975]).tolist(), 'draws_without_cells': int((~ok).sum())}


def two_level_interval(d, reps=REPS, seed=SEED):
    """Text -> task two-level bootstrap of the deck re-check (texts with replacement, then tasks within each drawn text;
    equal task weight). Reproduction of develop_logs/test_runs/20261108_deck_recheck only; not a registered batch interval."""
    task_mean = defaultdict(list)
    for x in d:
        task_mean[(x['text'], x['task'])].append(x['delta'])
    by_text = defaultdict(list)
    for (text, _), vs in sorted(task_mean.items()):
        by_text[text].append(np.mean(vs))
    texts = sorted(by_text); arr = [np.array(by_text[t]) for t in texts]
    rng = np.random.default_rng(seed); draws = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(texts), len(texts))
        draws[i] = np.mean(np.concatenate([arr[k][rng.integers(0, len(arr[k]), len(arr[k]))] for k in pick]))
    return {'mean': float(np.mean(np.concatenate(arr))), 'ci95': [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            'texts': len(texts), 'per_text_mean': {t: float(np.mean(a)) for t, a in zip(texts, arr)},
            'same_sign_across_texts': bool(len({np.sign(np.mean(a)) for a in arr}) == 1)}


def excludes0(ci):
    return bool(ci is not None and (ci[0] > 0 or ci[1] < 0))


def cluster_t(d):
    """Primary interval: cluster t on G - 1 df for the equal-task-weight mean; clusters are the registered parents (task;
    PRIMAL resampling group; Kimodo source motion). With one task per cluster this is native_analysis.cluster_t_interval
    over task means; otherwise the cluster-robust (CR1) variance of the task-weighted mean, which equals it when every
    cluster has the same number of tasks. flip_units: per-cluster sums of task means, whose sum is T x the mean."""
    tm, par = defaultdict(list), {}
    for x in d:
        tm[x['task']].append(x['delta']); par[x['task']] = x['parent']
    tasks = sorted(tm, key=str)
    means = np.array([np.mean(tm[t]) for t in tasks]); parents = [par[t] for t in tasks]
    g = len(set(parents))
    if g < 2:
        return {'clusters': g, 'mean': float(means.mean()), 'ci95': None, 'flip_units': None}
    if g == len(tasks):
        r = cluster_t_interval(means, parents)
        return {'clusters': g, 'mean': r['mean'], 'ci95': r['ci'], 'se': r['se'], 'df': r['df'], 'flip_units': r['cluster_means']}
    from scipy import stats
    mean = float(means.mean())
    dev, tot = defaultdict(float), defaultdict(float)
    for v, p in zip(means, parents):
        dev[p] += v-mean; tot[p] += v
    se = math.sqrt(g/(g-1)*sum(x*x for x in dev.values()))/len(tasks)
    q = float(stats.t.ppf(.975, g-1))
    return {'clusters': g, 'mean': mean, 'ci95': [mean-q*se, mean+q*se], 'se': se, 'df': g-1,
            'flip_units': [tot[p] for p in sorted(tot, key=str)]}


def summarize(d, reps=REPS, seed=SEED, crossed=False, tests=False):
    """Primary interval (cluster t), the task-cluster bootstrap as sensitivity, the crossed interval and the sign flip."""
    clusters = {x['parent'] for x in d}
    out = {'pairs': len(d), 'tasks': len({x['task'] for x in d}), 'clusters': len(clusters)}
    if not d:
        return {**out, 'mean': None, 'ci95': None, 'reason': 'not computable: no complete pair'}
    t = cluster_t(d)
    out.update(mean=t['mean'], ci95=t['ci95'], excludes_0=excludes0(t['ci95']), interval='cluster t, G - 1 df')
    if t['ci95'] is None:
        out['reason'] = f'not computable: {len(clusters)} cluster(s); the cluster t interval needs at least 2'
    p = clustered_interval([x['delta'] for x in d], [x['task'] for x in d], reps, seed, [x['parent'] for x in d])
    out['bootstrap_sensitivity'] = {'mean': p['mean_equal_group_weight'], 'ci95': p['ci95'], 'excludes_0': excludes0(p['ci95'])}
    if crossed:
        out['cross_paraphrase'] = {**crossed_interval(d, reps, seed), 'role': CROSSED_ROLE}
    if tests:
        if len(clusters) < 2:
            out.update(sign_flip_p=None, tests_reason=f'not computable: {len(clusters)} cluster(s)')
        else:
            f = sign_flip_p(t['flip_units'])
            out.update(sign_flip_p=f['p'], sign_flip_method=f['method'])
    return out


def robustness(primary, alts, holm_pass, in_family):
    """settings_robustness.rule: a conclusion needs Holm in the primary setting and, in every other applicable setting,
    the same sign and a raw 95% interval excluding 0; otherwise 'setting-dependent'."""
    if primary.get('ci95') is None:
        return primary.get('reason', 'not computable')
    if not in_family:
        return 'not in the Holm family (reported, no conclusion)'
    if holm_pass is None:
        return primary.get('holm_reason', 'not computable: Holm p value missing')
    if not holm_pass:
        return 'no conclusion (Holm not passed in the primary setting)'
    sign = np.sign(primary['mean'])
    bad = [k for k, a in alts.items() if a.get('mean') is None or np.sign(a['mean']) != sign or not a.get('excludes_0')]
    return 'setting-dependent' if bad else 'conclusion'


def consistency(on, off):
    """statistics.measurement_consistency: per-arm unmeasurable share, foot-offset median and IQR, share of rows inside
    the threshold band; any share differing by more than 5 percentage points or the offset median by more than 0.5 cm
    labels the effect 'measurement state differs'. on / off: lists of {unmeasurable_share, foot_offset_m, in_band}."""
    def arm(rows):
        un = [r['unmeasurable_share'] for r in rows if r.get('unmeasurable_share') is not None]
        off_m = [r['foot_offset_m'] for r in rows if r.get('foot_offset_m') is not None]
        band = [r['in_band'] for r in rows if r.get('in_band') is not None]
        return {'rows': len(rows), 'unmeasurable_share': float(np.mean(un)) if un else None,
                'foot_offset_median_m': float(np.median(off_m)) if off_m else None,
                'foot_offset_iqr_m': float(np.subtract(*np.percentile(off_m, [75, 25]))) if off_m else None,
                'in_band_share': float(np.mean(band)) if band else None}
    a, b = arm(on), arm(off)
    flags, missing = [], []
    for k, limit in (('unmeasurable_share', CONSISTENCY_SHARE), ('in_band_share', CONSISTENCY_SHARE),
                     ('foot_offset_median_m', CONSISTENCY_OFFSET_M)):
        if a[k] is None or b[k] is None:
            missing.append(k)
        elif abs(a[k]-b[k]) > limit:
            flags.append(k)
    verdict = 'measurement state differs' if flags else ('not checkable' if missing else 'same measurement state')
    return {'on': a, 'off': b, 'differs_in': flags, 'not_checkable': missing, 'verdict': verdict}


def dedup(units):
    """statistics.dedup: units with byte-identical inputs in the same cell (method, task, text, seed, condition) enter
    once; identical inputs across cells (e.g. control-off runs shared by tasks) remain, each in its own pair."""
    seen, out, dropped = set(), [], []
    for u in units:
        k = (u['method'], u['task'], u['text'], u['seed'], json.dumps(u['condition'], sort_keys=True), u['input_sha256'])
        if k in seen:
            dropped.append(u['id']); continue
        seen.add(k); out.append(u)
    return out, dropped


def check_split(units, final):
    bad = sorted({u['split'] for u in units} - {'development', 'deck_recheck'})
    if bad and not final:
        raise SystemExit(f'refused: rows of split {bad} present; held-out rows are analysed only with --final-comparison')


# ---------------------------------------------------------------- data

def layout(batch, split):
    base = Path(batch)/'evalkit'/split
    return {'manifest': Path(batch)/'evalkit'/f'manifest_{split}.json', 'method': base/'method', 'self_manifest': base/'self_manifest.json',
            'lumbar_manifest': base/'lumbar_manifest.json', 'self': base/'self'}


def weight_identity(anm, row):
    """analyze_native_matrix.main weight identity of one row (binding receipt, else the retrospective audit)."""
    binding = ROOT/row['run']/'weight_binding.json'
    if binding.exists():
        b = json.loads(binding.read_text())
        if 'after_run_unchanged' in b:
            return b['identity'] if b.get('after_run_unchanged') else None
    audit = anm.weight_audit.get(row['run'], {})
    return audit.get('identity') if audit.get('passed') else None


def source_rows(manifest_rows):
    """{(run relative to ROOT, PRIMAL variant or None): native-matrix row with the 0.1 s jerk keys and bone CV}."""
    import analyze_native_matrix as anm
    import v5_offline_recompute as v5
    from motiogen.motion import load_motion
    from motiogen.native_receipt_checks import intervention_checks
    anm.rows.clear(); anm.seen.clear(); anm.rejected.clear()
    runs = {}
    for r in manifest_rows:
        if r['method'] != 'kimodo':
            runs.setdefault(str(Path(r['run']).resolve().relative_to(ROOT)), int(r['task_id']))
    failed = {}
    for run, task in sorted(runs.items()):
        try:
            anm.read_run(ROOT/run, task, 'paused')
        except Exception as e:
            failed[run] = f'{type(e).__name__}: {e}'
    rows = list(anm.rows)
    per_run, _ = v5.recompute(rows)
    for row, pr in zip(rows, per_run):
        row['values']['bone_cv_mean'] = float(np.mean(list(pr['bone_cv'].values()))) if pr['bone_cv'] else None
        if row['method'] in ('dart', 'closd'):
            trace = json.loads((ROOT/row['run']/'trace.json').read_text())
            bounds = ([s['frame'] for s in trace['steps'] if s['frame'] > 0] if row['method'] == 'dart'
                      else sorted({p['frame'] for p in trace['planner'] if p['frame'] > 0}))
            motion = load_motion(ROOT/row['run']/'motion.npz')
            v = common_step_boundary_jerk(motion['joints'], float(motion['fps']), bounds) if bounds else None
            if v is not None:
                row['values']['boundary_jerk_common_0p1s'] = v
        row['intervention_checks'] = intervention_checks(row, ROOT)
        row['weight_identity'] = weight_identity(anm, row)
    return {(r['run'], r.get('variant')): r for r in rows}, failed, list(anm.rejected)


def kimodo_values(manifest_rows):
    """{case: v5 kimodo pair row} for the Kimodo cases of the manifest."""
    import v5_offline_recompute as v5
    from motiogen.config import data_root, load_config
    cases = {r['case'] for r in manifest_rows if r['method'] == 'kimodo'}
    if not cases:
        return {}
    _, rows = v5.kimodo_pairs(data_root(load_config())/'benchmarks/kimodo-stratified-20261008')
    return {r['case']: r for r in rows if r['case'] in cases}


def evalkit_values(lay, manifest_rows):
    """Per manifest row id: evalkit values by setting, evaluator gates and measurement-state fields."""
    from evalkit_batch import DEPTH_BAND_M, rows_summary
    from evalkit.spec import SPEC
    depth = SPEC['residual']['penetration_depth_m']
    msum = rows_summary(lay['method'], manifest_rows) if lay['method'].exists() else {}
    srows = []
    for name in ('lumbar_manifest', 'self_manifest'):
        if lay[name].exists():
            srows = json.loads(lay[name].read_text())['rows']; break
    ssum = rows_summary(lay['self'], srows) if srows and lay['self'].exists() else {}
    sby = {r['base_id']: (r, ssum.get(r['id'])) for r in srows}
    out = {}
    for r in manifest_rows:
        m, (srow, s) = msum.get(r['id']), sby.get(r['id'], (None, None))
        values, gates, meas = {}, {}, {}
        for floor, v in (('method', m), ('self', s)):
            evaluated = (Path(lay[floor])/(r['id'] if floor == 'method' else (srow or {}).get('id', ''))/'evalkit.json').exists()
            gates[f'residual@{floor}'] = int(v is not None and v.get('residual_rms_bw') is not None)
            gates[f'ergo@{floor}'] = int(v is not None and v.get('rula_certain_risk') is not None)
            gates[f'muscle@{floor}'] = int(v is not None and v.get('muscle_primary_muscle_effort_sum_a2_mean') is not None)
            meas[f'{floor}_evaluated'] = evaluated
            if v is None:
                continue
            for k, x in v.items():
                if isinstance(x, (int, float)) and not isinstance(x, bool) and k not in ('seconds',):
                    values[f'{k}@{floor}'] = float(x)
            meas[f'unmeasurable_share@{floor}'] = v['unmeasurable_share']
            le = v.get('lowest_elevation_median_m')
            meas[f'in_band@{floor}'] = None if le is None else int(abs(le+depth) <= BAND_M)
            meas[f'row_status@{floor}'] = v['row_status']
        if m is not None:
            for b in DEPTH_BAND_M:
                values[f'residual_positive_share@method_band{b*1000:+.0f}'] = m[f'residual_positive_share_depth_{b*1000:+.0f}mm']
            f = np.load(lay['method']/r['id']/'frames.npz', allow_pickle=True); sel = f['scored']
            low = f['lowest_elevation_m'][sel]
            if not np.isclose(float(np.mean(low < -depth)), m['penetration_share'], rtol=0, atol=1e-12):
                raise ValueError(f"{r['id']}: rescored penetration share differs from evalkit at the registered depth")
            for b in DEPTH_BAND_M:
                values[f'penetration_share@method_band{b*1000:+.0f}'] = float(np.mean(low < -(depth+b)))
            fo = lay['method']/r['id']/'foot_offset.json'
            meas['foot_offset_m'] = json.loads(fo.read_text())['foot_offset_m']['mean'] if fo.exists() else None
        if s is not None:
            if s.get('upper_lumbar_status') not in (None, 'not_sampled', 'deferred'):
                gates['upper_lumbar@self'] = int(s['upper_lumbar_status'] == 'measured')
            if r['method'] in NECK_ROTATION_METHODS:
                gates['neck_rotation@self'] = int(s.get('rula_neck_rotation_status') == 'ok')
                if s.get('rula_neck_rotation_certain_risk') is not None:
                    values['rula_certain_risk@self_neckrot'] = s['rula_neck_rotation_certain_risk']
        out[r['id']] = {'values': values, 'gates': gates, 'measure': meas}
    return out


def parent_of(method, task):
    if method == 'primal':
        t = next(t for t in TASKS['methods']['primal']['tasks'] if t['id'] == task)
        return str(t['resampling_group'])
    return str(task)


def build_units(manifest_rows, final=False):
    """Analysis units (one per manifest row) with source and evalkit values; source/evalkit failures are listed."""
    check_split(manifest_rows, final)
    src, src_failed, src_rejected = source_rows(manifest_rows)
    kim = kimodo_values(manifest_rows)
    units, missing = [], []
    for r in manifest_rows:
        m = r['method']
        u = {'id': r['id'], 'method': m, 'split': r['split'], 'input_sha256': r['input_sha256'], 'origin': r.get('origin'),
             'condition': code_condition(m, r['condition']), 'values': {}, 'gates': {}, 'measure': {}}
        if m == 'kimodo':
            k = kim.get(r['case'])
            u.update(task=r['case'], parent=r['cluster'], seed=r['seed'], text=None, paraphrase=None, zone=None)
            if k is None:
                missing.append({'id': r['id'], 'layer': 'source', 'reason': 'Kimodo pair not found'})
            else:
                i = u['condition']['correction']
                u['values']['jerk_common_0p1s@source'] = k['jerk_common_0p1s'][i]
                u['values']['bone_cv_mean@source'] = k['bone_cv'][i]
                for key in KIMODO_TARGET:
                    v = (k['metrics_raw'] if i == 0 else k['metrics_post']).get(key)
                    if v is not None:
                        u['values'][f'{key}@source'] = float(v)
        else:
            task = int(r['task_id'])
            u.update(task=task, parent=parent_of(m, task), seed=int(r['seed']), text=r['text_condition'],
                     paraphrase=r['paraphrase_id'], zone=r['target_zone'])
            run = str(Path(r['run']).resolve().relative_to(ROOT))
            variant = Path(r['source']).name.removesuffix('_canonical.npz') if m == 'primal' else None
            s = src.get((run, variant))
            if s is None:
                missing.append({'id': r['id'], 'layer': 'source', 'reason': src_failed.get(run, 'no source row (rejected or unreadable)')})
            else:
                for key in (TARGET.get(m),) + SOURCE_KEYS:
                    if key and s['values'].get(key) is not None:
                        u['values'][f'{key}@source'] = float(s['values'][key])
                u['block'] = {'signature': s['input_signature'], 'weights': s['weight_identity'], 'shape': (s['frames'], s['fps']),
                              'interventions': bool(s['intervention_checks']['passed'])}
        units.append(u)
    return units, missing, src_rejected


def validate_blocks(units):
    """A paired block (method, task, text, seed) enters only with equal inputs, weights and frame shape and executed
    interventions (as analyze_native_matrix); rejected blocks are listed, never zero-filled."""
    groups = defaultdict(list)
    for u in units:
        if 'block' in u:
            groups[u['method'], u['task'], u['text'], u['seed']].append(u)
    rejected = []
    for k, us in groups.items():
        reasons = []
        if len({u['block']['signature'] for u in us}) != 1:
            reasons.append('inputs differ')
        if any(u['block']['weights'] is None for u in us) or len({u['block']['weights'] for u in us}) != 1:
            reasons.append('weight identity missing or different')
        if len({tuple(u['block']['shape']) for u in us}) != 1:
            reasons.append('frame count / rate differ')
        if not all(u['block']['interventions'] for u in us):
            reasons.append('executed intervention check failed')
        for u in us:
            u['block_ok'] = not reasons
        if reasons:
            rejected.append({'block': list(map(str, k)), 'units': [u['id'] for u in us], 'reasons': reasons})
    return rejected


# ---------------------------------------------------------------- analysis

def contrast_list():
    out = []
    for role, table in (('primary', PRIMARY), ('secondary', SECONDARY)):
        for q, items in table.items():
            for method, spec in items:
                for text in (TEXTS if method in TEXT_METHODS and spec['kind'] != 'text' else (None,)):
                    out.append({'question': q, 'role': role, 'method': method, 'spec': spec, 'text': text})
    return out


def text_filter(units, text):
    return units if text is None else [u for u in units if u['text'] == text]


def effect_rows(units, by_id, c, reps, seed):
    """Every metric of one contrast: estimate, intervals, tests, settings, gate filter and measurement consistency."""
    m, spec, text = c['method'], c['spec'], c['text']
    mine = text_filter([u for u in units if u['method'] == m], text)
    names = [(n, k, c['role']) for n, k in primary_metrics(c['question'], m)]
    if c['role'] == 'primary' and c['question'] in ('Q1', 'Q2'):
        names += [(n, k, 'secondary') for n, k in SECONDARY_METRICS]
    if m == 'kimodo' and c['question'] == 'Q1':
        names = [x for x in names if x[0] != 'target_error'] + [(f'target_error_{t}', f'{t}@source', c['role']) for t in KIMODO_TARGET]
    out = []
    for name, key, role in names:
        e = {'question': c['question'], 'method': m, 'contrast': spec_name(m, spec), 'role': c['role'], 'metric_role': role,
             'text_condition': text, 'metric': name, 'key': key,
             'constrained': constrained(m, spec, 'target_error' if name.startswith('target_error') else name),
             'descriptive_only': descriptive_only(m, name)}
        if m == 'kimodo':
            e['label'] = 'PPT 已见数据的复核 (re-check of deck-seen data; neither development nor held-out)'
        if key is None:
            out.append({**e, 'pairs': 0, 'ci95': None, 'reason': 'not applicable: the method has no such quantity'})
            continue
        gate = evaluator(key)
        d = group_deltas(mine, m, spec, key, gate)
        e.update(summarize(d, reps, seed, crossed=m in TEXT_METHODS and spec['kind'] != 'text',
                           tests=c['role'] == 'primary' and role == 'primary'))
        if gate:
            e['gate'] = gate
            e['all_pairs_sensitivity'] = summarize(group_deltas(mine, m, spec, key), reps, seed)
        e['settings'] = {a: summarize(group_deltas(mine, m, spec, a, evaluator(a)), reps, seed) for a in alternates(key, m)}
        if gate and d and d[0]['arms'] is not None:
            floor = 'method' if gate.endswith('@method') else 'self'
            def arm(side):
                us = dedup(list({i: by_id[i] for x in d for i in x['arms'][side]}.values()))[0]
                return [{'unmeasurable_share': u['measure'].get(f'unmeasurable_share@{floor}'), 'foot_offset_m': u['measure'].get('foot_offset_m'),
                         'in_band': u['measure'].get(f'in_band@{floor}')} for u in us]
            e['measurement'] = consistency(arm('on'), arm('off'))
        out.append(e)
    return out


def gate_rows(units, by_id, c, reps, seed):
    """statistics.validity_gates: per evaluator, each arm's pass rate and the paired on - off difference."""
    m, spec, text = c['method'], c['spec'], c['text']
    mine = text_filter([u for u in units if u['method'] == m], text)
    keys = [k for _, k in primary_metrics('Q1', m) + primary_metrics('Q3', m) + SECONDARY_METRICS if k]
    out = []
    for g in sorted({evaluator(k) for k in keys} - {None} | ({'neck_rotation@self'} if m in NECK_ROTATION_METHODS else set())):
        d = group_deltas(mine, m, spec, None, value=lambda u, g=g: u['gates'].get(g))
        def rate(side):
            if not d or d[0]['arms'] is None:
                return []
            return [u['gates'][g] for u in dedup(list({i: by_id[i] for x in d for i in x['arms'][side]}.values()))[0]]
        on, off = rate('on'), rate('off')
        out.append({'question': c['question'], 'method': m, 'contrast': spec_name(m, spec), 'text_condition': text, 'gate': g,
                    'on_runs': len(on), 'off_runs': len(off), 'on_rate': float(np.mean(on)) if on else None,
                    'off_rate': float(np.mean(off)) if off else None, 'paired_difference': summarize(d, reps, seed)})
    return out


def dose_rows(units, c, reps, seed):
    """questions.Q1.dose: the contrast per target zone, and zone x switch against the deck's original tasks (secondary)."""
    m, spec, text = c['method'], c['spec'], c['text']
    mine = text_filter([u for u in units if u['method'] == m], text)
    out = []
    for name, key in [(n, k) for n, k in primary_metrics('Q1', m) + primary_metrics('Q3', m) if k]:
        d = group_deltas(mine, m, spec, key, evaluator(key))
        base = [x for x in d if x['zone'] == 'ppt_original']
        for z in sorted({x['zone'] for x in d}):
            dz = [x for x in d if x['zone'] == z]
            row = {'role': 'secondary (dose)', 'method': m, 'contrast': spec_name(m, spec), 'text_condition': text, 'metric': name,
                   'zone': z, 'effect': summarize(dz, reps, seed)}
            if z != 'ppt_original':
                row['zone_x_switch'] = stratified_difference(dz, base, reps, seed)
            out.append(row)
    return out


def mechanism_rows(units, reps, seed):
    """questions.Q1.mechanism_hypothesis: H_bone, H_dart effects on bone-length CV and H_fk levels per condition."""
    out = []
    for m, spec, text in [('omnicontrol', simple('spatial', {'realism': 1}), t) for t in TEXTS] + [('dart', simple('goal_update', {}), None)]:
        d = group_deltas(text_filter([u for u in units if u['method'] == m], text), m, spec, 'bone_cv_mean@source')
        row = {'hypothesis': 'H_bone' if m == 'omnicontrol' else 'H_dart', 'method': m, 'contrast': spec_name(m, spec),
               'text_condition': text, 'metric': 'bone_cv_mean', 'effect': summarize(d, reps, seed)}
        if m == 'omnicontrol':
            base = [x for x in d if x['zone'] == 'ppt_original']
            row['zone_x_spatial'] = {z: stratified_difference([x for x in d if x['zone'] == z], base, reps, seed)
                                     for z in sorted({x['zone'] for x in d} - {'ppt_original'})}
        out.append(row)
    for m in sorted({u['method'] for u in units}):
        for cond in sorted({json.dumps(u['condition'], sort_keys=True) for u in units if u['method'] == m}):
            us = dedup([u for u in units if u['method'] == m and json.dumps(u['condition'], sort_keys=True) == cond])[0]
            v = [u['values']['bone_cv_mean@source'] for u in us if u['values'].get('bone_cv_mean@source') is not None]
            out.append({'hypothesis': 'H_fk structural check' if m in ('ardy', 'kimodo', 'primal', 'chois') else 'level (descriptive)',
                        'method': m, 'condition': json.loads(cond), 'metric': 'bone_cv_mean', 'runs': len(v),
                        'median': float(np.median(v)) if v else None, 'max': float(np.max(v)) if v else None})
    return out


def analyse(units, reps=REPS, seed=SEED):
    by_id = {u['id']: u for u in units}
    contrasts = contrast_list()
    contrasts += [{**c, 'question': 'Q3'} for c in contrasts if c['role'] == 'primary']
    effects, gates, dose = [], [], []
    for c in contrasts:
        effects += effect_rows(units, by_id, c, reps, seed)
        if c['role'] == 'primary' and c['question'] in ('Q1', 'Q2'):
            gates += gate_rows(units, by_id, c, reps, seed)
            if c['method'] in TEXT_METHODS:
                dose += dose_rows(units, c, reps, seed)
    apply_holm(effects)
    for e in effects:
        e['verdict'] = robustness(e, e.get('settings', {}), e.get('holm_pass'), 'holm_family' in e or 'holm_reason' in e)
        if e.get('measurement', {}).get('verdict') == 'measurement state differs':
            e['verdict'] += '; measurement state differs (descriptive)'
    return {'effects': effects, 'gate_pass_rates': gates, 'dose': dose, 'mechanism': mechanism_rows(units, reps, seed)}


def stratified_difference(a, b, reps=REPS, seed=SEED):
    """Zone x switch: equal-task-weight effect in a zone minus in the deck's original tasks; Welch t interval over the
    task means of the two strata, the within-stratum task bootstrap as sensitivity."""
    if len({x['task'] for x in a}) < 2 or len({x['task'] for x in b}) < 2:
        return {'mean': None, 'ci95': None, 'reason': 'not computable: fewer than 2 tasks in a stratum'}
    def means(d):
        t = defaultdict(list)
        for x in d:
            t[x['task']].append(x['delta'])
        return np.array([np.mean(t[k]) for k in sorted(t, key=str)])
    from scipy import stats
    ma, mb = means(a), means(b)
    diff = float(ma.mean()-mb.mean())
    va, vb = ma.var(ddof=1)/len(ma), mb.var(ddof=1)/len(mb)
    if va+vb > 0:
        df = (va+vb)**2/(va**2/(len(ma)-1)+vb**2/(len(mb)-1))
        h = float(stats.t.ppf(.975, df))*math.sqrt(va+vb)
    else:
        df, h = None, 0.
    ci = [diff-h, diff+h]
    rng = np.random.default_rng(seed)
    da = ma[rng.integers(0, len(ma), (reps, len(ma)))].mean(1); db = mb[rng.integers(0, len(mb), (reps, len(mb)))].mean(1)
    boot = np.quantile(da-db, [.025, .975]).tolist()
    return {'mean': diff, 'ci95': ci, 'excludes_0': excludes0(ci), 'interval': 'Welch t', 'df': df, 'tasks': [len(ma), len(mb)],
            'bootstrap_sensitivity': {'ci95': boot, 'excludes_0': excludes0(boot)}}


def apply_holm(effects):
    """statistics.multiplicity: within each question and primary metric, over the methods' primary contrasts (text
    conditions separate), primary setting only; constrained and descriptive-only metrics stay out."""
    fam = defaultdict(list)
    for e in effects:
        if e['role'] == 'primary' and e['metric_role'] == 'primary' and not e.get('constrained') and not e.get('descriptive_only') \
                and e.get('key') and not e['metric'].startswith('target_error_'):
            fam[e['question'], e['metric']].append(e)
    for (q, metric), es in fam.items():
        tested = [e for e in es if e.get('sign_flip_p') is not None]
        adj = holm([e['sign_flip_p'] for e in tested]) if tested else []
        for e, p in zip(tested, adj):
            e.update(holm_p=p, holm_pass=bool(p < ALPHA), holm_family=f'{q} {metric} ({len(tested)} contrasts)')
        for e in es:
            if e.get('sign_flip_p') is None:
                e.update(holm_p=None, holm_pass=None, holm_reason=e.get('tests_reason', 'not computable'))


def fallback_reading(dev, held):
    """split_use.fallback_reading of one primary contrast and metric."""
    if dev.get('mean') is None or held.get('ci95') is None:
        return 'underpowered, cannot be judged'
    covers = held['ci95'][0] <= dev['mean'] <= held['ci95'][1]
    same = np.sign(held['mean']) == np.sign(dev['mean'])
    if same and covers:
        return 'consistent'
    if not same and not covers:
        return 'inconsistent'
    return 'underpowered, cannot be judged'


COUNT_KEYS = {'pairs', 'tasks', 'clusters', 'rows', 'runs', 'on_runs', 'off_runs', 'cells', 'paraphrases', 'manifest_rows',
              'units', 'dropped', 'revision_records', 'bootstrap_resamples', 'bootstrap_seed'}
RESULT_KEYS = {'verdict', 'reading', 'differs_in', 'not_checkable', 'sign_flip_method', 'interval_caveat'}


def structure(result):
    """Field names, row and pair counts only. Every other number, every boolean and every judgement string is replaced
    by a placeholder; explicit 'not computable' / 'not applicable' reasons stay readable."""
    def strip(x, key=''):
        if isinstance(x, dict):
            return {k: strip(v, k) for k, v in x.items()}
        if isinstance(x, list):
            if key in RESULT_KEYS:
                return '<judgement>'
            if x and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in x):
                return '<numbers>'
            return [strip(v) for v in x]
        if x is None:
            return None
        if isinstance(x, bool):
            return '<bool>'
        if isinstance(x, str):
            if key in RESULT_KEYS and not x.startswith(('not computable', 'not applicable')):
                return '<judgement>'
            return x
        if isinstance(x, int) and key in COUNT_KEYS:
            return x
        return '<number>'
    return strip(result)


def git_commit():
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batch', required=True, type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--structure-only', action='store_true', help='fields, row and pair counts only; no value is printed or written')
    p.add_argument('--final-comparison', action='store_true', help='open the held-out rows (split_use: once, with the final comparison)')
    a = p.parse_args()
    import scipy
    env = {'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__}
    if env != ENVIRONMENT:
        raise SystemExit(f'the analysis is pinned to {ENVIRONMENT} (analysis plan revision 2026-11-09); this interpreter has {env}')
    global CROSSED_ROLE
    val = json.loads(VALIDATION.read_text()) if VALIDATION.exists() else {}
    extra = json.loads(VALIDATION_EXTRA.read_text()) if VALIDATION_EXTRA.exists() else {}
    if not (val.get('passed') or (extra.get('passed') and extra.get('synthetic_json_sha256') == sha(VALIDATION))):
        raise SystemExit(f'analysis validation (b) has not passed: {VALIDATION.relative_to(ROOT)}')
    CROSSED_ROLE = val['crossed_bootstrap']['role']
    dev_lay = layout(a.batch, 'development')
    rows = json.loads(dev_lay['manifest'].read_text())['rows']
    units, missing, src_rejected = build_units(rows)
    units, dropped = dedup(units)
    rejected = validate_blocks(units)
    ev = evalkit_values(dev_lay, rows)
    for u in units:
        e = ev[u['id']]
        u['values'].update(e['values']); u['gates'] = e['gates']; u['measure'] = e['measure']
    plan = json.loads(PLAN.read_text())
    result = {'created': time.strftime('%Y-%m-%d %H:%M'), 'commit': git_commit(), 'batch': str(a.batch), 'split': 'development',
              'plan': {'path': str(PLAN.relative_to(ROOT)), 'sha256': sha(PLAN), 'revision_records': len(plan['revisions_after_freeze'])},
              'environment': env,
              'validation': {'path': str(VALIDATION.relative_to(ROOT)), 'sha256': sha(VALIDATION), 'crossed_bootstrap_role': CROSSED_ROLE,
                             'skew_null_extra': {'path': str(VALIDATION_EXTRA.relative_to(ROOT)), 'sha256': sha(VALIDATION_EXTRA)}
                             if VALIDATION_EXTRA.exists() else None},
              'parameters': {'primary_interval': 'cluster t, G - 1 df', 'bootstrap_resamples': REPS, 'bootstrap_seed': SEED, 'alpha': ALPHA,
                             'consistency_limits': {'share': CONSISTENCY_SHARE, 'offset_m': CONSISTENCY_OFFSET_M}, 'band_m': BAND_M},
              'inputs': {'manifest_rows': len(rows), 'units': len(units), 'dropped': len(dropped),
                         'units_per_method': {m: sum(u['method'] == m for u in units) for m in sorted({u['method'] for u in units})},
                         'missing_source': missing, 'source_rejected': src_rejected, 'rejected_blocks': rejected,
                         'evalkit_not_evaluated': {f: sum(not u['measure'].get(f'{f}_evaluated') for u in units) for f in ('method', 'self')}},
              **analyse(units),
              'not_implemented': {'overhead_strategy': 'Q3_ergonomics.overhead_strategy: its registered precondition (agreement with '
                                                       'task_reachability.py elevation() within 0.1 deg on existing evaluations) is not met yet',
                                  'fid': 'official evaluation batch only (statistics.fid_interval), outside this script'}}
    if a.final_comparison:
        held_lay = layout(a.batch, 'held_out')
        hrows = json.loads(held_lay['manifest'].read_text())['rows']
        hunits, hmissing, _ = build_units(hrows, final=True)
        hunits, _ = dedup(hunits); validate_blocks(hunits)
        hev = evalkit_values(held_lay, hrows)
        for u in hunits:
            e = hev[u['id']]; u['values'].update(e['values']); u['gates'] = e['gates']; u['measure'] = e['measure']
        held = analyse(hunits)
        idx = {(e['question'], e['method'], e['contrast'], e['text_condition'], e['metric']): e for e in held['effects']}
        reading = []
        for e in result['effects']:
            if e['role'] == 'primary' and e['metric_role'] == 'primary':
                h = idx.get((e['question'], e['method'], e['contrast'], e['text_condition'], e['metric']), {})
                reading.append({k: e[k] for k in ('question', 'method', 'contrast', 'text_condition', 'metric')} |
                               {'development_mean': e.get('mean'), 'held_out': {k: h.get(k) for k in ('pairs', 'mean', 'ci95')},
                                'reading': 'no held-out set' if e['method'] == 'ardy' else fallback_reading(e, h)})
        result['held_out'] = {'missing_source': hmissing, 'reading': reading}
        note = ROOT/'develop_logs/notes/held_out_openings.md'
        with open(note, 'a') as f:
            f.write(f"- {time.strftime('%Y-%m-%d %H:%M')} held-out set opened by analyze_batch.py --final-comparison, commit {result['commit']}, batch {a.batch}\n")
    if a.structure_only:
        s = structure(result)
        out = a.out or a.batch/'analysis'/'structure.json'
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(s, indent=1, ensure_ascii=False))
        print(json.dumps({'units': len(units), 'effects': len(result['effects']), 'missing_source': len(missing),
                          'rejected_blocks': len(rejected), 'out': str(out)}))
        return
    out = a.out or a.batch/'analysis'/('final_comparison.json' if a.final_comparison else 'development.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({'units': len(units), 'effects': len(result['effects']), 'out': str(out)}))


if __name__ == '__main__':
    main()
