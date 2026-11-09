#!/usr/bin/env python
"""Validation of python/scripts/analyze_batch.py before any batch result is seen (batch plan section 3).

  validate_batch_analysis.py native --out DIR       (a) native matrix reproduction
  validate_batch_analysis.py synthetic --out DIR [--datasets 1000] [--workers 14]
                                                    (b) synthetic null / injected effects on the development design,
                                                        plus the boundary cases of the automatic judgements

(a) 1. per-row values rebuilt from the native matrix run directories (analyze_native_matrix.main into DIR) equal
       20261024_native_matrix_analysis/analysis.json; 0.1 s jerk, boundary jerk and bone CV recomputed equal v5/per_run.json;
    2. the engine (group_deltas + summarize) reproduces every contrast and simple effect of analysis.json (2000 resamples,
       seed 20261024), the v5 effects (cluster t interval, sign-flip p, bootstrap) including the Kimodo pairs, and the
       deck re-check (its 72-item gate, task-cluster and two-level intervals; 10000 resamples, seed 20261108).
    Tolerance: equal, or relative difference <= 1e-9 (约定，未经推演).
(b) null: per primary contrast the raw interval excludes 0 in [3%, 7%] of the datasets, the Holm familywise error
    rate has a 95% upper bound <= 7%; injected 0.5, 1, 2 SD: the primary interval covers the truth in [90%, 98%], power
    reported (thresholds 约定，未经推演). Boundary cases: 4.9 / 5.1 percentage points, 0.49 / 0.51 cm, a sign flip
    across settings, held-out refusal, dedup.
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
sys.path.insert(0, str(ROOT/'python/scripts'))
import analyze_batch as ab  # noqa: E402
from motiogen.native_analysis import cluster_t_interval, factorial_block, sign_flip_p  # noqa: E402

ANALYSIS = ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
V5 = ROOT/'develop_logs/test_runs/20261030_next_round/v5'
RECHECK = ROOT/'develop_logs/test_runs/20261108_deck_recheck/recheck.json'
REL_TOL = 1e-9
DECK_REPS, DECK_SEED = 2000, 20261024


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (bool, str)) or isinstance(b, (bool, str)):
        return a == b
    a, b = float(a), float(b)
    return a == b or abs(a-b) <= REL_TOL*max(abs(a), abs(b))


def rel(a, b):
    if a is None or b is None or a == b:
        return 0.
    return abs(a-b)/max(abs(a), abs(b))


def flat(x):
    if isinstance(x, dict):
        return [v for k in sorted(x) for v in flat(x[k])]
    if isinstance(x, (list, tuple)):
        return [v for y in x for v in flat(y)]
    return [x]


class Tally:
    def __init__(self):
        self.checked, self.mismatches, self.max_rel = 0, [], 0.

    def add(self, what, mine, ref):
        self.checked += 1
        for x, y in zip(flat(mine), flat(ref)):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool) and not isinstance(y, bool):
                self.max_rel = max(self.max_rel, rel(float(x), float(y)))
        if not close(mine, ref):
            self.mismatches.append({'what': what, 'mine': mine, 'reference': ref})

    def report(self):
        return {'checked': self.checked, 'mismatches': len(self.mismatches), 'max_relative_difference': self.max_rel,
                'first_mismatches': self.mismatches[:20], 'passed': self.checked > 0 and not self.mismatches}


# ---------------------------------------------------------------- (a)

def rows_rebuild(out):
    import analyze_native_matrix as anm
    dest = out/'native_matrix_rerun.json'
    anm.main(dest)
    a, b = json.loads(ANALYSIS.read_text()), json.loads(dest.read_text())
    t = Tally()
    for k in sorted(set(a) | set(b)):
        if k not in ('updated_at', 'analysis_source_sha256'):
            t.add(k, b.get(k), a.get(k))
    return t.report()


def jerk_recompute(rows):
    import v5_offline_recompute as v5
    per_run, _ = v5.recompute(rows)
    ref = {(r['run'], r.get('variant')): r for r in json.loads((V5/'per_run.json').read_text())}
    t = Tally()
    for r in per_run:
        x = ref.get((r['run'], r['variant']))
        t.add(f"{r['run']} {r['variant']}", {k: r[k] for k in ('jerk_common_0p1s', 'boundary_jerk_common_0p1s', 'bone_cv')},
              None if x is None else {k: x[k] for k in ('jerk_common_0p1s', 'boundary_jerk_common_0p1s', 'bone_cv')})
    if len(per_run) != len(ref):
        t.mismatches.append({'what': 'row count', 'mine': len(per_run), 'reference': len(ref)})
    return t.report()


def matrix_units(rows, text_of=lambda r: r['clock'], valid=None):
    out = []
    for r in rows:
        out.append({'id': r['run'] + ('/' + r['variant'] if r.get('variant') else '') + '@' + r['clock'], 'method': r['method'],
                    'task': r['task'], 'parent': r.get('resampling_group', str(r['task'])), 'seed': r['seed'], 'text': text_of(r),
                    'paraphrase': None, 'zone': None, 'condition': r['condition'], 'values': r['values'], 'split': 'development',
                    'input_sha256': r['motion_sha256'], 'gates': {}, 'measure': {}, 'block_ok': True if valid is None else valid(r)})
    return out


def analysis_contrasts():
    """Every contrast and simple effect of analysis.json from the engine (2000 resamples, seed 20261024)."""
    a = json.loads(ANALYSIS.read_text())
    accepted = {(b['method'], b['task'], b['seed'], b['clock']) for b in a['blocks'] if b['complete']}
    blocks = defaultdict(list)
    for r in a['rows']:
        blocks[r['method'], r['task'], r['seed'], r['clock']].append(r)
    valid = {}
    for k, rs in blocks.items():
        ms = set.intersection(*(set(r['values']) for r in rs))
        valid[k] = {m for m in ms if all(isinstance(r['values'][m], (int, float)) and np.isfinite(r['values'][m]) for r in rs)}
    units = matrix_units(a['rows'], valid=lambda r: (r['method'], r['task'], r['seed'], r['clock']) in accepted)
    key_of = {u['id']: (u['method'], u['task'], u['seed'], u['text']) for u in units}
    value = lambda metric: (lambda u: u['values'].get(metric) if metric in valid[key_of[u['id']]] else None)
    ref_c = {(e['method'], e['clock'], tuple(e['factors']), e['metric']): e for e in a['contrasts']}
    ref_s = {(e['method'], e['clock'], e['factor'], json.dumps(e['fixed'], sort_keys=True), e['metric']): e for e in a['simple_effects']}
    tc, ts = Tally(), Tally()
    mine_c, mine_s = set(), set()
    for method, factors in ab.FACTORS.items():
        if method == 'kimodo':
            continue
        um = [u for u in units if u['method'] == method]
        metrics = sorted({m for u in um for m in u['values']})
        for clock in sorted({u['text'] for u in um}):
            uc = [u for u in um if u['text'] == clock]
            for order in range(1, len(factors)+1):
                for sel in itertools.combinations(factors, order):
                    for metric in metrics:
                        d = ab.group_deltas(uc, method, {'kind': 'factorial', 'factors': list(sel)}, metric, value=value(metric))
                        if not d:
                            continue
                        s = ab.summarize(d, DECK_REPS, DECK_SEED)
                        k = (method, clock, sel, metric); mine_c.add(k); e = ref_c.get(k)
                        b = s['bootstrap_sensitivity']
                        tc.add(str(k), [b['mean'], b['ci95'], s['pairs'], s['clusters']],
                               None if e is None else [e['mean_equal_group_weight'], e['ci95'], e['paired_n'], e['independent_groups']])
            if len(factors) > 1:
                for f in factors:
                    others = [x for x in factors if x != f]
                    for levels in itertools.product((0, 1), repeat=len(others)):
                        fixed = dict(zip(others, levels))
                        for metric in metrics:
                            d = ab.group_deltas(uc, method, ab.simple(f, fixed), metric, value=value(metric))
                            if not d:
                                continue
                            s = ab.summarize(d, DECK_REPS, DECK_SEED)
                            k = (method, clock, f, json.dumps(fixed, sort_keys=True), metric); mine_s.add(k); e = ref_s.get(k)
                            b = s['bootstrap_sensitivity']
                            ts.add(str(k), [b['mean'], b['ci95'], s['pairs'], s['clusters']],
                                   None if e is None else [e['mean_equal_group_weight'], e['ci95'], e['paired_n'], e['independent_groups']])
    for missing in set(ref_c) - mine_c:
        tc.mismatches.append({'what': str(missing), 'mine': None, 'reference': 'present'})
    for missing in set(ref_s) - mine_s:
        ts.mismatches.append({'what': str(missing), 'mine': None, 'reference': 'present'})
    return {'contrasts': {**tc.report(), 'reference_count': len(ref_c)}, 'simple_effects': {**ts.report(), 'reference_count': len(ref_s)}}


def v5_effects(paused):
    """v5.json paired effects on the rows after v5.recompute: marginal on - off within (task, seed), clusters = task or PRIMAL parent."""
    v = json.loads((V5/'v5.json').read_text())
    units = matrix_units(paused, text_of=lambda r: None)
    t = Tally()
    for e in v['effects']:
        um = [u for u in units if u['method'] == e['method']]
        d = ab.group_deltas(um, e['method'], ab.marginal(e['factor']), e['metric'])
        d = [{**x, 'task': str(x['parent'])} for x in d]
        if len({x['parent'] for x in d}) < 2:
            t.checked += 1
            t.mismatches.append({'what': f"{e['method']} {e['factor']} {e['metric']}", 'mine': f'{len(d)} pairs', 'reference': e['pairs']})
            continue
        s = ab.summarize(d, DECK_REPS, DECK_SEED, tests=True)
        t.add(f"{e['method']} {e['factor']} {e['metric']}",
              [s['pairs'], s['clusters'], s['mean'], s['ci95'], s['sign_flip_p'], s['bootstrap_sensitivity']['ci95']],
              [e['pairs'], e['groups'], e['mean_delta'], e['t_ci95'], e['sign_flip_p'], e['bootstrap_ci95']])
    return t.report()


def v5_kimodo():
    import v5_offline_recompute as v5
    from motiogen.config import data_root, load_config
    ref = json.loads((V5/'v5.json').read_text())['kimodo']
    _, rows = v5.kimodo_pairs(data_root(load_config())/'benchmarks/kimodo-stratified-20261008')
    units = []
    for r in rows:
        for c in (0, 1):
            vals = {k: r[k][c] for k in ('jerk_native', 'jerk_common_0p1s', 'bone_cv')}
            vals.update({k: x for k, x in (r['metrics_raw'] if c == 0 else r['metrics_post']).items()})
            units.append({'id': f"{r['case']}#{c}", 'method': 'kimodo', 'task': r['case'], 'parent': r['group'], 'seed': 0, 'text': None,
                          'paraphrase': None, 'zone': None, 'condition': {'correction': c}, 'values': vals, 'split': 'deck_recheck',
                          'input_sha256': f"{r['case']}#{c}", 'gates': {}, 'measure': {}})
    t = Tally()
    for key, e in ref['effects'].items():
        d = ab.group_deltas(units, 'kimodo', ab.simple('correction', {}), key)
        tcl = cluster_t_interval([x['delta'] for x in d], [x['parent'] for x in d])
        p = ab.sign_flip_p(tcl['cluster_means'])['p']
        t.add(f'kimodo {key}', [len(d), tcl['clusters'], tcl['mean'], tcl['ci'], p],
              [e['pairs'], e['groups'], e['mean_delta_post_minus_raw'], e['t_ci95'], e['sign_flip_p']])
    t.add('kimodo pairs', len(rows), ref['pairs'])
    return t.report()


RECHECK_EFFECTS = {'motionlcm': [('control_residual', {}, None)],
                   'omnicontrol': [('spatial', {}, None), ('realism', {}, None), ('spatial', {'realism': 0}, None), ('spatial', {'realism': 1}, None),
                                   ('realism', {'spatial': 0}, None), ('realism', {'spatial': 1}, None), ('realism', {}, 'spatial')]}


def recheck(paused):
    ref = json.loads(RECHECK.read_text())
    jerk = {r['run']: r['jerk_common_0p1s'] for r in json.loads((V5/'per_run.json').read_text())}
    rows = [{**r, 'values': {**r['values'], 'jerk_common_0p1s': jerk.get(r['run'])}} for r in paused if r['method'] in RECHECK_EFFECTS]
    units = matrix_units(rows, text_of=lambda r: r['task_spec']['prompt'])
    idx = {(e['method'], e['effect'], json.dumps(e['fixed'], sort_keys=True), e['metric']): e for e in ref['effects']}
    t, gate = Tally(), Tally()
    a = json.loads(ANALYSIS.read_text())
    deck = {}
    for e in a['contrasts']:
        if e['clock'] == 'paused':
            deck[(e['method'], 'x'.join(e['factors']) if len(e['factors']) == 2 else e['factors'][0], '', e['metric'])] = e
    for e in a['simple_effects']:
        if e['clock'] == 'paused':
            deck[(e['method'], e['factor'], json.dumps(e['fixed'], sort_keys=True), e['metric'])] = e
    seen = set()
    for method, effects in RECHECK_EFFECTS.items():
        um = [u for u in units if u['method'] == method]
        for factor, fixed, inter in effects:
            spec = ab.interaction(factor, inter) if inter else (ab.simple(factor, fixed) if fixed else ab.marginal(factor))
            name = f'{factor}x{inter}' if inter else factor
            for metric in ['hand_position_error', 'position_success_rate', 'jerk_rms', 'jerk_common_0p1s', 'trunk_inclination',
                           'right_upper_arm_elevation', 'left_upper_arm_elevation', 'foot_sliding_given_5cm_contact_10cms',
                           'foot_floor_penetration_fraction', 'model_rtf']:
                d = ab.group_deltas(um, method, spec, metric)
                if len({x['task'] for x in d}) < 2:
                    continue
                dt = [{**x, 'task': str(x['task']), 'parent': str(x['task'])} for x in d]
                tc = ab.summarize(dt, DECK_REPS, DECK_SEED)['bootstrap_sensitivity']
                tl = ab.two_level_interval(d)
                k = (method, name, json.dumps(fixed, sort_keys=True), metric); seen.add(k); e = idx.get(k)
                t.add(str(k), [len(d), tc['mean'], tc['ci95'], tl['mean'], tl['ci95'], tl['per_text_mean'], tl['same_sign_across_texts']],
                      None if e is None else [e['pairs'], e['task_cluster']['mean'], e['task_cluster']['ci95'], e['two_level']['mean'],
                                              e['two_level']['ci95'], e['two_level']['per_text_mean'], e['two_level']['same_sign_across_texts']])
                dk = (method, 'x'.join(sorted([factor, inter])) if inter else factor, json.dumps(fixed, sort_keys=True) if fixed else '', metric)
                if dk in deck:
                    gate.add(str(dk), [tc['mean'], tc['ci95']], [deck[dk]['mean_equal_group_weight'], deck[dk]['ci95']])
    for k in set(idx) - seen:
        t.mismatches.append({'what': str(k), 'mine': None, 'reference': 'present'})
    g = gate.report()
    g['reference_gate'] = {'checked': ref['reproduction_gate']['checked'], 'equal': ref['reproduction_gate']['equal']}
    g['passed'] = g['passed'] and g['checked'] == ref['reproduction_gate']['checked']
    return {'effects': {**t.report(), 'reference_count': len(idx)}, 'gate_72': g}


def native(args):
    import numpy, scipy
    from scipy import stats
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    res = {'created': time.strftime('%Y-%m-%d %H:%M'), 'commit': ab.git_commit(), 'tolerance_relative': REL_TOL,
           'environment': {'python': sys.version.split()[0], 'numpy': numpy.__version__, 'scipy': scipy.__version__,
                           't_ppf_0.975_df7': float(stats.t.ppf(.975, 7))}, 'parts': args.parts}
    paused = [r for r in json.loads(ANALYSIS.read_text())['rows'] if r['clock'] == 'paused']
    recomputed = [json.loads(json.dumps(r)) for r in paused]
    parts = []
    if args.parts == 'all':
        res['rows_rebuilt'] = rows_rebuild(out)
    res['jerk_per_run'] = jerk_recompute(recomputed)
    if args.parts == 'all':
        res['analysis_json'] = analysis_contrasts()
        parts += [res['rows_rebuilt'], res['analysis_json']['contrasts'], res['analysis_json']['simple_effects']]
    res['v5_effects'] = v5_effects(recomputed)
    res['v5_kimodo'] = v5_kimodo()
    parts += [res['jerk_per_run'], res['v5_effects'], res['v5_kimodo']]
    if args.parts == 'all':
        res['deck_recheck'] = recheck(paused)
        parts += [res['deck_recheck']['effects'], res['deck_recheck']['gate_72']]
    res['passed'] = all(p['passed'] for p in parts)
    res['seconds'] = round(time.time()-started, 1)
    (out/('native.json' if args.parts == 'all' else f'native_{args.parts}_scipy{scipy.__version__}.json')).write_text(
        json.dumps(res, indent=1, default=float))
    print(json.dumps({k: (v['passed'] if 'passed' in v else {kk: vv.get('passed') for kk, vv in v.items() if isinstance(vv, dict)})
                      if isinstance(v, dict) else v for k, v in res.items()}, indent=1))


# ---------------------------------------------------------------- (b)

SIGMA, TAU, BASE_SD = 1., .5, 1.
SYN_SEED = 20261109
CROSS_CHECK = 3
CHUNK = 10
SCENARIOS = ('normal', 'native_shape', 'native_shape_skew_aligned')
SHAPE_METRICS = sorted({'jerk_rms', 'boundary_jerk_rms', 'trunk_inclination', 'right_upper_arm_elevation', 'left_upper_arm_elevation',
                        'foot_sliding_given_5cm_contact_10cms'} | set(ab.TARGET.values()))
TASK_TABLE = json.loads((ROOT/'config/inference_tasks_20261109.json').read_text())


def design():
    """Development units of the frozen task table (method, task, seed, condition, text, paraphrase, zone, parent) and the
    Kimodo pairs (case, source-motion cluster); values are filled per dataset."""
    rows = development_rows()
    units = []
    for r in rows:
        m = r['method']
        conds = TASK_TABLE['methods'][m]['conditions'] if m == 'primal' else [TASK_TABLE['methods'][m]['conditions'][r['condition']]]
        task = next(t for t in TASK_TABLE['methods'][m]['tasks'] if t['id'] == r['task'])
        for c in conds:
            units.append({'id': f"{m}_{r['task']}_{r['seed']}_{r['text_level']}_{json.dumps(c, sort_keys=True)}", 'method': m,
                          'task': r['task'], 'parent': ab.parent_of(m, r['task']), 'seed': r['seed'], 'text': r['text_level'],
                          'paraphrase': r['paraphrase'], 'condition': ab.code_condition(m, c), 'split': r['split'],
                          'zone': (task['zone'] if task.get('tier') == 'risk' else 'ppt_original') if m in ab.TEXT_METHODS else None,
                          'input_sha256': None, 'gates': {}, 'measure': {}})
    for a in json.loads((ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json').read_text())['rows']:
        for c in (0, 1):
            units.append({'id': f"kimodo_{a['case']}_{c}", 'method': 'kimodo', 'task': a['case'], 'parent': a['cluster'], 'seed': a['seed'],
                          'text': None, 'paraphrase': None, 'condition': {'correction': c}, 'split': 'deck_recheck', 'zone': None,
                          'input_sha256': None, 'gates': {}, 'measure': {}})
    for i, u in enumerate(units):
        u['input_sha256'] = str(i)
    return units


def development_rows():
    h = TASK_TABLE['rows']['header']
    return [dict(zip(h, r)) for r in TASK_TABLE['rows']['data'] if r[h.index('split')] == 'development']


def primary_specs():
    return [(q, m, spec, text) for q, items in ab.PRIMARY.items() for m, spec in items
            for text in (ab.TEXTS if m in ab.TEXT_METHODS else (None,))]


def pair_sd(method, spec):
    """SD of one paired delta under the model: task-effect heterogeneity plus the noise of the on and off cells."""
    k = len(ab.FACTORS[method])
    cells = 2**(k-1) if spec['kind'] == 'factorial' else 1
    return math.sqrt(TAU**2 + 2*SIGMA**2/cells)


def residual_pool():
    """Shapes of the native matrix residuals: per (method, clock, switch, metric) the marginal on - off deltas within
    (task, seed) of analysis.json, centred and divided by their SD (ddof 0, so the pool has mean 0 and SD 1); groups with
    fewer than 20 deltas or SD 0 are skipped. 'aligned' multiplies each group by the sign of its skewness."""
    units = matrix_units(json.loads(ANALYSIS.read_text())['rows'])
    plain, aligned, groups = [], [], 0
    for method, factors in ab.FACTORS.items():
        um = [u for u in units if u['method'] == method]
        for clock in sorted({u['text'] for u in um}):
            uc = [u for u in um if u['text'] == clock]
            for f in factors:
                for metric in SHAPE_METRICS:
                    x = np.array([v['delta'] for v in ab.group_deltas(uc, method, ab.marginal(f), metric)])
                    if len(x) < 20 or x.std() == 0:
                        continue
                    z = (x-x.mean())/x.std()
                    plain.append(z); aligned.append(z*(1. if np.mean(z**3) >= 0 else -1.)); groups += 1
    plain, aligned = np.concatenate(plain), np.concatenate(aligned)
    def moments(z):
        return {'n': len(z), 'skewness': float(np.mean(z**3)), 'excess_kurtosis': float(np.mean(z**4)-3),
                'quantiles_0.1_1_50_99_99.9': np.percentile(z, [.1, 1, 50, 99, 99.9]).tolist()}
    return ({'native_shape': plain, 'native_shape_skew_aligned': aligned},
            {'groups': groups, 'metrics': SHAPE_METRICS, 'native_shape': moments(plain), 'native_shape_skew_aligned': moments(aligned)})


def fill(units, size, rng, pool=None):
    """y = parent + task + paraphrase + (task, text, seed) block + sum over switches of x * (effect + task heterogeneity) + noise.
    Effects: size x pair_sd of the method's primary contrast, the same for every switch of the method. Every random term
    is SD x a standard normal draw, or SD x a draw from the residual pool (mean 0, SD 1) when pool is given."""
    effect = {}
    for q, m, spec, _ in primary_specs():
        effect.setdefault(m, size*pair_sd(m, spec))
    z = (lambda: rng.normal()) if pool is None else (lambda: pool[rng.integers(len(pool))])
    draw = defaultdict(lambda: None)
    def r(key, sd):
        if draw[key] is None:
            draw[key] = sd*z()
        return draw[key]
    for u in units:
        m = u['method']
        y = r(('p', m, u['parent']), BASE_SD) + r(('t', m, u['task']), BASE_SD) + r(('b', m, u['task'], u['text'], u['seed']), .5)
        if u['paraphrase'] is not None:
            y += r(('q', m, u['text'], u['paraphrase']), .5)
        for f, x in u['condition'].items():
            if x:
                y += effect[m] + r(('h', m, f, u['parent']), TAU)
        u['values'] = {'y@source': y + SIGMA*z()}
    return effect


def split_sign_flip_p(cluster_means):
    """Same patterns and tolerance as native_analysis.sign_flip_p, enumerated as two halves."""
    x = np.asarray(cluster_means, float); g = len(x)
    if g > 24:
        return sign_flip_p(x)
    observed = abs(x.sum()); tolerance = 1e-12*max(1., np.abs(x).sum()); threshold = observed-tolerance
    def sums(part):
        s = np.zeros(1)
        for v in part:
            s = np.concatenate([s-v, s+v])
        return s
    left, right = sums(x[:g//2]), np.sort(sums(x[g//2:]))
    total = 1 << g
    if threshold <= 0:
        return {'p': 1., 'method': 'exact', 'patterns': total}
    hits = int((len(right)-np.searchsorted(right, threshold-left, 'left')).sum()
               + np.searchsorted(right, -threshold-left, 'right').sum())
    return {'p': hits/total, 'method': 'exact', 'patterns': total}


def sign_flip_equivalence(n=300):
    rng = np.random.default_rng(7); worst = 0.; mismatches = 0
    for i in range(n):
        g = int(rng.integers(2, 25))
        x = (rng.normal(size=g) if i % 3 == 0 else rng.integers(-3, 4, g).astype(float) if i % 3 == 1
             else rng.integers(-6, 7, g)/2.)
        a, b = sign_flip_p(x)['p'], split_sign_flip_p(x)['p']
        worst = max(worst, abs(a-b)); mismatches += a != b
    return {'vectors': n, 'cluster_counts': '2..24', 'kinds': 'continuous, integer ties, half-integer ties',
            'mismatches': mismatches, 'max_abs_difference': worst}


def covers(ci, truth):
    return bool(ci[0] <= truth <= ci[1])


def one_dataset(scenario, size, index, pool):
    registered = index < CROSS_CHECK
    ab.sign_flip_p = sign_flip_p if registered else split_sign_flip_p
    rng = np.random.default_rng([SYN_SEED, SCENARIOS.index(scenario), int(size*100), index])
    units = design()
    effect = fill(units, size, rng, pool)
    out = []
    for q, m, spec, text in primary_specs():
        d = ab.group_deltas(ab.text_filter([u for u in units if u['method'] == m], text), m, spec, 'y@source')
        s = ab.summarize(d, crossed=m in ab.TEXT_METHODS, tests=True)
        truth, b, c = effect[m], s['bootstrap_sensitivity'], s.get('cross_paraphrase')
        out.append({'question': q, 'method': m, 'contrast': ab.spec_name(m, spec), 'text': text, 'truth': truth, 'pairs': s['pairs'],
                    'clusters': s['clusters'], 'p': s['sign_flip_p'],
                    't_covers': covers(s['ci95'], truth), 't_excludes_0': s['excludes_0'],
                    'boot_covers': covers(b['ci95'], truth), 'boot_excludes_0': b['excludes_0'],
                    'crossed_covers': covers(c['ci95'], truth) if c else None, 'crossed_excludes_0': ab.excludes0(c['ci95']) if c else None,
                    'split_p': split_sign_flip_p(ab.cluster_t(d)['flip_units'])['p'] if registered else None})
    for q in ('Q1', 'Q2'):
        fam = [x for x in out if x['question'] == q]
        for x, p in zip(fam, ab.holm([x['p'] for x in fam])):
            x['holm_p'] = p
    return out


def dataset_chunk(task):
    scenario, size, start, count, pool = task
    return scenario, size, [one_dataset(scenario, size, i, pool) for i in range(start, start+count)]


def upper95(k, n):
    from scipy.stats import beta
    return float(beta.ppf(.95, k+1, n-k)) if k < n else 1.


def boundary_cases():
    cases = []
    def arm(n, un=0., off=0., band=0):
        return [{'unmeasurable_share': un, 'foot_offset_m': off, 'in_band': int(i < band)} for i in range(n)]
    for label, on, off, expect in [
            ('unmeasurable 4.9 pp', arm(1000, un=.049), arm(1000), 'same measurement state'),
            ('unmeasurable 5.1 pp', arm(1000, un=.051), arm(1000), 'measurement state differs'),
            ('in band 4.9 pp', arm(1000, band=49), arm(1000), 'same measurement state'),
            ('in band 5.1 pp', arm(1000, band=51), arm(1000), 'measurement state differs'),
            ('offset median 0.49 cm', arm(1000, off=.0049), arm(1000), 'same measurement state'),
            ('offset median 0.51 cm', arm(1000, off=.0051), arm(1000), 'measurement state differs')]:
        got = ab.consistency(on, off)['verdict']
        cases.append({'case': label, 'expected': expect, 'got': got, 'passed': got == expect})
    prim = {'mean': .3, 'ci95': [.1, .5], 'excludes_0': True}
    for label, alts, expect in [('every setting same sign, excludes 0', {'a': {'mean': .2, 'ci95': [.05, .4], 'excludes_0': True}}, 'conclusion'),
                                ('sign flips in another setting', {'a': {'mean': -.2, 'ci95': [-.4, -.05], 'excludes_0': True}}, 'setting-dependent'),
                                ('same sign, interval covers 0', {'a': {'mean': .2, 'ci95': [-.1, .4], 'excludes_0': False}}, 'setting-dependent')]:
        got = ab.robustness(prim, alts, True, True)
        cases.append({'case': label, 'expected': expect, 'got': got, 'passed': got == expect})
    held = [{'split': 'development'}, {'split': 'held_out'}]
    try:
        ab.check_split(held, False); got = 'accepted'
    except SystemExit:
        got = 'refused'
    cases.append({'case': 'held-out row without --final-comparison', 'expected': 'refused', 'got': got, 'passed': got == 'refused'})
    try:
        ab.check_split(held, True); got = 'accepted'
    except SystemExit:
        got = 'refused'
    cases.append({'case': 'held-out row with --final-comparison', 'expected': 'accepted', 'got': got, 'passed': got == 'accepted'})
    u = lambda i, task, sha: {'id': i, 'method': 'motionlcm', 'task': task, 'text': 'neutral', 'seed': 20, 'condition': {'control_residual': 0}, 'input_sha256': sha}
    kept, dropped = ab.dedup([u('a', 0, 'x'), u('b', 0, 'x'), u('c', 1, 'x')])
    got = {'kept': [k['id'] for k in kept], 'dropped': dropped}
    cases.append({'case': 'dedup: identical input in one cell once, across cells kept', 'expected': {'kept': ['a', 'c'], 'dropped': ['b']},
                  'got': got, 'passed': got == {'kept': ['a', 'c'], 'dropped': ['b']}})
    rng = np.random.default_rng(1)
    cells = [{'condition': dict(zip(ab.FACTORS['primal'], c)), 'values': {'y': float(rng.normal())}}
             for c in itertools.product((0, 1), repeat=3)]
    ref = {tuple(e['factors']): e['delta'] for e in factorial_block(cells, ab.FACTORS['primal'])['contrasts']}
    units = [{'id': str(i), 'method': 'primal', 'task': 0, 'parent': '0', 'seed': 0, 'text': None, 'paraphrase': None, 'zone': None,
              'condition': c['condition'], 'values': {'y@source': c['values']['y']}} for i, c in enumerate(cells)]
    ok = all(math.isclose(ab.group_deltas(units, 'primal', {'kind': 'factorial', 'factors': list(f)}, 'y@source')[0]['delta'], v,
                          rel_tol=1e-12, abs_tol=1e-15) for f, v in ref.items())
    cases.append({'case': 'factorial contrasts equal native_analysis.factorial_block (3 factors, all orders)', 'expected': True, 'got': ok, 'passed': ok})
    d = [{'task': t, 'paraphrase': 0, 'delta': 1., 'parent': str(t)} for t in range(5)]
    got = ab.crossed_interval(d).get('reason', '')
    cases.append({'case': 'crossed interval with one paraphrase', 'expected': 'not computable', 'got': got, 'passed': got.startswith('not computable')})
    rng = np.random.default_rng(2)
    d = [{'task': t, 'parent': f'g{t//2}', 'delta': float(v)} for t in range(16) for v in rng.normal(size=3)]
    tm = [np.mean([x['delta'] for x in d if x['task'] == t]) for t in range(16)]
    ref = cluster_t_interval([np.mean(tm[2*g:2*g+2]) for g in range(8)], [f'g{g}' for g in range(8)])
    got = ab.cluster_t(d)
    ok = math.isclose(got['mean'], ref['mean'], rel_tol=1e-12) and all(math.isclose(a, b, rel_tol=1e-12) for a, b in zip(got['ci95'], ref['ci']))
    cases.append({'case': 'cluster t with 2 tasks per cluster equals native_analysis.cluster_t_interval over cluster means',
                  'expected': True, 'got': ok, 'passed': ok})
    from scipy import stats
    a = [{'task': t, 'delta': float(v)} for t, v in enumerate(rng.normal(1, 1, 5))]
    b = [{'task': t, 'delta': float(v)} for t, v in enumerate(rng.normal(0, 2, 7))]
    ref = stats.ttest_ind([x['delta'] for x in a], [x['delta'] for x in b], equal_var=False).confidence_interval(.95)
    got = ab.stratified_difference(a, b)['ci95']
    ok = all(math.isclose(x, y, rel_tol=1e-9) for x, y in zip(got, (ref.low, ref.high)))
    cases.append({'case': 'zone x switch Welch interval equals scipy ttest_ind(equal_var=False)', 'expected': True, 'got': ok, 'passed': ok})
    one = ab.summarize([{'task': 0, 'parent': '0', 'paraphrase': 0, 'delta': 1.}], tests=True)
    got = ab.robustness(one, {}, None, True)
    cases.append({'case': 'one pair: interval and verdict not computable with a reason', 'expected': 'not computable', 'got': got,
                  'passed': got.startswith('not computable') and one['ci95'] is None and one['sign_flip_p'] is None})
    return cases


def synthetic(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sizes = (0., .5, 1., 2.)
    pools, pool_stats = residual_pool()
    print(json.dumps({'residual_pool': pool_stats}), flush=True)
    tasks = [(sc, sz, st, min(CHUNK, args.datasets-st), pools.get(sc)) for sc in SCENARIOS for sz in sizes
             for st in range(0, args.datasets, CHUNK)]
    results = defaultdict(list)
    equivalence = sign_flip_equivalence()
    print(json.dumps({'sign_flip_equivalence': equivalence}), flush=True)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    done = 0
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        for scenario, size, sets in pool.map(dataset_chunk, tasks, chunksize=1):
            results[scenario, size] += sets
            done += len(sets)
            if done % 500 < CHUNK:
                print(json.dumps({'datasets_done': done, 'of': len(SCENARIOS)*len(sizes)*args.datasets,
                                  'seconds': round(time.time()-started)}), flush=True)
    rate = lambda xs, k: float(np.mean([x[k] for x in xs]))
    summary, crossed_null = {}, {}
    for (scenario, size), sets in sorted(results.items()):
        per = defaultdict(list)
        for rows in sets:
            for x in rows:
                per[x['question'], x['method'], x['contrast'], x['text']].append(x)
        contrasts = []
        for (q, m, c, text), xs in per.items():
            row = {'question': q, 'method': m, 'contrast': c, 'text': text, 'datasets': len(xs), 'pairs': xs[0]['pairs'],
                   'clusters': xs[0]['clusters'], 'excludes_0_rate': rate(xs, 't_excludes_0'), 'coverage': rate(xs, 't_covers'),
                   'holm_reject_rate': float(np.mean([x['holm_p'] < ab.ALPHA for x in xs])),
                   'bootstrap_sensitivity': {'excludes_0_rate': rate(xs, 'boot_excludes_0'), 'coverage': rate(xs, 'boot_covers')}}
            if xs[0]['crossed_covers'] is not None:
                row['cross_paraphrase'] = {'excludes_0_rate': rate(xs, 'crossed_excludes_0'), 'coverage': rate(xs, 'crossed_covers')}
                if size == 0:
                    crossed_null[f'{scenario} {m} {text}'] = row['cross_paraphrase']['excludes_0_rate']
            row['passed'] = (.03 <= row['excludes_0_rate'] <= .07) if size == 0 else (.90 <= row['coverage'] <= .98)
            contrasts.append(row)
        fam = {}
        for q in ('Q1', 'Q2'):
            k = sum(any(x['holm_p'] < ab.ALPHA for x in rows if x['question'] == q) for rows in sets)
            fam[q] = {'familywise_rejections': k, 'datasets': len(sets), 'rate': k/len(sets), 'upper95': upper95(k, len(sets))}
            if size == 0:
                fam[q]['passed'] = fam[q]['upper95'] <= .07
        entry = {'contrasts': contrasts, 'holm_family': fam,
                 'passed': all(r['passed'] for r in contrasts) and all(f.get('passed', True) for f in fam.values())}
        if size == 0:
            by_g = defaultdict(list)
            for r in contrasts:
                by_g[r['clusters']].append(r)
            entry['null_rate_by_clusters'] = {g: {'contrasts': len(rs), 't': [min(r['excludes_0_rate'] for r in rs), max(r['excludes_0_rate'] for r in rs)],
                                                  'bootstrap': [min(r['bootstrap_sensitivity']['excludes_0_rate'] for r in rs),
                                                                max(r['bootstrap_sensitivity']['excludes_0_rate'] for r in rs)]}
                                              for g, rs in sorted(by_g.items())}
        summary.setdefault(scenario, {})[f'{size:g} SD'] = entry
    crossed_ok = all(.03 <= v <= .07 for v in crossed_null.values())
    crossed = {'null_excludes_0_rate': crossed_null, 'range': [.03, .07],
               'role': 'interval' if crossed_ok else 'descriptive',
               'rule': 'user 2026-11-09: reported separately; outside 3-7 % under the null in any scenario -> descriptive only'}
    cross = [x for sets in results.values() for rows in sets for x in rows if x['split_p'] is not None]
    equivalence['in_run_cross_check'] = {'datasets_per_cell': CROSS_CHECK, 'contrasts': len(cross),
                                         'mismatches': sum(x['p'] != x['split_p'] for x in cross),
                                         'max_abs_difference': max((abs(x['p']-x['split_p']) for x in cross), default=None)}
    equivalence['passed'] = equivalence['mismatches'] == 0 and equivalence['in_run_cross_check']['mismatches'] == 0
    cases = boundary_cases()
    passed = all(v['passed'] for sc in summary.values() for v in sc.values())
    res = {'created': time.strftime('%Y-%m-%d %H:%M'), 'commit': ab.git_commit(), 'datasets_per_size': args.datasets,
           'model': {'y': 'parent (1) + task (1) + paraphrase (0.5) + (task, text, seed) block (0.5) + switch x '
                          '(effect + parent heterogeneity (0.5)) + noise (1); SDs in brackets', 'effect': 'size x SD of one paired delta '
                          '(sqrt(tau^2 + 2 sigma^2 / cells per arm)); truth = the population effect', 'seed': SYN_SEED,
                     'scenarios': {'normal': 'every term normal',
                                   'native_shape': 'every term drawn from the standardized native-matrix residual pool',
                                   'native_shape_skew_aligned': 'the same pool with every group oriented to positive skew'},
                     'residual_pool': pool_stats,
                     'design': 'development rows of config/inference_tasks_20261109.json and the 416 Kimodo pairs (source-motion clusters)',
                     'convention': '约定，未经推演 (model variances and thresholds)'},
           'primary_interval': 'cluster t, G - 1 df (analysis plan revision, user 2026-11-09); bootstrap reported as sensitivity',
           'thresholds': {'null_excludes_0_rate': [.03, .07], 'holm_familywise_upper95': .07, 'coverage': [.90, .98],
                          'applies_to': 'primary interval (cluster t) and Holm, in every scenario'},
           'sign_flip': {'registered': 'native_analysis.sign_flip_p in the first %d datasets of each scenario and size' % CROSS_CHECK,
                         'other_datasets': 'split-half enumeration of the same 2^G patterns and tolerance (memory-bound speed)',
                         'equivalence': equivalence},
           'crossed_bootstrap': crossed,
           'results': summary, 'boundary_cases': cases,
           'passed': passed and all(c['passed'] for c in cases) and equivalence['passed'],
           'seconds': round(time.time()-started, 1)}
    (out/'synthetic.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({'passed': res['passed'], 'sign_flip_equivalence': equivalence['passed'], 'crossed_role': crossed['role'],
                      **{f'{sc} {k}': v['passed'] for sc, d in summary.items() for k, v in d.items()},
                      'boundary_cases': all(c['passed'] for c in cases)}, indent=1))


EXTRA_SCENARIO, EXTRA_RANGE, EXTRA_LIMIT = 'native_shape_skew_aligned', (1000, 5000), .07


def skew_extra(args):
    """Rule registered before this data (analysis plan revision, user 2026-11-09): 4000 further null datasets of the
    skew-aligned scenario (indices 1000-4999, never used before); combined with the first 1000, the Q2 Holm familywise
    point estimate <= 0.07 passes, > 0.07 fails; the 95 % upper bound is reported, not a gate. Applies only if every
    other criterion of synthetic.json passed."""
    out = Path(args.out); started = time.time()
    base = json.loads((out/'synthetic.json').read_text())
    failing = [(sc, size, q) for sc, d in base['results'].items() for size, v in d.items()
               for q, f in v['holm_family'].items() if f.get('passed') is False]
    failing += [(sc, size, r['method'], r['text']) for sc, d in base['results'].items() for size, v in d.items()
                for r in v['contrasts'] if not r['passed']]
    scope_ok = (failing == [(EXTRA_SCENARIO, '0 SD', 'Q2')] and all(c['passed'] for c in base['boundary_cases'])
                and base['sign_flip']['equivalence']['passed'])
    pools, pool_stats = residual_pool()
    if pool_stats != base['model']['residual_pool']:
        raise SystemExit('residual pool differs from synthetic.json')
    lo, hi = EXTRA_RANGE
    tasks = [(EXTRA_SCENARIO, 0., st, min(CHUNK, hi-st), pools[EXTRA_SCENARIO]) for st in range(lo, hi, CHUNK)]
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    sets = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        for _, _, chunk in pool.map(dataset_chunk, tasks, chunksize=1):
            sets += chunk
            if len(sets) % 500 < CHUNK:
                print(json.dumps({'datasets_done': len(sets), 'of': hi-lo, 'seconds': round(time.time()-started)}), flush=True)
    first = base['results'][EXTRA_SCENARIO]['0 SD']
    fam = {}
    for q in ('Q1', 'Q2'):
        k = sum(any(x['holm_p'] < ab.ALPHA for x in rows if x['question'] == q) for rows in sets)
        k0, n0 = first['holm_family'][q]['familywise_rejections'], first['holm_family'][q]['datasets']
        fam[q] = {'extra': {'familywise_rejections': k, 'datasets': len(sets), 'rate': k/len(sets)},
                  'combined': {'familywise_rejections': k+k0, 'datasets': len(sets)+n0, 'rate': (k+k0)/(len(sets)+n0),
                               'upper95_reported_not_a_gate': upper95(k+k0, len(sets)+n0)}}
    contrasts = []
    for r in first['contrasts']:
        xs = [x for rows in sets for x in rows if (x['question'], x['method'], x['text']) == (r['question'], r['method'], r['text'])]
        n0 = r['datasets']
        contrasts.append({'question': r['question'], 'method': r['method'], 'text': r['text'], 'clusters': r['clusters'],
                          'excludes_0_rate_combined': (sum(x['t_excludes_0'] for x in xs)+r['excludes_0_rate']*n0)/(len(xs)+n0),
                          'holm_reject_rate_combined': (sum(x['holm_p'] < ab.ALPHA for x in xs)+r['holm_reject_rate']*n0)/(len(xs)+n0),
                          'bootstrap_excludes_0_rate_combined': (sum(x['boot_excludes_0'] for x in xs)
                                                                 + r['bootstrap_sensitivity']['excludes_0_rate']*n0)/(len(xs)+n0)})
    rule_ok = fam['Q2']['combined']['rate'] <= EXTRA_LIMIT
    res = {'created': time.strftime('%Y-%m-%d %H:%M'), 'commit': ab.git_commit(), 'scenario': EXTRA_SCENARIO,
           'indices': list(EXTRA_RANGE), 'rule': 'Q2 Holm familywise point estimate over the combined 5000 null datasets <= 0.07 passes, > 0.07 fails; '
           'the 95 % upper bound is reported, not a gate (user 2026-11-09, registered before this data)',
           'synthetic_json_sha256': ab.sha(out/'synthetic.json'), 'other_criteria_of_synthetic_json': {'failing': failing, 'only_this_one': scope_ok},
           'holm_family': fam, 'contrasts_combined': contrasts, 'rule_passed': rule_ok, 'passed': bool(rule_ok and scope_ok),
           'seconds': round(time.time()-started, 1)}
    (out/'synthetic_skew_null_extra.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({'passed': res['passed'], 'rule_passed': rule_ok, 'only_this_criterion_failed_before': scope_ok}, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='step', required=True)
    n = sub.add_parser('native'); n.add_argument('--out', required=True); n.add_argument('--parts', choices=('all', 'v5'), default='all')
    s = sub.add_parser('synthetic'); s.add_argument('--out', required=True)
    s.add_argument('--datasets', type=int, default=1000); s.add_argument('--workers', type=int, default=14)
    x = sub.add_parser('skew-extra'); x.add_argument('--out', required=True); x.add_argument('--workers', type=int, default=12)
    sub.add_parser('boundary')
    a = p.parse_args()
    if a.step == 'boundary':
        print(json.dumps(boundary_cases(), indent=1, default=str)); return
    {'native': native, 'synthetic': synthetic, 'skew-extra': skew_extra}[a.step](a)


if __name__ == '__main__':
    main()
