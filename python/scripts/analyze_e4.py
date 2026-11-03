#!/usr/bin/env python
"""Analysis of E4 (config/e4_cross_method_20261103.json): the E criterion across the eight methods of experiment 1."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.native_analysis import holm, sign_flip_p
from motiogen.runtime import json_write

SPEC = json.loads((ROOT/'config/e4_cross_method_20261103.json').read_text(encoding='utf-8'))
OUT = ROOT/'develop_logs/test_runs/20261103_e4_cross_method'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e4_cross_method_20261103/runs')
PRIMARY = 'residual_force_rms_bw'
METRICS = SPEC['metrics']['primary']+SPEC['metrics']['secondary']
METHODS = list(SPEC['selection']['methods'])
TRIM = SPEC['pipeline']['trim_s']
REPS, SEED = 10000, 20261103


def key(cond):
    return json.dumps(cond, sort_keys=True)


def load():
    jobs = json.loads((OUT/'plan.json').read_text())['jobs']
    runs = collections.defaultdict(dict); missing, failed = [], []
    for job in jobs:
        rec = OUT/'runs'/job['id']
        if (rec/'metrics.json').exists():
            runs[(job['method'], key(job['condition']))][(job['task'], job['seed'])] = json.loads((rec/'metrics.json').read_text())
        elif (rec/'error.json').exists():
            failed.append(job['id'])
        else:
            missing.append(job['id'])
    return jobs, runs, missing, failed


def bootstrap(cluster_means, rng):
    x = np.asarray(cluster_means, float); idx = rng.integers(0, len(x), (REPS, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist()


def contrast(runs, method, factor, fixed, metric, valid_only):
    on, off = key({**fixed, factor: 1}), key({**fixed, factor: 0})
    a, b = runs.get((method, on), {}), runs.get((method, off), {})
    pairs = sorted(set(a) & set(b))
    if valid_only:
        pairs = [p for p in pairs if a[p]['valid'] and b[p]['valid']]
    by_cluster = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pairs:
        by_cluster[a[p]['cluster']][p[0]].append(a[p]['metrics'][metric]-b[p]['metrics'][metric])
    means = [np.mean([np.mean(v) for v in tasks.values()]) for tasks in by_cluster.values()]
    result = {'pairs': len(pairs), 'pairs_total': len(set(a) & set(b)), 'clusters': len(means)}
    if len(means) >= 2:
        rng = np.random.default_rng(SEED)
        result.update({'mean': float(np.mean(means)), 'ci95': bootstrap(means, rng), 'p_sign_flip': sign_flip_p(means)['p']})
        result['excludes_zero'] = bool(result['ci95'][0] > 0 or result['ci95'][1] < 0)
    return result


def levels(rows, metric, valid_only):
    v = [r['metrics'][metric] for r in rows.values() if r['valid'] or not valid_only]
    if not v:
        return {'n': 0}
    q = np.percentile(v, [25, 50, 75])
    return {'n': len(v), 'median': float(q[1]), 'iqr': [float(q[0]), float(q[2])]}


def lowpass_r(row, fc=6.0):
    """Post hoc sensitivity of the contact validation: both forces low-passed at fc before the correlation."""
    from scipy.signal import butter, filtfilt
    with np.load(Path(row['source'])/'motion.npz') as m:
        sim = np.asarray(m['contact_forces'], float)[..., 2].sum(1); fps = float(m['fps'])
    d = np.load(DATA/row['id']/'dynamics.npz'); t = d['times']; est = d['grf_r'][:, 1]+d['grf_l'][:, 1]
    s = np.interp(t, np.arange(len(sim))/fps, sim); dt = float(np.median(np.diff(t)))
    b, a = butter(4, fc*2*dt); keep = (t >= TRIM) & (t <= t[-1]-TRIM)
    return float(np.corrcoef(filtfilt(b, a, est)[keep], filtfilt(b, a, s)[keep])[0, 1])


def fmt_delta(c):
    if 'mean' not in c:
        return f'有效配对不足（{c["pairs"]}/{c["pairs_total"]}）'
    lo, hi = c['ci95']
    text = f'{c["mean"]:+.3f} [{lo:+.3f}, {hi:+.3f}]'
    if c['pairs'] < c['pairs_total']:
        text += f'\n有效 {c["pairs"]}/{c["pairs_total"]} 对'
    return text


def main():
    jobs, runs, missing, failed = load()
    result = {'protocol': SPEC['protocol'], 'jobs': len(jobs), 'evaluated': sum(len(v) for v in runs.values()),
              'missing': missing, 'failed': failed, 'statistics': SPEC['statistics'],
              'p_value_note': 'Two-sided sign-flip test over cluster means (exact up to 24 clusters), the p value used by the earlier analyses of this project; the registration names Holm but not the test'}
    passes, lv, contrasts, table = {}, {}, {}, {}
    for method in METHODS:
        mspec = SPEC['selection']['methods'][method]
        for cond in mspec['conditions']:
            rows = runs.get((method, key(cond)), {})
            name = f'{method} {key(cond)}'
            gates = collections.Counter(g for r in rows.values() for g, ok in r['validity'].items() if not ok)
            passes[name] = {'runs': len(rows), 'valid': sum(r['valid'] for r in rows.values()), 'failed_gates': dict(gates)}
            lv[name] = {m: {'valid': levels(rows, m, True), 'all': levels(rows, m, False)} for m in SPEC['metrics']['primary']}
        for i, (factor, fixed) in enumerate(mspec['contrasts']):
            name = f'{method} {factor} | {key(fixed)}'
            contrasts[name] = {'primary_contrast': i == 0, 'method': method,
                               'valid_pairs': {m: contrast(runs, method, factor, fixed, m, True) for m in METRICS},
                               'all_pairs': {m: contrast(runs, method, factor, fixed, m, False) for m in METRICS}}
        factor, fixed = mspec['contrasts'][0]
        off, on = lv[f'{method} {key({**fixed, factor: 0})}'][PRIMARY]['valid'], lv[f'{method} {key({**fixed, factor: 1})}'][PRIMARY]['valid']
        c = contrasts[f'{method} {factor} | {key(fixed)}']['valid_pairs'][PRIMARY]
        table[method] = {'levels_text': f'{off["median"]:.3f} → {on["median"]:.3f}' if off['n'] and on['n'] else '无有效运行',
                         'delta_text': fmt_delta(c), 'off': off, 'on': on, 'contrast': c}
    first = [n for n, c in contrasts.items() if c['primary_contrast'] and 'p_sign_flip' in c['valid_pairs'][PRIMARY]]
    for n, p in zip(first, holm([contrasts[n]['valid_pairs'][PRIMARY]['p_sign_flip'] for n in first])):
        contrasts[n]['valid_pairs'][PRIMARY]['p_holm'] = p
        table[contrasts[n]['method']]['contrast']['p_holm'] = p

    def verdict(name, direction):
        c = contrasts[name]['valid_pairs'][PRIMARY]
        if 'mean' not in c:
            return {'status': 'not_testable', **c}
        ok = c.get('p_holm', c['p_sign_flip']) < .05 and (direction == 0 or np.sign(c['mean']) == direction)
        return {'status': 'supported' if ok else 'not_supported', **c}
    hyp = {'H1_ardy': verdict('ardy correction | {"history_long": 0}', 1), 'H1_kimodo': verdict('kimodo correction | {}', 1),
           'H2': verdict('omnicontrol spatial | {"realism": 1}', 1),
           'H3': {m: verdict(f'{m} {SPEC["selection"]["methods"][m]["contrasts"][0][0]} | {key(SPEC["selection"]["methods"][m]["contrasts"][0][1])}', 0)
                  for m in ('motionlcm', 'chois', 'dart', 'primal')}}
    closd = [lv[f'closd {key(c)}'][PRIMARY]['valid'].get('median') for c in SPEC['selection']['methods']['closd']['conditions']]
    others = {n: l[PRIMARY]['valid'].get('median') for n, l in lv.items() if not n.startswith(('closd', 'chois'))}
    hyp['H4'] = {'closd_medians': closd, 'kinematic_medians': others, 'note': 'CHOIS excluded by the registered object-load boundary',
                 'status': 'not_testable' if None in closd or None in others.values()
                 else 'supported' if all(max(closd) < v for v in others.values()) else 'not_supported',
                 'violations': {n: v for n, v in others.items() if v is not None and None not in closd and v <= max(closd)}}
    cv = [(r['id'], r['contact_validation']) for c in SPEC['selection']['methods']['closd']['conditions']
          for r in runs.get(('closd', key(c)), {}).values() if 'contact_validation' in r]
    rs = [v['pearson_r_vertical'] for _, v in cv]
    contact = {'runs': len(rs), 'median_r': float(np.median(rs)) if rs else None,
               'iqr_r': np.percentile(rs, [25, 75]).tolist() if rs else None,
               'threshold': .8, 'passes': bool(rs) and float(np.median(rs)) >= .8,
               'simulator_flight_fraction_median': float(np.median([v['simulator_flight_fraction'] for _, v in cv])) if cv else None}
    try:
        post = [lowpass_r(r) for c in SPEC['selection']['methods']['closd']['conditions'] for r in runs.get(('closd', key(c)), {}).values()]
        contact['posthoc_lowpass_6hz'] = {'note': 'post hoc, not registered', 'median_r': float(np.median(post)) if post else None,
                                          'iqr_r': np.percentile(post, [25, 75]).tolist() if post else None}
    except Exception as exc:
        contact['posthoc_lowpass_6hz'] = {'error': repr(exc)}
    result.update({'validity': passes, 'levels': lv, 'contrasts': contrasts, 'hypotheses': hyp, 'contact_validation': contact,
                   'report_table': table})
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'evaluated': result['evaluated'], 'missing': len(missing), 'failed': len(failed),
                      'table': {m: [t['levels_text'], t['delta_text'], t['contrast'].get('p_holm')] for m, t in table.items()},
                      'H': {k: (v['status'] if 'status' in v else {m: x['status'] for m, x in v.items()}) for k, v in hyp.items()},
                      'contact': contact}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
