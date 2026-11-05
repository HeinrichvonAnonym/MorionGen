#!/usr/bin/env python
"""Registered analysis of the AddBiomechanics estimator calibration (config/val_a_addb_20261105.json).

Per variant and class: median and IQR over trials of V1-V4; the gate (walk and run V1, walk/run/jump V3);
tau_frame (99th percentile of per-frame residual, classes weighted equally, sts excluded) and tau_run (95th
percentile of per-trial residual RMS). Amendment 1 (config/val_a_addb_20261105_amend1.json): the same on kin30_a1,
the held-out walk V3, and tau_contact / tau_flight after the registered exclusions. Amendment 2
(config/val_a_addb_20261105_amend2.json): the same threshold rule on kin30 without running and yoga tasks.
Amendment 3 (config/val_a_addb_20261105_amend3.json): three-state thresholds (foot-event window, steady
contact, steady flight) and the human foot-event rate on the amendment-2 scope. Amendment 4
(config/val_a_addb_20261105_amend4.json): development checks of the horizontal-speed rule (kin30_a4) and the
amendment-3 thresholds recomputed with it.
Writes develop_logs/test_runs/20261105_val_a/analysis.json.
"""
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT/'develop_logs/test_runs/20261105_val_a'
SPEC = json.loads((ROOT/'config/val_a_addb_20261105.json').read_text())
AMEND = json.loads((ROOT/'config/val_a_addb_20261105_amend1.json').read_text())
AMEND2 = json.loads((ROOT/'config/val_a_addb_20261105_amend2.json').read_text())
AMEND3 = json.loads((ROOT/'config/val_a_addb_20261105_amend3.json').read_text())
AMEND4 = json.loads((ROOT/'config/val_a_addb_20261105_amend4.json').read_text())
WORK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/addb_val_20261105/work')
BALLET = ('balletsmalljump', 'ballethighleg', 'ballet_high_leg')
THRESHOLD_CLASSES = ('stand', 'walk', 'run', 'squat', 'jump', 'wholebody')
METRICS = ('V1_r', 'V1_rmse_bw', 'V1_bias_bw', 'V2_r', 'V2_rmse_bw', 'V3_balanced_accuracy_20N', 'V3_balanced_accuracy_50N',
           'V4_residual_rms_bw', 'V4_share_above_5pct', 'V4_residual_p95_bw', 'measured_vertical_mean_bw')


def weighted_quantile(values, weights, q):
    order = np.argsort(values); v, w = values[order], weights[order]
    c = np.cumsum(w)/w.sum()
    return float(v[np.searchsorted(c, q)])


def summary(xs):
    xs = np.array([x for x in xs if x is not None], float)
    if not len(xs):
        return None
    return {'median': float(np.median(xs)), 'q25': float(np.percentile(xs, 25)), 'q75': float(np.percentile(xs, 75)), 'n': int(len(xs))}


def ids(selection):
    return {r['id'] for r in json.loads((LOG/selection).read_text())['trials']}


def variant(name, keep_ids=None):
    rows = [json.loads(f.read_text()) for f in sorted((LOG/'runs'/name).glob('*.json'))]
    rows = [r for r in rows if keep_ids is None or r['id'] in keep_ids]
    ok = [r for r in rows if r['status'] == 'success']
    by = collections.defaultdict(list)
    for r in ok:
        by[r['class']].append(r)
    classes = {c: {'trials': len(v), **{m: summary([r.get(m) for r in v]) for m in METRICS},
                   'mass_header_over_dynamics': summary([r['mass_header_kg']/r['mass_kg'] for r in v])} for c, v in sorted(by.items())}
    values, weights, rms = [], [], []
    for c in THRESHOLD_CLASSES:
        frames = []
        for r in by.get(c, []):
            f = np.load(LOG/'runs'/name/f"{r['id']}_frames.npz")
            frames.append(f['residual_bw'][f['keep']])
        if frames:
            x = np.concatenate(frames); values.append(x); weights.append(np.full(len(x), 1/len(x)))
            rms += [r['V4_residual_rms_bw'] for r in by[c] if r.get('V4_residual_rms_bw') is not None]
    tau = None
    if values:
        v, w = np.concatenate(values), np.concatenate(weights)
        tau = {'tau_frame_bw': weighted_quantile(v, w, .99), 'tau_run_bw': float(np.percentile(rms, 95)),
               'frame_share_above_5pct_bw_equal_class_weight': float(np.sum(w*(v > .05))/w.sum()),
               'per_class_p99_bw': {c: float(np.percentile(x, 99)) for c, x in zip([c for c in THRESHOLD_CLASSES if by.get(c)], values)}}
    med = lambda c, m: (classes.get(c, {}).get(m) or {}).get('median')
    checks = {f'{c} V1 r >= 0.9': (med(c, 'V1_r') is not None and med(c, 'V1_r') >= .9) for c in ('walk', 'run')}
    checks.update({f'{c} V1 RMSE <= 0.15 BW': (med(c, 'V1_rmse_bw') is not None and med(c, 'V1_rmse_bw') <= .15) for c in ('walk', 'run')})
    checks.update({f'{c} V3 >= 0.85': (med(c, 'V3_balanced_accuracy_20N') is not None and med(c, 'V3_balanced_accuracy_20N') >= .85)
                   for c in ('walk', 'run', 'jump')})
    return {'runs': len(rows), 'success': len(ok), 'failed': [r['id'] for r in rows if r['status'] != 'success'], 'classes': classes,
            'threshold': tau, 'gate': {'checks': checks, 'pass': bool(all(checks.values()))}}


def contact_any(name, r, f):
    if 'contact_any' in f:
        return f['contact_any']
    d = np.load(WORK/name/r['id']/'dynamics.npz')
    dt = np.median(np.diff(d['times'])); trim = slice(int(.5/dt), len(d['times'])-int(.5/dt))
    return np.any([d[f'contact_{s}_{k}'][trim] for s in 'rl' for k in ('heel', 'toe')], axis=0)


def state_threshold(name, keep_ids, scope_tasks=()):
    ex = AMEND['threshold']['exclusions']
    rows = [json.loads(f.read_text()) for f in sorted((LOG/'runs'/name).glob('*.json'))]
    rows = [r for r in rows if r['id'] in keep_ids and r['status'] == 'success' and r['class'] in THRESHOLD_CLASSES]
    excluded = collections.Counter(); kept = collections.defaultdict(list)
    for r in rows:
        if r['task'] in scope_tasks or (scope_tasks and r['class'] in AMEND2['scope']['excluded_classes']):
            excluded[(r['class'], 'scope')] += 1
        elif r['task'] in ex['unmodeled_support_tasks']:
            excluded[(r['class'], 'unmodeled_support')] += 1
        elif r['ik_marker_error']['rms_mean_m'] > .03:
            excluded[(r['class'], 'ik_rms_over_3cm')] += 1
        else:
            kept[r['class']].append(r)
    out = {'kept_trials': {c: len(v) for c, v in kept.items()}, 'excluded_trials': {f'{c}/{k}': n for (c, k), n in excluded.items()}}
    per = {}
    for state in ('contact', 'flight'):
        values, weights, per[state] = [], [], {}
        for c, v in kept.items():
            x = []
            for r in v:
                f = np.load(LOG/'runs'/name/f"{r['id']}_frames.npz")
                ca = contact_any(name, r, f)
                sel = f['keep'] & (ca if state == 'contact' else ~ca)
                x.append(f['residual_bw'][sel])
            x = np.concatenate(x)
            if len(x):
                values.append(x); weights.append(np.full(len(x), 1/len(x)))
                per[state][c] = {'frames': int(len(x)), 'p99_bw': float(np.percentile(x, 99))}
        v, w = np.concatenate(values), np.concatenate(weights)
        out[f'tau_{state}_bw'] = weighted_quantile(v, w, .99)
        out[f'{state}_share_above_5pct_bw_equal_class_weight'] = float(np.sum(w*(v > .05))/w.sum())
    rms = [r['V4_residual_rms_bw'] for v in kept.values() for r in v if r.get('V4_residual_rms_bw') is not None]
    out['tau_run_bw'] = float(np.percentile(rms, 95)); out['per_class'] = per
    return out


def amendment():
    reg, held = ids('selection.json'), ids('selection_heldout.json')
    main = variant('kin30_a1', reg)
    h = variant('kin30_a1', held)
    hv3 = (h['classes'].get('walk', {}).get('V3_balanced_accuracy_20N') or {}).get('median')
    checks = {**main['gate']['checks'], 'held-out walk V3 >= 0.85': bool(hv3 is not None and hv3 >= .85)}
    return {'registered_trials': main, 'heldout_walk': h, 'gate': {'checks': checks, 'pass': bool(all(checks.values()))},
            'threshold': state_threshold('kin30_a1', reg)}


def frame_states(name, r, f, window_ms):
    d = np.load(WORK/name/r['id']/'dynamics.npz')
    dt = np.median(np.diff(d['times'])); trim = slice(int(.5/dt), len(d['times'])-int(.5/dt))
    foot = [(d[f'contact_{s}_heel'] | d[f'contact_{s}_toe'])[trim] for s in 'rl']
    events = np.concatenate([np.flatnonzero(np.diff(x.astype(int)) != 0)+.5 for x in foot])
    i = np.arange(len(foot[0]))
    near = (np.min(np.abs(i[:, None]-events[None, :]), 1)*dt*1000 <= window_ms) if len(events) else np.zeros(len(i), bool)
    contact = foot[0] | foot[1]
    states = {'event': near, 'contact': ~near & contact, 'flight': ~near & ~contact}
    return {k: v[f['keep']] for k, v in states.items()}, len(events)/(len(i)*dt)


def three_state(name, keep_ids, scope_tasks, window_ms):
    ex = AMEND['threshold']['exclusions']
    rows = [json.loads(f.read_text()) for f in sorted((LOG/'runs'/name).glob('*.json'))]
    kept = collections.defaultdict(list)
    for r in rows:
        if (r['id'] in keep_ids and r['status'] == 'success' and r['class'] in THRESHOLD_CLASSES and r['task'] not in scope_tasks
                and r['class'] not in AMEND2['scope']['excluded_classes'] and r['task'] not in ex['unmodeled_support_tasks']
                and r['ik_marker_error']['rms_mean_m'] <= .03):
            kept[r['class']].append(r)
    by = {k: collections.defaultdict(list) for k in ('event', 'contact', 'flight')}; rate = collections.defaultdict(list)
    for c, v in kept.items():
        for r in v:
            f = np.load(LOG/'runs'/name/f"{r['id']}_frames.npz")
            st, ev = frame_states(name, r, f, window_ms); rate[c].append(ev)
            for k, m in st.items():
                by[k][c].append(f['residual_bw'][f['keep']][m])
    minimum = AMEND3['threshold']['min_frames_per_class_state']
    out = {'window_ms': window_ms, 'kept_trials': {c: len(v) for c, v in kept.items()},
           'events_per_second': {c: {'median': float(np.median(x)), 'max': float(np.max(x))} for c, x in rate.items()},
           'events_per_second_max_all': float(max(max(x) for x in rate.values())), 'tau_bw': {}, 'per_class': {}, 'human_false_infeasible': {}}
    for k, d in by.items():
        xs = {c: np.concatenate(v) for c, v in d.items()}
        xs = {c: x for c, x in xs.items() if len(x) >= minimum}
        out['per_class'][k] = {c: {'frames': int(len(x)), 'p99_bw': float(np.percentile(x, 99))} for c, x in xs.items()}
        if xs:
            v = np.concatenate(list(xs.values())); w = np.concatenate([np.full(len(x), 1/len(x)) for x in xs.values()])
            tau = weighted_quantile(v, w, .99); out['tau_bw'][k] = tau
            out['human_false_infeasible'][k] = {c: float(np.mean(x > tau)) for c, x in xs.items()}
            out.setdefault('tau_envelope_bw', {})[k] = float(max(np.percentile(x, 99) for x in xs.values()))
    return out


def amendment3(name='kin30'):
    reg = ids('selection.json')
    yoga = tuple(AMEND2['scope']['excluded_tasks']['yoga'])
    return {f'window_{w}ms': three_state(name, reg, yoga, w) for w in AMEND3['threshold']['window_ms']}


def amendment4():
    rows = [json.loads(f.read_text()) for f in sorted((LOG/'runs/kin30_a4').glob('*.json'))]
    reg = ids('selection.json')
    dev_walk = {r['id'] for r in rows if r['class'] == 'walk' and 'tmf' not in r['task']}
    jump = {r['id'] for r in rows if r['class'] == 'jump' and r['id'] in reg
            and r['task'] not in AMEND['threshold']['exclusions']['unmodeled_support_tasks']}
    v = variant('kin30_a4', dev_walk | jump)
    med = lambda c, m: (v['classes'].get(c, {}).get(m) or {}).get('median')
    checks = {'dev walk V3 >= 0.85': bool((med('walk', 'V3_balanced_accuracy_20N') or 0) >= .85),
              'dev jump V3 >= 0.85': bool((med('jump', 'V3_balanced_accuracy_20N') or 0) >= .85),
              'dev walk V1 r >= 0.9': bool((med('walk', 'V1_r') or 0) >= .9),
              'dev walk V1 RMSE <= 0.15 BW': bool(med('walk', 'V1_rmse_bw') is not None and med('walk', 'V1_rmse_bw') <= .15)}
    grid = json.loads((LOG/'amend4_dev_grid.json').read_text())
    held = LOG/'amend4_heldout.json'
    out = {'v_h': grid['selected_v_h'], 'development': {'walk_trials': len(dev_walk), 'jump_trials': len(jump), 'classes': v['classes']},
           'development_checks': checks, 'thresholds': amendment3('kin30_a4')}
    if held.exists():
        h = json.loads(held.read_text())
        out['heldout'] = {k: h[k] for k in ('trials', 'subjects', 'median_v3', 'bridge') if k in h}
        checks['held-out walk V3 >= 0.85'] = bool(h.get('median_v3') is not None and h['median_v3'] >= .85)
        checks['bridge median |dV3| <= 0.02'] = bool(h.get('bridge', {}).get('median_abs_diff', 1) <= .02)
    out['gate'] = {'complete': held.exists(), 'pass': bool(held.exists() and all(checks.values()))}
    return out


def amendment2():
    reg = ids('selection.json')
    yoga = tuple(AMEND2['scope']['excluded_tasks']['yoga'])
    base = variant('kin30', reg)
    checks = {k: v for k, v in base['gate']['checks'].items() if not k.startswith('run ')}
    return {'gate': {'checks': checks, 'pass': bool(all(checks.values()))},
            'threshold': state_threshold('kin30', reg, yoga),
            'threshold_without_ballet': state_threshold('kin30', reg, yoga+BALLET)}


def main():
    result = {'protocol': SPEC['protocol'], 'variants': {}}
    for name in ('kin30', 'dyn30', 'kin_native'):
        if (LOG/'runs'/name).exists():
            result['variants'][name] = variant(name)
    if (LOG/'runs/kin30_a1').exists() and (LOG/'selection_heldout.json').exists():
        result['amendment1'] = amendment()
    result['amendment2'] = amendment2()
    result['amendment3'] = amendment3()
    if (LOG/'runs/kin30_a4').exists():
        result['amendment4'] = amendment4()
    (LOG/'analysis.json').write_text(json.dumps(result, indent=1))
    for name, v in result['variants'].items():
        print(f"== {name}: {v['success']}/{v['runs']} success; gate pass {v['gate']['pass']}", v['gate']['checks'])
        for c, s in v['classes'].items():
            g = lambda m: '   -  ' if s[m] is None else f"{s[m]['median']:6.3f}"
            print(f"  {c:9s} n={s['trials']:2d} V1 r {g('V1_r')} rmse {g('V1_rmse_bw')} bias {g('V1_bias_bw')} | V2 r {g('V2_r')} rmse {g('V2_rmse_bw')}"
                  f" | V3 {g('V3_balanced_accuracy_20N')} | resid rms {g('V4_residual_rms_bw')} >5% {g('V4_share_above_5pct')} p95 {g('V4_residual_p95_bw')}"
                  f" | meas Fy {g('measured_vertical_mean_bw')}")
        print('  threshold', json.dumps(v['threshold']))
    if 'amendment1' in result:
        a = result['amendment1']
        print('== amendment 1: gate pass', a['gate']['pass'], a['gate']['checks'])
        for tag, v in (('registered trials', a['registered_trials']), ('held-out walk', a['heldout_walk'])):
            for c, s in v['classes'].items():
                g = lambda m: '   -  ' if s[m] is None else f"{s[m]['median']:6.3f}"
                print(f"  {tag:17s} {c:9s} n={s['trials']:2d} V1 r {g('V1_r')} rmse {g('V1_rmse_bw')} | V3 {g('V3_balanced_accuracy_20N')}"
                      f" | resid rms {g('V4_residual_rms_bw')} >5% {g('V4_share_above_5pct')}")
        print('  threshold', json.dumps(a['threshold']))
    a = result['amendment2']
    print('== amendment 2: gate pass', a['gate']['pass'], a['gate']['checks'])
    for k in ('threshold', 'threshold_without_ballet'):
        print(' ', k, json.dumps(a[k]))
    for k, v in result['amendment3'].items():
        print('== amendment 3', k, json.dumps(v))
    if 'amendment4' in result:
        a = result['amendment4']
        print('== amendment 4: v_h', a['v_h'], 'checks', a['development_checks'], 'gate', a['gate'])
        for c, x in a['development']['classes'].items():
            g = lambda m: '   -  ' if x[m] is None else f"{x[m]['median']:6.3f}"
            print(f"  dev {c:6s} n={x['trials']:2d} V1 r {g('V1_r')} rmse {g('V1_rmse_bw')} | V3 {g('V3_balanced_accuracy_20N')} | resid rms {g('V4_residual_rms_bw')}")
        for k, v in a['thresholds'].items():
            print('  thresholds', k, 'envelope', {s: round(t, 3) for s, t in v['tau_envelope_bw'].items()}, 'events/s max', round(v['events_per_second_max_all'], 2))


if __name__ == '__main__':
    main()
