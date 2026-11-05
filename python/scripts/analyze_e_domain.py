#!/usr/bin/env python
"""Summary of the E criterion domain validation (config/e_criterion_domain_20261105.json).

Reads develop_logs/test_runs/20261105_e_domain/{static,inject,moco,bracket} and writes analysis.json there.
Sections whose runs are missing are reported as incomplete. OpenSim python.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils/motiogen'), str(ROOT/'utils'), str(ROOT/'python/scripts')]
BASE = ROOT/'develop_logs/test_runs/20261105_e_domain'
SPEC = json.loads((ROOT/'config/e_criterion_domain_20261105.json').read_text())
RULES = ('R0', 'R4', 'R5')
DETECT = .10
REPS, SEED = 10000, 20261105
BANDS = ((-np.inf, -.10), (-.10, -.05), (-.05, -.02), (-.02, .02), (.02, .05), (.05, np.inf))


def rate(x):
    x = list(x)
    return {'n': len(x), 'share': float(np.mean(x)) if x else None}


def boot_ci(means):
    means = np.asarray(means, float)
    rng = np.random.default_rng(SEED)
    b = means[rng.integers(0, len(means), (REPS, len(means)))].mean(1)
    return np.percentile(b, [2.5, 97.5]).tolist()


# ---------------------------------------------------------------- static
def hull_margin_e(work, kin_cache={}):
    """Diagnostic (post hoc): mean signed distance of the COM projection to the hull of E's R0 contact points."""
    import e_rules
    import e_domain_static as st
    times, _, _, kin = e_rules.kinematics(work)
    res = e_rules.evaluate(times, kin, 75., 'R0'); sel = e_rules.trim(times)
    import biomech
    pts = {s: biomech.sole_points(kin, s)[0] for s in 'rl'}
    out = []
    for t in np.flatnonzero(sel):
        p = [q[t][:, [0, 2]] for s in 'rl' for k, q in pts[s].items() if res['masks'][s][k][t]]
        poly = st.hull(np.concatenate(p)) if p else np.zeros((0, 2))
        out.append(st.signed_margin(res['com'][t, [0, 2]], poly))
    return float(np.mean(out))


def static():
    files = sorted((BASE/'static').glob('*.json'))
    rows = [json.loads(f.read_text()) for f in files]
    ok = [r for r in rows if r['status'] == 'success']
    out = {'poses': len(rows), 'success': len(ok), 'failed': [r['id'] for r in rows if r['status'] != 'success']}
    flag = lambda r, rule: r['rules'][rule]['envelope_infeasible_share'] > .5
    for rule in RULES:
        res = {'detection_truth_infeasible': rate(flag(r, rule) for r in ok if r['truth']['label'] == 'infeasible'),
               'false_alarm_truth_feasible': rate(flag(r, rule) for r in ok if r['truth']['label'] == 'feasible'),
               'flag_indeterminate': rate(flag(r, rule) for r in ok if r['truth']['label'] == 'indeterminate')}
        groups = collections.defaultdict(list)
        for r in ok:
            t = r['truth']; m = t['margin_m']
            band = 'no_support' if m is None else next(f'[{a:g},{b:g})' for a, b in BANDS if a <= m < b)
            groups[('band', band)].append(flag(r, rule))
            groups[('support', t['support'])].append(flag(r, rule))
            groups[('source_label', f"{r['source']} {'hover' if r.get('hover_m') else t['label']}")].append(flag(r, rule))
        for (kind, name), v in sorted(groups.items()):
            res.setdefault(f'flag_by_{kind}', {})[name] = rate(v)
        res['residual_force_rms_bw_max'] = float(max(r['rules'][rule]['residual_force_rms_bw'] for r in ok))
        out[rule] = res
    pitch = {src: [max(abs(v) for v in r['truth']['heel_minus_toe_m'].values()) for r in ok if r['source'] == src and not r.get('hover_m')]
             for src in ('human', 'generated')}
    out['foot_pitch_heel_minus_toe_abs_m'] = {s: np.percentile(v, [25, 50, 75]).tolist() for s, v in pitch.items()}
    out['truth_label_counts'] = dict(collections.Counter(f"{r['source']} {'hover' if r.get('hover_m') else r['truth']['label']}" for r in ok))
    disk = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_domain_20261105/static/work')
    cache = BASE/'static_hull_diagnostic.json'
    hull = json.loads(cache.read_text()) if cache.exists() else {}
    for r in ok:
        if r['id'] not in hull:
            hull[r['id']] = hull_margin_e(disk/r['id'])
    cache.write_text(json.dumps(hull, indent=1))
    inside = [r for r in ok if hull[r['id']] >= 0]; outside = [r for r in ok if hull[r['id']] < 0]
    out['diagnostic_e_contact_hull'] = {
        'note': 'post hoc: COM projection against the hull of the sole points E itself counts as contact (R0); explains the static blind spot',
        'com_inside_e_hull': {'n': len(inside), 'truth_infeasible': sum(r['truth']['label'] == 'infeasible' for r in inside),
                              'flagged_R0': sum(flag(r, 'R0') for r in inside)},
        'com_outside_e_hull': {'n': len(outside), 'flagged_R0': sum(flag(r, 'R0') for r in outside),
                               'margins_m': sorted(round(hull[r['id']], 3) for r in outside)}}
    return out


# ---------------------------------------------------------------- inject
def window_metrics(z, rule, sel):
    if not sel.any():
        return None
    c = z[f'{rule}_contact'][sel]
    return {'envelope_infeasible_share': float(z[f'{rule}_infeasible'][sel].mean()),
            'residual_force_rms_bw': float(np.sqrt(np.mean(z[f'{rule}_residual_bw'][sel]**2))),
            'foot_skate_share': float(z[f'{rule}_skate'][sel].sum()/c.sum()) if c.sum() else None,
            'landmark_contact_share': float(c.mean())}


def trim_sel(t, s):
    return (t >= t[0]+s) & (t <= t[-1]-s)


def inject():
    jobs = json.loads((BASE/'inject_jobs.json').read_text())
    rows = {j['id']: json.loads((BASE/'inject'/f"{j['id']}.json").read_text()) for j in jobs if (BASE/'inject'/f"{j['id']}.json").exists()}
    out = {'jobs': len(jobs), 'done': len(rows), 'status': dict(collections.Counter(r['status'] for r in rows.values())),
           'failed': [k for k, r in rows.items() if r['status'] == 'failed']}
    deltas = collections.defaultdict(list)
    for j in jobs:
        r = rows.get(j['id']); c = rows.get(f"{j['base']}__control")
        if j['defect'] == 'control' or not r or not c or r['status'] != 'success' or c['status'] != 'success':
            continue
        z = np.load(BASE/'inject'/f"{j['id']}.npz"); zc = np.load(BASE/'inject'/f"{j['base']}__control.npz")
        t, tc = z['times'], zc['times']
        if r['window_s']:
            a, b = r['window_s']; sel, selc = (t >= a) & (t < b), (tc >= a) & (tc < b)
        else:
            s = .25 if j['defect'] == 'time_compression' else .5
            sel, selc = trim_sel(t, s), trim_sel(tc, s)
        for rule in RULES:
            m, mc = window_metrics(z, rule, sel), window_metrics(zc, rule, selc)
            if m is None or mc is None:
                continue
            d = {k: (None if m[k] is None or mc[k] is None else m[k]-mc[k]) for k in m}
            d['detected'] = d['envelope_infeasible_share'] >= DETECT
            deltas[(j['defect'], j['size'], rule)].append((j['class'], d, m, mc))
    table = {}
    for (defect, size, rule), v in sorted(deltas.items()):
        row = {}
        for cls in ('all', 'walk', 'squat'):
            vv = [d for c, d, *_ in v if cls in ('all', c)]
            if not vv:
                continue
            med = lambda k: float(np.median([d[k] for d in vv if d[k] is not None])) if any(d[k] is not None for d in vv) else None
            row[cls] = {'n': len(vv), 'detection_rate': float(np.mean([d['detected'] for d in vv])),
                        'median_delta_envelope_share': med('envelope_infeasible_share'), 'median_delta_residual_rms_bw': med('residual_force_rms_bw'),
                        'median_delta_foot_skate_share': med('foot_skate_share'), 'median_delta_contact_share': med('landmark_contact_share')}
        table.setdefault(f'{defect} {size:g}', {})[rule] = row
    out['table'] = table
    ctl = collections.defaultdict(list)
    for k, r in rows.items():
        if r['defect'] == 'control' and r['status'] == 'success':
            z = np.load(BASE/'inject'/f'{k}.npz')
            for rule in RULES:
                ctl[(rule, r['class'])].append(window_metrics(z, rule, trim_sel(z['times'], .5))['envelope_infeasible_share'])
    out['control_envelope_share_median'] = {f'{a} {b}': float(np.median(v)) for (a, b), v in sorted(ctl.items())}
    return out


# ---------------------------------------------------------------- moco
def moco():
    out = {}
    for d in sorted((BASE/'moco').glob('*')):
        solve = json.loads((d/'solve.json').read_text()) if (d/'solve.json').exists() else None
        ev = json.loads((d/'evaluation.json').read_text()) if (d/'evaluation.json').exists() else None
        out[d.name] = {'solve': solve, 'evaluation': None if ev is None else
                       {k: ev[k] for k in ('frames', 'moco_vertical_grf_mean_bw', 'ik')} |
                       {'rules': {r: {k: ev['rules'][r][k] for k in ('envelope_infeasible_share', 'residual_force_rms_bw',
                                                                     'unsupported_fraction_5pct_bw', 'vertical_grf', 'contact_agreement',
                                                                     'contact_confusion', 'foot_skate_share')} for r in RULES}}}
    return out


# ---------------------------------------------------------------- bracket
METRICS = ('residual_force_rms_bw', 'unsupported_fraction_5pct_bw', 'envelope_infeasible_share', 'reba_r3_certain_acceptable_ungated')


def flat(rule_row):
    out = {k: rule_row[k] for k in METRICS[:3]}
    out['reba_r3_certain_acceptable_ungated'] = rule_row['reba_r3_certain_acceptable']['ungated']
    return out


def load_bracket(set_name):
    rows = {}
    for f in (BASE/'bracket'/set_name).glob('*.json'):
        r = json.loads(f.read_text())
        if r['status'] == 'success':
            rows[r['id']] = r
    return rows


def paired_interval(values, clusters):
    keys = sorted(set(clusters))
    means = [np.mean([v for v, c in zip(values, clusters) if c == k]) for k in keys]
    if len(means) < 2:
        return {'pairs': len(values), 'clusters': len(means)}
    ci = boot_ci(means)
    return {'pairs': len(values), 'clusters': len(means), 'mean': float(np.mean(means)), 'ci95': ci,
            'excludes_zero': bool(ci[0] > 0 or ci[1] < 0)}


def comparisons(set_name, rows):
    """{name: {metric: {rule: interval}}} for the registered comparisons of each experiment."""
    out = {}
    if set_name == 'e4':
        import analyze_e4 as a4
        _, runs, _, _ = a4.load()
        for method, mspec in a4.SPEC['selection']['methods'].items():
            for factor, fixed in mspec['contrasts']:
                name = f'{method} {factor} | {a4.key(fixed)}'
                out[name] = {}
                for metric in METRICS:
                    out[name][metric] = {}
                    for rule in RULES:
                        sub = {}
                        for cond, by in runs.items():
                            if cond[0] != method:
                                continue
                            sub[cond] = {p: {**r, 'metrics': flat(rows[r['id']]['rules'][rule])} for p, r in by.items() if r['id'] in rows}
                        out[name][metric][rule] = a4.contrast(sub, method, factor, fixed, metric, True)
        return out
    key = (lambda r: (r['method'], r['task'], r['seed'])) if set_name == 'x3' else \
          (lambda r: (r['method'], r['task'], r['seed'], r['offset_m'], r.get('info', {}).get('epsilon_m'), r.get('info', {}).get('block')))
    by = collections.defaultdict(dict)
    for r in rows.values():
        by[key(r)][r['arm']] = r
    arms = sorted({r['arm'] for r in rows.values()} - {'C0'})
    methods = sorted({r['method'] for r in rows.values()})
    for method in methods:
        for arm in arms:
            name = f'{method} {arm} - C0'
            pairs = [(k, v[arm], v['C0']) for k, v in by.items() if k[0] == method and arm in v and 'C0' in v and v[arm]['valid'] and v['C0']['valid']]
            if not pairs:
                continue
            out[name] = {}
            for metric in METRICS:
                out[name][metric] = {}
                for rule in RULES:
                    vals = [flat(a['rules'][rule])[metric]-flat(b['rules'][rule])[metric] for _, a, b in pairs]
                    out[name][metric][rule] = paired_interval(vals, [k[1] for k, *_ in pairs])
    return out


def consistency(comp):
    out = {}
    for name, metrics in comp.items():
        for metric, by_rule in metrics.items():
            res = [by_rule[r] for r in RULES]
            if any('mean' not in r for r in res):
                verdict = 'not_testable'
            else:
                signs = {np.sign(r['mean']) for r in res}
                if all(r['excludes_zero'] for r in res) and len(signs) == 1:
                    verdict = 'conclusion'
                elif not any(r['excludes_zero'] for r in res):
                    verdict = 'null_under_all_rules'
                else:
                    verdict = 'rule_dependent'
            out[f'{name} :: {metric}'] = {'verdict': verdict, 'R0_excludes_zero': res[0].get('excludes_zero'),
                                         **{r: {k: by_rule[r].get(k) for k in ('mean', 'ci95', 'pairs')} for r in RULES}}
    return out


def bracket():
    out = {}
    for set_name in ('e4', 'x3', 'm1', 'e1'):
        rows = load_bracket(set_name)
        if not rows:
            out[set_name] = {'runs': 0}
            continue
        r0 = [r['rules']['R0']['r0_matches_original'] for r in rows.values()]
        desc = {}
        for rule in RULES:
            by_m = collections.defaultdict(list)
            for r in rows.values():
                if r['valid']:
                    by_m[r['method']].append(r['rules'][rule])
            desc[rule] = {m: {k: float(np.median([x[k] for x in v if x[k] is not None])) for k in
                              ('residual_force_rms_bw', 'envelope_infeasible_share', 'foot_skate_share', 'landmark_contact_share', 'contact_lost_vs_R0')}
                          for m, v in sorted(by_m.items())}
        comp = comparisons(set_name, rows)
        cons = consistency(comp)
        out[set_name] = {'runs': len(rows), 'r0_recompute_max_abs_diff': float(max(r0)), 'descriptive_valid_median': desc,
                         'verdict_counts': dict(collections.Counter(v['verdict'] for v in cons.values())),
                         'R0_significant_but_rule_dependent': [k for k, v in cons.items() if v['R0_excludes_zero'] and v['verdict'] == 'rule_dependent'],
                         'consistency': cons}
    return out


def main():
    sections = sys.argv[1:] or ['static', 'inject', 'moco', 'bracket']
    path = BASE/'analysis.json'
    result = json.loads(path.read_text()) if path.exists() else {}
    result['protocol'] = SPEC['protocol']
    for s in sections:
        result[s] = globals()[s]()
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    for s in sections:
        print(s, json.dumps(result[s], ensure_ascii=False, default=float)[:3000])


if __name__ == '__main__':
    main()
