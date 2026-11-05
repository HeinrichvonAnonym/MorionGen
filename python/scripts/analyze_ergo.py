#!/usr/bin/env python
"""RULA/REBA analysis (config/x3_cross_method_layers_20261104_amend3.json, config/ergo_rula_reba_20261105.json).

--set x3 --stage confirm|top_up: registered ergo family R1-R3 per method (Holm within method), pairs and
gates exactly as the E tests of analyze_x3. --set e4: descriptive component contrasts, no tests.
--revision r3: the same contrasts on the revision 3 bounds (post hoc for X3/E4/E1): primary REBA r3 certain-acceptable
share; the registered metric, the common version (r3_common: neck and shoulder at their minimum for all methods) and
the repeated-action upper bound (r3_repeat1) are reported as sensitivity.
Writes develop_logs/test_runs/20261105_ergo/analysis_<set>[_<stage>][_r3].json.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils')); sys.path.insert(0, str(ROOT/'python/scripts'))
from motiogen.native_analysis import holm, sign_flip_p
from motiogen.runtime import json_write

ERGO = ROOT/'develop_logs/test_runs/20261105_ergo'
PRIMARY = 'reba_acceptable_lower'
CLASSES = ('infeasible', 'certain_risk', 'indeterminate', 'certain_acceptable')
SENSITIVITY = ['reba_conditional_certain_risk', 'reba_ungated_lower_acceptable', 'reba_certain_acceptable_share',
               'rula_acceptable_lower', 'reba_neck_source_acceptable_lower']
CLASS_PREFIX = 'reba'
REPORTED = [PRIMARY, *[f'reba_{c}_share' for c in CLASSES], *SENSITIVITY, 'feasible_share']


def use_r3():
    global PRIMARY, SENSITIVITY, CLASS_PREFIX, REPORTED
    PRIMARY, CLASS_PREFIX = 'reba_r3_certain_acceptable_share', 'reba_r3'
    SENSITIVITY = ['reba_r3_common_certain_acceptable_share', 'reba_r3_repeat1_certain_acceptable_share', 'reba_r3_acceptable_lower', 'reba_r3_conditional_certain_risk',
                   'reba_r3_ungated_certain_acceptable', 'rula_r3_certain_acceptable_share', 'reba_acceptable_lower']
    REPORTED = [PRIMARY, *[f'reba_r3_{c}_share' for c in CLASSES], *SENSITIVITY, 'feasible_share']


def ergo_rows(set_name):
    rows = {}
    for f in sorted((ERGO/set_name).glob('*.json')):
        r = json.loads(f.read_text())
        if r.get('status') == 'success':
            rows[r['id']] = r
    return rows


def flat(r):
    out = {'feasible_share': r['feasible_share'], 'ergo_frames': r['frames']}
    for m in ('reba', 'rula', 'reba_r3', 'rula_r3', 'reba_r3_common', 'rula_r3_common', 'reba_r3_repeat1'):
        out.update({f'{m}_{k}': v for k, v in r.get(m, {}).items()})
    # Registered revision: neck-from-source needs skeleton-neutral offsets; not reported until computed.
    out.pop('reba_neck_source_acceptable_lower', None); out.pop('rula_neck_source_acceptable_lower', None)
    return out


def pooled_classes(rows):
    frames = np.array([r['ergo_frames'] for r in rows], float)
    if not len(frames):
        return None
    return {c: float(np.sum([r[f'{CLASS_PREFIX}_{c}_share'] for r in rows]*frames)/frames.sum()) for c in CLASSES}


def x3(stage):
    import analyze_x3 as ax
    reg = ax.xt.prereg()
    table, _, missing = ax.collect(stage)
    rows = ergo_rows('x3'); joined = 0
    for (method, arm, d, task, seed), row in table.items():
        e = rows.get(f'{method}_t{task:02d}_s{seed}_{arm}') if d == 1.0 else None
        if e:
            row.update(flat(e)); joined += 1
    result = {'stage': stage, 'protocol': 'x3 amendment 3 ergo family', 'joined_runs': joined, 'methods': {}}
    for method in reg['methods']:
        T = reg['methods'][method]['T_arm']
        contrasts = {'R1': (T, 'C0'), 'R2': ('O', T)}
        if method in ('dart', 'omnicontrol'):
            contrasts['R3'] = ('S', T)
        tests = {k: ax.paired(table, method, on, off, PRIMARY, [1.0], valid=True) for k, (on, off) in contrasts.items()}
        names = [k for k, r in tests.items() if r is not None and 'sign_flip_p' in r]
        adjusted = dict(zip(names, holm([tests[k]['sign_flip_p'] for k in names]))) if names else {}
        c0 = [r for (m, a, d, *_), r in table.items() if m == method and a == 'C0' and d == 1.0 and PRIMARY in r
              and r.get('E_valid')]
        pooled = pooled_classes(c0)
        insensitive = bool(pooled and max(pooled.values()) >= .9)
        out = {}
        for k, (on, off) in contrasts.items():
            r = tests[k]
            if r is None or k not in adjusted:
                out[k] = {'contrast': f'{on} - {off}', 'status': 'not_testable', 'result': r}; continue
            excludes = r['ci95'][0] > 0 or r['ci95'][1] < 0
            out[k] = {**r, 'contrast': f'{on} - {off}', 'holm_p': adjusted[k],
                      'supported': bool(adjusted[k] < .05 and excludes and not insensitive),
                      'direction': 'higher' if r['mean'] > 0 else 'lower',
                      'all_pairs_sensitivity': ax.paired(table, method, on, off, PRIMARY, [1.0], valid=False, fk_gate=False),
                      'sensitivity': {s: ax.paired(table, method, on, off, s, [1.0], valid=True) for s in SENSITIVITY},
                      'class_shares': {c: ax.paired(table, method, on, off, f'{CLASS_PREFIX}_{c}_share', [1.0], valid=True) for c in CLASSES}}
        secondary = {'kimodo': [('TO', 'T')], 'omnicontrol': [('TS', 'T')], 'dart': [('W0', 'G4')]}.get(method, [])
        arms = sorted({a for (m, a, d, *_), r in table.items() if m == method and d == 1.0 and PRIMARY in r})
        result['methods'][method] = {
            'family': out, 'insensitive_at_walking': insensitive, 'c0_pooled_classes': pooled,
            'secondary': {f'{a} - {b}': ax.paired(table, method, a, b, PRIMARY, [1.0], valid=True) for a, b in secondary},
            'levels': {a: {m: ax.levels(table, method, a, 1.0, m) for m in REPORTED} for a in arms}}
    result['missing_kinematic_cells'] = len(missing)
    return ax.finite(result)


def e4():
    import analyze_e4 as a4
    _, runs, _, _ = a4.load()
    rows = ergo_rows('e4')
    for cell in runs.values():
        for m in cell.values():
            e = rows.get(m['id'])
            m['ergo'] = flat(e) if e else None
    rng = np.random.default_rng(a4.SEED)
    result = {'protocol': 'E4 descriptive RULA/REBA (amendment 3)', 'note': 'methods ran different tasks; levels are not a method ranking',
              'scored_runs': len(rows), 'levels': {}, 'contrasts': {}}
    for method, spec in a4.SPEC['selection']['methods'].items():
        for cond in spec['conditions']:
            cell = [m['ergo'] for m in runs.get((method, a4.key(cond)), {}).values() if m['ergo'] and m['valid']]
            result['levels'][f'{method} {a4.key(cond)}'] = {
                'runs': len(cell), **{k: (float(np.median([c[k] for c in cell if c.get(k) is not None])) if any(c.get(k) is not None for c in cell) else None)
                                       for k in REPORTED}}
        for factor, fixed in spec['contrasts']:
            on, off = a4.key({**fixed, factor: 1}), a4.key({**fixed, factor: 0})
            a, b = runs.get((method, on), {}), runs.get((method, off), {})
            pairs = [p for p in sorted(set(a) & set(b)) if a[p]['ergo'] and b[p]['ergo'] and a[p]['valid'] and b[p]['valid']]
            entry = {}
            for metric in REPORTED:
                by = collections.defaultdict(list)
                for p in pairs:
                    if a[p]['ergo'].get(metric) is not None and b[p]['ergo'].get(metric) is not None:
                        by[a[p]['cluster']].append(a[p]['ergo'][metric]-b[p]['ergo'][metric])
                means = [np.mean(v) for v in by.values()]
                if len(means) >= 2:
                    entry[metric] = {'mean': float(np.mean(means)), 'ci95': a4.bootstrap(means, rng), 'p_sign_flip': sign_flip_p(means)['p'],
                                     'clusters': len(means), 'pairs': sum(map(len, by.values()))}
            result['contrasts'][f'{method} {factor} | {a4.key(fixed)}'] = entry
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', required=True, choices=['x3', 'e4'])
    p.add_argument('--stage', default='confirm', choices=['pilot', 'confirm', 'top_up'])
    p.add_argument('--revision', default='registered', choices=['registered', 'r3'])
    args = p.parse_args()
    if args.revision == 'r3':
        use_r3()
    if args.set == 'x3':
        result = x3(args.stage); name = f'analysis_x3_{args.stage}.json'
        for method, r in result['methods'].items():
            print(method, 'insensitive' if r['insensitive_at_walking'] else 'sensitive', r['c0_pooled_classes'])
            for k, t in r['family'].items():
                print(' ', k, t['contrast'], t.get('mean'), t.get('ci95'), t.get('holm_p'), t.get('supported'), t.get('status', ''))
    else:
        result = e4(); name = 'analysis_e4.json'
        for k, c in result['contrasts'].items():
            v = c.get(PRIMARY)
            print(k, None if v is None else (round(v['mean'], 3), [round(x, 3) for x in v['ci95']], round(v['p_sign_flip'], 4)))
    if args.revision == 'r3':
        result['revision'] = 'r3 (post hoc, config/ergo_rula_reba_20261105.json)'; name = name.replace('.json', '_r3.json')
    json_write(ERGO/name, result)


if __name__ == '__main__':
    main()
