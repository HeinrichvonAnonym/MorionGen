#!/usr/bin/env python
"""ARDY fair-target control (config/ardy_fair_20261105.json): registered F1-F4, per-offset contrasts,
descriptive RULA/REBA levels of the E1 M2 arms, and F3/F4 recomputed on the revision 3 bounds (post hoc, outside the Holm family). Writes develop_logs/test_runs/20261105_ardy_fair/analysis.json.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils')); sys.path.insert(0, str(ROOT/'python/scripts'))
import analyze_ergo as ae
from analyze_e1 import METRICS
from analyze_next_round import test
from motiogen.native_analysis import holm
from motiogen.runtime import json_write

SPEC = json.loads((ROOT/'config/ardy_fair_20261105.json').read_text(encoding='utf-8'))
OUT = ROOT/'develop_logs/test_runs/20261105_ardy_fair'
OFFSETS = SPEC['data']['offsets_m']
RES = 'residual_force_rms_bw'
ERG = ae.PRIMARY
POST_HOC = ('reba_r3_certain_acceptable_share', 'reba_r3_common_certain_acceptable_share')
REPORTED = [*METRICS, *ae.REPORTED, *POST_HOC]


def load(mode='registered'):
    """(arm, offset, task, seed) -> E metrics + ergo values; mode 'registered' keeps valid runs only."""
    ergo = ae.ergo_rows('m1')
    table, invalid, missing_ergo = {}, [], []
    for path in sorted((OUT/'e/runs').glob('*/metrics.json')):
        row = json.loads(path.read_text()); info = row['m1']
        if mode == 'registered' and not row['valid']:
            invalid.append({'id': row['id'], 'validity': row['validity']}); continue
        values = dict(row['metrics'])
        if row['id'] in ergo:
            values.update(ae.flat(ergo[row['id']]))
        else:
            missing_ergo.append(row['id'])
        table[(info['arm'], info['offset_m'], info['task'], info['seed'])] = values
    failed = sorted(p.parent.name for p in (OUT/'e/runs').glob('*/error.json'))
    return table, invalid, failed, missing_ergo


def diffs(table, offset, metric):
    deltas, clusters = [], []
    for (a, o, task, seed), on in sorted(table.items()):
        off = table.get(('C0', o, task, seed))
        if a == 'C1' and o == offset and off is not None and metric in on and metric in off:
            deltas.append(on[metric]-off[metric]); clusters.append(task)
    return deltas, clusters


def slopes(table, metric):
    """Per task: least-squares slope over offset of the seed-mean C1 - C0 difference (tasks with every offset)."""
    by = collections.defaultdict(dict)
    for o in OFFSETS:
        d, c = diffs(table, o, metric)
        for task in set(c):
            by[task][o] = float(np.mean([x for x, t in zip(d, c) if t == task]))
    tasks = sorted(t for t, v in by.items() if len(v) == len(OFFSETS))
    return [float(np.polyfit(OFFSETS, [by[t][o] for o in OFFSETS], 1)[0]) for t in tasks], tasks


def cell(table, arm, offset, metric):
    by = collections.defaultdict(list)
    for (a, o, task, _), v in table.items():
        if a == arm and o == offset and metric in v:
            by[task].append(v[metric])
    return float(np.mean([np.mean(x) for x in by.values()])) if by else None


def primary(table):
    c0 = [cell(table, 'C0', o, RES) for o in OFFSETS]
    valid = bool(all(v is not None and v <= .03 for v in c0))
    pooled = ae.pooled_classes([v for (a, *_), v in table.items() if a == 'C0' and 'ergo_frames' in v])
    insensitive = bool(pooled is not None and max(pooled.values()) >= .9)
    f = {'F1': test(*diffs(table, .1, RES)), 'F2': test(*slopes(table, RES)),
         'F3': test(*diffs(table, .1, ERG)), 'F4': test(*slopes(table, ERG))}
    for key, p in zip(f, holm([h.get('sign_flip_p', 1.) for h in f.values()])):
        ok = valid and (key in ('F1', 'F2') or not insensitive)
        f[key].update(holm_p=p, supported=bool(p < .05 and f[key].get('excludes_zero', False) and ok))
    mech = (f['F1']['supported'] and f['F1']['mean'] >= SPEC['conventions_not_derived']['mechanism_cost_threshold_bw']) or f['F3']['supported']
    conflict = (f['F2']['supported'] and f['F2']['mean'] > 0) or (f['F4']['supported'] and f['F4']['mean'] < 0)
    return {'pipeline_valid': valid, 'c0_residual_by_offset': c0, 'c0_pooled_classes': pooled, 'ergo_insensitive': insensitive,
            'tests': f, 'mechanism_cost': bool(mech), 'conflict_cost': bool(conflict)}


def e1_levels():
    """Descriptive RULA/REBA cells of the E1 M2 arms (valid E1 runs; heading-unclamped re-evaluations replace originals)."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in ae.ergo_rows('e1').values():
        if not r['valid']:
            continue
        v = ae.flat(r)
        for m in (*ae.REPORTED, *POST_HOC):
            if m in v:
                by[f"{r['arm']}|d{r['offset_m']:g}"][m].append((r['task'], v[m]))
    out = {}
    for k, ms in sorted(by.items()):
        out[k] = {}
        for m, pairs in ms.items():
            t = collections.defaultdict(list)
            for task, x in pairs:
                t[task].append(x)
            out[k][m] = float(np.mean([np.mean(x) for x in t.values()]))
        out[k]['runs'] = len(next(iter(ms.values())))
    return out


def main():
    table, invalid, failed, missing = load()
    result = {'protocol': SPEC['protocol'], 'runs': len(table)+len(invalid), 'valid_runs': len(table), 'invalid': invalid,
              'failed': failed, 'missing_ergo': missing, 'primary': primary(table),
              'cells': {f'{a}|d{o:g}': {m: cell(table, a, o, m) for m in REPORTED} for a in ('C0', 'C1') for o in OFFSETS},
              'per_offset': {f'd{o:g}': {m: test(*diffs(table, o, m)) for m in REPORTED if diffs(table, o, m)[0]} for o in OFFSETS}}
    everything = load('all')[0]
    result['all_pairs_sensitivity'] = {'F1': test(*diffs(everything, .1, RES)), 'F2': test(*slopes(everything, RES)),
                                       'F3': test(*diffs(everything, .1, ERG)), 'F4': test(*slopes(everything, ERG))}
    result['post_hoc_r3'] = {m: {'F3': test(*diffs(table, .1, m)), 'F4': test(*slopes(table, m))} for m in POST_HOC}
    result['e1_ergo_levels'] = e1_levels()
    json_write(OUT/'analysis.json', result)
    p = result['primary']
    print(json.dumps({'valid_runs': len(table), 'pipeline_valid': p['pipeline_valid'], 'insensitive': p['ergo_insensitive'],
                      **{k: (round(v['mean'], 4), [round(x, 4) for x in v['ci95']], v.get('holm_p'), v['supported'])
                         for k, v in p['tests'].items()},
                      'mechanism_cost': p['mechanism_cost'], 'conflict_cost': p['conflict_cost']}, indent=1))


if __name__ == '__main__':
    main()
