#!/usr/bin/env python
"""Analysis of E2 (config/e2_biomech_20261102.json): musculoskeletal costs of the P0 goal-lookahead arms."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_e1 import METRICS, PRIMARY, limb_valid
from analyze_next_round import test
from motiogen.config import ROOT
from motiogen.native_analysis import holm
from motiogen.runtime import json_write

OUT = ROOT/'develop_logs/test_runs/20261102_e2_biomech'
P0_SPEC = json.loads((ROOT/'config/p0_20261102.json').read_text(encoding='utf-8'))
ARMS = ['G0', 'G4', 'W0g', 'G4W']; OFFSETS = P0_SPEC['offsets_m']
MARGIN = .03


def load_table(mode='registered'):
    """mode: 'registered' (preregistered validity), 'all' (no exclusion), 'limb' (limb-marker IK validity)."""
    table, invalid = {}, []
    for path in sorted((OUT/'runs').glob('*/metrics.json')):
        row = json.loads(path.read_text()); info = row['p0']
        keep = {'registered': row['valid'], 'all': True, 'limb': limb_valid(row, path)}[mode]
        if not keep:
            invalid.append({'id': row['id'], 'arm': info['arm'], 'offset_m': info['offset_m'], 'validity': row['validity'],
                            'ik': row['ik'], 'posthoc_missing': keep is None}); continue
        table[(info['arm'], info['offset_m'], info['task'], info['seed'])] = row['metrics']
    return table, invalid, [p.parent.name for p in (OUT/'runs').glob('*/error.json')]


def paired(table, on, off, offsets, metric, shift=0.):
    deltas, clusters = [], []
    for (a, o, task, seed), values in sorted(table.items()):
        base = table.get((off, o, task, seed))
        if a == on and o in offsets and base is not None and metric in values and metric in base:
            deltas.append(values[metric]-base[metric]-shift); clusters.append(task)
    return deltas, clusters


def cell(table, arm, offset, metric):
    by = collections.defaultdict(list)
    for (a, o, task, _), values in table.items():
        if a == arm and o == offset and metric in values:
            by[task].append(values[metric])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def primary(table):
    g0 = [cell(table, 'G0', o, PRIMARY) for o in OFFSETS]
    valid = bool(all(v is not None and v <= .03 for v in g0))
    hypotheses = {'K1': test(*paired(table, 'W0g', 'G4', OFFSETS, PRIMARY)),
                  'K2': test(*paired(table, 'G4', 'G0', OFFSETS, PRIMARY, MARGIN)),
                  'K3': test(*paired(table, 'G4W', 'G4', OFFSETS, PRIMARY, MARGIN))}
    for key in ('K2', 'K3'):
        hypotheses[key]['margin'] = MARGIN; hypotheses[key]['mean_difference'] = hypotheses[key]['mean']+MARGIN
    side = {'K1': lambda ci: ci[0] > 0, 'K2': lambda ci: ci[1] < 0, 'K3': lambda ci: ci[1] < 0}
    for key, p in zip(hypotheses, holm([h['sign_flip_p'] for h in hypotheses.values()])):
        hypotheses[key]['holm_p'] = p
        hypotheses[key]['supported'] = bool(p < .05 and side[key](hypotheses[key]['ci95']) and valid)
    return valid, hypotheses


def validity_by_arm():
    counts = collections.defaultdict(collections.Counter)
    for path in (OUT/'runs').glob('*/metrics.json'):
        row = json.loads(path.read_text()); key = f"{row['p0']['arm']}|d{row['p0']['offset_m']:g}"
        counts[key]['runs'] += 1
        counts[key]['registered_invalid'] += not row['valid']
        for check, ok in row['validity'].items():
            counts[key][f'failed_{check}'] += not ok
        limb = limb_valid(row, path)
        counts[key]['limb_invalid'] += limb is False
        counts[key]['posthoc_missing'] += limb is None
    return {k: dict(v) for k, v in sorted(counts.items())}


def main():
    table, invalid, failed = load_table()
    result = {'runs': len(table), 'invalid_runs': invalid, 'failed_runs': failed}
    result['cells'] = {f'{a}|d{o:g}': {m: cell(table, a, o, m) for m in METRICS} for a in ARMS for o in OFFSETS}
    result['pipeline_valid'], hypotheses = primary(table)
    result['primary'] = hypotheses
    result['validity_by_arm'] = validity_by_arm()
    result['posthoc_sensitivity'] = {}
    for mode in ('all', 'limb'):
        alt, alt_invalid, _ = load_table(mode)
        valid, alt_hypotheses = primary(alt)
        result['posthoc_sensitivity'][mode] = {'runs': len(alt), 'excluded': len(alt_invalid), 'pipeline_valid': valid,
                                               'primary': alt_hypotheses,
                                               'cells': {f'{a}|d{o:g}': cell(alt, a, o, PRIMARY) for a in ARMS for o in OFFSETS}}
    result['contrasts'] = {f'{on}-{off}': {'pooled': {m: test(*paired(table, on, off, OFFSETS, m)) for m in METRICS},
                                           'per_offset': {str(o): {m: test(*paired(table, on, off, [o], m)) for m in METRICS} for o in OFFSETS}}
                           for on, off in [('W0g', 'G0'), ('G4', 'G0'), ('W0g', 'G4'), ('G4W', 'G4')]}
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'runs': result['runs'], 'pipeline_valid': result['pipeline_valid'], 'invalid': len(invalid), 'failed': len(failed),
                      'primary': {k: {x: v.get(x) for x in ('mean', 'ci95', 'holm_p', 'supported')} for k, v in hypotheses.items()}}))


if __name__ == '__main__':
    main()
