#!/usr/bin/env python
"""Analysis of E1 (config/e1_biomech_20261101.json): musculoskeletal costs of the M2 arms."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_next_round import test
from motiogen.config import ROOT
from motiogen.native_analysis import holm
from motiogen.runtime import json_write

SPEC = json.loads((ROOT/'config/e1_biomech_20261101.json').read_text(encoding='utf-8'))
M2_SPEC = json.loads((ROOT/'config/m2_20261101.json').read_text(encoding='utf-8'))
OUT = ROOT/'develop_logs/test_runs/20261101_e1_biomech'
ARMS = list(M2_SPEC['arms']); OFFSETS = M2_SPEC['offsets_m']
PRIMARY = 'residual_force_rms_bw'
METRICS = [PRIMARY, 'unsupported_fraction_5pct_bw', 'residual_moment_rms_m', 'lumbar_extension_moment_rms_nm_kg',
           *[f'{n}_moment_{k}_nm_kg' for n in ('hip_flexion', 'hip_adduction', 'knee_angle', 'ankle_angle') for k in ('rms', 'peak')],
           'muscle_effort_sum_a2_mean', 'muscle_activation_mean', 'reserve_rms_nm', 'reserve_peak_nm']


def load_table():
    table, invalid, failed = {}, [], []
    for path in sorted((OUT/'runs').glob('*/metrics.json')):
        row = json.loads(path.read_text()); info = row['m2']
        if not row['valid']:
            invalid.append({'id': row['id'], 'validity': row['validity'], 'ik': row['ik']}); continue
        table[(info['arm'], info['offset_m'], info['task'], info['seed'])] = row['metrics']
    failed = [p.parent.name for p in (OUT/'runs').glob('*/error.json')]
    return table, invalid, failed


def paired(table, on, off, offsets, metric):
    deltas, clusters = [], []
    for (a, o, task, seed), values in sorted(table.items()):
        base = table.get((off, o, task, seed))
        if a == on and o in offsets and base is not None and metric in values and metric in base:
            deltas.append(values[metric]-base[metric]); clusters.append(task)
    return deltas, clusters


def cell(table, arm, offset, metric):
    by = collections.defaultdict(list)
    for (a, o, task, _), values in table.items():
        if a == arm and o == offset and metric in values:
            by[task].append(values[metric])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def kinematic_link(table):
    """Task-level Spearman correlation across arm-offset cells between M2 kinematic costs and E1 dynamic costs."""
    from analyze_m2 import load_table as m2_table
    kin = m2_table()
    result = {}
    for kmetric in ('boundary_jerk_0p1s', 'foot_sliding_5cm_10cms', 'jerk_0p1s'):
        for dmetric in (PRIMARY, 'muscle_effort_sum_a2_mean'):
            rhos = []
            for task in sorted({k[2] for k in table}):
                x, y = [], []
                for a in ARMS:
                    for o in OFFSETS:
                        ks = [kin[(a, o, task, s)][kmetric] for s in M2_SPEC['seeds'] if (a, o, task, s) in kin and (a, o, task, s) in table]
                        ds = [table[(a, o, task, s)][dmetric] for s in M2_SPEC['seeds'] if (a, o, task, s) in kin and (a, o, task, s) in table]
                        if ks:
                            x.append(np.mean(ks)); y.append(np.mean(ds))
                if len(x) > 3:
                    rhos.append(spearmanr(x, y).correlation)
            result[f'{kmetric}~{dmetric}'] = {'median_rho': float(np.median(rhos)) if rhos else None,
                                               'tasks': len(rhos), 'rho_ci95_t': test(rhos, list(range(len(rhos))))['ci95'] if len(rhos) > 1 else None}
    return result


def main():
    table, invalid, failed = load_table()
    result = {'runs': len(table), 'expected_runs': M2_SPEC['runs'], 'invalid_runs': invalid, 'failed_runs': failed}
    result['cells'] = {f'{a}|d{o:g}': {m: cell(table, a, o, m) for m in METRICS} for a in ARMS for o in OFFSETS}
    c0 = [result['cells'][f'C0|d{o:g}'][PRIMARY] for o in OFFSETS]
    result['pipeline_valid'] = bool(all(v is not None and v <= .03 for v in c0))
    hypotheses = {'H1': test(*paired(table, 'W0X', 'C0', OFFSETS, PRIMARY)),
                  'H2': test(*paired(table, 'W0X', 'W0', OFFSETS, PRIMARY)),
                  'H3': test(*paired(table, 'XFh', 'C0', OFFSETS, PRIMARY))}
    side = {'H1': lambda ci: ci[0] > 0, 'H2': lambda ci: ci[0] > 0 or ci[1] < 0, 'H3': lambda ci: ci[0] > 0}
    for key, p in zip(hypotheses, holm([h['sign_flip_p'] for h in hypotheses.values()])):
        hypotheses[key]['holm_p'] = p
        hypotheses[key]['supported'] = bool(p < .05 and side[key](hypotheses[key]['ci95']) and result['pipeline_valid'])
    result['primary'] = hypotheses
    result['contrasts'] = {f'{on}-{off}': {'pooled': {m: test(*paired(table, on, off, OFFSETS, m)) for m in METRICS},
                                           'per_offset': {str(o): {m: test(*paired(table, on, off, [o], m)) for m in (PRIMARY, 'muscle_effort_sum_a2_mean', 'knee_angle_moment_rms_nm_kg', 'lumbar_extension_moment_rms_nm_kg')} for o in OFFSETS}}
                           for on, off in [('W0X', 'C0'), ('W0', 'C0'), ('XFh', 'C0'), ('XFo', 'C0'), ('W0X', 'XFo'), ('W0X', 'W0')]}
    result['kinematic_link'] = kinematic_link(table)
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'runs': result['runs'], 'pipeline_valid': result['pipeline_valid'], 'invalid': len(invalid), 'failed': len(failed),
                      'primary': {k: {x: v.get(x) for x in ('mean', 'ci95', 'holm_p', 'supported')} for k, v in hypotheses.items()}}))


if __name__ == '__main__':
    main()
