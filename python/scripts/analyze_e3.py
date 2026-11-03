#!/usr/bin/env python
"""Analysis of E3 (config/e3_upper_body_20261103.json): upper-body ergonomic costs of the E1 and E2 arms."""
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_next_round import test
from motiogen.config import ROOT
from motiogen.native_analysis import holm
from motiogen.runtime import json_write

SPEC = json.loads((ROOT/'config/e3_upper_body_20261103.json').read_text(encoding='utf-8'))
OUT = ROOT/'develop_logs/test_runs/20261103_e3_upper'
OFFSETS = {'e1': [.1, .5, 1.], 'e2': [.5, 1.]}
ARMS = {'e1': ['C0', 'W0', 'XFh', 'XFo', 'W0X'], 'e2': ['G0', 'G4', 'W0g', 'G4W']}
REGIONS = {'lumbar': ('lumbar', 'lumbar_L5_S1_compression_rms_bw'), 'neck': ('neck', 'neck_C7_T1_compression_rms_bw'),
           'shoulder': ('lumbar', 'shoulder_moment_rms_nm_kg')}
CONTRASTS = {'e1': ('W0X', 'C0'), 'e2': ('W0g', 'G4')}
SECONDARY_CONTRASTS = {'e1': [('W0', 'C0'), ('XFh', 'C0'), ('XFo', 'C0'), ('W0X', 'W0')], 'e2': [('G4', 'G0'), ('G4W', 'G4'), ('W0g', 'G0')]}
POSTURE = ['trunk_inclination_deg_abs_p95', 'trunk_flexion_deg_mean', 'trunk_asymmetry_deg_abs_p95', 'head_inclination_deg_mean',
           'neck_flexion_deg_mean', 'neck_asymmetry_deg_abs_p95', 'upper_arm_elevation_deg_abs_p95', 'head_angular_speed_deg_s_rms',
           'trunk_inclination_deg_over_20_fraction', 'head_inclination_deg_over_25_fraction', 'neck_flexion_deg_over_25_fraction',
           'upper_arm_elevation_deg_over_20_fraction', 'upper_arm_elevation_deg_over_60_fraction', 'trunk_asymmetry_deg_over_10_fraction',
           'neck_asymmetry_deg_over_10_fraction']
LUMBAR = ['lumbar_L5_S1_compression_rms_bw', 'lumbar_L5_S1_compression_peak_n', 'lumbar_L5_S1_shear_rms_n', 'lumbar_L4_L5_compression_rms_bw',
          'lumbar_L1_L2_compression_rms_bw', 'lumbar_L5_S1_over_niosh_fraction', 'lumbar_flex_extension_moment_rms_nm_kg',
          'lumbar_lat_bending_moment_rms_nm_kg', 'lumbar_axial_rotation_moment_rms_nm_kg', 'lumbar_erector_spinae_activation_mean',
          'lumbar_multifidus_activation_mean', 'lumbar_quadratus_lumborum_activation_mean', 'lumbar_psoas_activation_mean',
          'lumbar_abdominals_activation_mean', 'lumbar_muscle_effort_sum_a2_mean', 'lumbar_reserve_rms_nm',
          'shoulder_moment_rms_nm_kg', 'shoulder_moment_peak_nm_kg', 'elbow_r_moment_rms_nm_kg', 'residual_force_rms_bw']
NECK = ['neck_C7_T1_compression_rms_bw', 'neck_C7_T1_compression_peak_n', 'neck_C7_T1_shear_rms_n', 'neck_C0_C1_compression_rms_bw',
        'neck_C0_C1_shear_rms_n', *[f'neck_{c}_moment_rms_nm' for c in ('pitch2', 'roll2', 'yaw2', 'pitch1', 'roll1', 'yaw1')],
        'neck_extensors_activation_mean', 'neck_flexors_activation_mean', 'neck_muscle_effort_sum_a2_mean', 'neck_reserve_rms_nm']
METRICS = {'lumbar': LUMBAR, 'neck': NECK+POSTURE}


def load_table(part, exclude=()):
    """(experiment, arm, offset, task, seed) -> metrics of the valid records of one region; neck records carry the posture."""
    table, invalid, failed = {}, [], sorted(p.parent.name for p in (OUT/'runs').glob(f'*/{part}_error.json'))
    for path in sorted((OUT/'runs').glob(f'*/{part}.json')):
        row = json.loads(path.read_text()); info = row['cell']
        if row['id'] in exclude:
            continue
        if not row['valid']:
            invalid.append({'id': row['id'], 'experiment': row['experiment'], 'arm': info['arm'], 'validity': row['validity']}); continue
        table[(row['experiment'], info['arm'], info['offset_m'], info['task'], info['seed'])] = {**row['metrics'], **row.get('posture', {})}
    return table, invalid, failed


def paired(table, experiment, on, off, metric):
    deltas, clusters = [], []
    for (e, a, o, task, seed), values in sorted(table.items()):
        base = table.get((e, off, o, task, seed))
        if e == experiment and a == on and base is not None and metric in values and metric in base:
            deltas.append(values[metric]-base[metric]); clusters.append(task)
    return deltas, clusters


def cell(table, experiment, arm, offset, metric):
    by = collections.defaultdict(list)
    for (e, a, o, task, _), values in table.items():
        if e == experiment and a == arm and o == offset and metric in values:
            by[task].append(values[metric])
    return float(np.mean([np.mean(v) for v in by.values()])) if by else None


def primary(tables):
    hypotheses = {}
    for experiment, (on, off) in CONTRASTS.items():
        for region, (part, metric) in REGIONS.items():
            hypotheses[f'{experiment}_{region}'] = {'contrast': f'{on}-{off}', 'metric': metric,
                                                    **test(*paired(tables[part], experiment, on, off, metric))}
    for key, p in zip(hypotheses, holm([h.get('sign_flip_p', 1.) for h in hypotheses.values()])):
        hypotheses[key]['holm_p'] = p
        hypotheses[key]['supported'] = bool(p < .05 and hypotheses[key]['ci95'][0] > 0)
    return hypotheses


def validity_by_arm(part):
    counts = collections.defaultdict(collections.Counter)
    for path in (OUT/'runs').glob(f'*/{part}.json'):
        row = json.loads(path.read_text()); key = f"{row['experiment']}|{row['cell']['arm']}"
        counts[key]['runs'] += 1; counts[key]['invalid'] += not row['valid']
        for check, ok in row['validity'].items():
            counts[key][f'failed_{check}'] += not ok
    return {k: dict(v) for k, v in sorted(counts.items())}


def main():
    tables, result = {}, {'scope': SPEC['data'], 'regions': {}}
    for part in ('lumbar', 'neck'):
        tables[part], invalid, failed = load_table(part)
        result['regions'][part] = {'records': len(tables[part]), 'invalid': invalid, 'failed': failed, 'validity_by_arm': validity_by_arm(part),
                                   'cells': {f'{e}|{a}|d{o:g}': {m: cell(tables[part], e, a, o, m) for m in METRICS[part]}
                                             for e in ARMS for a in ARMS[e] for o in OFFSETS[e]}}
    result['primary'] = primary(tables)
    result['secondary'] = {}
    for experiment, pairs in SECONDARY_CONTRASTS.items():
        for on, off in [CONTRASTS[experiment], *pairs]:
            result['secondary'][f'{experiment}|{on}-{off}'] = {
                part: {m: test(*paired(tables[part], experiment, on, off, m)) for m in METRICS[part]} for part in ('lumbar', 'neck')}
    development = set(SPEC['development_runs'])
    sensitivity = {part: load_table(part, development)[0] for part in ('lumbar', 'neck')}
    result['sensitivity_without_development_runs'] = primary(sensitivity)
    json_write(OUT/'analysis.json', result)
    print(json.dumps({'records': {p: len(t) for p, t in tables.items()},
                      'primary': {k: {x: v.get(x) for x in ('mean', 'ci95', 'holm_p', 'supported')} for k, v in result['primary'].items()}}))


if __name__ == '__main__':
    main()
