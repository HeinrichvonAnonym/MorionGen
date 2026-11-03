#!/usr/bin/env python
"""Analysis of E3 (config/e3_upper_body_20261103.json, revision 2): every run is classified per region into
type 1 (conversion distortion, run level), type 2 (unreachable frames) and type 3 (loads of the reachable frames)."""
import argparse
import collections
import json
import multiprocessing
import os
import subprocess
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
OPENSIM_PY = Path.home()/'.local/share/opensim-api/bin/python'
OFFSETS = {'e1': [.1, .5, 1.], 'e2': [.5, 1.]}
ARMS = {'e1': ['C0', 'W0', 'XFh', 'XFo', 'W0X'], 'e2': ['G0', 'G4', 'W0g', 'G4W']}
CONTROLS = ('C0', 'G0', 'G4')
CONTRASTS = {'e1': ('W0X', 'C0'), 'e2': ('W0g', 'G4')}
SECONDARY_CONTRASTS = {'e1': [('W0', 'C0'), ('XFh', 'C0'), ('XFo', 'C0'), ('W0X', 'W0')], 'e2': [('G4', 'G0'), ('G4W', 'G4'), ('W0g', 'G0')]}
MASS, G = 75., 9.80665
# Thresholds of the preregistration; 约定，未经推演 (listed there under conventions_not_derived).
IK_RMS_M, NECK_FIT = .03, {'chest_deg': 1., 'c2_deg': 2., 'skull_deg': 2., 't1_m': .005}
REACTION_VALID, GATE = .95, .05
RESIDUAL_BW, RESERVE_NM, LIMIT_DEG = .05, 1., .5
SHEAR_N = (700., 1000.)
NEAR_GOAL_S = .5  # 约定，未经推演: window for the timing of type-2 frames relative to the correction goals
PRIMARY = {'type2_lumbar': ('lumbar', 'type2_fraction', 'directional'), 'type2_neck': ('neck', 'type2_fraction', 'directional'),
           'type3_lumbar': ('lumbar', 'L5_S1_compression_p95_bw', 'two_sided'),
           'type3_neck': ('neck', 'C7_T1_compression_p95_bw', 'two_sided')}
POSTURE = ['trunk_inclination_deg_abs_p95', 'trunk_flexion_deg_mean', 'trunk_asymmetry_deg_abs_p95', 'head_inclination_deg_mean',
           'neck_flexion_deg_mean', 'neck_asymmetry_deg_abs_p95', 'upper_arm_elevation_deg_abs_p95', 'head_angular_speed_deg_s_rms',
           'trunk_inclination_deg_over_20_fraction', 'head_inclination_deg_over_25_fraction', 'neck_flexion_deg_over_25_fraction',
           'upper_arm_elevation_deg_over_20_fraction', 'upper_arm_elevation_deg_over_60_fraction', 'trunk_asymmetry_deg_over_10_fraction',
           'neck_asymmetry_deg_over_10_fraction']


def geometry(out):
    """Spine tangents and coordinate ranges from the generic models (OpenSim environment, once per analysis directory)."""
    path = out/'geometry.json'
    if not path.exists():
        code = ('import json, sys; sys.path.insert(0, "utils/motiogen"); sys.path.insert(0, "utils"); '
                'from motiogen.upper_body import geometry; print("GEOMETRY" + json.dumps(geometry()))')
        env = {**os.environ, 'PYTHONNOUSERSITE': '1'}
        text = subprocess.run([str(OPENSIM_PY), '-c', code], cwd=ROOT, env=env, capture_output=True, text=True, check=True).stdout
        path.write_text(json.dumps(json.loads(next(l for l in text.splitlines() if l.startswith('GEOMETRY'))[8:]), indent=1))
    return json.loads(path.read_text())


def read_sto(path):
    lines = Path(path).read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == 'endheader')+1
    labels = lines[start].split('\t')
    data = np.array([[float(v) for v in l.split('\t')] for l in lines[start+1:] if l.strip()])
    return data[:, 0], labels[1:], data[:, 1:]


def column(labels, data, name):
    return data[:, labels.index(name)]


def at(times, source_times, values):
    return np.interp(times, source_times, values)


def rows_at(source_times, times):
    mask = np.zeros(len(source_times), bool)
    mask[[int(np.argmin(np.abs(source_times-x))) for x in times]] = True
    return mask


def near_limit(values, spec):
    tol = np.deg2rad(LIMIT_DEG)
    radians = np.deg2rad(values) if spec['degrees_in_files'] else values
    return (radians <= spec['min']+tol) | (radians >= spec['max']-tol)


def project(labels, data, joint, child, axis):
    """Reaction on the child resolved along the spine tangent: compression, anterior-posterior and medio-lateral shear."""
    F = np.stack([column(labels, data, f'{joint}_on_{child}_in_{child}_f{a}') for a in 'xyz'], 1)
    u = np.asarray(axis); ap = np.cross(u, [0., 0., 1.]); ap /= np.linalg.norm(ap); ml = np.cross(ap, u)
    return F@u, F@ap, F@ml


def group_means(labels, act, groups, rows):
    out = {}
    for name, members in groups.items():
        cols = [k for k, l in enumerate(labels) if not l.startswith('reserve_') and any(l.startswith(m) for m in members)]
        if cols and rows.any():
            out[f'{name}_activation_mean'] = float(act[np.ix_(rows, cols)].mean())
    return out


def goal_seconds(run):
    try:
        condition = json.loads((Path(run)/'intervention.json').read_text()).get('condition', {})
    except (OSError, ValueError):
        return []
    return [float(s) for s in condition.get('goal_seconds') or []]


def timing(times, flagged, goals):
    if not goals:
        return {}
    near = np.min(np.abs(times[:, None]-np.asarray(goals)[None]), 1) <= NEAR_GOAL_S
    return {'near_goal_frame_fraction': float(near.mean()),
            'type2_near_goal_fraction': float(near[flagged].mean()) if flagged.any() else None}


def classify_lumbar(work, run, geo):
    work, bw, flags = Path(work), MASS*G, {}
    receipt = json.loads((work/'receipt.json').read_text())
    duration = (receipt['frames']-1)/receipt['fps']
    t_ik, l_ik, d_ik = read_sto(work/'ik/_ik_marker_errors.sto')
    trim = (t_ik >= geo['trim_s']) & (t_ik <= duration-geo['trim_s'])
    flags['ik_fit'] = bool(np.median(column(l_ik, d_ik, 'marker_error_RMS')[trim]) > IK_RMS_M)
    flags['non_anatomical_limit'] = bool(set(receipt.get('frames_at_coordinate_limit', {})) & set(geo['pelvis']))
    t, labels, jr = read_sto(next((work/'jr').glob('*ReactionLoads.sto')))
    finite = np.isfinite(jr).all(1) & (np.abs(jr) < 1e6).all(1)
    flags['reaction'] = bool(finite.mean() < REACTION_VALID)
    dyn = np.load(work/'dynamics.npz')
    external = at(t, dyn['times'], np.linalg.norm(dyn['residual_force'], axis=1)/bw) > RESIDUAL_BW
    t_act, l_act, act = read_sto(work/'so/so_so_activation.sto')
    internal = np.zeros(len(t), bool)
    for c in geo['lumbar']['coordinates']:
        if f'reserve_{c}' in l_act:
            internal |= np.abs(at(t, t_act, column(l_act, act, f'reserve_{c}'))) > RESERVE_NM
    t_q, l_q, q = read_sto(work/'coordinates.mot')
    kinematic = np.zeros(len(t), bool)
    for c, spec in geo['lumbar']['coordinates'].items():
        kinematic |= near_limit(at(t, t_q, column(l_q, q, c)), spec)
    type2 = external | internal | kinematic
    reachable = ~type2 & finite
    region = {'type1': any(flags.values()), 'type1_flags': flags, 'frames': int(len(t)), 'reachable_frames': int(reachable.sum()),
              'metrics': {'type2_fraction': float(type2.mean()), 'type2_external_fraction': float(external.mean()),
                          'type2_internal_fraction': float(internal.mean()), 'type2_kinematic_fraction': float(kinematic.mean()),
                          **timing(t, type2, goal_seconds(run))}}
    series = {'times': np.round(t, 3), 'reachable': reachable}
    if reachable.any():
        m = region['metrics']
        for key, (joint, child) in geo['joints']['lumbar'].items():
            comp, ap, ml = project(labels, jr, joint, child, geo['lumbar']['tangents'][key]['axis'])
            m[f'{key}_compression_p95_bw'] = float(np.percentile(comp[reachable], 95)/bw)
            m[f'{key}_compression_mean_bw'] = float(comp[reachable].mean()/bw)
            if key == 'L5_S1':
                series['L5_S1_compression'] = comp
                m['L5_S1_over_niosh_fraction'] = float(np.mean(comp[reachable] > geo['niosh_n']))
                m['L5_S1_ap_shear_p95_n'] = float(np.percentile(np.abs(ap[reachable]), 95))
                m['L5_S1_ml_shear_p95_n'] = float(np.percentile(np.abs(ml[reachable]), 95))
                for limit in SHEAR_N:
                    m[f'L5_S1_ap_shear_over_{limit:g}_fraction'] = float(np.mean(np.abs(ap[reachable]) > limit))
        id_labels = list(dyn['id_labels'])
        moment = lambda c: next(i for i, l in enumerate(id_labels) if l.startswith(c+'_'))
        shoulder = []
        for s in 'rl':
            vector = np.stack([at(t, dyn['times'], dyn['id'][:, moment(f'{c}_{s}')]) for c in ('arm_flex', 'arm_add', 'arm_rot')], 1)
            shoulder.append(np.linalg.norm(vector, axis=1)[reachable]/MASS)
        m['shoulder_moment_p95_nm_kg'] = float(np.mean([np.percentile(x, 95) for x in shoulder]))
        m['shoulder_moment_mean_nm_kg'] = float(np.mean([x.mean() for x in shoulder]))
        m.update(group_means(l_act, act, geo['groups']['lumbar'], rows_at(t_act, t[reachable])))
    return region, series


def classify_neck(work, run, geo):
    work, bw, flags = Path(work), MASS*G, {}
    receipt = json.loads((work/'receipt.json').read_text())
    fps, n = receipt['fps'], receipt['frames']
    fit_path = work/'fit_frames.npz'
    if fit_path.exists():
        errors = np.load(fit_path)['errors_chest_c2_skull_deg_t1_m'][int(geo['trim_s']*fps):n-int(geo['trim_s']*fps)]
        medians = dict(zip(('chest_deg', 'c2_deg', 'skull_deg', 't1_m'), np.median(errors, 0)))
        fit_source = 'per_frame'
    else:
        fit = receipt['fit_error']
        medians = {'chest_deg': fit['chest_deg_max'], 'c2_deg': fit['c2_deg_rms'], 'skull_deg': fit['skull_deg_rms'], 't1_m': fit['t1_m_max']}
        fit_source = 'receipt_upper_bound'
    flags['fit'] = bool(any(medians[k] > v for k, v in NECK_FIT.items()))
    flags['non_anatomical_limit'] = bool(set(receipt.get('frames_at_coordinate_limit', {})) & set(geo['base']))
    t, labels, jr = read_sto(next((work/'jr').glob('*ReactionLoads.sto')))
    finite = np.isfinite(jr).all(1) & (np.abs(jr) < 1e6).all(1)
    flags['reaction'] = bool(finite.mean() < REACTION_VALID)
    t_act, l_act, act = read_sto(work/'so/so_so_activation.sto')
    internal = np.zeros(len(t), bool)
    for c in geo['neck']['coordinates']:
        if f'reserve_{c}' in l_act:
            internal |= np.abs(at(t, t_act, column(l_act, act, f'reserve_{c}'))) > RESERVE_NM
    t_q, l_q, q = read_sto(work/'coordinates.mot')
    kinematic = np.zeros(len(t), bool)
    for c, spec in geo['neck']['coordinates'].items():
        kinematic |= near_limit(at(t, t_q, column(l_q, q, c)), spec)
    type2 = internal | kinematic
    reachable = ~type2 & finite
    region = {'type1': any(flags.values()), 'type1_flags': flags, 'fit_medians': {k: float(v) for k, v in medians.items()},
              'fit_source': fit_source, 'frames': int(len(t)), 'reachable_frames': int(reachable.sum()),
              'metrics': {'type2_fraction': float(type2.mean()), 'type2_internal_fraction': float(internal.mean()),
                          'type2_kinematic_fraction': float(kinematic.mean()), **timing(t, type2, goal_seconds(run))}}
    series = {'times': np.round(t, 3), 'reachable': reachable}
    if reachable.any():
        m = region['metrics']
        for key, (joint, child) in geo['joints']['neck'].items():
            comp, ap, _ = project(labels, jr, joint, child, geo['neck']['tangents'][key]['axis'])
            m[f'{key}_compression_p95_bw'] = float(np.percentile(comp[reachable], 95)/bw)
            m[f'{key}_compression_mean_bw'] = float(comp[reachable].mean()/bw)
            m[f'{key}_ap_shear_p95_n'] = float(np.percentile(np.abs(ap[reachable]), 95))
            if key == 'C7_T1':
                series['C7_T1_compression'] = comp
        m.update(group_means(l_act, act, geo['groups']['neck'], rows_at(t_act, t[reachable])))
    return region, series


def classify(job):
    record, geo = job
    out = {'id': record['id'], 'experiment': record['experiment'], 'cell': record['cell'], 'part': record['part']}
    try:
        fn = classify_lumbar if record['part'] == 'lumbar' else classify_neck
        out['region'], out['series'] = fn(record['work'], record['run'], geo)
        out['posture'] = record.get('posture', {})
    except Exception as exc:  # a missing or unreadable output is reported, not dropped silently
        out['error'] = repr(exc)
    return out


def load(runs, geo, workers):
    records = []
    for part in ('lumbar', 'neck'):
        for path in sorted(Path(runs).glob(f'*/{part}.json')):
            row = json.loads(path.read_text()); row['part'] = part; records.append(row)
    failed = sorted(f'{p.parent.name}/{p.stem[:-6]}' for p in Path(runs).glob('*/*_error.json'))
    with multiprocessing.Pool(workers) as pool:
        results = pool.map(classify, [(r, geo) for r in records], chunksize=4)
    regions = {'lumbar': {}, 'neck': {}}
    unreadable = []
    for r in results:
        if 'error' in r:
            unreadable.append({'id': r['id'], 'part': r['part'], 'error': r['error']}); continue
        info = r['cell']
        regions[r['part']][(r['experiment'], info['arm'], info['offset_m'], info['task'], info['seed'])] = r
    return regions, failed, unreadable


def type1_report(regions):
    report = {}
    for part, table in regions.items():
        by = collections.defaultdict(collections.Counter)
        for (e, a, *_), r in table.items():
            c = by[f'{e}|{a}']; c['runs'] += 1; c['type1'] += r['region']['type1']
            for flag, on in r['region']['type1_flags'].items():
                c[f'flag_{flag}'] += on
        report[part] = {k: {**v, 'rate': v['type1']/v['runs']} for k, v in sorted(by.items())}
    gate = {f'{part}|{arm}': v['rate'] for part, arms in report.items() for arm, v in arms.items() if arm.split('|')[1] in CONTROLS}
    return report, {'control_rates': gate, 'passed': all(r <= GATE for r in gate.values()), 'threshold': GATE}


def usable(regions, exclude=()):
    """Type-1-free records of each region; posture needs both regions type-1-free."""
    table = {part: {k: r for k, r in t.items() if not r['region']['type1'] and r['id'] not in exclude} for part, t in regions.items()}
    table['posture'] = {k: r for k, r in table['neck'].items() if k in table['lumbar']}
    return table


def value(record, part, metric):
    return record['posture'].get(metric) if part == 'posture' else record['region']['metrics'].get(metric)


def paired(table, experiment, on, off, metric, part):
    deltas, clusters = [], []
    for (e, a, o, task, seed), r in sorted(table.items()):
        base = table.get((e, off, o, task, seed))
        if e != experiment or a != on or base is None:
            continue
        x, y = value(r, part, metric), value(base, part, metric)
        if x is not None and y is not None:
            deltas.append(x-y); clusters.append(task)
    return deltas, clusters


def time_matched(table, experiment, on, off, part, series_key):
    """Type-3 95th percentile on the frames reachable in both runs of a pair."""
    deltas, clusters = [], []
    for (e, a, o, task, seed), r in sorted(table.items()):
        base = table.get((e, off, o, task, seed))
        if e != experiment or a != on or base is None or series_key not in r['series'] or series_key not in base['series']:
            continue
        common, i, j = np.intersect1d(r['series']['times'], base['series']['times'], return_indices=True)
        both = r['series']['reachable'][i] & base['series']['reachable'][j]
        if both.any():
            p95 = lambda s, k: np.percentile(s['series'][series_key][k][both], 95)/(MASS*G)
            deltas.append(float(p95(r, i)-p95(base, j))); clusters.append(task)
    return deltas, clusters


def safe_test(deltas, clusters):
    return test(deltas, clusters) if len(set(clusters)) > 1 else {'pairs': len(deltas), 'clusters': len(set(clusters))}


def primary(table):
    hypotheses = {}
    for experiment, (on, off) in CONTRASTS.items():
        for name, (part, metric, sided) in PRIMARY.items():
            hypotheses[f'{experiment}_{name}'] = {'contrast': f'{on}-{off}', 'region': part, 'metric': metric, 'test': sided,
                                                  **safe_test(*paired(table[part], experiment, on, off, metric, part))}
    for key, p in zip(hypotheses, holm([h.get('sign_flip_p', 1.) for h in hypotheses.values()])):
        h = hypotheses[key]; h['holm_p'] = p
        significant = 'ci95' in h and h['ci95'][0] is not None and 'sign_flip_p' in h and p < .05
        h['supported'] = bool(significant and (h['ci95'][0] > 0 if h['test'] == 'directional' else h['ci95'][0] > 0 or h['ci95'][1] < 0))
    return hypotheses


def metric_names(table, part):
    names = set()
    for r in table[part].values():
        names |= set(r['posture'] if part == 'posture' else r['region']['metrics'])
    return sorted(names & set(POSTURE)) if part == 'posture' else sorted(names)


def cells(table):
    out = {}
    for part in ('lumbar', 'neck', 'posture'):
        metrics = metric_names(table, part)
        for e in ARMS:
            for a in ARMS[e]:
                for o in OFFSETS[e]:
                    by = collections.defaultdict(lambda: collections.defaultdict(list))
                    for (e2, a2, o2, task, _), r in table[part].items():
                        if (e2, a2, o2) == (e, a, o):
                            for m in metrics:
                                v = value(r, part, m)
                                if v is not None:
                                    by[m][task].append(v)
                    out.setdefault(f'{e}|{a}|d{o:g}', {})[part] = {m: float(np.mean([np.mean(v) for v in by[m].values()])) for m in by}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, default=OUT/'runs'); parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    geo = geometry(args.out)
    regions, failed, unreadable = load(args.runs, geo, args.workers)
    type1, gate = type1_report(regions)
    table = usable(regions)
    result = {'scope': SPEC['data'], 'records': {p: len(t) for p, t in regions.items()}, 'failed': failed, 'unreadable': unreadable,
              'geometry': {p: geo[p] for p in ('lumbar', 'neck')},
              'type1': {'by_arm': type1, 'gate': gate,
                        'note': 'gate failed: the conversion is treated as faulty and types 2 and 3 are not interpreted' if not gate['passed'] else 'gate passed'},
              'usable': {p: len(t) for p, t in table.items()}, 'cells': cells(table), 'primary': primary(table)}
    secondary = {}
    for experiment, pairs in SECONDARY_CONTRASTS.items():
        for on, off in [CONTRASTS[experiment], *pairs]:
            secondary[f'{experiment}|{on}-{off}'] = {part: {m: safe_test(*paired(table[part], experiment, on, off, m, part)) for m in metric_names(table, part)}
                                                     for part in ('lumbar', 'neck', 'posture')}
    result['secondary'] = secondary
    result['time_matched_type3'] = {f'{e}|{on}-{off}': {'lumbar': safe_test(*time_matched(table['lumbar'], e, on, off, 'lumbar', 'L5_S1_compression')),
                                                        'neck': safe_test(*time_matched(table['neck'], e, on, off, 'neck', 'C7_T1_compression'))}
                                    for e, (on, off) in CONTRASTS.items()}
    result['sensitivity_without_development_runs'] = primary(usable(regions, set(SPEC['development_runs'])))
    json_write(args.out/'analysis.json', result)
    print(json.dumps({'records': result['records'], 'usable': result['usable'], 'failed': len(failed), 'unreadable': len(unreadable),
                      'gate': gate, 'primary': {k: {x: v.get(x) for x in ('mean', 'ci95', 'pairs', 'holm_p', 'supported')}
                                                for k, v in result['primary'].items()}}, default=str))


if __name__ == '__main__':
    main()
