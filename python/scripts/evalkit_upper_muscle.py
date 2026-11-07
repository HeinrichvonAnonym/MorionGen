#!/usr/bin/env python
"""Validation of the upper-body muscle dimension (config/evalkit_upper_muscle_20261108.json). OpenSim python.

  evalkit_upper_muscle.py run [--workers 5] [--only V1,V2,V3,V4]   solve every validation arm (resumable)
  evalkit_upper_muscle.py report                                    develop_logs/test_runs/20261107_evalkit/upper_muscle/validation.json

V1 equivalence: 3 E3 runs (first valid by id of e1 C0, e1 W0X, e2 G4) in the e3 mode against analyze_e3.classify_lumbar
on the E3 work directories. V2 ground-truth reactions: the heldout2 synthetic control clips, the same FBLS kinematics
with the evalkit and with the truth ground reactions. V3 floor: first smoke8x5 row by id of each method, floor -2 / +2 cm
(v11 rows_floor2cm). V4 front end: the same rows in the e3 mode. V5 (AddB) is not executable (registration).
"""
import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils/motiogen'), str(ROOT/'utils'), str(ROOT/'python/scripts')]
DATA = Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])/'outputs/evalkit_20261107'
OUT = ROOT/'develop_logs/test_runs/20261107_evalkit/upper_muscle'
E3 = ROOT/'develop_logs/test_runs/20261103_e3_upper'
V11 = ROOT/'develop_logs/test_runs/20261107_evalkit/v11/smoke8x5'
SMOKE_MANIFEST = ROOT/'develop_logs/test_runs/20261107_evalkit/smoke8x5/manifest.json'
EQUIVALENCE_CELLS = (('e1', 'C0'), ('e1', 'W0X'), ('e2', 'G4'))
SHARED_FIT = ('receipt.json', 'model.osim', 'scaled.osim', 'markers.trc', 'calibration.trc', 'ik', 'calib')
COMPARE = ('L5_S1_compression_p95_bw', 'L5_S1_compression_mean_bw', 'type2_fraction')


def equivalence_runs():
    first = {}
    for p in sorted((E3/'runs').glob('*/lumbar.json')):
        d = json.loads(p.read_text()); k = (d['experiment'], d['cell']['arm'])
        if d['valid'] and k in EQUIVALENCE_CELLS and k not in first:
            first[k] = d
    return [first[k] for k in EQUIVALENCE_CELLS]


def truth_clips():
    return sorted(p.name for p in (DATA/'tool/heldout2').glob('*__control'))


def floor_rows():
    rows = json.loads(SMOKE_MANIFEST.read_text())['rows']
    return [sorted((r for r in rows if r['method'] == m), key=lambda r: r['id'])[0] for m in sorted({r['method'] for r in rows})]


def tasks(only):
    out = []
    if 'V1' in only:
        out += [('V1', d['id']) for d in equivalence_runs()]
    if 'V2' in only:
        out += [('V2', c) for c in truth_clips()]
    if 'V3' in only:
        out += [('V3', f"{r['id']}__floor{s}") for r in floor_rows() for s in ('-2cm', '+2cm')]
    if 'V4' in only:
        out += [('V4', r['id']) for r in floor_rows()]
    return out


def solve(task):
    arm, name = task
    os.nice(19)
    try:
        import opensim as osim
        osim.Logger.setLevelString('error')
        from evalkit import run, upper_muscle
        if arm == 'V1':
            d = next(x for x in equivalence_runs() if x['id'] == name)
            upper_muscle.lumbar_fbls(d['run'], OUT/'V1'/name/'lumbar', 75., 'e3')
        elif arm == 'V2':
            tool = DATA/'tool/heldout2'/name; e = json.loads((tool/'evalkit.json').read_text())
            f = np.load(tool/'frames.npz', allow_pickle=True)
            contact = (f['times'], {s: {k: f[f'contact_{s}_{k}'] for k in ('heel', 'toe')} for s in 'rl'})
            floor, src = e['receipt']['floor_eval_m'], e['receipt']['run']
            est = OUT/'V2'/name/'evalkit'
            upper_muscle.lumbar_fbls(src, est, e['receipt']['mass_kg'], 'evalkit', floor, contact)
            tru = OUT/'V2'/name/'truth'; tru.mkdir(parents=True, exist_ok=True)
            for item in SHARED_FIT:
                if not (tru/item).exists():
                    (shutil.copytree if (est/item).is_dir() else shutil.copy)(est/item, tru/item)
            upper_muscle.lumbar_fbls(src, tru, e['receipt']['mass_kg'], 'truth', floor, contact, DATA/'truth_eval/heldout2'/name/'grf.xml')
        elif arm == 'V3':
            d = V11/'rows_floor2cm'/name
            run.add_upper(json.loads((d/'evalkit.json').read_text())['receipt']['run'], d)
        else:
            d = V11/'rows'/name
            run.add_upper(json.loads((d/'evalkit.json').read_text())['receipt']['run'], d, mode='e3')
        return arm, name, 'ok'
    except Exception as e:
        path = OUT/'errors'/f'{arm}_{name}.json'; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-3000:]}))
        return arm, name, 'failed'


def run_all(args):
    todo = tasks(args.only.split(','))
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        for arm, name, status in p.map(solve, todo):
            print(json.dumps({'arm': arm, 'id': name, 'status': status}), flush=True)


def lumbar(path):
    p = Path(path)/'lumbar.json'
    return json.loads(p.read_text()) if p.exists() else None


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0]) if len(a) > 2 else None


def q(x):
    x = np.asarray([v for v in x if v is not None], float)
    return {'median': float(np.median(x)), 'q25': float(np.percentile(x, 25)), 'q75': float(np.percentile(x, 75)),
            'max_abs': float(np.abs(x).max()), 'n': int(len(x))} if len(x) else None


def e3_reference(d):
    """analyze_e3.classify_lumbar on the E3 work directory, in the system python E3 analysis runs in."""
    import subprocess
    code = ('import json, sys; sys.path[:0] = ["python/scripts", "utils"]; import analyze_e3 as A; '
            f'print("REF" + json.dumps(A.classify_lumbar({d["work"]!r}, {d["run"]!r}, A.geometry(A.OUT))[0]))')
    text = subprocess.run(['python3', '-c', code], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return json.loads(next(l for l in text.splitlines() if l.startswith('REF'))[3:])


def report_v1():
    rows, ok = [], True
    for d in equivalence_runs():
        mine = lumbar(OUT/'V1'/d['id']/'lumbar')
        if mine is None:
            rows.append({'id': d['id'], 'status': 'missing'}); ok = False; continue
        ref = e3_reference(d)
        keys = [k for k in ref['metrics'] if k in mine['metrics'] and (k in COMPARE or k.endswith('_activation_mean'))]
        delta = {k: mine['metrics'][k]-ref['metrics'][k] for k in keys}
        tol = {k: .01 for k in keys}
        passed = all(abs(delta[k]) <= tol[k] for k in keys) and mine['status'] == ('type1' if ref['type1'] else 'measured')
        ok &= passed
        rows.append({'id': d['id'], 'experiment': d['experiment'], 'arm': d['cell']['arm'], 'pass': passed, 'delta': delta,
                     'e3': {k: ref['metrics'][k] for k in keys}, 'type1_e3': ref['type1'], 'status_new': mine['status'],
                     'max_abs_delta': max(abs(v) for v in delta.values())})
    return {'rule': 'per run |delta| <= 0.01 (BW for compression; fraction; activation) and the same type-1 status', 'pass': ok, 'runs': rows}


def report_v2():
    per = []
    for c in truth_clips():
        a, b = lumbar(OUT/'V2'/c/'evalkit'), lumbar(OUT/'V2'/c/'truth')
        if a is None or b is None:
            per.append({'clip': c, 'status': 'missing'}); continue
        row = {'clip': c, 'status': [a['status'], b['status']]}
        for k in COMPARE+('L5_S1_ap_shear_p95_n', 'erector_spinae_activation_mean', 'psoas_activation_mean'):
            x, y = a['metrics'].get(k), b['metrics'].get(k)
            row[k] = {'estimated': x, 'truth': y, 'signed': None if x is None or y is None else x-y,
                      'relative': None if x is None or y is None or k.startswith('type2') or y == 0 else (x-y)/abs(y)}
        tool = json.loads((DATA/'tool/heldout2'/c/'evalkit.json').read_text())['receipt']
        import biomech
        import xml.etree.ElementTree as ET
        tg, lg, g = biomech.read_storage(ET.parse(DATA/'truth_eval/heldout2'/c/'grf.xml').getroot().find('.//datafile').text.strip())
        row['frames'] = {'floor_m': tool['floor_m'], 'floor_eval_m': tool['floor_eval_m'],
                         'truth_cop_height_m': float(np.median(g[:, lg.index('ground_force_r_py')]))}
        per.append(row)
    done = [r for r in per if r['status'] != 'missing' and r['status'][0] == 'measured' and r['status'][1] == 'measured']
    stats = {}
    for k in ('L5_S1_compression_p95_bw', 'L5_S1_compression_mean_bw', 'L5_S1_ap_shear_p95_n', 'erector_spinae_activation_mean',
              'psoas_activation_mean', 'type2_fraction'):
        pairs = [(r[k]['estimated'], r[k]['truth']) for r in done if r[k]['estimated'] is not None and r[k]['truth'] is not None]
        stats[k] = {'abs_relative': q([abs((x-y)/y) for x, y in pairs if y]) if not k.startswith('type2') else None,
                    'signed': q([x-y for x, y in pairs]), 'spearman': spearman(*zip(*pairs)) if len(pairs) > 2 else None}
    p95 = stats['L5_S1_compression_p95_bw']
    gate = {'spearman': p95['spearman'], 'median_abs_relative': p95['abs_relative']['median'] if p95['abs_relative'] else None}
    gate['pass'] = bool(gate['spearman'] is not None and gate['spearman'] >= .8 and gate['median_abs_relative'] is not None
                        and gate['median_abs_relative'] <= .15)
    return {'rule': 'Spearman >= 0.8 and median |relative error| <= 0.15 of L5_S1_compression_p95_bw over the clips (约定，未经推演)',
            'clips': len(per), 'both_measured': len(done), 'gate': gate, 'statistics': stats, 'per_clip': per}


def diff(a, b, k):
    """a - b for one metric; None when either run has no reachable frame (the metric is absent)."""
    x, y = a['metrics'].get(k), b['metrics'].get(k)
    return None if x is None or y is None else x-y


def report_v3_v4():
    v3, v4 = [], []
    for r in floor_rows():
        base = json.loads((V11/'rows'/r['id']/'evalkit.json').read_text()).get('upper', {}).get('lumbar')
        for s in ('-2cm', '+2cm'):
            shifted = json.loads((V11/'rows_floor2cm'/f"{r['id']}__floor{s}"/'evalkit.json').read_text()).get('upper', {}).get('lumbar')
            row = {'id': r['id'], 'method': r['method'], 'shift': s, 'status': [base and base['status'], shifted and shifted['status']],
                   'reachable_frames': [base and base.get('reachable_frames'), shifted and shifted.get('reachable_frames')]}
            if base and shifted and base['status'] == shifted['status'] == 'measured':
                row.update({k: diff(shifted, base, k) for k in COMPARE})
                p = row['L5_S1_compression_p95_bw']
                row['p95_relative'] = None if p is None else p/base['metrics']['L5_S1_compression_p95_bw']
            v3.append(row)
        e3 = lumbar(V11/'rows'/r['id']/'upper_e3/lumbar')
        if base and e3:
            v4.append({'id': r['id'], 'method': r['method'], 'status': [base['status'], e3['status']],
                       'reachable_frames': [base.get('reachable_frames'), e3.get('reachable_frames')],
                       'evalkit': base['metrics'], 'e3': e3['metrics'],
                       **({f'{k}_evalkit_minus_e3': diff(base, e3, k) for k in COMPARE}
                          if base['status'] == e3['status'] == 'measured' else {})})
    rel = [abs(x['p95_relative']) for x in v3 if x.get('p95_relative') is not None]
    t2 = [abs(x['type2_fraction']) for x in v3 if x.get('type2_fraction') is not None]
    return ({'descriptive': True, 'median_abs_relative_change_p95': float(np.median(rel)) if rel else None,
             'max_abs_relative_change_p95': float(np.max(rel)) if rel else None,
             'median_abs_change_type2_fraction': float(np.median(t2)) if t2 else None, 'rows': v3},
            {'descriptive': True, 'rows': v4})


def report(args):
    v3, v4 = report_v3_v4()
    out = {'protocol': 'config/evalkit_upper_muscle_20261108.json', 'V1_equivalence': report_v1(), 'V2_ground_truth_grf': report_v2(),
           'V3_floor': v3, 'V4_front_end': v4,
           'V5_addb': 'not executable: AddB measured.npz holds force vectors only (no centre of pressure, no free moment)',
           'errors': sorted(p.stem for p in (OUT/'errors').glob('*.json')) if (OUT/'errors').exists() else []}
    (OUT/'validation.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v.get('pass') if isinstance(v, dict) and 'pass' in v else v.get('gate') if isinstance(v, dict) and 'gate' in v else
                          {kk: vv for kk, vv in v.items() if kk != 'rows'} if isinstance(v, dict) else v) for k, v in out.items()}, indent=1))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest='step', required=True)
    r = sub.add_parser('run'); r.add_argument('--workers', type=int, default=5); r.add_argument('--only', default='V1,V2,V3,V4')
    sub.add_parser('report')
    args = a.parse_args()
    {'run': run_all, 'report': report}[args.step](args)


if __name__ == '__main__':
    main()
