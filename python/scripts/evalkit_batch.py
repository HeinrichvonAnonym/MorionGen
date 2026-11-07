#!/usr/bin/env python
"""evalkit over a manifest of generated runs (residual force, RULA/REBA classes, muscle load, kinematics). OpenSim python.

  evalkit_batch.py manifest-e4 --per-method N --out M.json   draw N runs per method from the E4 control arm (C0: every
                                                             intervention off) with numpy default_rng(crc32(--draw-key))
  evalkit_batch.py run --manifest M.json --out DIR [--dims residual,ergo,muscle,kinematic] [--workers 6] [--dry-run]
  evalkit_batch.py manifest-shift --manifest M.json --out M2.json [--shift 0.02]   the rows with the floor moved by +-shift
  evalkit_batch.py check --manifest M.json --rows DIR --out C.json [--shift-manifest M2.json --shift-rows DIR2]
                                                             per-method sanity medians, references, floor sensitivity

Manifest rows: {id, method, source, condition?, mass_kg?, floor_m?, reference?, object_free?, reach_task?}; floor, reference and
object-free default to evalkit.sources. Each row writes DIR/<id>/ (evalkit outputs); a row is skipped when its
evalkit.json has the same input sha256, dims, registration version, fit preparation and contact margin (resume); a row
missing only the kinematic dimension gets it from its saved frames without refitting.
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
E4_PLAN = ROOT/'develop_logs/test_runs/20261103_e4_cross_method/plan.json'
# --dry-run only: smoke8x5 median 18.2 wall s per motion s for the three (p90 24.1), split 5:3:6 as timed on the synthetic set
SECONDS_PER_MOTION_SECOND = {'residual': 6.5, 'ergo': 3.9, 'muscle': 7.8, 'kinematic': 0.}


def control_arm(job):
    """E4 C0: every intervention flag off; PRIMAL keeps ground_snap and reprojection on (its scored setting)."""
    c = job['condition']
    if job['method'] == 'primal':
        return c.get('inertialization') == 0
    return all(v == 0 for v in c.values())


def manifest_e4(args):
    jobs = [j for j in json.loads(E4_PLAN.read_text())['jobs'] if control_arm(j)]
    rng = np.random.default_rng(zlib.crc32(args.draw_key.encode()))
    rows = []
    for method in sorted({j['method'] for j in jobs}):
        pool = sorted((j for j in jobs if j['method'] == method), key=lambda j: j['id'])
        pick = sorted(rng.choice(len(pool), min(args.per_method, len(pool)), replace=False))
        rows += [{'id': pool[i]['id'], 'method': method, 'source': pool[i]['source'], 'condition': pool[i]['condition']} for i in pick]
    out = {'created': time.strftime('%Y-%m-%d %H:%M'), 'draw': {'plan': str(E4_PLAN.relative_to(ROOT)), 'arm': 'C0', 'per_method': args.per_method,
                                                              'key': args.draw_key, 'rng': 'numpy default_rng(crc32(key)), methods in sorted order'},
           'rows': rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f'{len(rows)} rows -> {args.out}')


def input_file(source):
    p = Path(source)
    return p if p.is_file() else p/'motion.npz'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def resolve(row):
    from evalkit import sources
    source = str(ROOT/row['source']) if not Path(row['source']).is_absolute() else row['source']
    if 'reference' in row and 'floor_m' in row:
        ref, floor = row['reference'], row['floor_m']
    else:
        ref, floor = sources.floor_variant(row['method'], source, row.get('condition'))
    free = row['object_free'] if 'object_free' in row else sources.object_free(row['method'], source)
    return source, ref, float(floor), bool(free)


def key(row, dims):
    from evalkit.fit import PREPARE
    from evalkit.spec import SPEC
    source = resolve(row)[0]
    return {'input_sha256': sha256(input_file(source)), 'dims': list(dims), 'spec_version': SPEC['version'], 'fit_prepare': PREPARE,
            'contact_margin_m': SPEC['residual']['contact_margin_m']}


def job(task):
    row, out_root, dims = task
    out = Path(out_root)/row['id']
    try:
        os.nice(19)
        k = key(row, dims)
        from evalkit import run
        source, ref, floor, free = resolve(row)
        if (out/'evalkit.json').exists():
            old = json.loads((out/'evalkit.json').read_text()).get('batch', {})
            if old == k:
                return row['id'], 'skip'
            if old and {**old, 'dims': k['dims']} == k and set(k['dims'])-set(old['dims']) == {'kinematic'}:
                run.add_kinematic(source, out, row.get('reach_task'))
                rec = json.loads((out/'evalkit.json').read_text()); rec['batch'] = k
                (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
                return row['id'], 'ok_kinematic_added'
        run.evaluate(source, out, floor, ref, row.get('mass_kg', 75.), method=row['method'], dims=dims, object_free=free,
                     reach_task=row.get('reach_task'))
        rec = json.loads((out/'evalkit.json').read_text())
        rec['batch'] = k; rec['row'] = row
        (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
        (out/'error.json').unlink(missing_ok=True)
        return row['id'], 'ok'
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        (out/'error.json').write_text(json.dumps({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-3000:]}))
        return row['id'], 'failed'


def run_batch(args):
    rows = json.loads(Path(args.manifest).read_text())['rows']
    dims = tuple(args.dims.split(','))
    if args.dry_run:
        import biomech_sources
        per = sum(SECONDS_PER_MOTION_SECOND[d] for d in dims)
        length = [(lambda p: p[0].shape[0]/p[2])(biomech_sources.load(resolve(r)[0])) for r in rows]
        print(json.dumps({'rows': len(rows), 'dims': dims, 'motion_s_total': round(sum(length), 1),
                          'estimated_wall_min': round(sum(length)*per/args.workers/60, 1),
                          'methods': {m: sum(r['method'] == m for r in rows) for m in sorted({r['method'] for r in rows})}}, indent=1))
        return
    Path(args.out).mkdir(parents=True, exist_ok=True)
    counts = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        for rid, status in p.map(job, [(r, args.out, dims) for r in rows]):
            counts[status] = counts.get(status, 0)+1
            print(json.dumps({'id': rid, 'status': status, **counts}), flush=True)
    (Path(args.out)/'batch.json').write_text(json.dumps({'manifest': str(args.manifest), 'dims': dims, 'counts': counts,
                                                          'finished': time.strftime('%Y-%m-%d %H:%M')}, indent=1))


def manifest_shift(args):
    """Same rows with the resolved floor moved by +-shift (reference unchanged): the floor-sensitivity arm."""
    src = json.loads(Path(args.manifest).read_text())
    rows = []
    for r in src['rows']:
        _, ref, floor, free = resolve(r)
        for s in (-args.shift, args.shift):
            rows.append({**r, 'id': f"{r['id']}__floor{s*100:+.0f}cm", 'base_id': r['id'], 'floor_shift_m': s,
                         'floor_m': floor+s, 'reference': ref, 'object_free': free})
    Path(args.out).write_text(json.dumps({'created': time.strftime('%Y-%m-%d %H:%M'), 'from': str(args.manifest),
                                          'shift_m': args.shift, 'rows': rows}, indent=1))
    print(f'{len(rows)} rows -> {args.out}')


def rows_summary(rows_dir, rows):
    out = {}
    for r in rows:
        d = Path(rows_dir)/r['id']
        if not (d/'evalkit.json').exists():
            continue
        e = json.loads((d/'evalkit.json').read_text()); f = np.load(d/'frames.npz', allow_pickle=True)
        sel = f['scored']; um = f['unmeasurable'][sel]; pos = np.flatnonzero(um)/max(len(um)-1, 1)
        rf = f['residual_bw'][sel]
        out[r['id']] = {
            'method': r['method'], 'seconds': float(f['times'][sel][-1]-f['times'][sel][0]),
            'residual_rms_bw': float(np.sqrt(np.mean(rf**2))), 'residual_p95_bw': float(np.percentile(rf, 95)),
            'residual_positive_share': e['residual']['positive_share'], 'penetration_share': e['residual']['penetration_share'],
            'no_contact_share': e['residual']['no_contact_share'], 'lowest_elevation_median_m': e['residual']['lowest_elevation_m']['median'],
            'unmeasurable_share': float(um.mean()),
            'unmeasurable_by_third': [float(np.mean((pos >= a) & (pos < b))) if len(pos) else 0. for a, b in ((0, 1/3), (1/3, 2/3), (2/3, 1.01))],
            'unmeasurable_ik_error_median_m': float(np.median(f['ik_max_error_m'][sel][um])) if um.any() else None,
            **{f'{m}_{c}': e['ergo'][m][f'{c}_share'] for m in ('reba', 'rula') if 'ergo' in e
               for c in ('unmeasurable', 'certain_risk', 'indeterminate', 'certain_acceptable')},
            **({f'muscle_{k}': e['muscle'][k] for k in ('muscle_effort_sum_a2_mean', 'muscle_activation_peak', 'reserve_rms_nm', 'reserve_peak_nm')}
               if 'muscle' in e else {}),
            **({'penetration_depth_p95_m': e['kinematic']['floor']['penetration_depth_p95_m'],
                'hover_height_p95_m': e['kinematic']['floor']['hover_height_p95_m'],
                'max_slip_m': e['kinematic']['skating']['max_slip_m'], 'jerk_0p1s': e['kinematic']['jerk_0p1s'],
                'bone_cv_max': e['kinematic']['bone_cv_max']} if 'kinematic' in e else {})}
    return out


def synthetic_controls():
    """Tool residual and muscle reserve on the development control clips (synthetic negatives, truth model)."""
    root = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/evalkit_20261107/tool/dev')
    rms, res = [], []
    for d in sorted(root.glob('*__control')):
        f = np.load(d/'frames.npz', allow_pickle=True); rms.append(float(np.sqrt(np.mean(f['residual_bw'][f['scored']]**2))))
        e = json.loads((d/'evalkit.json').read_text())
        if 'muscle' in e:
            res.append(e['muscle']['reserve_rms_nm'])
    q = lambda x: {'median': float(np.median(x)), 'max': float(np.max(x)), 'n': len(x)}
    return {'residual_rms_bw': q(rms), 'reserve_rms_nm': q(res) if res else None}


def check(args):
    """Sanity table of a smoke batch: per method medians, unmeasurable location, references, floor sensitivity."""
    rows = json.loads(Path(args.manifest).read_text())['rows']
    per = rows_summary(args.rows, rows)
    keys = sorted({k for v in per.values() for k in v if k not in ('method', 'unmeasurable_by_third')})
    methods = {}
    for m in sorted({v['method'] for v in per.values()}):
        vs = [v for v in per.values() if v['method'] == m]
        methods[m] = {'rows': len(vs), **{k: float(np.median([v[k] for v in vs if v.get(k) is not None]))
                                          for k in keys if any(v.get(k) is not None for v in vs)},
                      'unmeasurable_by_third': np.mean([v['unmeasurable_by_third'] for v in vs], 0).tolist()}
    addb = json.loads((ROOT/'develop_logs/test_runs/20261105_val_a/analysis.json').read_text())['variants']['kin30']['classes']
    out = {'rows_dir': str(args.rows), 'rows': len(per), 'methods': methods,
           'references': {'addb_kin30_residual_rms_bw_median': {c: v['V4_residual_rms_bw']['median'] for c, v in addb.items()},
                          'addb_note': 'AddBiomechanics kin30 (val_a 2026-11-05) whole-trial residual RMS per activity class, real people',
                          'synthetic_dev_controls': synthetic_controls()}}
    if args.shift_rows and args.shift_manifest:
        srows = json.loads(Path(args.shift_manifest).read_text())['rows']
        sper = rows_summary(args.shift_rows, srows)
        delta = {}
        for r in srows:
            if r['id'] in sper and r['base_id'] in per:
                a, b = per[r['base_id']], sper[r['id']]
                d = delta.setdefault(r['method'], {}).setdefault(f"{r['floor_shift_m']*100:+.0f}cm", [])
                d.append({k: b[k]-a[k] for k in ('residual_rms_bw', 'residual_positive_share', 'penetration_share', 'no_contact_share',
                                                  'reba_certain_risk', 'reba_certain_acceptable', 'rula_certain_risk', 'rula_certain_acceptable')
                          if a.get(k) is not None and b.get(k) is not None})
        out['floor_sensitivity'] = {m: {s: {k: float(np.median([x[k] for x in xs])) for k in xs[0]} for s, xs in d.items()}
                                    for m, d in delta.items()}
    out['per_row'] = per
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_row'}, indent=1))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest='step', required=True)
    m = sub.add_parser('manifest-e4'); m.add_argument('--per-method', type=int, default=5); m.add_argument('--out', required=True)
    m.add_argument('--draw-key', default='evalkit_p4_smoke')
    s = sub.add_parser('manifest-shift'); s.add_argument('--manifest', required=True); s.add_argument('--out', required=True)
    s.add_argument('--shift', type=float, default=.02)
    c = sub.add_parser('check'); c.add_argument('--manifest', required=True); c.add_argument('--rows', required=True)
    c.add_argument('--out', required=True); c.add_argument('--shift-manifest'); c.add_argument('--shift-rows')
    r = sub.add_parser('run'); r.add_argument('--manifest', required=True); r.add_argument('--out', required=True)
    r.add_argument('--dims', default='residual,ergo,muscle,kinematic'); r.add_argument('--workers', type=int, default=6)
    r.add_argument('--dry-run', action='store_true')
    args = a.parse_args()
    {'manifest-e4': manifest_e4, 'manifest-shift': manifest_shift, 'check': check, 'run': run_batch}[args.step](args)


if __name__ == '__main__':
    main()
