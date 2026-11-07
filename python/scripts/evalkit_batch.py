#!/usr/bin/env python
"""evalkit over a manifest of generated runs (residual force, RULA/REBA classes, muscle load, kinematics). OpenSim python.

  evalkit_batch.py manifest-e4 --per-method N --out M.json   draw N runs per method from the E4 control arm (C0: every
                                                             intervention off) with numpy default_rng(crc32(--draw-key))
  evalkit_batch.py run --manifest M.json --out DIR [--dims residual,ergo,muscle,kinematic] [--workers 6] [--dry-run]
  evalkit_batch.py manifest-shift --manifest M.json --out M2.json [--shift 0.02]   the rows with the floor moved by +-shift
  evalkit_batch.py manifest-self-floor --manifest M.json --rows DIR --out M3.json   descriptive self-floor arm (floor at each
                                                             row's own still-landmark level, self_floor_shift)
  evalkit_batch.py check --manifest M.json --rows DIR --out C.json [--shift-manifest M2.json --shift-rows DIR2]
                                                             per-method sanity medians, references, floor sensitivity

Manifest rows: {id, method, source, condition?, mass_kg?, floor_m?, reference?, object_free?, reach_task?}; floor, reference and
object-free default to evalkit.sources. run has two stages: every row is fitted and its foot offset written
(DIR/<id>/foot_offset.json), then the per-method median offsets (DIR/floor_offsets.json, spec floor.per_clip_correction)
set the corrected floor of every row's evaluation. Each row writes DIR/<id>/ (evalkit outputs); a row is skipped when its
evalkit.json has the same input sha256, dims, registration version, fit preparation, contact margin and method offset
(resume; adding rows of a method moves its median, so its rows are re-scored on the cached fit); a row missing only the
kinematic dimension gets it from its saved frames without refitting.
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


NATIVE_MATRIX = ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
TEXT_ONLY_C0 = {'motionlcm': {'control_residual': 0}, 'omnicontrol': {'realism': 0, 'spatial': 0}}


def manifest_unique_c0(args):
    """Text-only C0 supplement (user decision 2026-11-07 23:26, no new generation): with control off these methods see only
    the text, and the 12 tasks share 2 texts, so E4 holds 4 distinct motions per method. The native matrix (seeds 20-24)
    holds 10; one row per distinct input (lowest task), ids as E4."""
    rows = json.loads(NATIVE_MATRIX.read_text())['rows']
    out, seen = [], {}
    for method, c0 in TEXT_ONLY_C0.items():
        pool = sorted((r for r in rows if r['method'] == method and r['clock'] == 'paused'
                       and all(r['condition'].get(k) == v for k, v in c0.items())), key=lambda r: (r['seed'], r['task']))
        for r in pool:
            h = sha256(input_file(ROOT/r['run'] if not Path(r['run']).is_absolute() else r['run']))
            if h in seen:
                seen[h]['duplicates'] += 1; continue
            name = '_'.join(f'{k}{v}' for k, v in sorted(c0.items()))
            seen[h] = {'id': f"{method}_{name}_t{r['task']}_s{r['seed']}", 'method': method, 'source': str(ROOT/r['run']) if not Path(r['run']).is_absolute() else r['run'],
                       'condition': c0, 'input_sha256': h, 'duplicates': 0}
            out.append(seen[h])
    Path(args.out).write_text(json.dumps({'created': time.strftime('%Y-%m-%d %H:%M'), 'from': str(NATIVE_MATRIX.relative_to(ROOT)),
                                          'arm': 'C0 text-only methods, distinct inputs', 'rows': out}, indent=1))
    print(json.dumps({m: sum(r['method'] == m for r in out) for m in TEXT_ONLY_C0}), f'-> {args.out}')


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


def key(row, dims, offset_reference=None):
    from evalkit.fit import PREPARE
    from evalkit.spec import SPEC
    source = resolve(row)[0]
    return {'input_sha256': sha256(input_file(source)), 'dims': list(dims), 'spec_version': SPEC['version'], 'fit_prepare': PREPARE,
            'contact_margin_m': SPEC['residual']['contact_margin_m'], 'offset_reference_m': offset_reference}


def offset_job(task):
    """Stage 1: fit and foot offset of one row (spec floor.per_clip_correction), cached as DIR/<id>/foot_offset.json."""
    row, out_root = task
    out = Path(out_root)/row['id']
    try:
        os.nice(19)
        from evalkit import run
        from evalkit.fit import PREPARE
        source, _, floor, _ = resolve(row)
        k = {'input_sha256': sha256(input_file(source)), 'fit_prepare': PREPARE, 'floor_m': floor, 'mass_kg': row.get('mass_kg', 75.)}
        path = out/'foot_offset.json'
        if path.exists() and json.loads(path.read_text()).get('key') == k:
            return row['id'], 'skip'
        d = run.foot_offset(source, out/'work', k['mass_kg'], floor)[0]
        path.write_text(json.dumps({'key': k, 'foot_offset_m': d}))
        return row['id'], 'ok'
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        (out/'error.json').write_text(json.dumps({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-3000:]}))
        return row['id'], 'failed'


def offset_references(rows, out_root):
    """D per method: median foot offset over the manifest rows of the method whose stage 1 succeeded (rows with
    per_clip_correction false, e.g. the self-floor arm, neither use nor enter it)."""
    by = {}
    for r in rows:
        p = Path(out_root)/r['id']/'foot_offset.json'
        if p.exists() and r.get('per_clip_correction', True):
            by.setdefault(r['method'], []).append(json.loads(p.read_text())['foot_offset_m']['mean'])
    return {m: {'offset_reference_m': float(np.median(v)), 'rows': len(v)} for m, v in sorted(by.items())}


def job(task):
    row, out_root, dims, offset_reference = task
    out = Path(out_root)/row['id']
    try:
        os.nice(19)
        k = key(row, dims, offset_reference)
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
                     reach_task=row.get('reach_task'), offset_reference=offset_reference)
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
    counts, refs = {}, None
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        if any(d in dims for d in ('residual', 'ergo', 'muscle')):
            for rid, status in p.map(offset_job, [(r, args.out) for r in rows]):
                counts[f'offset_{status}'] = counts.get(f'offset_{status}', 0)+1
                print(json.dumps({'id': rid, 'stage': 'offset', 'status': status}), flush=True)
            refs = offset_references(rows, args.out)
            (Path(args.out)/'floor_offsets.json').write_text(json.dumps(refs, indent=1))
            print(json.dumps({'offset_reference': refs}), flush=True)
        use = lambda r: refs and r['method'] in refs and r.get('per_clip_correction', True)
        tasks = [(r, args.out, dims, refs[r['method']]['offset_reference_m'] if use(r) else None) for r in rows]
        for rid, status in p.map(job, tasks):
            counts[status] = counts.get(status, 0)+1
            print(json.dumps({'id': rid, 'status': status, **counts}), flush=True)
    (Path(args.out)/'batch.json').write_text(json.dumps({'manifest': str(args.manifest), 'dims': dims, 'counts': counts,
                                                          'floor_offsets': refs, 'finished': time.strftime('%Y-%m-%d %H:%M')}, indent=1))


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


DEPTH_BAND_M = (-.005, .005)


def positive_share(f, sel, depth):
    """Residual-positive share of the scored frames at the registered tau and the given penetration depth."""
    from evalkit.spec import SPEC
    tau = SPEC['residual']['tau_bw']; st = f['state']
    bound = np.select([st == 'event', st == 'contact'], [tau['event'], tau['contact']], tau['flight'])
    return float(((f['residual_bw'] > bound) | (f['lowest_elevation_m'] < -depth) | f['unmeasurable'])[sel].mean())


SELF_FLOOR_STILL_M_S = .05


def self_floor_shift(f, still=SELF_FLOOR_STILL_M_S):
    """Median over the scored frames of the lowest sole-landmark elevation among the landmarks whose vertical speed
    is below `still` (m/s, convention, not derived). Checked on known-floor data before use
    (scratch/self_floor_estimators.py): synthetic walk -0.6 cm, other synthetic -0.3 cm, CLoSD -0.2 cm, PRIMAL +0.1 cm;
    the 10th percentile of the lowest elevation was biased by the rolling foot in walking (-1.8 / -1.3 cm)."""
    E = np.stack([f[f'elevation_{s}_{k}'] for s in 'rl' for k in ('heel', 'toe')], 1)
    E = np.where(np.abs(np.gradient(E, f['times'], axis=0)) < still, E, np.inf).min(1)
    ok = f['scored'] & np.isfinite(E)
    return float(np.median(E[ok])) if ok.any() else float(np.median(f['lowest_elevation_m'][f['scored']]))


def manifest_self_floor(args):
    """Descriptive self-floor arm (user decision 2026-11-07 21:19): every evaluated row again with its floor moved by
    self_floor_shift of its main-arm frames under DIR (relative to the floor that arm used), refitted on that floor,
    no per-clip correction, all dimensions. Separates 'does the method keep its own floor' from the dynamics,
    ergonomics and muscle load of the motion."""
    src = json.loads(Path(args.manifest).read_text())
    rows = []
    for r in src['rows']:
        d = Path(args.rows)/r['id']
        if not (d/'evalkit.json').exists():
            continue
        rec = json.loads((d/'evalkit.json').read_text())['receipt']; f = np.load(d/'frames.npz', allow_pickle=True)
        base = rec.get('floor_eval_m', rec['floor_m'])
        shift = self_floor_shift(f)
        _, ref, _, free = resolve(r)
        rows.append({**r, 'id': f"{r['id']}__selffloor", 'base_id': r['id'], 'floor_m': base+shift, 'reference': ref, 'object_free': free,
                     'per_clip_correction': False,
                     'self_floor': {'estimator': f'still-landmark median, {SELF_FLOOR_STILL_M_S} m/s', 'main_floor_eval_m': base, 'shift_m': shift}})
    Path(args.out).write_text(json.dumps({'created': time.strftime('%Y-%m-%d %H:%M'), 'from': str(args.manifest), 'rows_dir': str(args.rows),
                                          'arm': 'self_floor (descriptive)', 'rows': rows}, indent=1))
    print(f'{len(rows)} rows -> {args.out}')


def upper_body(d, e):
    """Descriptive lumbar / shoulder moments (config/evalkit_upper_body_20261107.json); read from the row's static
    optimization when the row predates them."""
    from evalkit import muscle
    keys = [f'{p}_moment_nm_kg_{s}' for p in muscle.UPPER_BODY for s in ('mean', 'p95')]
    if all(k in e['muscle'] for k in keys):
        return {f'muscle_{k}': e['muscle'][k] for k in keys}
    if not (d/'work/so/so_so_force.sto').exists():
        return {}
    up = muscle.upper_body(d/'work/so', d/'work/model_so.osim')
    return {f'muscle_{k}': up[k] for k in keys}


def rows_summary(rows_dir, rows):
    """Per-row sanity values; residual_positive_share_depth_{-,+}5mm: descriptive band at the registered penetration
    depth -+ 0.5 cm (generated feet often sit near it, so the binary share can flip there)."""
    from evalkit.spec import SPEC
    from evalkit import sources
    depth = SPEC['residual']['penetration_depth_m']
    out = {}
    for r in rows:
        d = Path(rows_dir)/r['id']
        if not (d/'evalkit.json').exists():
            continue
        e = json.loads((d/'evalkit.json').read_text()); f = np.load(d/'frames.npz', allow_pickle=True)
        sel = f['scored']; um = f['unmeasurable'][sel]; pos = np.flatnonzero(um)/max(len(um)-1, 1)
        rf = f['residual_bw'][sel]
        out[r['id']] = {
            'method': r['method'], 'input_sha256': e.get('batch', {}).get('input_sha256'),
            'prompt': e['receipt'].get('prompt') or sources.prompt(e['receipt']['run']),
            'seconds': float(f['times'][sel][-1]-f['times'][sel][0]),
            'residual_rms_bw': float(np.sqrt(np.mean(rf**2))), 'residual_p95_bw': float(np.percentile(rf, 95)),
            'residual_positive_share': e['residual']['positive_share'], 'penetration_share': e['residual']['penetration_share'],
            **{f'residual_positive_share_depth_{b*1000:+.0f}mm': positive_share(f, sel, depth+b) for b in DEPTH_BAND_M},
            'floor_correction_m': e['receipt']['floor_eval_m']-e['receipt']['floor_m'] if 'floor_eval_m' in e['receipt'] else None,
            'no_contact_share': e['residual']['no_contact_share'], 'lowest_elevation_median_m': e['residual']['lowest_elevation_m']['median'],
            'unmeasurable_share': float(um.mean()),
            'unmeasurable_by_third': [float(np.mean((pos >= a) & (pos < b))) if len(pos) else 0. for a, b in ((0, 1/3), (1/3, 2/3), (2/3, 1.01))],
            'unmeasurable_ik_error_median_m': float(np.median(f['ik_max_error_m'][sel][um])) if um.any() else None,
            **{f'{m}_{c}': e['ergo'][m][f'{c}_share'] for m in ('reba', 'rula') if 'ergo' in e
               for c in ('unmeasurable', 'certain_risk', 'indeterminate', 'certain_acceptable')},
            **({f'muscle_{k}': e['muscle'][k] for k in ('muscle_effort_sum_a2_mean', 'muscle_activation_peak', 'reserve_rms_nm', 'reserve_peak_nm')}
               if 'muscle' in e else {}),
            **(upper_body(d, e) if 'muscle' in e else {}),
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
    """Sanity table of a smoke batch: per method medians, unmeasurable location, references, floor sensitivity.
    Rows with byte-identical inputs (E4 C0 MotionLCM / OmniControl: the 12 tasks share 2 texts, so control-off runs
    repeat per text and seed) enter the per-method medians once; rows and unique_inputs are both reported. Methods whose
    distinct inputs share a few prompts also get per-prompt medians (by_prompt): outcomes can split by prompt (E4 C0
    OmniControl: left-hand reach floats at the method floor, right-hand reach does not), where one median misleads."""
    rows = json.loads(Path(args.manifest).read_text())['rows']
    per = rows_summary(args.rows, rows)
    keys = sorted({k for v in per.values() for k in v if k not in ('method', 'unmeasurable_by_third', 'input_sha256', 'prompt')})
    medians = lambda vs: {k: float(np.median([v[k] for v in vs if v.get(k) is not None])) for k in keys if any(v.get(k) is not None for v in vs)}
    methods = {}
    for m in sorted({v['method'] for v in per.values()}):
        allv = [v for v in per.values() if v['method'] == m]
        seen, vs = set(), []
        for v in allv:
            if v['input_sha256'] is None or v['input_sha256'] not in seen:
                seen.add(v['input_sha256']); vs.append(v)
        prompts = sorted({v['prompt'] for v in vs if v['prompt']})
        methods[m] = {'rows': len(allv), 'unique_inputs': len(vs), **medians(vs),
                      'unmeasurable_by_third': np.mean([v['unmeasurable_by_third'] for v in vs], 0).tolist(),
                      **({'by_prompt': {p: {'unique_inputs': sum(v['prompt'] == p for v in vs), **medians([v for v in vs if v['prompt'] == p])}
                                        for p in prompts}} if 1 < len(prompts) < len(vs) else {})}
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
    f = sub.add_parser('manifest-self-floor'); f.add_argument('--manifest', required=True); f.add_argument('--rows', required=True)
    f.add_argument('--out', required=True)
    u = sub.add_parser('manifest-unique-c0'); u.add_argument('--out', required=True)
    args = a.parse_args()
    {'manifest-e4': manifest_e4, 'manifest-shift': manifest_shift, 'manifest-self-floor': manifest_self_floor, 'check': check,
     'run': run_batch, 'manifest-unique-c0': manifest_unique_c0}[args.step](args)


if __name__ == '__main__':
    main()
