#!/usr/bin/env python
"""evalkit over a manifest of generated runs (residual force, RULA/REBA classes, muscle load, kinematics). OpenSim python.

  evalkit_batch.py manifest-e4 --per-method N --out M.json   draw N runs per method from the E4 control arm (C0: every
                                                             intervention off) with numpy default_rng(crc32(--draw-key))
  evalkit_batch.py run --manifest M.json --out DIR [--dims residual,ergo,muscle,kinematic] [--workers 6] [--dry-run]
  evalkit_batch.py manifest-shift --manifest M.json --out M2.json [--shift 0.02]   the rows with the floor moved by +-shift
  evalkit_batch.py manifest-self-floor --manifest M.json --rows DIR --out M3.json   self-floor arm, primary for the loads (floor at each
                                                             row's own still-landmark level, self_floor_shift)
  evalkit_batch.py check --manifest M.json --rows DIR --out C.json [--shift-manifest M2.json --shift-rows DIR2]
                         [--self-manifest M3.json --self-rows DIR3]
                                                             per-method sanity medians, references, floor sensitivity;
                                                             with the self-floor arm the v1.2 merged view (residual from
                                                             DIR, loads from DIR3, method_floor_* alongside)
  evalkit_batch.py refresh --manifest M.json --rows DIR      v1.2 forced metrics on existing rows (no new solve)
  evalkit_batch.py neck-rotation --manifest M.json --rows DIR   RULA with the rotation neck reading on rows with ergo
                                                             (sensitivity; ARDY / CHOIS / Kimodo / PRIMAL, cached fit)

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
import re
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
SECONDS_PER_MOTION_SECOND = {'residual': 6.5, 'ergo': 3.9, 'muscle': 7.8, 'kinematic': 0., 'upper': 30.}


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


def manifest_offset_reference(args):
    """Registered set of the frozen offset reference (config/evalkit_batch_protocol_20261108.json, offset_reference;
    user 2026-11-08 18:38): native matrix, paused clock, per method x condition x task the lowest seed, one row per
    distinct input (control-off MotionLCM / OmniControl repeat inputs across tasks); Kimodo (not in the native matrix)
    from E4, one case per text x constraint type x correction."""
    rows = json.loads(NATIVE_MATRIX.read_text())['rows']
    cells = {}
    for r in rows:
        if r['clock'] == 'paused':
            cells.setdefault((r['method'], json.dumps(r['condition'], sort_keys=True), r['task']), []).append(r)
    out, seen = [], set()
    for (method, cond, task), rs in sorted(cells.items()):
        r = min(rs, key=lambda r: r['seed'])
        run = ROOT/r['run'] if not Path(r['run']).is_absolute() else Path(r['run'])
        source = run/'factorial'/f"{r['variant']}_canonical.npz" if method == 'primal' else run
        h = sha256(input_file(source))
        if h in seen:
            continue
        seen.add(h)
        c = json.loads(cond)
        out.append({'id': f"{method}_{'_'.join(f'{k}{v}' for k, v in sorted(c.items()))}_t{task}_s{r['seed']}", 'method': method,
                    'source': str(source), 'condition': c, 'input_sha256': h})
    out += [{'id': j['id'], 'method': 'kimodo', 'source': j['source'], 'condition': j['condition']}
            for j in sorted(json.loads(E4_PLAN.read_text())['jobs'], key=lambda j: j['id']) if j['method'] == 'kimodo']
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({'created': time.strftime('%Y-%m-%d %H:%M'), 'arm': 'offset reference set',
                                          'from': [str(NATIVE_MATRIX.relative_to(ROOT)), str(E4_PLAN.relative_to(ROOT))], 'rows': out}, indent=1))
    print(json.dumps({m: sum(r['method'] == m for r in out) for m in sorted({r['method'] for r in out})}), f'-> {args.out}')


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


# dimensions an existing evaluation can gain from its saved fit and frames without refitting (muscle: only when the
# row has residual, whose contact decision and grf it uses)
ADDABLE = {'kinematic', 'upper', 'ergo', 'muscle'}
TASK_SEED = re.compile(r'_t(\d+)_s(\d+)$')


def row_dims(row, dims):
    """dims of one row: upper only on the lumbar subsample when the manifest marks it (row['upper'] False: not sampled)."""
    return tuple(d for d in dims if not (d == 'upper' and row.get('upper') is False))


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
            added = set(k['dims'])-set(old.get('dims', ()))
            if (old and {**old, 'dims': k['dims']} == k and set(old['dims']) <= set(k['dims']) and added <= ADDABLE
                    and ('muscle' not in added or 'residual' in old['dims'])):
                if added & {'ergo', 'muscle'}:
                    run.add_dims(source, out, [d for d in ('ergo', 'muscle') if d in added], object_free=free)
                if 'kinematic' in added:
                    run.add_kinematic(source, out, row.get('reach_task'))
                if 'upper' in added:
                    run.add_upper(source, out)
                rec = json.loads((out/'evalkit.json').read_text()); rec['batch'] = k
                (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
                return row['id'], 'ok_added_' + '_'.join(sorted(added))
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
        length = [(lambda p: p[0].shape[0]/p[2])(biomech_sources.load(resolve(r)[0])) for r in rows]
        per = [sum(SECONDS_PER_MOTION_SECOND[d] for d in row_dims(r, dims)) for r in rows]
        print(json.dumps({'rows': len(rows), 'dims': dims, 'upper_rows': sum('upper' in row_dims(r, dims) for r in rows),
                          'motion_s_total': round(sum(length), 1),
                          'estimated_wall_min': round(sum(x*y for x, y in zip(length, per))/args.workers/60, 1),
                          'methods': {m: sum(r['method'] == m for r in rows) for m in sorted({r['method'] for r in rows})}}, indent=1))
        return
    frozen = json.loads(Path(args.offset_reference).read_text()) if args.offset_reference else None
    if frozen:
        missing = sorted({r['method'] for r in rows if r.get('per_clip_correction', True)} - set(frozen['methods']))
        if missing:
            raise SystemExit(f'{args.offset_reference} has no offset reference for {missing}')
    Path(args.out).mkdir(parents=True, exist_ok=True)
    counts, refs = {}, None
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        if any(d in dims for d in ('residual', 'ergo', 'muscle')):
            for rid, status in p.map(offset_job, [(r, args.out) for r in rows]):
                counts[f'offset_{status}'] = counts.get(f'offset_{status}', 0)+1
                print(json.dumps({'id': rid, 'stage': 'offset', 'status': status}), flush=True)
            refs = batch = offset_references(rows, args.out)
            if frozen:
                refs = {m: {**v, 'source': str(args.offset_reference)} for m, v in frozen['methods'].items()}
            (Path(args.out)/'floor_offsets.json').write_text(json.dumps(
                refs if not frozen else {**refs, '_batch_median': batch, '_note': 'frozen reference used; _batch_median: drift diagnostic'},
                indent=1))
            print(json.dumps({'offset_reference': refs, **({'batch_median': batch} if frozen else {})}), flush=True)
        use = lambda r: refs and r['method'] in refs and r.get('per_clip_correction', True)
        tasks = [(r, args.out, row_dims(r, dims), refs[r['method']]['offset_reference_m'] if use(r) else None) for r in rows]
        for rid, status in p.map(job, tasks):
            counts[status] = counts.get(status, 0)+1
            print(json.dumps({'id': rid, 'status': status, **counts}), flush=True)
    (Path(args.out)/'batch.json').write_text(json.dumps({'manifest': str(args.manifest), 'dims': dims, 'counts': counts,
                                                          'floor_offsets': refs, 'finished': time.strftime('%Y-%m-%d %H:%M')}, indent=1))


def offset_reference(args):
    """Frozen per-method offset reference (config/evalkit_batch_protocol_20261108.json, offset_reference): stage 1 on
    the registered rows, their per-method median written once; batches then use it via run --offset-reference, so
    rows streamed in later do not move it."""
    rows = json.loads(Path(args.manifest).read_text())['rows']
    Path(args.rows).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        status = dict(p.map(offset_job, [(r, args.rows) for r in rows]))
    refs = offset_references(rows, args.rows)
    Path(args.out).write_text(json.dumps({
        'created': time.strftime('%Y-%m-%d %H:%M'), 'manifest': str(args.manifest), 'rows_dir': str(args.rows),
        'status': 'frozen once registered; never recomputed after any batch row is evaluated',
        'rows': {m: sorted(r['id'] for r in rows if r['method'] == m and status.get(r['id']) in ('ok', 'skip')) for m in refs},
        'failed': sorted(i for i, s in status.items() if s == 'failed'), 'methods': refs}, indent=1))
    print(json.dumps(refs, indent=1))


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


def lumbar_cell(row):
    """Stratum of the lumbar subsample (config/evalkit_batch_protocol_20261108.json, lumbar_subsample): method x condition
    x task x text condition x split (development / held-out); the seed orders rows inside it. task / seed from the row
    fields (output contract L2) or the row id (_t<task>_s<seed>); a row with neither (Kimodo cases: one sample per case)
    is its own stratum."""
    m = TASK_SEED.search(row['id'])
    task = row.get('task_id', int(m.group(1)) if m else row['id'])
    seed = row.get('seed', m and int(m.group(2)))
    return (row['method'], json.dumps(row.get('condition'), sort_keys=True), task, row.get('text_condition'), row.get('split')), seed


def manifest_lumbar_sample(args):
    """Same rows, row['upper'] True on the first --per-cell seeds of every stratum and False elsewhere (0: every row).
    Raising --per-cell later and rerunning adds upper to the newly sampled rows only (batch key, ADDABLE).
    --defer-split: sampled rows of these splits get upper False here; a later run without it adds their upper last."""
    src = json.loads(Path(args.manifest).read_text())
    cells = {}
    for r in src['rows']:
        c, seed = lumbar_cell(r)
        cells.setdefault(c, []).append((seed is None, seed, r['id']))
    chosen = {i for v in cells.values() for _, _, i in sorted(v)[:args.per_cell or None]}
    defer = set(filter(None, args.defer_split.split(',')))
    deferred = {r['id'] for r in src['rows'] if r['id'] in chosen and r.get('split') in defer}
    rows = [{**r, 'upper': r['id'] in chosen and r['id'] not in deferred} for r in src['rows']]
    Path(args.out).write_text(json.dumps({**{k: v for k, v in src.items() if k != 'rows'}, 'from': str(args.manifest),
                                          'lumbar_subsample': {'per_cell': args.per_cell, 'cells': len(cells), 'rows': len(chosen),
                                                               'deferred': len(deferred), 'defer_split': sorted(defer),
                                                               'created': time.strftime('%Y-%m-%d %H:%M')},
                                          'rows': rows}, indent=1))
    print(f'{len(chosen)-len(deferred)}/{len(rows)} rows sampled over {len(cells)} cells, {len(deferred)} deferred -> {args.out}')


def manifest_self_floor(args):
    """Descriptive self-floor arm (user decision 2026-11-07 21:19): every evaluated row again with its floor moved by
    self_floor_shift of its main-arm frames under DIR (relative to the floor that arm used), refitted on that floor,
    no per-clip correction, all dimensions. Separates 'does the method keep its own floor' from the dynamics,
    ergonomics and muscle load of the motion; since v1.2 (config/evalkit_v12_20261108.json) the primary arm of the
    load dimensions."""
    from evalkit.floor import SELF_FLOOR_STILL_M_S, self_floor_shift
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
                                          'arm': 'self_floor (v1.2: primary for the load dimensions)', 'rows': rows}, indent=1))
    print(f'{len(rows)} rows -> {args.out}')


def net_moments(d, e):
    """Net joint moments of the torque-driven Rajagopal trunk and arms (config/evalkit_upper_body_20261107.json, renamed
    by config/evalkit_upper_muscle_20261108.json: no trunk, neck or shoulder muscles, so not a load); read under the
    pre-rename keys or from the row's static optimization when the row predates them."""
    from evalkit import muscle, run
    new = [f'net_moment_{p}_nm_kg_{s}' for p in muscle.UPPER_BODY for s in ('mean', 'p95')]
    if all(k in e['muscle'] for k in new):
        return {k: e['muscle'][k] for k in new}
    if all(f'{p}_moment_nm_kg_{s}' in e['muscle'] for p in muscle.UPPER_BODY for s in ('mean', 'p95')):
        return run.net_moments(e['muscle'])
    if not (d/'work/so/so_so_force.sto').exists():
        return {}
    return run.net_moments(muscle.upper_body(d/'work/so', d/'work/model_so.osim'))


UPPER_METRICS = ('L5_S1_compression_p95_bw', 'L5_S1_compression_mean_bw', 'L5_S1_ap_shear_p95_n', 'L5_S1_over_niosh_fraction',
                 'type2_fraction', 'type2_external_fraction', 'type2_internal_fraction', 'type2_kinematic_fraction',
                 'erector_spinae_activation_mean', 'multifidus_activation_mean', 'quadratus_lumborum_activation_mean',
                 'psoas_activation_mean', 'abdominals_activation_mean', 'latissimus_activation_mean')


FORCED_UPPER = ('forced_L5_S1_compression_p95_bw', 'forced_L5_S1_compression_mean_bw', 'forced_residual_bw_p95')
MUSCLE_KEYS = ('muscle_effort_sum_a2_mean', 'muscle_activation_peak', 'reserve_rms_nm', 'reserve_peak_nm',
               'primary_muscle_effort_sum_a2_mean', 'forced_muscle_effort_sum_a2_mean', 'forced_residual_positive_share')


def upper(e):
    """Upper-body muscle dimension (config/evalkit_upper_muscle_20261108.json): region statuses and lumbar metrics;
    the lumbar metrics of type-1 rows are not reported; forced metrics of v1.2 (config/evalkit_v12_20261108.json)."""
    u = e['upper']; lumbar = u['lumbar']
    out = {f'upper_{r}_status': u[r]['status'] for r in ('lumbar', 'neck', 'shoulder')}
    if lumbar['status'] == 'measured':
        out.update({f'upper_lumbar_{k}': lumbar['metrics'].get(k) for k in UPPER_METRICS})
        out.update({f'upper_lumbar_{k}': lumbar.get('forced_metrics', {}).get(k) for k in FORCED_UPPER})
        if 'forced_frames' in lumbar:
            out['upper_lumbar_forced_external_share'] = lumbar['forced_external_frames']/max(lumbar['frames'], 1)
    return out


def refresh_job(task):
    """v1.2 forced metrics on an existing row from its saved files (run.add_forced, no new solve)."""
    rid, out = task
    try:
        os.nice(19)
        import opensim as osim
        osim.Logger.setLevelString('error')
        from evalkit import run
        run.add_forced(Path(out)/rid)
        return rid, 'ok'
    except Exception as e:
        return rid, f'failed: {type(e).__name__}: {e}'


def refresh(args):
    rows = [r for r in json.loads(Path(args.manifest).read_text())['rows'] if (Path(args.rows)/r['id']/'evalkit.json').exists()]
    counts = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        for rid, status in p.map(refresh_job, [(r['id'], args.rows) for r in rows]):
            counts[status.split(':')[0]] = counts.get(status.split(':')[0], 0)+1
            print(json.dumps({'id': rid, 'status': status, **counts}), flush=True)


def neck_rotation_job(task):
    """Rotation-reading RULA of one evaluated row (evalkit.neck_rotation); skipped while its key matches the row."""
    row, out_root = task
    out = Path(out_root)/row['id']
    try:
        os.nice(19)
        import opensim as osim
        osim.Logger.setLevelString('error')
        from evalkit import neck_rotation
        rec = json.loads((out/'evalkit.json').read_text())
        if rec.get('neck_rotation', {}).get('key') == neck_rotation.key(rec):
            return row['id'], 'skip'
        return row['id'], neck_rotation.evaluate(resolve(row)[0], out, rec['receipt']['method'] or row['method'])['status']
    except Exception as e:
        from evalkit import neck_rotation
        rec = json.loads((out/'evalkit.json').read_text())
        rec['neck_rotation'] = {'status': 'failed', 'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-3000:],
                                'key': neck_rotation.key(rec)}
        (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
        return row['id'], 'failed'


def neck_rotation(args):
    rows = [r for r in json.loads(Path(args.manifest).read_text())['rows'] if (Path(args.rows)/r['id']/'evalkit.json').exists()]
    counts = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        for rid, status in p.map(neck_rotation_job, [(r, args.rows) for r in rows]):
            counts[status] = counts.get(status, 0)+1
    print(json.dumps({'rows': len(rows), **counts}), flush=True)


def neck_rotation_fields(e):
    """check fields of the rotation-reading RULA (sensitivity; the evalkit rula_* stay the joint-line reading)."""
    n = e.get('neck_rotation')
    if n is None:
        return {}
    out = {'rula_neck_rotation_status': n['status']}
    if n['status'] == 'ok':
        out.update({f'rula_neck_rotation_{c}': n['rula'][f'{c}_share'] for c in ('unmeasurable', 'certain_risk', 'indeterminate', 'certain_acceptable')},
                   rula_neck_rotation_class_changed_share=n['class_changed_share'],
                   rula_neck_rotation_band_changed_share=n['neck_band_changed_share'],
                   rula_neck_rotation_b_minus_a_flexion_median_deg=n['b_minus_a_flexion_median_deg'])
    return out


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
            'row_status': 'partial_failed' if (d/'error.json').exists() else 'ok',
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
            **neck_rotation_fields(e),
            **({f'muscle_{k}': e['muscle'][k] for k in MUSCLE_KEYS if k in e['muscle']} if 'muscle' in e else {}),
            **(net_moments(d, e) if 'muscle' in e else {}),
            **(upper(e) if 'upper' in e else {'upper_lumbar_status': 'failed'} if (d/'error.json').exists()
               else {'upper_lumbar_status': 'not_sampled'} if r.get('upper') is False else {}),
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


LOAD_PREFIXES = ('reba_', 'rula_', 'muscle_', 'net_moment_', 'upper_')


def merge_self_floor(per, srows, srows_dir):
    """v1.2 merged view (config/evalkit_v12_20261108.json): residual and kinematics from the method-floor row, every load
    dimension from its self-floor row (primary), the method-floor loads as method_floor_* (descriptive); rows without a
    self-floor evaluation keep no primary load."""
    sper = rows_summary(srows_dir, srows)
    by_base = {r['base_id']: (r, sper.get(r['id'])) for r in srows}
    out = {}
    for rid, v in per.items():
        load = {k: x for k, x in v.items() if k.startswith(LOAD_PREFIXES)}
        row = {**{k: x for k, x in v.items() if k not in load}, **{f'method_floor_{k}': x for k, x in load.items()}}
        r, s = by_base.get(rid, (None, None))
        if s is not None:
            row.update({k: x for k, x in s.items() if k.startswith(LOAD_PREFIXES)})
            row.update(self_floor_shift_m=r['self_floor']['shift_m'], self_floor_residual_positive_share=s['residual_positive_share'],
                       self_floor_no_contact_share=s['no_contact_share'])
        out[rid] = row
    return out


def check(args):
    """Sanity table of a smoke batch: per method medians, unmeasurable location, references, floor sensitivity.
    Rows with byte-identical inputs (E4 C0 MotionLCM / OmniControl: the 12 tasks share 2 texts, so control-off runs
    repeat per text and seed) enter the per-method medians once; rows and unique_inputs are both reported. Methods whose
    distinct inputs share a few prompts also get per-prompt medians (by_prompt): outcomes can split by prompt (E4 C0
    OmniControl: left-hand reach floats at the method floor, right-hand reach does not), where one median misleads."""
    rows = json.loads(Path(args.manifest).read_text())['rows']
    raw = per = rows_summary(args.rows, rows)
    selffloor = bool(args.self_manifest and args.self_rows)
    per = merge_self_floor(per, json.loads(Path(args.self_manifest).read_text())['rows'] if selffloor else [], args.self_rows)
    status = lambda k: k.endswith('_status') and k != 'row_status'
    keys = sorted({k for v in per.values() for k in v if k not in ('method', 'unmeasurable_by_third', 'input_sha256', 'prompt', 'row_status')
                   and not status(k)})
    medians = lambda vs: {k: float(np.median([v[k] for v in vs if v.get(k) is not None])) for k in keys if any(v.get(k) is not None for v in vs)}
    methods = {}
    for m in sorted({v['method'] for v in per.values()}):
        allv = [v for v in per.values() if v['method'] == m]
        seen, vs = set(), []
        for v in allv:
            if v['input_sha256'] is None or v['input_sha256'] not in seen:
                seen.add(v['input_sha256']); vs.append(v)
        prompts = sorted({v['prompt'] for v in vs if v['prompt']})
        statuses = {k: {s: sum(v.get(k) == s for v in vs) for s in sorted({v[k] for v in vs if k in v})}
                    for k in sorted({k for v in vs for k in v if status(k)})}
        methods[m] = {'rows': len(allv), 'unique_inputs': len(vs),
                      'row_status': {s: sum(v['row_status'] == s for v in allv) for s in ('ok', 'partial_failed')},
                      'not_evaluated': sum(r['method'] == m and r['id'] not in per for r in rows), **medians(vs),
                      **{name: part for name, part in (('upper_status', {k: v for k, v in statuses.items() if 'upper_' in k}),
                                                       ('neck_rotation_status', {k: v for k, v in statuses.items() if 'neck_rotation' in k})) if part},
                      'unmeasurable_by_third': np.mean([v['unmeasurable_by_third'] for v in vs], 0).tolist(),
                      **({'by_prompt': {p: {'unique_inputs': sum(v['prompt'] == p for v in vs), **medians([v for v in vs if v['prompt'] == p])}
                                        for p in prompts}} if 1 < len(prompts) < len(vs) else {})}
    addb = json.loads((ROOT/'develop_logs/test_runs/20261105_val_a/analysis.json').read_text())['variants']['kin30']['classes']
    out = {'rows_dir': str(args.rows), 'rows': len(per),
           'load_arm': 'self_floor (primary), method_floor_* descriptive' if selffloor
           else 'method_floor_* only (descriptive; the primary self-floor loads were not evaluated)', 'methods': methods,
           'references': {'addb_kin30_residual_rms_bw_median': {c: v['V4_residual_rms_bw']['median'] for c, v in addb.items()},
                          'addb_note': 'AddBiomechanics kin30 (val_a 2026-11-05) whole-trial residual RMS per activity class, real people',
                          'synthetic_dev_controls': synthetic_controls()}}
    if args.shift_rows and args.shift_manifest:
        srows = json.loads(Path(args.shift_manifest).read_text())['rows']
        sper = rows_summary(args.shift_rows, srows)
        delta = {}
        for r in srows:
            if r['id'] in sper and r['base_id'] in raw:
                a, b = raw[r['base_id']], sper[r['id']]
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
    c.add_argument('--self-manifest', help='v1.2: the self-floor manifest; loads are then taken from --self-rows (primary)')
    c.add_argument('--self-rows')
    x = sub.add_parser('refresh', help='v1.2 forced metrics on existing rows from their saved files (no new solve)')
    x.add_argument('--manifest', required=True); x.add_argument('--rows', required=True); x.add_argument('--workers', type=int, default=6)
    r = sub.add_parser('run'); r.add_argument('--manifest', required=True); r.add_argument('--out', required=True)
    r.add_argument('--dims', default='residual,ergo,muscle,kinematic', help='add upper for the upper-body muscle dimension')
    r.add_argument('--workers', type=int, default=6)
    r.add_argument('--dry-run', action='store_true')
    r.add_argument('--offset-reference', help='frozen per-method offset reference (offset-reference); default: the manifest median')
    mo = sub.add_parser('manifest-offset-reference', help='the registered rows of the frozen offset reference')
    mo.add_argument('--out', required=True)
    o = sub.add_parser('offset-reference', help='freeze the per-method offset reference from registered rows')
    o.add_argument('--manifest', required=True); o.add_argument('--rows', required=True); o.add_argument('--out', required=True)
    o.add_argument('--workers', type=int, default=6)
    f = sub.add_parser('manifest-self-floor'); f.add_argument('--manifest', required=True); f.add_argument('--rows', required=True)
    f.add_argument('--out', required=True)
    ls = sub.add_parser('manifest-lumbar-sample', help='mark the lumbar (upper) subsample per method x condition x task x text')
    ls.add_argument('--manifest', required=True); ls.add_argument('--out', required=True)
    ls.add_argument('--per-cell', type=int, default=1, help='seeds per stratum; 0: every row')
    ls.add_argument('--defer-split', default='', help='comma-separated split values whose sampled rows get upper in a final fill-in')
    u = sub.add_parser('manifest-unique-c0'); u.add_argument('--out', required=True)
    n = sub.add_parser('neck-rotation', help='RULA with the rotation neck reading on evaluated rows (sensitivity)')
    n.add_argument('--manifest', required=True); n.add_argument('--rows', required=True); n.add_argument('--workers', type=int, default=6)
    args = a.parse_args()
    {'manifest-e4': manifest_e4, 'manifest-shift': manifest_shift, 'manifest-self-floor': manifest_self_floor, 'check': check,
     'run': run_batch, 'manifest-unique-c0': manifest_unique_c0, 'refresh': refresh, 'neck-rotation': neck_rotation,
     'manifest-lumbar-sample': manifest_lumbar_sample, 'offset-reference': offset_reference,
     'manifest-offset-reference': manifest_offset_reference}[args.step](args)


if __name__ == '__main__':
    main()
