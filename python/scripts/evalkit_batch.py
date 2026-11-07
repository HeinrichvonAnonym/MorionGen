#!/usr/bin/env python
"""evalkit over a manifest of generated runs (residual force, RULA/REBA classes, muscle load, kinematics). OpenSim python.

  evalkit_batch.py manifest-e4 --per-method N --out M.json   draw N runs per method from the E4 control arm (C0: every
                                                             intervention off) with numpy default_rng(crc32(--draw-key))
  evalkit_batch.py run --manifest M.json --out DIR [--dims residual,ergo,muscle,kinematic] [--workers 6] [--dry-run]

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
SECONDS_PER_MOTION_SECOND = {'residual': 5., 'ergo': 3., 'muscle': 6., 'kinematic': 0.}  # timed on the synthetic set (3 s clips), --dry-run only


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


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest='step', required=True)
    m = sub.add_parser('manifest-e4'); m.add_argument('--per-method', type=int, default=5); m.add_argument('--out', required=True)
    m.add_argument('--draw-key', default='evalkit_p4_smoke')
    r = sub.add_parser('run'); r.add_argument('--manifest', required=True); r.add_argument('--out', required=True)
    r.add_argument('--dims', default='residual,ergo,muscle,kinematic'); r.add_argument('--workers', type=int, default=6)
    r.add_argument('--dry-run', action='store_true')
    args = a.parse_args()
    (manifest_e4 if args.step == 'manifest-e4' else run_batch)(args)


if __name__ == '__main__':
    main()
