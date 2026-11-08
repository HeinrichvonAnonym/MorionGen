#!/usr/bin/env python
"""Generation output contract check (config/motion_output_contract_20261108.json).

  motion_contract_check.py run DIR [--method M]   one run directory (Kimodo case directory, PRIMAL factorial file)
  motion_contract_check.py scan [--out DIR]       every legacy run: native matrix (analysis.json) + Kimodo pairs (audit.json)
  motion_contract_check.py matrix [--out DIR]     method x evaluator measurability from the capability / requirement tables

Each layer of a run is ok, legacy (the pre-contract stand-in listed in the contract is present) or gap (missing items).
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = json.loads((ROOT/'config/motion_output_contract_20261108.json').read_text())
DATA = Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
NATIVE_MATRIX = ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
KIMODO_AUDIT = ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json'
KIMODO_ROOT = DATA/'benchmarks/kimodo-stratified-20261008/generated_raw'
LAYERS = CONTRACT['layers']
CAP = {k: v for k, v in CONTRACT['method_capability'].items() if not k.startswith('_')}
REQ = {k: v for k, v in CONTRACT['evaluator_requirements'].items() if not k.startswith('_')}


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def layer(missing, legacy=False):
    return {'state': 'gap' if missing else 'legacy' if legacy else 'ok', 'missing': missing}


def detect(path, method=None):
    path = Path(path)
    if method:
        return method
    if path.suffix == '.npz' and path.parent.name == 'factorial':
        return 'primal'
    if (path/'meta.json').exists() and (path/'constraints.json').exists():
        return 'kimodo'
    m = read_json(path/'manifest.json')
    return m.get('baseline') if m else None


def check_l1(d, method):
    if method == 'kimodo':
        return layer([f for f in ('meta.json',) if not (d/f).exists()], legacy=True)
    m = read_json(d/'manifest.json') or {}
    miss = [f'manifest.json:{k}' for k in LAYERS['L1_identity']['required']['manifest.json'] if k not in m]
    miss += [f for f in ('weight_binding.json', 'intervention.json') if not (d/f).exists()]
    if not ((d/'environment.json').exists() or (d/'implementation.json').exists()):
        miss.append('environment.json|implementation.json')
    legacy = method == 'primal' and 'intervention.json' in miss
    return layer([f for f in miss if not (legacy and f == 'intervention.json')], legacy=legacy)


def check_l2(d, method):
    if method == 'kimodo':
        return layer([] if (d/'constraints.json').exists() and (d/'meta.json').exists() else ['constraints.json|meta.json'], legacy=True)
    s = read_json(d/'scenario.json')
    if s is None:
        s = (read_json(d/'manifest.json') or {}).get('scenario')
        legacy = s is not None
    else:
        legacy = False
    if s is None:
        return layer(['scenario.json'])
    fmt = ('root_goal_change' if 'goal_start' in s else 'continuation' if 'initial_history' in s else None) if 'kind' not in s else None
    if fmt:
        out = layer([], legacy=True)
        out.update(legacy_format=fmt, new_batch_missing=list(LAYERS['L2_task']['required_new_batch']['scenario.json']))
        return out
    miss = [f'scenario:{k}' for k in LAYERS['L2_task']['required']['scenario.json'] if k not in s]
    for c in s.get('constraints') or []:
        miss += [f'constraint:{k}' for k in LAYERS['L2_task']['constraint_item'] if k not in c and f'constraint:{k}' not in miss]
    new = [k for k in LAYERS['L2_task']['required_new_batch']['scenario.json'] if k not in s]
    out = layer(miss, legacy=legacy or bool(new))
    out['new_batch_missing'] = new
    return out


def motion_file(path, method):
    path = Path(path)
    return path if path.suffix == '.npz' else path/'motion.npz'


def check_l3(path, method):
    f = motion_file(path, method)
    if not f.exists():
        return layer([f.name])
    try:
        m = np.load(f, allow_pickle=False)
    except Exception as e:
        return layer([f'{f.name}: {type(e).__name__}'])
    if method == 'kimodo':
        j = m['posed_joints'] if 'posed_joints' in m.files else None
        return layer([] if j is not None and np.isfinite(j).all() else ['posed_joints'], legacy=True)
    miss = []
    j = m['joints'] if 'joints' in m.files else None
    if j is None:
        return layer(['joints'])
    j = j[0] if j.ndim == 4 else j
    if j.ndim != 3 or j.shape[-1] != 3:
        miss.append(f'joints shape {j.shape}')
    if not np.isfinite(j).all():
        miss.append('joints finite')
    legacy = method == 'primal'
    if 'joint_names' in m.files:
        if len(m['joint_names']) != j.shape[1]:
            miss.append('joint_names length')
    elif not legacy:
        miss.append('joint_names')
    if 'fps' in m.files:
        if not float(m['fps']) > 0:
            miss.append('fps')
    elif not legacy:
        miss.append('fps')
    if 'metadata_json' in m.files:
        meta = json.loads(str(m['metadata_json']))
        if meta.get('units') != 'm' or meta.get('axes') != 'right-handed Z-up':
            miss.append('metadata units / axes')
    elif not legacy:
        miss.append('metadata_json')
    return layer(miss, legacy=legacy)


def npz_keys(p):
    try:
        with np.load(p, allow_pickle=False) as m:
            return set(m.files)
    except Exception:
        return set()


LEGACY_KEYS = {'global_rotations': {'global_rotations'}, 'global_rot_mats': {'global_rot_mats'}, 'object_positions': {'object_positions'},
               'object_rotations': {'object_rotations'}, 'contact_forces': {'contact_forces'}, 'smplx_params': {'pose'}}


def check_l4(path, method):
    cap = CAP.get(method)
    if cap is None:
        return layer([f'unknown method {method}'])
    path = Path(path); d = path.parent.parent if method == 'primal' else path
    native = npz_keys(d/'native.npz') if (d/'native.npz').exists() else set()
    miss, legacy = [], False
    for k in cap['native_required']:
        if k in native:
            continue
        loc = cap.get('native_legacy', {}).get(k)
        if loc and k == 'features_263' and (d/loc).exists():
            legacy = True; continue
        if loc:
            p = path if (loc.startswith('factorial/') and method == 'primal') else d/loc
            if method == 'primal':
                p = path.with_name(path.name.replace('_canonical', ''))
            if npz_keys(p) & LEGACY_KEYS.get(k, {k}) and (method != 'primal' or 'betas' in npz_keys(p.parent/'initial_history.npz')):
                legacy = True; continue
        miss.append(k)
    if method == 'chois' and 'object_mass_kg' in miss:
        miss = [m for m in miss if m != 'object_mass_kg'] + ['object_mass_kg (no mass in any CHOIS output)']
    return layer(miss, legacy=legacy)


def check_l5(path, method):
    d = Path(path).parent.parent if method == 'primal' else Path(path)
    w = read_json(d/'world.json')
    if w is None:
        return layer(['world.json (legacy floors: config/evalkit_floor_reference_20261107.json)'])
    miss = [k for k in LAYERS['L5_world']['required'] if k not in w]
    if method == 'chois':
        miss += [k for k in LAYERS['L5_world']['objects']['required_if_object'] if k not in w]
    if method == 'closd' and 'contact_forces' not in npz_keys(d/'native_physics.npz'):
        miss.append('native_physics.npz contact_forces')
    return layer(miss)


def check_l6(path, method):
    d = Path(path).parent.parent if method == 'primal' else Path(path)
    integ = read_json(d/'integrity.json')
    if integ is None:
        m = read_json(d/'manifest.json') or {}
        if method == 'kimodo' or m.get('status') == 'success':
            return layer([], legacy=True)
        return layer(['integrity.json', f"manifest status {m.get('status')}"])
    miss = [k for k in ('sha256', 'done') if k not in integ]
    for name, h in (integ.get('sha256') or {}).items():
        p = d/name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != h:
            miss.append(f'sha256 {name}')
    return layer(miss)


def check_run(path, method=None):
    path = Path(path); method = detect(path, method)
    d = path.parent.parent if method == 'primal' else path
    return {'run': str(path), 'method': method,
            'L1_identity': check_l1(d, method), 'L2_task': check_l2(d, method), 'L3_motion': check_l3(path, method),
            'L4_native': check_l4(path, method), 'L5_world': check_l5(path, method), 'L6_integrity': check_l6(path, method)}


def legacy_runs():
    rows = json.loads(NATIVE_MATRIX.read_text())['rows']
    seen, out = set(), []
    for r in rows:
        run = ROOT/r['run']
        if r['method'] == 'primal':
            c = r['condition']
            run = run/'factorial'/f"ground{c['ground_snap']}_reproject{c['reprojection']}_inertialize{c['inertialization']}_canonical.npz"
        if run not in seen:
            seen.add(run); out.append((run, r['method']))
    for r in json.loads(KIMODO_AUDIT.read_text())['rows']:
        out.append((KIMODO_ROOT/r['case'], 'kimodo'))
    return out


def scan(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    results = [check_run(p, m) for p, m in legacy_runs()]
    summary = {}
    for m in sorted({r['method'] for r in results}):
        rs = [r for r in results if r['method'] == m]
        summary[m] = {'runs': len(rs)}
        for L in LAYERS:
            states = collections.Counter(r[L]['state'] for r in rs)
            missing = collections.Counter(x for r in rs for x in r[L]['missing'])
            summary[m][L] = {'states': dict(states), 'missing': dict(missing.most_common(8))}
            if L == 'L2_task':
                summary[m][L]['legacy_format'] = dict(collections.Counter(r[L].get('legacy_format', 'scenario') for r in rs))
                summary[m][L]['new_batch_missing'] = dict(collections.Counter(x for r in rs for x in r[L].get('new_batch_missing', [])))
    (out/'scan.json').write_text(json.dumps({'contract': CONTRACT['version'], 'summary': summary, 'runs': results}, indent=1))
    print(json.dumps(summary, indent=1))


def measurable(method, evaluator):
    """measurable: every needed capability native; to_verify: none absent, some still to verify; with_caveat: a
    conditional item the method lacks (e.g. object mass); not_measurable otherwise."""
    cap, req = CAP[method], REQ[evaluator]
    states = [cap.get(k, 'absent').split(' ')[0] for k in req['needs']]
    if 'absent' in states:
        return 'not_measurable'
    if 'to_verify' in states:
        return 'to_verify'
    caveats = [f'{k}: {v}' for k, v in req.get('conditional', {}).items()
               if cap.get(k, 'absent').split(' ')[0] == 'native' and v.split(' ')[0] not in cap.get(k, '')]
    return 'with_caveat' if caveats else 'measurable'


def matrix(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    mat = {m: {e: measurable(m, e) for e in REQ} for m in CAP}
    notes = {f'{m}/{e}': REQ[e]['conditional'] for m, row in mat.items() for e, v in row.items() if v == 'with_caveat'}
    (out/'matrix.json').write_text(json.dumps({'contract': CONTRACT['version'], 'matrix': mat, 'notes': notes}, indent=1))
    width = max(len(e) for e in REQ)
    print(' '*12 + ' '.join(e[:width].ljust(width) for e in REQ))
    for m, row in mat.items():
        print(m.ljust(12) + ' '.join(row[e].ljust(width) for e in REQ))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = a.add_subparsers(dest='step', required=True)
    r = sub.add_parser('run'); r.add_argument('path'); r.add_argument('--method')
    default = str(ROOT/'develop_logs/test_runs/20261108_contract')
    s = sub.add_parser('scan'); s.add_argument('--out', default=default)
    x = sub.add_parser('matrix'); x.add_argument('--out', default=default)
    args = a.parse_args()
    if args.step == 'run':
        return print(json.dumps(check_run(args.path, args.method), indent=1))
    {'scan': scan, 'matrix': matrix}[args.step](args)


if __name__ == '__main__':
    main()
