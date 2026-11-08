#!/usr/bin/env python
"""Contract layers L4-L6 of new batch runs (config/motion_output_contract_20261108.json) written after the generation,
without touching any file the generator wrote (the generation system is not changed: contract scope).

  complete_run_contract.py complete --batch DIR    every run of DIR/queue*/state.json
  complete_run_contract.py scan --batch DIR        contract check of those runs and of the linked earlier runs (DIR/plans/links.json)

complete, per successful run (method from manifest.json):
  native.npz  the method_capability.<method>.native_required arrays (and native_optional ones present) copied from the
              native_legacy locations, with native_sources (json: key -> file:array); DART smplx_params is packed as one
              (T, D) float32 array with smplx_params_layout (json: name -> [start, end, shape]; gender, pelvis_delta).
              PRIMAL: none; one run holds the 8 switch combinations (contract L1 legacy), its native data stay in factorial/.
  world.json  method_world, method_world_sha256, floor_variant (ARDY correction=1 when corrected, else default); CHOIS also
              object_id, object_mass_kg null, object_mass_source (contract method_capability.chois.object_mass).
  integrity.json  sha256 of the L2-L5 files (and the native source files), done: true; written last.
Runs whose status is not success keep their directory and get failure.json (status, reason); never replaced by another seed.
Rerunning skips runs whose integrity.json verifies.

scan gate (exit 1 otherwise): new runs ok in every layer except the contract-listed PRIMAL legacy (L1 identity, L3
motion, L4 native); linked earlier runs legacy or ok (no gap) in every layer.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import motion_contract_check as mcc  # noqa: E402

CONTRACT = json.loads((ROOT/'config/motion_output_contract_20261108.json').read_text())
CAP = {k: v for k, v in CONTRACT['method_capability'].items() if not k.startswith('_')}
PRIMAL_LEGACY = {'L1_identity', 'L3_motion', 'L4_native'}
LAYERS = ('L1_identity', 'L2_task', 'L3_motion', 'L4_native', 'L5_world', 'L6_integrity')
NATIVE_SOURCE = {'ardy': {'global_rotations': 'motion.npz'},
                 'chois': {'global_rotations': 'raw/native_prediction.npz', 'object_positions': 'motion.npz', 'object_rotations': 'motion.npz'},
                 'closd': {'rotations_xyzw': 'native_physics.npz', 'contact_forces': 'native_physics.npz', 'body_names': 'native_physics.npz',
                           'dof_positions': 'native_physics.npz'},
                 'omnicontrol': {'features_263': 'raw/features.npy'}}
SMPLX_FIELDS = ('transl', 'global_orient', 'body_pose', 'betas')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def runs(batch):
    out = []
    for state in sorted(Path(batch).glob('queue*/state.json')):
        for job, rec in json.loads(state.read_text())['jobs'].items():
            for a in rec.get('attempts', []):
                if a.get('run_dir'):
                    out.append((job, Path(a['run_dir']), a['status']))
    return out


def native(d, method):
    arrays, sources = {}, {}
    if method == 'dart':
        with np.load(d/'smplx_params.npz') as z:
            parts, layout, start = [], {}, 0
            for k in SMPLX_FIELDS:
                a = np.asarray(z[k], np.float32)
                flat = a.reshape(a.shape[0], -1)
                layout[k] = [start, start+flat.shape[1], list(a.shape[1:])]
                start += flat.shape[1]; parts.append(flat)
            arrays['smplx_params'] = np.concatenate(parts, 1)
            arrays['smplx_params_layout'] = np.array(json.dumps({'fields': layout, 'gender': str(z['gender']),
                                                                 'pelvis_delta': np.asarray(z['pelvis_delta']).tolist()}))
            sources['smplx_params'] = 'smplx_params.npz:' + ','.join(SMPLX_FIELDS)
    for key, rel in NATIVE_SOURCE.get(method, {}).items():
        p = d/rel
        if not p.exists():
            if key in CAP[method]['native_required']:
                raise FileNotFoundError(f'{p} (native {key})')
            continue
        if p.suffix == '.npy':
            arrays[key] = np.load(p, allow_pickle=False)
        else:
            with np.load(p, allow_pickle=False) as z:
                arrays[key] = z[key]
        sources[key] = f'{rel}:{key}'
    return arrays, sources


def complete(d):
    man = json.loads((d/'manifest.json').read_text())
    method = man['baseline']
    if man.get('status') != 'success':
        if not (d/'failure.json').exists():
            (d/'failure.json').write_text(json.dumps({'status': man.get('status'), 'reason': man.get('reason'),
                                                      'written_by': 'python/scripts/complete_run_contract.py'}, indent=1))
        return 'failure_recorded'
    integ = d/'integrity.json'
    if integ.exists():
        rec = json.loads(integ.read_text())
        if rec.get('done') and all((d/n).exists() and sha(d/n) == h for n, h in rec['sha256'].items()):
            return 'skip'
        raise SystemExit(f'{d}: integrity.json does not verify; not overwritten')
    files = ['scenario.json']
    if method == 'primal':
        files += sorted(str(p.relative_to(d)) for p in (d/'factorial').glob('*.npz'))
    else:
        files.append('motion.npz')
        arrays, sources = native(d, method)
        if arrays:
            np.savez(d/'native.npz', native_sources=np.array(json.dumps(sources)), **arrays)
            files.append('native.npz')
        files += sorted({s.split(':')[0] for s in sources.values()} - {'motion.npz'})
    world_path = ROOT/f'config/motion_world/{method}.json'
    cfg = json.loads((d/'config.json').read_text()) if (d/'config.json').exists() else {}
    corrected = method == 'ardy' and int(cfg.get('_intervention', {}).get('correction', 0)) == 1
    world = {'method_world': str(world_path.relative_to(ROOT)), 'method_world_sha256': sha(world_path),
             'floor_variant': 'correction=1' if corrected else 'default', 'written_by': 'python/scripts/complete_run_contract.py'}
    if method == 'chois':
        scenario = json.loads((d/'scenario.json').read_text())
        world.update(object_id=scenario['native_sequence'].split('_')[1], object_mass_kg=None,
                     object_mass_source='not used in this batch (contract method_capability.chois.object_mass; '
                                        'config/chois_object_mass_lookup_20261109.json status)')
    (d/'world.json').write_text(json.dumps(world, indent=1))
    files.append('world.json')
    integ.write_text(json.dumps({'sha256': {n: sha(d/n) for n in files}, 'done': True, 'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                 'written_by': 'python/scripts/complete_run_contract.py'}, indent=1))
    return 'completed'


def targets(d, method):
    return sorted((d/'factorial').glob('*_canonical.npz')) if method == 'primal' else [d]


def gate(result, method, linked):
    bad = []
    for layer in LAYERS:
        state = result[layer]['state']
        if state == 'gap' or (not linked and state == 'legacy' and not (method == 'primal' and layer in PRIMAL_LEGACY)):
            bad.append(f'{layer} {state} {result[layer].get("missing") or result[layer].get("new_batch_missing") or ""}'.strip())
    return bad


def scan(batch):
    out, problems = {'new': {}, 'linked': {}}, []
    for job, d, status in runs(batch):
        if status != 'success':
            continue
        method = json.loads((d/'manifest.json').read_text())['baseline']
        for t in targets(d, method):
            r = mcc.check_run(t, method)
            bad = gate(r, method, False)
            out['new'][str(t)] = {'job': job, 'method': method, **{L: r[L]['state'] for L in LAYERS}, 'problems': bad}
            problems += [f'{t}: {b}' for b in bad]
    links = Path(batch)/'plans/links.json'
    for x in json.loads(links.read_text())['links'] if links.exists() else []:
        d, method = ROOT/x['run'], x['row']['method']
        for t in targets(d, method):
            r = mcc.check_run(t, method)
            bad = gate(r, method, True)
            out['linked'][str(t)] = {'method': method, **{L: r[L]['state'] for L in LAYERS}, 'problems': bad}
            problems += [f'{t}: {b}' for b in bad]
    summary = {k: {'runs': len(v), 'with_problems': sum(bool(r['problems']) for r in v.values())} for k, v in out.items()}
    (Path(batch)/'contract_scan.json').write_text(json.dumps({'summary': summary, 'problems': problems, **out}, indent=1))
    print(json.dumps({'summary': summary, 'problems': problems[:20]}))
    return 1 if problems else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['complete', 'scan'])
    p.add_argument('--batch', required=True, type=Path)
    a = p.parse_args()
    if a.step == 'scan':
        raise SystemExit(scan(a.batch))
    counts = {}
    for job, d, status in runs(a.batch):
        r = complete(d)
        counts[r] = counts.get(r, 0) + 1
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
