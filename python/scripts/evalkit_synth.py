#!/usr/bin/env python
"""Build the evalkit synthetic validation set (config/evalkit_synth_20261107.json). OpenSim python.

One task per (split, subject seed, base class): the subject model, the base truth motion, every defect clip with its
truth (truth/truth.npz, coordinates.mot, grf.mot / grf.xml for static optimization) and the generator-like input
(input/motion.npz, input/meta.json). Index: develop_logs/test_runs/20261107_evalkit/synth/index.json.

  --only CLASS[,CLASS]  --split dev|heldout|heldout2|heldout3  --limit N  --workers N
"""
import argparse
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
SPEC = json.loads((ROOT/'config/evalkit_synth_20261107.json').read_text())
LOG = ROOT/'develop_logs/test_runs/20261107_evalkit/synth'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/evalkit_20261107/synth')
BASES = SPEC['constructed']['classes']+['demo_walk']


def tasks(split=None, only=None):
    out = []
    for sp, v in SPEC['splits'].items():
        if not isinstance(v, dict) or (split and sp != split):
            continue
        for seed in v['subject_seeds']:
            out += [{'split': sp, 'seed': seed, 'base': b} for b in BASES if not only or b in only]
    return out


def defects(base):
    out = [('control', 0)]
    for d, sizes in SPEC['defects'].items():
        if d == 'slide_m_s' and base == 'demo_walk':
            continue
        out += [(d, s) for s in sizes]
    return out


def clip_id(task, defect, size):
    return f"{task['split']}_{task['seed']}_{task['base']}__{defect}" + (f'_{size:g}' if defect != 'control' else '')


def export_seed(cid):
    return zlib.crc32(cid.encode())


def reexport(base_dir):
    """Re-draw the generator-like input of every clip of a built base from its truth (export seed amendment)."""
    os.nice(19)
    import biomech
    from evalkit import synth
    root = Path(base_dir); done = json.loads((root/'done.json').read_text()); info = done['info']
    for cid in done['clips']:
        d = root/cid; m = json.loads((d/'input/meta.json').read_text()); tr = np.load(d/'truth/truth.npz')
        _, labels, coords = biomech.read_storage(d/'truth/coordinates.mot')
        clip = {'labels': labels, 'coords': coords, 'fps': m['fps_truth'], 'floor': m['truth_floor_m'],
                'stretch': tr['stretch'] if 'stretch' in tr.files else None}
        ex = synth.export(root/'model.osim', clip, info, seed=export_seed(cid))
        np.savez_compressed(d/'input/motion.npz', joints=ex['joints'], joint_names=np.array(ex['joint_names']), fps=ex['fps'])
        np.savez_compressed(d/'truth/clean_joints.npz', joints=ex['clean_joints'], joint_names=np.array(ex['joint_names']), fps=ex['fps'])
        m.update(floor_height_m=ex['floor_height_m'], floor_shift_m=ex['floor_shift_m'], export_seed=export_seed(cid))
        (d/'input/meta.json').write_text(json.dumps(m, indent=1))
    return str(root)


def build(task):
    os.nice(19)
    import opensim as osim
    import biomech
    from evalkit import synth
    root = DISK/task['split']/f"{task['seed']}_{task['base']}"
    done = root/'done.json'
    if done.exists():
        return json.loads(done.read_text())
    root.mkdir(parents=True, exist_ok=True)
    start = time.time(); rows = []
    try:
        if task['base'] == 'demo_walk':
            info = synth.demo_model(root/'model.osim')
            walk = synth.demo_walk(root/'model.osim')
            ref = synth.demo_reference(root/'model.osim', walk)
            base = {**walk, 'ref': ref, 'contact': {s: {'heel': walk['contact_plate'][s], 'toe': walk['contact_plate'][s]} for s in synth.SIDES}}
            source = 'demo_walk'
        else:
            info = synth.subject_model(task['seed'], root/'model.osim')
            base = synth.construct(root/'model.osim', info, task['base'])
            source = 'constructed'
        info['base_seconds'] = round(time.time()-start, 1)
        for k, (defect, size) in enumerate(defects(task['base'])):
            cid = clip_id(task, defect, size); d = root/cid
            (d/'truth').mkdir(parents=True, exist_ok=True); (d/'input').mkdir(exist_ok=True)
            if defect == 'slide_m_s':
                clip = synth.apply_defect(synth.construct(root/'model.osim', info, task['base'], slide_m_s=size), 'slide_m_s', size)
            else:
                clip = synth.apply_defect(base, defect, size)
            tr = synth.truth(root/'model.osim', clip, info['mass_kg'], source)
            times = np.arange(len(clip['coords']))/clip['fps']
            synth.write_mot(times, clip['labels'], clip['coords'], d/'truth/coordinates.mot')
            biomech.write_loads(times, tr['grf'], clip['floor'], d/'truth/grf.mot', d/'truth/grf.xml')
            link = d/'truth/model.osim'
            if not link.exists():
                link.symlink_to(root/'model.osim')
            np.savez_compressed(d/'truth/truth.npz', times=times, residual_bw=tr['residual_bw'], e_label=tr['e_label'],
                                unmeasurable_label=tr['unmeasurable_label'], lowest_elevation=tr['lowest_elevation'],
                                **{f'contact_{s}_{k2}': tr['contact'][s][k2] for s in synth.SIDES for k2 in ('heel', 'toe')},
                                **{f'elevation_{s}_{k2}': tr['elevation'][s][k2] for s in synth.SIDES for k2 in ('heel', 'toe')},
                                **({'stretch': clip['stretch']} if clip.get('stretch') is not None else {}))
            ex = synth.export(root/'model.osim', clip, info, seed=export_seed(cid))
            np.savez_compressed(d/'input/motion.npz', joints=ex['joints'], joint_names=np.array(ex['joint_names']), fps=ex['fps'])
            np.savez_compressed(d/'truth/clean_joints.npz', joints=ex['clean_joints'], joint_names=np.array(ex['joint_names']), fps=ex['fps'])
            meta = {'id': cid, 'split': task['split'], 'seed': task['seed'], 'base': task['base'], 'source': source, 'defect': defect,
                    'size': size, 'method': 'synthetic', 'mass_kg': info['mass_kg'], 'floor_height_m': ex['floor_height_m'],
                    'floor_shift_m': ex['floor_shift_m'], 'truth_floor_m': clip['floor'], 'truth_reference_m': clip['ref'],
                    'fps_truth': clip['fps'], 'fps_input': ex['fps'], 'window_frames': clip['window'],
                    'foot_slip_m': clip.get('foot_slip_m'), 'object_free': True,
                    'qc_ok': clip.get('foot_slip_m') is None or clip['foot_slip_m'] <= SPEC['constructed']['foot_slip_max_m'],
                    'truth_shares': {'e_positive': float(np.mean(tr['e_label'] == 1)), 'e_ambiguous': float(np.mean(tr['e_label'] == -1)),
                                     'unmeasurable_positive': float(np.mean(tr['unmeasurable_label'] == 1))}}
            (d/'input/meta.json').write_text(json.dumps(meta, indent=1))
            rows.append(meta)
        out = {'task': task, 'status': 'ok', 'info': info, 'seconds': round(time.time()-start, 1), 'clips': [r['id'] for r in rows]}
    except Exception as e:
        out = {'task': task, 'status': 'failed', 'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-2000:]}
    (root/'done.json').write_text(json.dumps(out, indent=1, default=float)) if out['status'] == 'ok' else (root/'error.json').write_text(json.dumps(out, indent=1))
    return out


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--split', choices=('dev', 'heldout', 'heldout2', 'heldout3')); a.add_argument('--only'); a.add_argument('--limit', type=int)
    a.add_argument('--workers', type=int, default=1)
    a.add_argument('--reexport', action='store_true', help='re-draw the inputs of built bases from their truth')
    args = a.parse_args()
    if args.reexport:
        bases = [str(p.parent) for p in sorted(DISK.glob('*/*/done.json'))]
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
            for r in pool.map(reexport, bases):
                print(r, flush=True)
        return
    todo = tasks(args.split, args.only.split(',') if args.only else None)[:args.limit]
    LOG.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        for r in pool.map(build, todo):
            results.append(r); print(json.dumps({k: r.get(k) for k in ('task', 'status', 'seconds', 'error')}), flush=True)
    index = LOG/'index.json'
    prev = json.loads(index.read_text()) if index.exists() else {}
    for r in results:
        prev[f"{r['task']['split']}_{r['task']['seed']}_{r['task']['base']}"] = {k: r.get(k) for k in ('status', 'seconds', 'clips', 'error', 'info')}
    index.write_text(json.dumps(prev, indent=1, default=float))


if __name__ == '__main__':
    main()
