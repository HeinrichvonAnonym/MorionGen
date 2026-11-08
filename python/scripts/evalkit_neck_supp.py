#!/usr/bin/env python
"""Neck supplement (config/evalkit_neck_supp_20261108.json). OpenSim python.

  evalkit_neck_supp.py real   [--workers N] [--limit K]   joint-line vs rotation neck reading and RULA class on the
                                                          rotation-carrying rows of the registered offset-reference set
  evalkit_neck_supp.py synth  [--workers N] [--limit K]   RULA kill rates with the neck / head landmark offsets scaled
                                                          (heldout2 + 3 + 4 method-floor v1.1 ergo pass clips)
  --out DIR: output root (default develop_logs/test_runs/20261108_neck_supp, the registered run)
"""
import argparse
import json
import multiprocessing as mp
import shutil
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'python/scripts')]
import evalkit_validate as V  # noqa: E402
import evalkit_rula_supp as S  # noqa: E402

REG = json.loads((ROOT/'config/evalkit_neck_supp_20261108.json').read_text())
OFFSET_SET = ROOT/'develop_logs/test_runs/20261108_offset_reference'
FROZEN = json.loads((ROOT/'config/evalkit_offset_reference_20261108.json').read_text())
OUT = ROOT/'develop_logs/test_runs/20261108_neck_supp'
METHODS = ('ardy', 'chois', 'kimodo', 'primal')
CONVENTION_GATE_DEG, FRAME_GATE_DEG = .5, .05
BANDS = (-np.inf, 0., 10., 20., np.inf)
P22 = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def angle(a, b):
    a = a/np.linalg.norm(a, axis=-1, keepdims=True); b = b/np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a*b, -1), -1, 1)))


def native(method, source):
    """Native joints (T, J, 3), names, global rotations (T, J, 3, 3) and the neck-to-head chain (joint indices)."""
    from evalkit import ergo
    from motion import SOMA77_NAMES
    src = Path(source)
    if method == 'ardy':
        z = np.load(src/'motion.npz'); names = [str(n) for n in z['joint_names']]
        return z['joints'].astype(float), names, z['global_rotations'].astype(float), [names.index('neck'), names.index('head')]
    if method == 'chois':
        z = np.load(src/'raw/native_prediction.npz')
        return z['joints'][:, :22].astype(float), ergo.SMPL22_NAMES, z['global_rotations'][:, :22].astype(float), [12, 15]
    if method == 'kimodo':
        z = np.load(src/'motion.npz'); names = list(SOMA77_NAMES)
        return z['posed_joints'].astype(float), names, z['global_rot_mats'].astype(float), [names.index(n) for n in ('neck', 'Neck2', 'head')]
    from scipy.spatial.transform import Rotation
    z = np.load(str(src).replace('_canonical.npz', '.npz')); pose = z['pose'][0].astype(float)
    local = Rotation.from_rotvec(pose[:, 3:].reshape(-1, 3)).as_matrix().reshape(len(pose), 22, 3, 3)
    glob = np.zeros_like(local)
    for k in range(22):
        glob[:, k] = local[:, k] if P22[k] < 0 else glob[:, P22[k]] @ local[:, k]
    return z['joints'][0].astype(float), ergo.SMPL22_NAMES, glob, [12, 15]


def readings(method, source):
    """A (joint line) and B (rest neck-to-head vector carried by the head rotation) per native frame, uncorrected,
    with the convention check (max angle between the per-frame rest rebuild and its median)."""
    from evalkit import ergo
    j, names, R, chain = native(method, source)
    rebuild = sum(np.einsum('tji,tj->ti', R[:, p], j[:, c]-j[:, p]) for p, c in zip(chain[:-1], chain[1:]))
    rest = np.median(rebuild, 0)
    idx = ergo.posture_index(names)
    head = j[:, chain[0]]+np.einsum('tij,j->ti', R[:, chain[-1]], rest)
    jb = j.copy(); jb[:, idx['head']] = head
    return ergo.neck_angles(j, idx), ergo.neck_angles(jb, idx), float(angle(rebuild, rest[None]).max())


def neck_input(flex_side):
    """Revision 3 neck inputs from a zero-corrected [flexion, side bend]."""
    from evalkit import ergo
    flex = float(flex_side[0])
    if -ergo.R3['neck_extension_tolerance_deg'] <= flex < 0:
        flex = 0.
    return {'neck_flexion_deg': flex, 'neck_side_bent': int(abs(flex_side[1]) > ergo.CONV['twist_and_side_bend_threshold_deg'])}


def real_job(task):
    row, root = task
    try:
        V._quiet()
        import evalkit_batch as B
        import biomech_sources
        from evalkit import run, ergo, residual, fit as fit_mod
        from evalkit.spec import reference_offsets, absolute_reference
        out = Path(root)/'real'/row['id']; work = out/'work'
        if not work.exists():
            shutil.copytree(OFFSET_SET/'rows'/row['id']/'work', work)
        source, ref, floor, free = B.resolve(row)
        mass = row.get('mass_kg', 75.)
        src = biomech_sources.load(source)
        offset, f, kin = run.foot_offset(source, work, mass, floor, src)
        floor_eval = floor+offset['mean']-FROZEN['methods'][row['method']]['offset_reference_m']
        offsets = reference_offsets(ref) if isinstance(ref, str) else ref
        res = residual.evaluate(kin, f['times'], mass, floor_eval, absolute_reference(offsets, floor_eval))
        zero = ergo.neutral(row['method'], source)
        e, inp = S.capture_score(work/'model.osim', f['times'], f['labels'], f['coords'], res['masks'], fit_mod.scored(f['times']),
                                 src, zero, free, f['unmeasurable'])
        a_nat, b_nat, conv = readings(row['method'], source)
        rec = {'id': row['id'], 'method': row['method'], 'convention_deg': conv, 'frames': int(len(e['times']))}
        if conv > CONVENTION_GATE_DEG:
            return {**rec, 'status': 'failed_convention'}
        joints, names, fps = src
        k = np.clip(np.round(e['times']*fps).astype(int), 0, len(joints)-1)
        a_src = ergo.neck_angles(joints[k], ergo.posture_index(names))
        if len(a_nat) != len(joints):
            return {**rec, 'status': 'failed_frame_match', 'frames_native': len(a_nat), 'frames_source': len(joints)}
        rec['frame_match_deg'] = float(np.abs(a_nat[k]-a_src).max())
        if rec['frame_match_deg'] > FRAME_GATE_DEG:
            return {**rec, 'status': 'failed_frame_match'}
        neck0 = zero[1]
        a, b = a_nat[k]-neck0, b_nat[k]-neck0
        repro = np.array([S.rula_class(sides) for sides in inp])
        scored = e['rula_class'] != 0
        rec['reproduced'] = bool(np.array_equal(repro[scored], e['rula_class'][scored]))
        b_class = e['rula_class'].copy()
        for i in np.flatnonzero(scored):
            b_class[i] = S.rula_class({s: {**v, **neck_input(b[i])} for s, v in inp[i].items()})
        np.savez_compressed(out/'neck.npz', times=e['times'], a=a, b=b, a_class=e['rula_class'], b_class=b_class,
                            a_neck_input=np.array([inp[i]['r']['neck_flexion_deg'] for i in range(len(inp))]),
                            b_neck_input=np.array([neck_input(x)['neck_flexion_deg'] for x in b]))
        return {**rec, 'status': 'ok'}
    except Exception as ex:
        import traceback
        return {'id': row['id'], 'method': row['method'], 'status': 'failed', 'error': f'{type(ex).__name__}: {ex}',
                'trace': traceback.format_exc()[-1500:]}


def pct(x):
    return {'median': float(np.median(x)), 'p5': float(np.percentile(x, 5)), 'p95': float(np.percentile(x, 95))} if len(x) else None


def band(x):
    return np.digitize(x, BANDS[1:-1], right=False)


def real_summary(recs, root):
    from evalkit.ergo import CLASSES
    per = {}
    for m in METHODS + ('pooled',):
        ok = [r for r in recs if r['status'] == 'ok' and (m == 'pooled' or r['method'] == m)]
        if not ok:
            continue
        z = [dict(np.load(Path(root)/'real'/r['id']/'neck.npz')) for r in ok]
        sc = [d['a_class'] != 0 for d in z]
        d_flex = np.concatenate([(d['b'][:, 0]-d['a'][:, 0])[s] for d, s in zip(z, sc)])
        d_side = np.concatenate([(d['b'][:, 1]-d['a'][:, 1])[s] for d, s in zip(z, sc)])
        ab = np.concatenate([d['a_neck_input'][s] for d, s in zip(z, sc)]); bb = np.concatenate([d['b_neck_input'][s] for d, s in zip(z, sc)])
        ac = np.concatenate([d['a_class'][s] for d, s in zip(z, sc)]); bc = np.concatenate([d['b_class'][s] for d, s in zip(z, sc)])
        per[m] = {'rows': len(ok), 'scored_frames': int(len(ac)),
                  'b_minus_a_flexion_deg': pct(d_flex), 'b_minus_a_side_bend_deg': pct(d_side),
                  'row_median_b_minus_a_flexion_deg': pct([np.median((d['b'][:, 0]-d['a'][:, 0])[s]) for d, s in zip(z, sc) if s.any()]),
                  'neck_band_changed_share': float(np.mean(band(ab) != band(bb))) if len(ab) else None,
                  'rula_transitions': {CLASSES[i]: {CLASSES[j]: int(((ac == i) & (bc == j)).sum()) for j in (1, 2, 3)} for i in (1, 2, 3)},
                  'rula_class_changed_share': float(np.mean(ac != bc)) if len(ac) else None,
                  'certain_risk_share': {'A': float(np.mean(ac == 1)), 'B': float(np.mean(bc == 1))} if len(ac) else None,
                  'rows_with_class_change': int(sum((d['a_class'] != d['b_class']).any() for d in z))}
    return per


def real(args):
    (args.out/'real').mkdir(parents=True, exist_ok=True)
    rows = [r for r in json.loads((OFFSET_SET/'manifest.json').read_text())['rows'] if r['method'] in METHODS][:args.limit]
    recs = S.pool_results(real_job, [(r, str(args.out)) for r in rows], args.workers)
    res = {'registration': 'config/evalkit_neck_supp_20261108.json', 'part': 'real', 'rows': len(rows),
           'status': {s: sum(r['status'] == s for r in recs) for s in sorted({r['status'] for r in recs})},
           'reproduced': {'ok_rows': sum(r['status'] == 'ok' for r in recs), 'reproduced': sum(r.get('reproduced', False) for r in recs)},
           'convention_deg_max': {m: max([r['convention_deg'] for r in recs if r['method'] == m and 'convention_deg' in r], default=None) for m in METHODS},
           'frame_match_deg_max': {m: max([r['frame_match_deg'] for r in recs if r['method'] == m and 'frame_match_deg' in r], default=None) for m in METHODS},
           'summary': real_summary(recs, args.out), 'failed': [r for r in recs if r['status'] != 'ok']}
    (args.out/'real.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('status', 'reproduced', 'convention_deg_max', 'frame_match_deg_max')}, indent=1))


def synth_job(clip):
    try:
        V._quiet()
        import biomech
        import biomech_sources
        from evalkit import synth, ergo
        split, cid = clip['split'], clip['id']
        d = Path(clip['dir']); tl_dir = V.DATA/'tool'/split/cid; te_dir = V.DATA/'truth_eval'/split/cid
        m = json.loads((d/'input/meta.json').read_text()); tr = np.load(d/'truth/truth.npz')
        _, labels, coords = biomech.read_storage(d/'truth/coordinates.mot')
        spec = {'labels': labels, 'coords': coords, 'fps': m['fps_truth'], 'floor': m['truth_floor_m'],
                'stretch': tr['stretch'] if 'stretch' in tr.files else None}
        seed = zlib.crc32(cid.encode()); stored = np.load(d/'input/motion.npz')
        ex1 = synth.export(Path(clip['base_dir'])/'model.osim', spec, clip['info'], seed=seed)
        sd = synth.SPEC['export']['landmark_offset_sd_m']
        synth.SPEC['export']['landmark_offset_sd_m'] = 0.
        try:
            ex0 = synth.export(Path(clip['base_dir'])/'model.osim', spec, clip['info'], seed=seed)
        finally:
            synth.SPEC['export']['landmark_offset_sd_m'] = sd
        rec = {'id': cid, 'split': split, 'export_repro_m': float(np.abs(ex1['joints']-stored['joints']).max()),
               'shift_equal': bool(ex0['floor_shift_m'] == ex1['floor_shift_m'] == m['floor_shift_m'])}
        if rec['export_repro_m'] > 1e-9 or not rec['shift_equal']:
            return {**rec, 'error': 'export gate failed'}
        names = list(ex1['joint_names']); cols = [names.index('neck'), names.index('head')]
        f = np.load(tl_dir/'frames.npz', allow_pickle=True)
        x_times, x_labels, x_coords = biomech.read_storage(tl_dir/'work/coordinates.mot')
        masks = {s: {k: f[f'contact_{s}_{k}'] for k in V.LANDMARKS} for s in V.SIDES}
        te = np.load(te_dir/'truth_eval.npz', allow_pickle=True)
        t_src = biomech_sources.load(d/'truth/clean_joints.npz')
        zero_t = ergo.neutral('synthetic')
        t_neck = ergo.source_posture(t_src, te['ergo_times'], zero_t)[1][:, 0]
        for k in REG['part_synth']['scales']:
            xk = ex1['joints'].copy()
            xk[:, cols] = ex0['joints'][:, cols]+k*(ex1['joints'][:, cols]-ex0['joints'][:, cols])
            src = (biomech.to_opensim(xk), names, float(ex1['fps']))
            zero = ergo.neutral('synthetic', d/'input')
            e = ergo.score(tl_dir/'work/model.osim', x_times, x_labels, x_coords, masks, f['scored'], src, zero,
                           ergo.object_free(None), f['unmeasurable'])
            rec[f'scale_{k:g}'] = {'times': e['times'].tolist(), 'rula': e['rula_class'].tolist(),
                                   'neck': ergo.source_posture(src, e['times'], zero)[1][:, 0].tolist()}
        rec['reproduced'] = bool(np.array_equal(np.array(rec['scale_1']['rula']), f['ergo_rula_class']))
        rec['truth'] = {'times': te['ergo_times'].tolist(), 'rula': te['ergo_rula_class'].tolist(), 'neck': t_neck.tolist()}
        return rec
    except Exception as ex:
        import traceback
        return {'id': clip['id'], 'split': clip['split'], 'error': f'{type(ex).__name__}: {ex}', 'trace': traceback.format_exc()[-1500:]}


def synth_summary(results):
    from evalkit.validate import match, rate
    out = {}
    for k in REG['part_synth']['scales']:
        t_all, x_all, subj, task, dneck = [], [], [], [], []
        for r in results:
            t = {q: np.array(v) for q, v in r['truth'].items()}; x = {q: np.array(v) for q, v in r[f'scale_{k:g}'].items()}
            j = match(t['times'], x['times']); ok = np.abs(x['times'][j]-t['times']) <= .02
            t_all.append(t['rula'][ok]); x_all.append(x['rula'][j][ok])
            subj.append(np.full(ok.sum(), S.subject(r['id']))); task.append(np.full(ok.sum(), S.base_task(r['id'])))
            acc = t['rula'][ok] == 3
            dneck.append((x['neck'][j][ok]-t['neck'][ok])[acc])
        t, x, subj, task = map(np.concatenate, (t_all, x_all, subj, task))
        per = {s: S.fk_mk(t[subj == s], x[subj == s]) for s in np.unique(subj)}
        out[f'{k:g}'] = {'pooled': S.fk_mk(t, x),
                         'per_subject_false_kill': {s: v['false_kill']['rate'] for s, v in per.items()},
                         'subjects_over_5pct': sum(v['false_kill']['rate'] is not None and v['false_kill']['rate'] > .05 for v in per.values()),
                         'per_task_false_kill': {b: S.fk_mk(t[task == b], x[task == b])['false_kill'] for b in np.unique(task)},
                         'neck_flexion_tool_minus_truth_deg_on_truth_acceptable': pct(np.concatenate(dneck))}
    return out


def synth_run(args):
    (args.out/'synth').mkdir(parents=True, exist_ok=True)
    items = [{**c, 'split': s} for s in S.SPLITS if (V.SYNTH/s).exists() for c in V.clips(s) if V.ergo_pass(V.meta({**c, 'split': s}))]
    items = [c for c in items if (V.DATA/'tool'/c['split']/c['id']/'frames.npz').exists()][:args.limit]
    results = S.pool_results(synth_job, items, args.workers)
    errors = [r for r in results if 'error' in r]; ok = [r for r in results if 'error' not in r]
    (args.out/'synth'/'clips.json').write_text(json.dumps(results))
    res = {'registration': 'config/evalkit_neck_supp_20261108.json', 'part': 'synth', 'clips': len(items), 'ok': len(ok),
           'errors': [{q: r.get(q) for q in ('id', 'error', 'export_repro_m', 'shift_equal')} for r in errors],
           'reproduced_scale_1': sum(r['reproduced'] for r in ok),
           'export_repro_m_max': max([r['export_repro_m'] for r in ok], default=None),
           'scales': synth_summary(ok)}
    (args.out/'synth.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({q: v for q, v in res.items() if q != 'scales'}, indent=1))
    print(json.dumps({k: {'pooled_fk': v['pooled']['false_kill'], 'over5': v['subjects_over_5pct'],
                          'neck_diff': v['neck_flexion_tool_minus_truth_deg_on_truth_acceptable']} for k, v in res['scales'].items()}, indent=1))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('step', choices=('real', 'synth')); a.add_argument('--workers', type=int, default=6)
    a.add_argument('--limit', type=int); a.add_argument('--out', type=Path, default=OUT)
    args = a.parse_args()
    {'real': real, 'synth': synth_run}[args.step](args)


if __name__ == '__main__':
    main()
