"""Versioned K rotation entry point; historical RULA sensitivity remains explicit legacy code."""
import json
from pathlib import Path

import numpy as np

from . import ergo, fit as fit_mod
from .spec import SIDES, LANDMARKS

METHODS = ('ardy', 'chois', 'kimodo', 'primal', 'dart', 'closd')
CONVENTION_GATE_DEG, FRAME_GATE_DEG = .5, .05
BANDS = (0., 10., 20.)
VERSION = 2


def native(method, source):
    """Native joints (T, J, 3), names, global rotations (T, J, 3, 3) and the neck-to-head chain (joint indices)."""
    from .k import load
    motion = load(source, method)
    if motion.rotations is None:
        raise ValueError('Native rotations unavailable in this export')
    names = motion.names
    joints = motion.positions
    rotations = motion.rotations
    if method == 'chois':
        names = ergo.SMPL22_NAMES
        joints = joints[:, :22]
    if len(names) != rotations.shape[1]:
        raise ValueError('Historical joint-line sensitivity requires matching joint and rotation layouts; use K native segment output')
    chain = [names.index(n) for n in (('neck','Neck2','head') if method=='kimodo' else ('neck','head'))]
    return joints, names, rotations, chain


def readings(method, source):
    """A and B per native frame (uncorrected [flexion, side bend]) and the convention check: max angle between the
    per-frame rest rebuild sum R_parent^T (child - parent) along the chain and its median."""
    j, names, R, chain = native(method, source)
    rebuild = sum(np.einsum('tji,tj->ti', R[:, p], j[:, c]-j[:, p]) for p, c in zip(chain[:-1], chain[1:]))
    rest = np.median(rebuild, 0)
    cos = rebuild @ rest/np.linalg.norm(rebuild, axis=1)/np.linalg.norm(rest)
    idx = ergo.posture_index(names)
    jb = j.copy(); jb[:, idx['head']] = j[:, chain[0]]+np.einsum('tij,j->ti', R[:, chain[-1]], rest)
    return ergo.neck_angles(j, idx), ergo.neck_angles(jb, idx), float(np.degrees(np.arccos(np.clip(cos, -1, 1))).max())


def neck_input(flex_side):
    """Revision 3 neck inputs from a zero-corrected [flexion, side bend]."""
    flex = float(flex_side[0])
    if -ergo.R3['neck_extension_tolerance_deg'] <= flex < 0:
        flex = 0.
    return {'neck_flexion_deg': flex, 'neck_side_bent': int(abs(flex_side[1]) > ergo.CONV['twist_and_side_bend_threshold_deg'])}


def capture_score(*args):
    """ergo.score with the per-frame, per-side RULA inputs recorded."""
    calls, orig = [], ergo.score_union

    def rec(inputs, method):
        calls.append((method, dict(inputs)))
        return orig(inputs, method)
    ergo.score_union = rec
    try:
        e = ergo.score(*args)
    finally:
        ergo.score_union = orig
    rula = [c[1] for c in calls if c[0] == 'RULA']
    return e, [dict(zip(SIDES, rula[i*len(SIDES):(i+1)*len(SIDES)])) for i in range(len(e['times']))]


def rula_class(side_inputs):
    lo = hi = None
    for inp in side_inputs.values():
        a, b = ergo.score_union(inp, 'RULA')
        lo, hi = (a, b) if lo is None else (max(lo, a), max(hi, b))
    risk = ergo.RISK['RULA']
    return 1 if lo >= risk else 2 if hi >= risk else 3


def key(rec):
    return {'version': VERSION, 'batch': rec.get('batch'), 'commit': rec['receipt'].get('commit'), 'added_commit': rec['receipt'].get('added_commit')}


def evaluate(run, out, method):
    """Attach current K neck reliability to an evaluated row; legacy sensitivity is opt-in below."""
    out = Path(out); rec = json.loads((out/'evalkit.json').read_text()); r = rec['receipt']
    from .k import load as load_k, evaluate as evaluate_k
    from .k.posture import summary
    k = evaluate_k(load_k(run, method))
    res = summary(k)
    res['status'] = k['metrics']['neck']['status']
    res['reason'] = k['metrics']['neck']['reason']
    rec['neck_rotation'] = {**res, 'key': key(rec)}
    (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
    return rec['neck_rotation']


def evaluate_legacy(run, out, method):
    """Historical r3 sensitivity only, on a pinned historical implementation."""
    out = Path(out); rec = json.loads((out/'evalkit.json').read_text()); r = rec['receipt']
    if r.get('fit_prepare') != fit_mod.PREPARE:
        return {'status':'legacy_requires_pinned_checkout', 'reason':'historical fit is preserved'}
    if method not in METHODS:
        res = {'status': 'not_applicable', 'reason': 'no neck and head rotations in this method\'s output (contract L4)'}
    elif 'ergo' not in r['dims']:
        res = {'status': 'not_applicable', 'reason': 'ergo not evaluated'}
    else:
        res = _evaluate(run, out, method, r)
    rec['neck_rotation'] = {**res, 'key': key(rec)}
    (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
    return rec['neck_rotation']


def _evaluate(run, out, method, r):
    import biomech_sources
    frames = np.load(out/'frames.npz', allow_pickle=True)
    f = fit_mod.fit(run, out/'work', r['mass_kg'], r['floor_m'])
    if not np.array_equal(f['times'], frames['times']):
        raise ValueError('cached fit does not match frames.npz')
    masks = {s: {k: frames[f'contact_{s}_{k}'] for k in LANDMARKS} for s in SIDES}
    src = biomech_sources.load(run); zero = ergo.neutral(method, run)
    e, inp = capture_score(out/'work/model.osim', f['times'], f['labels'], f['coords'], masks, frames['scored'], src, zero,
                           r['object_free'], f['unmeasurable'])
    a_nat, b_nat, conv = readings(method, run)
    res = {'convention_deg': conv}
    if conv > CONVENTION_GATE_DEG:
        return {**res, 'status': 'failed_convention'}
    joints, names, fps = src
    if len(a_nat) != len(joints):
        return {**res, 'status': 'failed_frame_match'}
    k = np.clip(np.round(e['times']*fps).astype(int), 0, len(joints)-1)
    res['frame_match_deg'] = float(np.abs(a_nat[k]-ergo.neck_angles(joints[k], ergo.posture_index(names))).max())
    if res['frame_match_deg'] > FRAME_GATE_DEG:
        return {**res, 'status': 'failed_frame_match'}
    if not np.array_equal(e['rula_class'], frames['ergo_rula_class']):
        return {**res, 'status': 'failed_reproduction'}
    a, b = a_nat[k]-zero[1], b_nat[k]-zero[1]
    scored = e['rula_class'] != 0
    b_class = e['rula_class'].copy()
    for i in np.flatnonzero(scored):
        b_class[i] = rula_class({s: {**v, **neck_input(b[i])} for s, v in inp[i].items()})
    a_in = np.array([x['r']['neck_flexion_deg'] for x in inp]); b_in = np.array([neck_input(x)['neck_flexion_deg'] for x in b])
    np.savez_compressed(out/'neck_rotation.npz', times=e['times'], a=a, b=b, a_class=e['rula_class'], b_class=b_class,
                        a_neck_input=a_in, b_neck_input=b_in)
    share = lambda x: float(np.mean(x)) if scored.any() else None
    return {**res, 'status': 'ok',
            'rula': {f'{c}_share': float(np.mean(b_class == i)) for i, c in enumerate(ergo.CLASSES)},
            'class_changed_share': share((b_class != e['rula_class'])[scored]),
            'neck_band_changed_share': share((np.digitize(a_in, BANDS) != np.digitize(b_in, BANDS))[scored]),
            'b_minus_a_flexion_median_deg': float(np.median((b-a)[scored, 0])) if scored.any() else None}
