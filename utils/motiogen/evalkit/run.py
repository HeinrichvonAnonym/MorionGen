"""One evaluation: shared IK, then residual force, RULA/REBA classes and muscle load, none gating another.

Writes <out>/work (OpenSim files), <out>/frames.npz (per-frame arrays) and <out>/evalkit.json (summary and receipt).
OpenSim python.
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .spec import SPEC, SPEC_PATH, ROOT, SIDES, LANDMARKS, reference_offsets, absolute_reference
from . import fit as fit_mod, residual, ergo, muscle, kinematic

DIMS = ('residual', 'ergo', 'muscle', 'kinematic')
# upper: muscle-level upper-body load (config/evalkit_upper_muscle_20261108.json), about 20 min per clip, so opt-in
ALL_DIMS = DIMS+('upper',)


def _commit():
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def _stats(x):
    x = np.asarray(x, float)
    return {k: float(np.percentile(x, q)) for k, q in (('p5', 5), ('median', 50), ('p95', 95))} if len(x) else None


def canonical_joints(run):
    """Canonical Z-up joints, names and fps of a run directory or file (Kimodo posed_joints: Y-up SOMA77, 30 fps)."""
    from motion import SOMA77_NAMES, Y_TO_Z
    from biomech_sources import KIMODO_FPS
    with np.load(Path(run)/'motion.npz' if Path(run).is_dir() else run, allow_pickle=False) as z:
        if 'posed_joints' in z.files:
            j = np.asarray(z['posed_joints'], float)
            return (j[0] if j.ndim == 4 else j) @ Y_TO_Z.T, list(SOMA77_NAMES), KIMODO_FPS
        return np.asarray(z['joints'], float), [str(n) for n in z['joint_names']], float(z['fps'])


def kinematic_summary(run, frames, reach_task=None):
    k = kinematic.evaluate(*canonical_joints(run), frames, reach_task)
    k['skating'] = {n: v for n, v in k['skating'].items() if n != 'per_span'}
    return k


def add_kinematic(run, out, reach_task=None):
    """Kinematic dimension on an existing evaluation (frames.npz, evalkit.json), without refitting."""
    out = Path(out); rec = json.loads((out/'evalkit.json').read_text())
    rec['kinematic'] = kinematic_summary(run, dict(np.load(out/'frames.npz', allow_pickle=True)), reach_task)
    rec['receipt']['dims'] = sorted(set(rec['receipt']['dims']) | {'kinematic'}, key=ALL_DIMS.index)
    rec['receipt']['kinematic_commit'] = _commit()
    (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
    return rec['kinematic']


def net_moments(up):
    """Net joint moments of the torque-driven Rajagopal trunk and arms (no trunk, neck or shoulder muscles), not loads."""
    return {f'net_moment_{p}_nm_kg_{s}': up[f'{p}_moment_nm_kg_{s}'] for p in muscle.UPPER_BODY for s in ('mean', 'p95')}


def upper_summary(result):
    return {k: ({n: x for n, x in v.items() if n not in ('key', 'prepare_key')} if isinstance(v, dict) else v)
            for k, v in result.items()}


def add_upper(run, out, mode='evalkit', truth=None):
    """Upper-body muscle dimension on an existing evaluation: its contact decision (frames.npz) and floor_eval, no
    refit of the shared IK. mode evalkit writes out/upper and evalkit.json['upper']; e3 and truth (truth: external-loads
    xml) write out/upper_<mode> only (validation arms)."""
    from . import upper_muscle
    out = Path(out); rec = json.loads((out/'evalkit.json').read_text())
    f = np.load(out/'frames.npz', allow_pickle=True)
    contact = (f['times'], {s: {k: f[f'contact_{s}_{k}'] for k in LANDMARKS} for s in SIDES})
    floor = rec['receipt'].get('floor_eval_m', rec['receipt']['floor_m'])
    result = upper_muscle.evaluate(run, out/('upper' if mode == 'evalkit' else f'upper_{mode}'), rec['receipt']['mass_kg'],
                                   mode, floor, contact, truth)
    if mode == 'evalkit':
        rec['upper'] = upper_summary(result)
        rec['receipt']['dims'] = sorted(set(rec['receipt']['dims']) | {'upper'}, key=ALL_DIMS.index)
        rec['receipt']['upper_commit'] = _commit()
        (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
    return result


def foot_offset(run, work, mass, floor, source=None):
    """Fit (cached in work) and fit.foot_offset of one run: the per-clip input of the floor correction."""
    import opensim as osim
    import biomech
    import biomech_sources
    osim.Logger.setLevelString('error')
    f = fit_mod.fit(run, work, mass, floor)
    kin = biomech.body_kinematics(osim.Model(str(Path(work)/'model.osim')), f['labels'], f['coords'])
    return fit_mod.foot_offset(*(source or biomech_sources.load(run)), kin, f['times']), f, kin


def evaluate(run, out, floor, reference, mass=75., method=None, prompt=None, dims=DIMS, source=None, object_free=None,
             reach_task=None, offset_reference=None):
    """run: directory with motion.npz (canonical Z-up joints). floor: floor height (m, canonical Z) of this motion.
    reference: floor-reference variant (spec.reference_offsets) or {side: {landmark: m}} offsets above the floor.
    method: ergo neutral key (default: the reference variant). prompt: task text (None: object-free).
    object_free: overrides the text rule (evalkit.sources.object_free). reach_task: kinematic.evaluate wrist target.
    offset_reference: the method's median foot offset (spec floor.per_clip_correction); the residual, ergo and muscle
    dimensions then use floor + this clip's foot offset - offset_reference. None: the input floor (no correction)."""
    import opensim as osim
    import biomech
    import biomech_sources
    osim.Logger.setLevelString('error')
    start = time.time(); out = Path(out).resolve(); work = out/'work'
    variant = reference if isinstance(reference, str) else None
    offsets = reference_offsets(reference) if variant else reference
    src = source or biomech_sources.load(run)
    offset, f, kin = foot_offset(run, work, mass, floor, src)
    delta = 0. if offset_reference is None else offset['mean']-offset_reference
    floor_eval = floor+delta
    ref = absolute_reference(offsets, floor_eval)
    times, labels, coords = f['times'], f['labels'], f['coords']
    sel = fit_mod.scored(times)
    res = residual.evaluate(kin, times, mass, floor_eval, ref)
    biomech.write_loads(times, res['grf'], floor_eval, work/'grf.mot', work/'grf.xml')
    frames = {'times': times, 'scored': sel, 'unmeasurable': f['unmeasurable'], 'ik_max_error_m': f['ik_max_error_m'],
              'residual_bw': res['residual_bw'], 'residual_force': res['residual_force'], 'com': res['com'],
              'lowest_elevation_m': res['lowest_elevation'], 'state': res['state'],
              **{f'contact_{s}_{k}': res['masks'][s][k] for s in SIDES for k in LANDMARKS},
              **{f'elevation_{s}_{k}': res['elevation'][s][k] for s in SIDES for k in LANDMARKS}}
    summary = {'ik': {**f['receipt']['ik_marker_error'], 'unmeasurable_share': float(f['unmeasurable'][sel].mean())}}
    if 'residual' in dims:
        positive = residual.judge(res) if SPEC['residual']['tau_bw'] else None
        if positive is not None:
            frames['residual_positive'] = positive
        contact_any = np.any([res['masks'][s][k] for s in SIDES for k in LANDMARKS], 0)
        summary['residual'] = {'residual_bw': _stats(res['residual_bw'][sel]), 'lowest_elevation_m': _stats(res['lowest_elevation'][sel]),
                               'no_contact_share': float(np.mean(~contact_any[sel])),
                               'penetration_share': float(np.mean(res['lowest_elevation'][sel] < -SPEC['residual']['penetration_depth_m']))
                               if SPEC['residual']['penetration_depth_m'] is not None else None,
                               'positive_share': float(positive[sel].mean()) if positive is not None else None,
                               'state_share': {k: float(np.mean(res['state'][sel] == k)) for k in ('event', 'contact', 'flight')}}
    if 'ergo' in dims:
        method = method or (variant.split('__')[0] if variant else None)
        e = ergo.score(work/'model.osim', times, labels, coords, res['masks'], sel, src, ergo.neutral(method, run),
                       ergo.object_free(prompt) if object_free is None else object_free, f['unmeasurable'])
        frames.update({f'ergo_{k}': v for k, v in e.items() if k != 'source_mode'})
        summary['ergo'] = {'source_mode': e['source_mode'], 'frames': int(len(e['index']))}
        for m in ergo.RISK:
            c = e[f'{m.lower()}_class']
            summary['ergo'][m.lower()] = {**{f'{n}_share': float(np.mean(c == i)) for i, n in enumerate(ergo.CLASSES)},
                                          'lower_median': float(np.median(e[f'{m.lower()}_bounds'][:, 0])),
                                          'upper_median': float(np.median(e[f'{m.lower()}_bounds'][:, 1]))}
    if 'muscle' in dims:
        t0, t1 = float(times[sel][0]), float(times[sel][-1])
        mu = muscle.evaluate(work, t0, t1)
        frames.update(muscle_times=mu['times'], muscle_effort=mu['effort'], muscle_reserve_abs_sum_nm=mu['reserve_abs_sum_nm'])
        summary['muscle'] = mu['summary']
        up = muscle.upper_body(work/'so', work/'model_so.osim')
        frames.update(net_moment_lumbar_nm_kg=up['lumbar'], net_moment_shoulder_nm_kg=up['shoulder'])
        summary['muscle'].update(net_moments(up))
    if 'kinematic' in dims:
        summary['kinematic'] = kinematic_summary(run, frames, reach_task)
    if 'upper' in dims:
        from . import upper_muscle
        summary['upper'] = upper_summary(upper_muscle.evaluate(run, out/'upper', mass, 'evalkit', floor_eval, (times, res['masks'])))
    np.savez_compressed(out/'frames.npz', **frames)
    receipt = {'run': str(run), 'floor_m': float(floor), 'foot_offset_m': offset, 'offset_reference_m': offset_reference,
               'floor_eval_m': float(floor_eval), 'reference': variant or offsets, 'mass_kg': mass, 'method': method,
               'prompt': prompt, 'dims': list(dims), 'spec': str(SPEC_PATH.relative_to(ROOT)), 'spec_version': SPEC['version'],
               'fit_prepare': fit_mod.PREPARE, 'contact_margin_m': residual.C['contact_margin_m'],
               'object_free': None if 'ergo' not in dims else bool(ergo.object_free(prompt) if object_free is None else object_free),
               'commit': _commit(), 'seconds': round(time.time()-start, 1)}
    (out/'evalkit.json').write_text(json.dumps({'receipt': receipt, **summary}, indent=1))
    return summary
