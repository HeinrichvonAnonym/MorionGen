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
    from .k import load as load_k
    from motion import Y_TO_Z
    motion = load_k(run)
    return motion.positions @ Y_TO_Z.T, motion.names, motion.fps


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


def muscle_residual(frames):
    """Index of the nearest residual frame of every static-optimization frame and its residual-positive flag."""
    t, times = frames['muscle_times'], frames['times']
    i = np.clip(np.searchsorted(times, t), 1, len(times)-1)
    i = np.where(np.abs(times[i-1]-t) <= np.abs(times[i]-t), i-1, i)
    return i, np.asarray(frames['residual_positive'], bool)[i]


def muscle_split(frames):
    """Leg-muscle effort of v1.2 (config/evalkit_v12_20261108.json, forced.leg_muscle): primary over the static-
    optimization frames whose nearest residual frame is residual-negative, forced over every frame (the v1.1 value)."""
    if 'residual_positive' not in frames or 'muscle_effort' not in frames:
        return {}
    eff = frames['muscle_effort']; i, pos = muscle_residual(frames)
    eligible = ~pos
    if 'k_force_eligible' in frames:
        eligible = eligible & np.asarray(frames['k_force_eligible'],bool)[i]
    return {'primary_muscle_effort_sum_a2_mean': float(eff[eligible].mean()) if eligible.any() else None,
            'primary_frames': int(eligible.sum()), 'forced_muscle_effort_sum_a2_mean': float(eff.mean()),
            'forced_frames': int(len(eff)), 'forced_residual_positive_share': float(pos.mean()),
            'forced_residual_bw_p95': float(np.percentile(frames['residual_bw'][i][pos], 95)) if pos.any() else 0.}


def add_forced(out):
    """Forced metrics of v1.2 on an existing evaluation from its saved files (no new solve): leg-muscle split and the
    lumbar reclassification."""
    out = Path(out); rec = json.loads((out/'evalkit.json').read_text())
    if 'muscle' in rec:
        rec['muscle'].update(muscle_split(dict(np.load(out/'frames.npz', allow_pickle=True))))
    if 'upper' in rec and (out/'upper/lumbar/lumbar.json').exists():
        from . import upper_muscle
        rec['upper']['lumbar'] = upper_summary({'lumbar': upper_muscle.reclassify(out/'upper/lumbar', rec['receipt']['mass_kg'])})['lumbar']
    rec['receipt']['forced_commit'] = _commit()
    (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
    return rec


def ergo_dim(run, work, f, masks, sel, src, method, prompt, object_free, frames, k_config=None):
    """RULA / REBA classes of one fit (f: fit_mod.fit result); adds the ergo_* arrays to frames, returns the summary."""
    from .k import load as load_k, evaluate as evaluate_k
    k = evaluate_k(load_k(run, method, config=k_config), config=k_config)
    e = ergo.score(work/'model.osim', f['times'], f['labels'], f['coords'], masks, sel, src, ('unknown', None, None),
                   ergo.object_free(prompt) if object_free is None else object_free, f['unmeasurable'], k_result=k, arm_valid=f['arm_valid'])
    frames.update({f'ergo_{k}': v for k, v in e.items() if k != 'source_mode'})
    summary = {'source_mode': e['source_mode'], 'frames': int(len(e['index']))}
    for m in ergo.RISK:
        c = e[f'{m.lower()}_class']
        summary[m.lower()] = {**{f'{n}_share': float(np.mean(c == i)) for i, n in enumerate(ergo.CLASSES)},
                              'lower_median': float(np.median(e[f'{m.lower()}_bounds'][:, 0])),
                              'upper_median': float(np.median(e[f'{m.lower()}_bounds'][:, 1]))}
    return summary


def muscle_dim(work, times, sel, frames):
    """Leg-muscle static optimization on work (grf.xml of the residual step); adds the muscle arrays to frames."""
    mu = muscle.evaluate(work, float(times[sel][0]), float(times[sel][-1]))
    frames.update(muscle_times=mu['times'], muscle_effort=mu['effort'], muscle_reserve_abs_sum_nm=mu['reserve_abs_sum_nm'])
    summary = mu['summary']
    up = muscle.upper_body(work/'so', work/'model_so.osim')
    frames.update(net_moment_lumbar_nm_kg=up['lumbar'], net_moment_shoulder_nm_kg=up['shoulder'])
    summary.update(net_moments(up))
    summary.update(muscle_split(frames))
    return summary


def add_dims(run, out, dims, object_free=None, source=None):
    """ergo and / or muscle on an existing evaluation from its cached fit (work), contact masks and grf (the residual
    step at floor_eval), without refitting: the same inputs evaluate would use, so the same result.
    object_free: as in evaluate (the receipt holds it only when ergo was evaluated)."""
    import opensim as osim
    from .k import load as load_k
    osim.Logger.setLevelString('error')
    out = Path(out).resolve(); work = out/'work'; rec = json.loads((out/'evalkit.json').read_text()); r = rec['receipt']
    frames = dict(np.load(out/'frames.npz', allow_pickle=True))
    sel = frames['scored']
    k_config = r.get('k_config')
    if 'ergo' in dims:
        f = fit_mod.fit(run, work, r['mass_kg'], r['floor_m'], config=k_config)
        if not np.array_equal(f['times'], frames['times']):
            raise ValueError('cached fit does not match frames.npz')
        masks = {s: {k: frames[f'contact_{s}_{k}'] for k in LANDMARKS} for s in SIDES}
        variant = r['reference'] if isinstance(r['reference'], str) else None
        method = r['method'] or (variant.split('__')[0] if variant else None)
        rec['ergo'] = ergo_dim(run, work, f, masks, sel, source or load_k(run, config=k_config).legacy(), method, r['prompt'],
                               object_free, frames, k_config=k_config)
        r['method'] = method
        r['object_free'] = bool(ergo.object_free(r['prompt']) if object_free is None else object_free)
    if 'muscle' in dims:
        rec['muscle'] = muscle_dim(work, frames['times'], sel, frames)
    np.savez_compressed(out/'frames.npz', **frames)
    r['dims'] = sorted(set(r['dims']) | set(dims), key=ALL_DIMS.index)
    r['added_commit'] = _commit()
    (out/'evalkit.json').write_text(json.dumps(rec, indent=1))
    return rec


def foot_offset(run, work, mass, floor, source=None, k_config=None):
    """Fit (cached in work) and fit.foot_offset of one run: the per-clip input of the floor correction."""
    import opensim as osim
    import biomech
    from .k import load as load_k
    osim.Logger.setLevelString('error')
    f = fit_mod.fit(run, work, mass, floor, config=k_config)
    kin = biomech.body_kinematics(osim.Model(str(Path(work)/'model.osim')), f['labels'], f['coords'])
    return fit_mod.foot_offset(*(source or load_k(run, config=k_config).legacy()), kin, f['times']), f, kin


def evaluate(run, out, floor, reference, mass=75., method=None, prompt=None, dims=DIMS, source=None, object_free=None,
             reach_task=None, offset_reference=None, k_config=None):
    """run: directory with motion.npz (canonical Z-up joints). floor: floor height (m, canonical Z) of this motion.
    reference: floor-reference variant (spec.reference_offsets) or {side: {landmark: m}} offsets above the floor.
    method: ergo neutral key (default: the reference variant). prompt: task text (None: object-free).
    object_free: overrides the text rule (evalkit.sources.object_free). reach_task: kinematic.evaluate wrist target.
    offset_reference: the method's median foot offset (spec floor.per_clip_correction); the residual, ergo and muscle
    dimensions then use floor + this clip's foot offset - offset_reference. None: the input floor (no correction)."""
    import opensim as osim
    import biomech
    from .k import load as load_k
    osim.Logger.setLevelString('error')
    start = time.time(); out = Path(out).resolve(); work = out/'work'
    variant = reference if isinstance(reference, str) else None
    offsets = reference_offsets(reference) if variant else reference
    src = source or load_k(run, config=k_config).legacy()
    offset, f, kin = foot_offset(run, work, mass, floor, src, k_config=k_config)
    delta = 0. if offset_reference is None else offset['mean']-offset_reference
    floor_eval = floor+delta
    ref = absolute_reference(offsets, floor_eval)
    times = f['times']
    sel = fit_mod.scored(times)
    res = residual.evaluate(kin, times, mass, floor_eval, ref)
    biomech.write_loads(times, res['grf'], floor_eval, work/'grf.mot', work/'grf.xml')
    frames = {'times': times, 'scored': sel, 'unmeasurable': f['unmeasurable'], 'ik_max_error_m': f['ik_max_error_m'],
              **{'arm_'+s+'_fit_valid':v for s,v in f['arm_valid'].items()},
              'residual_bw': res['residual_bw'], 'residual_force': res['residual_force'], 'com': res['com'],
              'lowest_elevation_m': res['lowest_elevation'], 'state': res['state'],
              **{f'contact_{s}_{k}': res['masks'][s][k] for s in SIDES for k in LANDMARKS},
              **{f'elevation_{s}_{k}': res['elevation'][s][k] for s in SIDES for k in LANDMARKS}}
    for category in ['region_angle_valid','region_registered_valid','region_position_valid']:
        frames.update({f'k_{category}_{region}':mask for region,mask in f.get(category,{}).items()})
    if f.get('region_registered_valid'):
        frames['k_force_eligible'] = np.logical_and.reduce([
            f[category][region] for category in ['region_registered_valid','region_position_valid']
            for region in f[category]])
    from .k import load as load_k, evaluate as evaluate_k
    from .k.posture import summary as k_summary
    summary = {'ik': {**f['receipt']['ik_marker_error'], 'unmeasurable_share': float(f['unmeasurable'][sel].mean()),
                      'arm_source_fidelity':f['receipt']['arm_quality']},
               'k': k_summary(evaluate_k(load_k(run, method, config=k_config), config=k_config))}
    if 'ik_marker_error_scope' in f['receipt']:
        summary['ik']['error_scope'] = f['receipt']['ik_marker_error_scope']
        summary['k_downstream_support'] = {
            'force_eligible_share':float(frames['k_force_eligible'][sel].mean()),
            'scope':'registered angular and spatial input fidelity required in every model region; force and muscle accuracy require independent validation',
            'residual_statistics':'all-frame diagnostics; per-frame eligibility saved separately',
            'muscle_primary':'residual-negative and K-force-eligible frames; forced result retains all solved frames'}
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
        summary['ergo'] = ergo_dim(run, work, f, res['masks'], sel, src, method, prompt, object_free, frames, k_config=k_config)
    if 'muscle' in dims:
        summary['muscle'] = muscle_dim(work, times, sel, frames)
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
    if k_config is not None:
        from .k.config import resolve_config
        receipt['k_config'] = resolve_config(k_config)
        receipt['k_support_scope'] = f['receipt'].get('support_scope')
    (out/'evalkit.json').write_text(json.dumps({'receipt': receipt, **summary}, indent=1))
    return summary
