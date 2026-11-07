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
from . import fit as fit_mod, residual, ergo, muscle

DIMS = ('residual', 'ergo', 'muscle')


def _commit():
    try:
        return subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    except OSError:
        return None


def _stats(x):
    x = np.asarray(x, float)
    return {k: float(np.percentile(x, q)) for k, q in (('p5', 5), ('median', 50), ('p95', 95))} if len(x) else None


def evaluate(run, out, floor, reference, mass=75., method=None, prompt=None, dims=DIMS, source=None):
    """run: directory with motion.npz (canonical Z-up joints). floor: floor height (m, canonical Z) of this motion.
    reference: floor-reference variant (spec.reference_offsets) or {side: {landmark: m}} offsets above the floor.
    method: ergo neutral key (default: the reference variant). prompt: task text (None: object-free)."""
    import opensim as osim
    import biomech
    import biomech_sources
    osim.Logger.setLevelString('error')
    start = time.time(); out = Path(out).resolve(); work = out/'work'
    variant = reference if isinstance(reference, str) else None
    offsets = reference_offsets(reference) if variant else reference
    ref = absolute_reference(offsets, floor)
    f = fit_mod.fit(run, work, mass, floor)
    times, labels, coords = f['times'], f['labels'], f['coords']
    sel = fit_mod.scored(times)
    kin = biomech.body_kinematics(osim.Model(str(work/'model.osim')), labels, coords)
    res = residual.evaluate(kin, times, mass, floor, ref)
    biomech.write_loads(times, res['grf'], floor, work/'grf.mot', work/'grf.xml')
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
        src = source or biomech_sources.load(run)
        e = ergo.score(work/'model.osim', times, labels, coords, res['masks'], sel, src, ergo.neutral(method, run),
                       ergo.object_free(prompt), f['unmeasurable'])
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
    np.savez_compressed(out/'frames.npz', **frames)
    receipt = {'run': str(run), 'floor_m': float(floor), 'reference': variant or offsets, 'mass_kg': mass, 'method': method,
               'prompt': prompt, 'dims': list(dims), 'spec': str(SPEC_PATH.relative_to(ROOT)), 'spec_version': SPEC['version'],
               'fit_prepare': fit_mod.PREPARE, 'contact_margin_m': residual.C['contact_margin_m'],
               'commit': _commit(), 'seconds': round(time.time()-start, 1)}
    (out/'evalkit.json').write_text(json.dumps({'receipt': receipt, **summary}, indent=1))
    return summary
