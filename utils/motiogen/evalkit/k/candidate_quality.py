"""Whole-clip candidate ranking using D only, at exact observed timestamps.

No reference motions, reference lengths, source-jump labels or window IDs enter
this module. This is an input-fidelity selector, not accuracy qualification.
"""
import numpy as np
from motiogen.degradation.metrics import points, angle, REGIONS
from .geometry import valid_rotation


DIRECTIONS = {
    'trunk_axis': ('pelvis_mid', 'shoulder_mid'),
    'shoulder_line': ('left_shoulder', 'right_shoulder'),
    'pelvis_line': ('left_hip', 'right_hip'),
    'head_neck_axis': ('neck', 'head'),
}
for side in ('left', 'right'):
    for part, start, end in [('upper_arm','shoulder','elbow'), ('forearm','elbow','wrist'),
                             ('thigh','hip','knee'), ('shank','knee','ankle'), ('foot_axis','ankle','foot')]:
        DIRECTIONS[f'{side}_{part}'] = (f'{side}_{start}', f'{side}_{end}')


def exact_indices(observed_times, output_times):
    observed_times, output_times = np.asarray(observed_times), np.asarray(output_times)
    if (not np.isfinite(output_times).all() or np.any(np.diff(output_times) <= 0)
            or not np.isfinite(observed_times).all() or np.any(np.diff(observed_times) <= 0)):
        raise ValueError('Candidate clocks must be finite and strictly increasing')
    index = np.searchsorted(output_times, observed_times)
    index = np.clip(index, 0, len(output_times)-1)
    previous = np.maximum(index-1, 0)
    index = np.where(abs(output_times[previous]-observed_times) < abs(output_times[index]-observed_times), previous, index)
    if not np.allclose(output_times[index], observed_times, atol=1e-8, rtol=0):
        raise ValueError('Candidate ranking requires exact D knots; resampling is a separate experiment')
    return index


def _unit(values):
    length = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.where(length > 1e-9, values/np.maximum(length, 1e-9), np.nan)


def assess(motion, fitted, *, head_rotation=None, thresholds=None, include_export_path=False, ranking_policy='flat_v1'):
    """Assess model FK on fixed D support. Missing F never reduces denominators."""
    limits = dict(angle_deg=5., position_cm=5., direction_rate_deg_s=10., added_step_deg=5.)
    if ranking_policy not in ('flat_v1','observed_then_path_v1'):
        raise ValueError('Unknown candidate-ranking policy')
    if thresholds is not None:
        if set(thresholds) != set(limits):
            raise ValueError('Declare every candidate-selection threshold explicitly')
        limits.update(thresholds)
    if not all(np.isfinite(x) and x > 0 for x in limits.values()):
        raise ValueError('Invalid candidate-selection thresholds')
    from .motion import Motion
    index = exact_indices(motion.times, fitted['times'])
    decoded = points(motion)
    # Priors are explicitly unsupported even when the reader emits finite points.
    for key in motion.legacy_prior_names():
        if key in decoded:
            decoded[key] = np.full_like(decoded[key], np.nan)
    decoded['pelvis_mid'] = (decoded['left_hip']+decoded['right_hip'])/2
    decoded['shoulder_mid'] = (decoded['left_shoulder']+decoded['right_shoulder'])/2
    model = points(Motion(np.asarray(fitted['positions'])[index], list(fitted['names']), motion.times, motion.method))
    metrics = {}

    def record(name, error, observed, threshold, statistic='p95'):
        observed = np.asarray(observed, bool)
        valid = observed & np.isfinite(error)
        values = np.asarray(error)[valid]
        value = float(np.percentile(values,95) if statistic == 'p95' else np.max(values)) if len(values) else None
        expected, present = int(observed.sum()), int(valid.sum())
        metrics[name] = dict(expected=expected, present=present, missing=expected-present,
                             value=value, statistic=statistic, threshold=threshold,
                             passed=None if expected == 0 else present == expected and value <= threshold)

    for names in REGIONS.values():
        for name in names:
            record('position/'+name, np.linalg.norm(model[name]-decoded[name],axis=-1)*100,
                   np.isfinite(decoded[name]).all(-1), limits['position_cm'])
    for name, (start, end) in DIRECTIONS.items():
        d, f = decoded[end]-decoded[start], model[end]-model[start]
        observed = np.isfinite(_unit(d)).all(-1)
        record('direction/'+name, angle(d,f), observed, limits['angle_deg'])
        support = observed[1:] & observed[:-1]
        rate = np.diff(_unit(f)-_unit(d),axis=0)/np.diff(motion.times)[:,None]
        record('rate/'+name, np.linalg.norm(rate,axis=-1)*180/np.pi, support, limits['direction_rate_deg_s'])
        # Per-step added jump, rather than subtracting maxima at unrelated times.
        added = np.maximum(angle(f[:-1],f[1:])-angle(d[:-1],d[1:]),0.)
        record('added_step/'+name, added, support, limits['added_step_deg'], 'max')
    for side in ('left','right'):
        for joint, parent, child in [('knee','hip','ankle'), ('elbow','shoulder','wrist')]:
            def flex(p):
                return 180-angle(p[f'{side}_{parent}']-p[f'{side}_{joint}'], p[f'{side}_{child}']-p[f'{side}_{joint}'])
            d, f = flex(decoded), flex(model)
            record(f'angle/{side}_{joint}', abs(f-d), np.isfinite(d), limits['angle_deg'])
    from .native_frames import read
    native = read(motion)
    if native['status'] == 'declared_native_frames':
        d = native['head_relative_thorax']
        f = np.full_like(d,np.nan) if head_rotation is None else np.asarray(head_rotation)[index]
        error = np.full(len(d),np.nan)
        good = valid_rotation(d) & valid_rotation(f)
        from scipy.spatial.transform import Rotation
        error[good] = np.rad2deg(Rotation.from_matrix(d[good].swapaxes(-1,-2)@f[good]).magnitude())
        record('rotation/head_relative_thorax',error,native['head_relative_valid'],limits['angle_deg'])
    if include_export_path:
        # Diagnostic only: a fitted-anchor path is not a dense observation.
        # Inspect actual exported FK between D knots to catch compensating
        # joint branches that the native-only selector cannot see.
        full_times=np.asarray(fitted['times'])
        if full_times[0]<motion.times[0]-1e-8 or full_times[-1]>motion.times[-1]+1e-8:
            raise ValueError('Export-path inspection requires bounded input clock')
        dense=points(Motion(np.asarray(fitted['positions']),list(fitted['names']),full_times,motion.method))
        dense['pelvis_mid']=(dense['left_hip']+dense['right_hip'])/2
        dense['shoulder_mid']=(dense['left_shoulder']+dense['right_shoulder'])/2
        left=np.clip(np.searchsorted(motion.times,full_times,side='right')-1,0,len(motion.times)-2)
        alpha=(full_times-motion.times[left])/(motion.times[left+1]-motion.times[left])
        inserted=np.ones(len(full_times),bool);inserted[index]=False
        for name,(start,end) in DIRECTIONS.items():
            native=model[end]-model[start]
            target=(1-alpha[:,None])*native[left]+alpha[:,None]*native[left+1]
            supported_native=np.isfinite(_unit(decoded[end]-decoded[start])).all(-1)
            observed=supported_native[left]&supported_native[left+1]
            observed[index]=supported_native
            actual=dense[end]-dense[start]
            record('interpolation_direction/'+name,angle(target,actual),observed&inserted,limits['angle_deg'],'max')
            steps=observed[1:]&observed[:-1]
            rate=np.diff(_unit(actual)-_unit(target),axis=0)/np.diff(full_times)[:,None]
            record('interpolation_rate/'+name,np.linalg.norm(rate,axis=-1)*180/np.pi,steps,limits['direction_rate_deg_s'])
    supported = [m for m in metrics.values() if m['expected']]
    if not supported or not any(m['present'] for m in supported):
        raise ValueError('No supported candidate observations')
    ratios = [m['value']/m['threshold'] for m in supported if m['value'] is not None]
    missing = sum(m['missing'] for m in supported)
    failures = sum(not m['passed'] for m in supported)
    # Lexicographic whole-clip ranking. Declaration order is the final tie-break.
    rank = [missing, failures, max(ratios, default=float('inf')), sum(ratios)/len(ratios) if ratios else float('inf')]
    if ranking_policy=='observed_then_path_v1':
        observed=[v for k,v in metrics.items() if v['expected'] and not k.startswith('interpolation_')]
        path=[v for k,v in metrics.items() if v['expected'] and k.startswith('interpolation_')]
        def quality(group):
            ratios=[v['value']/v['threshold'] for v in group if v['value'] is not None]
            return [sum(v['missing'] for v in group),sum(not v['passed'] for v in group),
                    max(ratios,default=0.),sum(ratios)/len(ratios) if ratios else 0.]
        obs,pth=quality(observed),quality(path)
        rank=obs[:2]+pth[:2]+obs[2:]+pth[2:]
    return dict(metrics=metrics, rank=rank, passed=bool(supported) and missing == 0 and failures == 0,
                ranking_policy=ranking_policy,
                scope='D-only full native clock; no source-jump exclusions; whole-clip selection, no per-frame switching',
                frames=len(motion.times), thresholds=limits,
                export_path_diagnostic=bool(include_export_path),
                export_path_scope='Fitted-anchor linear path assumption gated by D support; no R or dense observation qualification')


def choose(qualities):
    if not qualities:
        raise ValueError('No fitted candidates')
    return min(qualities, key=lambda name: tuple(qualities[name]['rank']))
