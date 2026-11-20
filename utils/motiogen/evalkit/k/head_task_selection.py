"""D-only head task selection on model FK; body priors remain diagnostics.

Thresholds and margin are inherited, not fitted to held-out outcomes. This
module receives D and F only. Relative twist is required only when D observes it.
"""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .candidate_quality import assess as body_assess, exact_indices
from .native_frames import read
from .geometry import valid_rotation

VERSION = 'optical_head_task_d_only_v1'
HEAD_KEYS = {
    'direction/head_neck_axis', 'rate/head_neck_axis',
    'added_step/head_neck_axis', 'interpolation_direction/head_neck_axis',
    'interpolation_rate/head_neck_axis', 'rotation/head_relative_thorax',
}


def assess(motion, fitted, *, head_rotation=None, thresholds=None,
           include_export_path=False, ranking_policy='observed_then_path_v1'):
    full = body_assess(motion, fitted, head_rotation=head_rotation,
                       thresholds=thresholds, include_export_path=include_export_path,
                       ranking_policy=ranking_policy)
    metrics = {k: v for k, v in full['metrics'].items() if k in HEAD_KEYS}
    limits = full['thresholds']

    def record(name, error, support, threshold, statistic='p95'):
        good = support & np.isfinite(error)
        n = int(support.sum()); present = int(good.sum())
        value = float(np.percentile(error[good], 95) if statistic == 'p95'
                      else np.max(error[good])) if present else None
        metrics[name] = dict(expected=n, present=present, missing=n-present,
                             value=value, statistic=statistic, threshold=threshold,
                             passed=None if not n else present == n and value <= threshold)

    def distance(a, b):
        good = valid_rotation(a) & valid_rotation(b)
        result = np.full(len(a), np.nan)
        result[good] = np.rad2deg(Rotation.from_matrix(a[good].swapaxes(-1, -2) @ b[good]).magnitude())
        return result

    def velocity(a, b, times):
        good = valid_rotation(a[:-1]) & valid_rotation(a[1:]) & valid_rotation(b[:-1]) & valid_rotation(b[1:])
        result = np.full(len(times)-1, np.nan)
        va = Rotation.from_matrix(a[1:][good] @ a[:-1][good].swapaxes(-1,-2)).as_rotvec()
        vb = Rotation.from_matrix(b[1:][good] @ b[:-1][good].swapaxes(-1,-2)).as_rotvec()
        result[good] = np.rad2deg(np.linalg.norm(vb-va, axis=-1))/np.diff(times)[good]
        return result

    native = read(motion)
    if native['status'] == 'declared_native_frames':
        d = native['head_relative_thorax']; observed = native['head_relative_valid']
        ft = np.asarray(fitted['times']); idx = exact_indices(motion.times, ft)
        dense = np.full((len(ft),3,3), np.nan) if head_rotation is None else np.asarray(head_rotation)
        f = dense[idx]; steps = observed[:-1] & observed[1:]
        record('rotation_rate/head_relative_thorax', velocity(d,f,motion.times), steps, limits['direction_rate_deg_s'])
        record('rotation_added_step/head_relative_thorax', np.maximum(distance(f[:-1],f[1:])-distance(d[:-1],d[1:]),0), steps, limits['added_step_deg'], 'max')
        if include_export_path:
            # Fixed fitted anchors define an interpolation hypothesis, not new D.
            target = np.full_like(dense,np.nan); supported = np.zeros(len(ft),bool)
            for i in range(len(d)-1):
                mask=(ft>=motion.times[i]) & (ft<=motion.times[i+1])
                if observed[i] and observed[i+1]:
                    supported[mask]=True
                    if valid_rotation(f[i:i+2]).all():
                        target[mask]=Slerp(motion.times[i:i+2],Rotation.from_matrix(f[i:i+2]))(ft[mask]).as_matrix()
            inserted=np.ones(len(ft),bool);inserted[idx]=False
            record('interpolation_rotation/head_relative_thorax',distance(target,dense),supported&inserted,limits['angle_deg'],'max')
            record('interpolation_rotation_rate/head_relative_thorax',velocity(target,dense,ft),supported[:-1]&supported[1:],limits['direction_rate_deg_s'])

    def group_quality(items):
        ratios=[v['value']/v['threshold'] for v in items if v['value'] is not None]
        return [sum(v['missing'] for v in items),sum(not v['passed'] for v in items),
                max(ratios,default=0.),sum(ratios)/len(ratios) if ratios else 0.]
    supported=[v for v in metrics.values() if v['expected']]
    if not supported:raise ValueError('No supported head observations')
    obs=group_quality([v for k,v in metrics.items() if v['expected'] and not k.startswith('interpolation_')])
    path=group_quality([v for k,v in metrics.items() if v['expected'] and k.startswith('interpolation_')])
    return dict(full,metrics=metrics,rank=obs[:2]+path[:2]+obs[2:]+path[2:],
                passed=all(v['passed'] for v in supported),selection_scope=VERSION,
                body_prior_diagnostics={k:v for k,v in full['metrics'].items() if k not in HEAD_KEYS},
                scope='D-supported head direction/relative rotation, coverage, native temporal and export path; no R and no body-prior stopping targets')
