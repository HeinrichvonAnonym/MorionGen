import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def coordinate_precision_valid(positions, resolution_m=1e-4):
    """Reject world coordinates whose stored floating-point spacing erases sub-mm geometry."""
    p=np.asarray(positions)
    magnitude=np.max(np.abs(p),axis=(-2,-1))
    return np.isfinite(magnitude)&(np.spacing(magnitude)<=resolution_m)


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.full_like(v, np.nan), where=n > 1e-6)


def frame(up, right):
    y = unit(up)
    x = unit(right - (right*y).sum(-1, keepdims=True)*y)
    return np.stack([x, y, np.cross(x, y)], axis=-1)


def valid_rotation(r, tolerance=2e-3):
    finite = np.isfinite(r).all(axis=(-2, -1))
    safe = np.where(finite[..., None, None], r, np.eye(3))
    return finite & (np.max(np.abs(safe @ safe.swapaxes(-1, -2)-np.eye(3)), axis=(-2,-1)) < tolerance) & (np.abs(np.linalg.det(safe)-1) < tolerance)


def distance(a, b):
    relative = a.swapaxes(-1, -2) @ b
    return np.degrees(np.arccos(np.clip((np.trace(relative, axis1=-2, axis2=-1)-1)/2, -1, 1)))


def chain(local, parents):
    result = np.empty_like(local)
    for j, p in enumerate(parents):
        result[:, j] = local[:, j] if p < 0 else result[:, p] @ local[:, j]
    return result


def recover_short_gaps(times, rotations, observed, max_seconds):
    """Offline predictions only; no extrapolation or promotion to observations."""
    result = np.asarray(rotations).copy()
    predicted = np.zeros(len(times), bool)
    good = np.flatnonzero(observed & valid_rotation(rotations))
    for a, b in zip(good[:-1], good[1:]):
        if b-a <= 1 or times[b]-times[a] > max_seconds+np.median(np.diff(times))*1.01:
            continue
        result[a+1:b] = Slerp(times[[a,b]], Rotation.from_matrix(rotations[[a,b]]))(times[a+1:b]).as_matrix()
        predicted[a+1:b] = True
    return result, predicted


def tiers(errors, total=None):
    e = np.asarray(errors, float)
    e = e[np.isfinite(e)]
    return {'valid': len(e), 'total': int(total if total is not None else len(errors)),
            'median_deg': float(np.median(e)) if len(e) else None,
            'p95_deg': float(np.percentile(e,95)) if len(e) else None,
            **{f'within_{k}_deg': float(np.mean(e<=k)) if len(e) else None for k in (5,10,20)}}


def rotation_errors(reference, predicted, total=None):
    valid = valid_rotation(reference) & valid_rotation(predicted)
    relative = reference[valid].swapaxes(-1,-2) @ predicted[valid]
    bias = Rotation.from_matrix(relative).as_rotvec().mean(0)*180/np.pi if len(relative) else None
    return {**tiers(distance(reference[valid], predicted[valid]), total if total is not None else len(reference)),
            'signed_rotvec_bias_deg': bias.tolist() if bias is not None else None,
            'bias_axes': 'reference segment frame XYZ; rotation-vector mean'}
