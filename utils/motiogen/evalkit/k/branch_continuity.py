"""Input-direction parallel transport for weakly observed leg axial rotation."""
import numpy as np


def transport_direction(previous, current):
    a = np.asarray(previous, float); b = np.asarray(current, float)
    if not np.isfinite([a, b]).all() or min(np.linalg.norm(a), np.linalg.norm(b)) < 1e-9:
        return None
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
    c = float(np.clip(a@b, -1, 1))
    if c < -1 + 1e-8:
        return None  # half-turn transport is not uniquely observed
    v = np.cross(a, b)
    skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + skew + skew@skew/(1+c)


def weak_flexion_weight(flexion, threshold_deg):
    if not np.isfinite(threshold_deg) or not 0 < threshold_deg < 90:
        raise ValueError('Weak-observation threshold must be between zero and ninety degrees')
    return float(max(0., 1. - abs(flexion)/np.deg2rad(threshold_deg)))
