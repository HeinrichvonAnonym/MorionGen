"""Input-only direction-error rate targets, in physical radians per second.

The derivative is that of (fitted unit direction - observed unit direction),
not of the fitted motion alone. It therefore preserves observed fast motion.
This is the existing chord-vector rate diagnostic, not an SO(3) angular rate.
"""
import numpy as np


def direction_error_rate(current, previous, dt):
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite physical time step required')
    current, previous = np.asarray(current), np.asarray(previous)
    if current.shape != previous.shape or current.ndim != 2 or current.shape[-1] != 3:
        raise ValueError('Matching per-direction three-vector errors required')
    if not np.isfinite(current).all() or not np.isfinite(previous).all():
        raise ValueError('Finite observed direction errors required')
    return (current - previous) / dt


def rate_excess_residual(rate, scale, margin, weight):
    if not np.isfinite([scale, margin, weight]).all() or scale <= 0 or not 0 < margin <= 1 or weight < 0:
        raise ValueError('Positive rate scale, margin in (0,1], and nonnegative weight required')
    return weight * np.maximum(np.linalg.norm(rate, axis=-1) / scale - margin, 0.)
