"""Bounds for native returned-state observations, not interpolated execution."""
import numpy as np


def observed_hold_bounds(inside, lower, upper, request_seconds=5., hold_seconds=.5):
    """Bound a consecutive-observation streak using timestamp intervals.

    No statement is made about motion between observations. An uncertain event
    boundary is excluded from the definite test and retained in the possible
    test. Interval endpoints need not be probabilistic confidence limits.
    """
    inside=np.asarray(inside,dtype=bool)
    lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
    if inside.shape!=lower.shape or lower.shape!=upper.shape or lower.ndim!=1:
        raise ValueError('Aligned one-dimensional observations required')
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>upper):
        raise ValueError('Finite ordered timestamp bounds required')
    if np.any(np.diff(lower)<0) or np.any(np.diff(upper)<0):
        raise ValueError('Observation bounds must be chronological')
    def check(definite):
        beginning=None
        for i,is_inside in enumerate(inside):
            eligible=(lower[i] if definite else upper[i])>=request_seconds
            if not eligible or not is_inside:
                beginning=None
                continue
            if beginning is None:beginning=i
            span=(lower[i]-upper[beginning]) if definite else (upper[i]-max(lower[beginning],request_seconds))
            if i>beginning and span>=hold_seconds:return True
        return False
    return {'definite':check(True),'possible':check(False)}
