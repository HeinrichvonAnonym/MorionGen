"""Clock boundary equality for double-precision resampling arithmetic."""
import numpy as np

BOUNDARY_ROUNDOFF_S=1e-9
IK_CLOCK_ROUNDTRIP_LIMIT_S=1e-6


def supported(times,start,end):
    t=np.asarray(times)
    return np.isfinite(t)&(t>=start-BOUNDARY_ROUNDOFF_S)&(t<=end+BOUNDARY_ROUNDOFF_S)


def native_clock(recorded,input_times):
    recorded=np.asarray(recorded);absolute=np.asarray(input_times)
    canonical=absolute-absolute[0]
    if recorded.shape!=canonical.shape or not np.isfinite(recorded).all():
        raise ValueError('IK/input clock layout mismatch')
    error=float(np.max(abs(recorded-canonical)))
    if error>IK_CLOCK_ROUNDTRIP_LIMIT_S:raise ValueError('IK time mismatch exceeds serialization roundtrip limit')
    return canonical,{'max_roundtrip_error_s':error,'roundtrip_limit_s':IK_CLOCK_ROUNDTRIP_LIMIT_S,
                      'source':'Declared D frame timestamps; native IK frame correspondence verified'}
