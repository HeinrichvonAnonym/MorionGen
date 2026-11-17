"""Input-only export clocks; preserve native knots without changing native poses."""
import numpy as np


def native_multiple_clock(times,minimum_rate=100.):
    """Uniform subdivisions of a verified uniform D clock, at >=minimum_rate.

    The rate follows D, never R. A 30 Hz input gives 120 Hz for a 100 Hz minimum.
    Existing uniform100 exports remain a separate control. Irregular inputs must
    choose a declared alternative; they are never silently regularized here.
    """
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or len(times)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError('Finite increasing native timestamps required')
    if not np.isfinite(minimum_rate) or minimum_rate<=0:raise ValueError('Positive minimum export rate required')
    dt=np.diff(times);mean=float(np.mean(dt));jitter=float(abs(dt-mean).max())
    if not np.allclose(dt,mean,rtol=1e-9,atol=1e-9):
        raise ValueError('native_multiple requires a uniform input clock')
    subdivisions=max(1,int(np.ceil(minimum_rate*mean-1e-9)))
    output=np.r_[(times[:-1,None]+dt[:,None]*(np.arange(subdivisions)/subdivisions)).ravel(),times[-1]]
    if not np.array_equal(output[::subdivisions],times):raise ValueError('Native timestamp preservation failed')
    return output,{'policy':'native_multiple','minimum_rate_hz':float(minimum_rate),'actual_rate_hz':subdivisions/mean,
                   'subdivisions_per_native_interval':subdivisions,'native_frames':len(times),'output_frames':len(output),
                   'native_dt_jitter_s':jitter,'scope':'Kinematic export only; source timestamps preserved; no reference access or force qualification'}
