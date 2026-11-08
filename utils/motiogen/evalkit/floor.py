"""Floor repair of evalkit v1.2 (config/evalkit_v12_20261108.json, floor_repair): one vertical shift of the floor per
clip, estimated from the evaluation frames on the floor the clip was first evaluated on."""
import numpy as np

SELF_FLOOR_STILL_M_S = .05


def self_floor_shift(f, still=SELF_FLOOR_STILL_M_S):
    """Median over the scored frames of the lowest sole-landmark elevation among the landmarks whose vertical speed
    is below `still` (m/s, convention, not derived). Checked on known-floor data before use
    (scratch/self_floor_estimators.py): synthetic walk -0.6 cm, other synthetic -0.3 cm, CLoSD -0.2 cm, PRIMAL +0.1 cm;
    the 10th percentile of the lowest elevation was biased by the rolling foot in walking (-1.8 / -1.3 cm)."""
    E = np.stack([f[f'elevation_{s}_{k}'] for s in 'rl' for k in ('heel', 'toe')], 1)
    E = np.where(np.abs(np.gradient(E, f['times'], axis=0)) < still, E, np.inf).min(1)
    ok = f['scored'] & np.isfinite(E)
    return float(np.median(E[ok])) if ok.any() else float(np.median(f['lowest_elevation_m'][f['scored']]))
