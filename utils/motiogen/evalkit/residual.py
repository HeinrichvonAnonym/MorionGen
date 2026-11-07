"""Residual force: contact from landmark height above the input floor (height only), contact-constrained GRF fit
(friction cone, sole points on the input floor) and the whole-body force it leaves unexplained. OpenSim python."""
import numpy as np

from .spec import SPEC, LANDMARKS, SIDES
import biomech

C = SPEC['residual']


def contact(heights, reference, margin=C['contact_margin_m']):
    """A landmark is in contact when its height minus its absolute flat-foot reference is <= margin."""
    elevation = {s: {k: h-reference[s][k] for k, h in hs.items()} for s, hs in heights.items()}
    return {s: {k: e <= margin for k, e in hs.items()} for s, hs in elevation.items()}, elevation


def states(masks, dt, window_ms=C['event_window_ms']):
    """event: within window_ms of a foot contact change; otherwise contact (some landmark down) or flight."""
    foot = [masks[s]['heel'] | masks[s]['toe'] for s in SIDES]
    events = np.concatenate([np.flatnonzero(np.diff(x.astype(int)) != 0)+.5 for x in foot])
    i = np.arange(len(foot[0]))
    near = (np.min(np.abs(i[:, None]-events[None, :]), 1)*dt*1000 <= window_ms) if len(events) else np.zeros(len(i), bool)
    down = foot[0] | foot[1]
    return {'event': near, 'contact': ~near & down, 'flight': ~near & ~down}


def evaluate(kin, times, mass, floor, reference, margin=None):
    dt = float(np.median(np.diff(times)))
    F, M, com = biomech.required_wrench(kin, dt)
    points, heights = {}, {}
    for s in SIDES:
        points[s], heights[s] = biomech.sole_points(kin, s)
    masks, elevation = contact(heights, reference, C['contact_margin_m'] if margin is None else margin)
    bw = mass*9.80665
    grf, rF, rM = biomech.estimate_grf(F, M, com, points, masks, floor, bw)
    st = states(masks, dt)
    return {'dt': dt, 'masks': masks, 'elevation': elevation, 'grf': grf, 'residual_force': rF, 'residual_moment': rM,
            'residual_bw': np.linalg.norm(rF, axis=1)/bw, 'com': com,
            'lowest_elevation': np.min([elevation[s][k] for s in SIDES for k in LANDMARKS], 0),
            'state': np.select([st['event'], st['contact']], ['event', 'contact'], 'flight')}


def judge(res, tau=None, depth=None):
    """Per-frame positive: residual above the state's tau (BW) or the lowest landmark deeper than depth below its
    reference. Defaults: the registered values."""
    tau = tau or C['tau_bw']; depth = C['penetration_depth_m'] if depth is None else depth
    bound = np.array([tau[s] for s in res['state']])
    return (res['residual_bw'] > bound) | (res['lowest_elevation'] < -depth)
