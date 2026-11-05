"""E under the three registered contact rules (config/e_criterion_domain_20261105.json), without writing into the work
directory: contacts, GRF fit and residual from coordinates.mot and model.osim. OpenSim python."""
import json
from pathlib import Path

import numpy as np

import biomech

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT/'config/e_criterion_domain_20261105.json').read_text())
RULES = {k: SPEC['rules'][k].get('biomech', {}) for k in ('R0', 'R4', 'R5')}
TAU = {k: SPEC['rules'][k]['tau_envelope_bw'] for k in RULES}
SKATE_M_S = .3
WINDOW_MS = 40.


def kinematics(work):
    import opensim as osim
    osim.Logger.setLevelString('error')
    work = Path(work)
    times, labels, data = biomech.read_storage(work/'coordinates.mot')
    kin = biomech.body_kinematics(osim.Model(str(work/'model.osim')), labels, data)
    return times, labels, data, kin


def contact_v2(heights, reference, margin, dt, hmax=None, onset=None, horizontal=None):
    """E v2: a landmark is in contact when its height above its flat-foot standing height (reference, from the method's
    own floor convention) is <= margin; one world floor for all landmarks. Returns masks and elevations (m)."""
    elevation = {s: {k: h-reference[s][k] for k, h in hs.items()} for s, hs in heights.items()}
    masks = {}
    for s, hs in elevation.items():
        masks[s] = {}
        for k, e in hs.items():
            m = e <= margin
            if hmax is not None or onset is not None:
                v = np.linalg.norm(np.gradient(horizontal[s][k], dt, axis=0), axis=1)
                if hmax is not None:
                    m = m & (v <= hmax)
                if onset is not None:
                    m = biomech.onset_contact(m, v, onset)
            masks[s][k] = m
    return masks, elevation


def evaluate(times, kin, mass, rule, reference=None, world_floor=0.):
    """reference=None: v1 (per-landmark 5th-percentile floor). reference={side: {landmark: m}}: v2 (contact_v2, GRF
    contact points on world_floor, elevation and per-frame lowest elevation returned)."""
    dt = float(np.median(np.diff(times)))
    F, M, C = biomech.required_wrench(kin, dt)
    points, heights = {}, {}
    for side in ('r', 'l'):
        points[side], heights[side] = biomech.sole_points(kin, side)
    horizontal = {side: biomech.landmark_ground(kin, side) for side in ('r', 'l')}
    kw = RULES[rule]
    extra = {}
    if reference is None:
        floor, masks = biomech.contact_masks(heights, dt=dt, hmax=kw.get('hmax'), onset=kw.get('onset'), horizontal=horizontal)
    else:
        masks, elevation = contact_v2(heights, reference, biomech.CONTACT_MARGIN_M, dt, kw.get('hmax'), kw.get('onset'), horizontal)
        floor = world_floor
        extra = {'elevation': elevation, 'lowest_elevation': np.min([e for hs in elevation.values() for e in hs.values()], 0)}
    bw = mass*9.80665
    grf, rF, rM = biomech.estimate_grf(F, M, C, points, masks, floor, bw)
    speed = {s: {k: np.linalg.norm(np.gradient(horizontal[s][k], dt, axis=0), axis=1) for k in horizontal[s]} for s in horizontal}
    return {'dt': dt, 'floor': floor, 'masks': masks, 'grf': grf, 'residual_force': rF, 'residual_moment': rM,
            'residual_bw': np.linalg.norm(rF, axis=1)/bw, 'speed': speed, 'com': C, 'required_force': F, **extra}


def states(masks, dt):
    foot = [masks[s]['heel'] | masks[s]['toe'] for s in ('r', 'l')]
    events = np.concatenate([np.flatnonzero(np.diff(x.astype(int)) != 0)+.5 for x in foot])
    i = np.arange(len(foot[0]))
    near = (np.min(np.abs(i[:, None]-events[None, :]), 1)*dt*1000 <= WINDOW_MS) if len(events) else np.zeros(len(i), bool)
    contact = foot[0] | foot[1]
    return {'event': near, 'contact': ~near & contact, 'flight': ~near & ~contact}


def infeasible(res, rule, tau=None):
    """tau: {state: BW} envelope; default the v1 registered TAU of the rule."""
    st = states(res['masks'], res['dt'])
    tau = tau or TAU[rule]; bound = np.zeros(len(res['residual_bw']))
    for k, m in st.items():
        bound[m] = tau[k]
    return res['residual_bw'] > bound, st


def summary(res, rule, sel, reference_masks=None, tau=None):
    """Metrics over the boolean frame selection sel."""
    bad, st = infeasible(res, rule, tau)
    r = res['residual_bw'][sel]
    contact = np.concatenate([res['masks'][s][k][sel] for s in ('r', 'l') for k in ('heel', 'toe')])
    fast = np.concatenate([res['speed'][s][k][sel] > SKATE_M_S for s in ('r', 'l') for k in ('heel', 'toe')])
    out = {'frames': int(sel.sum()), 'residual_force_rms_bw': float(np.sqrt(np.mean(r**2))) if len(r) else None,
           'unsupported_fraction_5pct_bw': float(np.mean(r > .05)) if len(r) else None,
           'envelope_infeasible_share': float(np.mean(bad[sel])) if len(r) else None,
           'state_share': {k: float(np.mean(m[sel])) for k, m in st.items()},
           'foot_skate_share': float(np.mean(fast[contact])) if contact.any() else None,
           'landmark_contact_share': float(np.mean(contact)) if len(contact) else None}
    if reference_masks is not None:
        ref = np.concatenate([reference_masks[s][k][sel] for s in ('r', 'l') for k in ('heel', 'toe')])
        out['contact_lost_vs_R0'] = float(np.mean(~contact[ref])) if ref.any() else None
    if 'lowest_elevation' in res and len(r):
        low = res['lowest_elevation'][sel]
        out['lowest_elevation_m'] = {'median': float(np.median(low)), 'p5': float(np.percentile(low, 5)),
                                     'p95': float(np.percentile(low, 95))}
    return out


def trim(times, seconds=.5):
    dt = float(np.median(np.diff(times))); n = len(times); k = int(round(seconds/dt))
    sel = np.zeros(n, bool); sel[k:n-k] = True
    return sel


def write_shadow(work, shadow, times, res):
    """A directory that ergo_frames.score_run can read: links to model.osim and coordinates.mot, a dynamics.npz of this rule."""
    shadow = Path(shadow); shadow.mkdir(parents=True, exist_ok=True)
    for name in ('model.osim', 'coordinates.mot'):
        link = shadow/name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path(work).resolve()/name)
    np.savez_compressed(shadow/'dynamics.npz', times=times, residual_force=res['residual_force'], residual_moment=res['residual_moment'],
                        **{f'contact_{s}_{k}': res['masks'][s][k] for s in res['masks'] for k in res['masks'][s]},
                        **{f'grf_{s}': res['grf'][s]['force'] for s in res['grf']})
    return shadow
