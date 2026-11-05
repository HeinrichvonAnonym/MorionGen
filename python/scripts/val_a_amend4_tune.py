#!/usr/bin/env python
"""Amendment 4 development step (config/val_a_addb_20261105_amend4.json): contact balanced accuracy (V3, 20 N) of the
height + horizontal-speed rule over the registered grid on the development trials, from the pipeline kinematics.

Caches heel/toe trajectories (100 Hz, filtered coordinates -> FK) to <DATA>/addb_val_20261105/amend4_cache/<id>.npz and
writes develop_logs/test_runs/20261105_val_a/amend4_dev_grid.json. --evaluate: V3 from the jointCenter landmarks
(val_a_amend4_landmarks.py) for the bridge check and the held-out trials -> amend4_heldout.json. Runs in the OpenSim python.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
import biomech  # noqa: E402

LOG = ROOT/'develop_logs/test_runs/20261105_val_a'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/addb_val_20261105')
SPEC = json.loads((ROOT/'config/val_a_addb_20261105_amend4.json').read_text())
SCOPE = json.loads((ROOT/'config/val_a_addb_20261105_amend2.json').read_text())
UNMODELED = json.loads((ROOT/'config/val_a_addb_20261105_amend1.json').read_text())['threshold']['exclusions']['unmodeled_support_tasks']


def development():
    reg = json.loads((LOG/'selection.json').read_text())['trials']; held = json.loads((LOG/'selection_heldout.json').read_text())['trials']
    walk = [(r, DATA/'work/kin30'/r['id']) for r in reg if r['class'] == 'walk']
    walk += [(r, DATA/'work/kin30_a1'/r['id']) for r in held if r['class'] == 'walk']
    walk = [(r, w) for r, w in walk if 'tmf' not in r['task']]
    jump = [(r, DATA/'work/kin30'/r['id']) for r in reg if r['class'] == 'jump' and r['task'] not in UNMODELED]
    return walk, jump


def landmarks(r, work):
    cache = DATA/'amend4_cache'/f"{r['id']}.npz"
    if cache.exists():
        return dict(np.load(cache))
    import opensim as osim
    times, labels, data = biomech.read_storage(work/'coordinates.mot')
    kin = biomech.body_kinematics(osim.Model(str(work/'model.osim')), labels, data)
    out = {'times': times}
    for s in 'rl':
        for k, (body, _, _) in biomech.SOLE.items():
            out[f'{s}_{k}'] = kin['origin'][:, kin['names'].index(f'{body}_{s}')]
    cache.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(cache, **out)
    return out


def balanced(pred, true):
    if true.all() or (~true).all():
        return None
    return float(.5*(pred[true].mean()+(~pred[~true]).mean()))


def v3(lm, measured, hmax, threshold_n=20.):
    t = lm['times']; dt = float(np.median(np.diff(t)))
    heights = {s: {k: lm[f'{s}_{k}'][:, 1] for k in ('heel', 'toe')} for s in 'rl'}
    horizontal = {s: {k: lm[f'{s}_{k}'][:, [0, 2]] for k in ('heel', 'toe')} for s in 'rl'}
    _, masks = biomech.contact_masks(heights, dt=dt, hmax=hmax, horizontal=horizontal if hmax is not None else None)
    m = np.load(measured) if isinstance(measured, (str, Path)) else measured; trim = slice(int(.5/dt), len(t)-int(.5/dt)); tt = t[trim]
    near = np.clip(np.round(tt/np.median(np.diff(m['times']))).astype(int), 0, len(m['times'])-1)
    ok = m['valid'][near] & ~m['excluded'][near]
    acc = []
    for s in 'rl':
        pred = (masks[s]['heel'] | masks[s]['toe'])[trim][ok]
        true = np.interp(tt, m['times'], m[f'grf_{s}'][:, 1])[ok] > threshold_n
        acc.append(balanced(pred, true))
    acc = [a for a in acc if a is not None]
    return float(np.mean(acc)) if acc else None


def main():
    walk, jump = development()
    grid = [None]+SPEC['development']['grid_m_s']
    result = {'walk_trials': len(walk), 'jump_trials': len(jump), 'per_value': {}}
    per_trial = {}
    for cls, rows in (('walk', walk), ('jump', jump)):
        for r, work in rows:
            lm = landmarks(r, work); meas = Path(r['sources']['kin30'])/'measured.npz'
            per_trial[r['id']] = {'class': cls, **{str(h): v3(lm, meas, h) for h in grid}}
            print(cls, r['id'], flush=True)
    for h in grid:
        row = {}
        for cls in ('walk', 'jump'):
            xs = [p[str(h)] for p in per_trial.values() if p['class'] == cls and p[str(h)] is not None]
            row[cls] = {'median': float(np.median(xs)), 'n': len(xs)}
        result['per_value'][str(h)] = row
    vals = {h: result['per_value'][str(h)]['walk']['median'] for h in SPEC['development']['grid_m_s']
            if result['per_value'][str(h)]['jump']['median'] >= .85}
    best = max(vals.values())
    result['selected_v_h'] = max(h for h, v in vals.items() if v >= best-.01)
    result['per_trial'] = per_trial
    (LOG/'amend4_dev_grid.json').write_text(json.dumps(result, indent=1))
    for h in grid:
        r = result['per_value'][str(h)]
        print(f"v_h {str(h):5s} walk V3 {r['walk']['median']:.3f} (n {r['walk']['n']}) jump V3 {r['jump']['median']:.3f}")
    print('selected', result['selected_v_h'])




def bridge_and_heldout():
    """V3 from the jointCenter landmarks: development (bridge against the pipeline) and held-out; amend4_heldout.json."""
    grid = json.loads((LOG/'amend4_dev_grid.json').read_text()); h = grid['selected_v_h']
    def jc_v3(tid):
        from scipy.interpolate import CubicSpline
        from scipy.signal import butter, sosfiltfilt
        z = dict(np.load(DATA/'amend4_landmarks'/f'{tid}.npz')); t30 = z['t30']
        z['times'] = np.arange(t30[0], t30[-1]+.5/100, 1/100.)
        for side in 'rl':
            for k in ('heel', 'toe'):
                p = sosfiltfilt(butter(4, 6., fs=30., output='sos'), z[f'raw_{side}_{k}'], axis=0)
                z[f'{side}_{k}'] = CubicSpline(t30, p, axis=0)(z['times'])
        m = {'times': z['m_times'], 'grf_r': z['grf_r'], 'grf_l': z['grf_l'], 'valid': z['valid'], 'excluded': z['excluded']}
        return v3(z, m, h)
    walk, _ = development()
    diffs = {}
    for r, _ in walk:
        a, b = jc_v3(r['id']), grid['per_trial'][r['id']][str(h)]
        if a is not None and b is not None:
            diffs[r['id']] = a-b
    out = {'v_h': h, 'bridge': {'trials': len(diffs), 'median_abs_diff': float(np.median(np.abs(list(diffs.values())))),
                                'median_diff': float(np.median(list(diffs.values()))), 'per_trial': diffs}}
    sel = LOG/'amend4_heldout_selection.json'
    if sel.exists():
        rows = json.loads(sel.read_text())['trials']
        per = {r['id']: jc_v3(r['id']) for r in rows}
        xs = [x for x in per.values() if x is not None]
        out.update(trials=len(rows), subjects=len({r['subject'] for r in rows}), median_v3=float(np.median(xs)) if xs else None,
                   per_trial=per, per_study={s: float(np.median([per[r['id']] for r in rows if r['study'] == s and per[r['id']] is not None]))
                                             for s in sorted({r['study'] for r in rows})})
    (LOG/'amend4_heldout.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ('per_trial',)} | {'bridge': {k: v for k, v in out['bridge'].items() if k != 'per_trial'}}))


if __name__ == '__main__':
    bridge_and_heldout() if sys.argv[1:] == ['--evaluate'] else main()
