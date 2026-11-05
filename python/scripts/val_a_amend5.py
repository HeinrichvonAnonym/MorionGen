#!/usr/bin/env python
"""Amendment 5 (config/val_a_addb_20261105_amend5.json): onset contact rule.

  --heldout    select and export the held-out trials (Camargo2021 AB18, AB24, AB25, AB28) -> amend5_heldout_selection.json;
               motiogen-addb environment (nimblephysics)
  --evaluate   V3 (balanced accuracy, 20 N) on the development trials over the grid (pipeline landmarks), the bridge at
               v_on, the held-out trials at v_on and the robustness values, and the amendment-4 held-out trials (secondary)
               -> amend5_eval.json; OpenSim python
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python/scripts'))
LOG = ROOT/'develop_logs/test_runs/20261105_val_a'
SPEC = json.loads((ROOT/'config/val_a_addb_20261105_amend5.json').read_text())
SUBJECTS = ('AB18', 'AB24', 'AB25', 'AB28')
GRID = [.1, .2, .3, .5, .7, 1.]
PER_SUBJECT, MAX_TRIALS = 10, 40


def heldout():
    import nimblephysics as nimble
    import val_a_amend4_landmarks as L
    rows = []
    for f in sorted(L.FILES.glob('*/No_Arm/Camargo2021_Formatted_No_Arm/*/*.b3d')):
        if f.parent.name.split('_split')[0] not in SUBJECTS:
            continue
        s = nimble.biomechanics.SubjectOnDisk(str(f))
        for t in range(s.getNumTrials()):
            name = s.getTrialName(t); seconds = s.getTrialLength(t)*s.getTrialTimestep(t)
            if L.level_walk('Camargo2021', name) and seconds >= 2.0:
                rows.append({'file': str(f.relative_to(L.FILES)), 'trial_index': t, 'study': 'Camargo2021',
                             'subject': f"Camargo2021/{f.parent.name.split('_split')[0]}", 'trial': name, 'seconds': seconds})
    pool = sorted(rows, key=lambda r: (r['file'], r['trial_index']))
    rng = np.random.default_rng(20261105); per = collections.Counter(); chosen = []
    for k in rng.permutation(len(pool)):
        r = pool[k]
        if per[r['subject']] >= PER_SUBJECT:
            continue
        r['id'] = re.sub(r'[^A-Za-z0-9]+', '_', f"Camargo2021_{Path(r['file']).parent.name}_{r['trial']}")
        r['grf_valid_share'], _ = L.export(r['file'], r['trial_index'], r['id'])
        if r['grf_valid_share'] >= .5:
            chosen.append(r); per[r['subject']] += 1
        if len(chosen) == MAX_TRIALS:
            break
    (LOG/'amend5_heldout_selection.json').write_text(json.dumps({'pool': len(pool), 'trials': chosen}, indent=1))
    print(json.dumps({'level_walk_pool': len(pool), 'chosen': len(chosen), 'per_subject': dict(per)}))


def v3(lm, m, von, threshold_n=20.):
    import biomech
    import val_a_amend4_tune as T
    t = lm['times']; dt = float(np.median(np.diff(t)))
    heights = {s: {k: lm[f'{s}_{k}'][:, 1] for k in ('heel', 'toe')} for s in 'rl'}
    horizontal = {s: {k: lm[f'{s}_{k}'][:, [0, 2]] for k in ('heel', 'toe')} for s in 'rl'}
    _, masks = biomech.contact_masks(heights, dt=dt, horizontal=horizontal, onset=von)
    m = dict(np.load(m)) if isinstance(m, (str, Path)) else m
    trim = slice(int(.5/dt), len(t)-int(.5/dt)); tt = t[trim]
    near = np.clip(np.round(tt/np.median(np.diff(m['times']))).astype(int), 0, len(m['times'])-1)
    ok = m['valid'][near] & ~m['excluded'][near]
    acc = []
    for s in 'rl':
        pred = (masks[s]['heel'] | masks[s]['toe'])[trim][ok]
        true = np.interp(tt, m['times'], m[f'grf_{s}'][:, 1])[ok] > threshold_n
        acc.append(T.balanced(pred, true))
    acc = [a for a in acc if a is not None]
    return float(np.mean(acc)) if acc else None


def jointcenters(tid):
    import val_a_amend4_tune as T
    from scipy.interpolate import CubicSpline
    from scipy.signal import butter, sosfiltfilt
    z = dict(np.load(T.DATA/'amend4_landmarks'/f'{tid}.npz')); t30 = z['t30']
    z['times'] = np.arange(t30[0], t30[-1]+.5/100, 1/100.)
    for side in 'rl':
        for k in ('heel', 'toe'):
            p = sosfiltfilt(butter(4, 6., fs=30., output='sos'), z[f'raw_{side}_{k}'], axis=0)
            z[f'{side}_{k}'] = CubicSpline(t30, p, axis=0)(z['times'])
    return z, {'times': z['m_times'], 'grf_r': z['grf_r'], 'grf_l': z['grf_l'], 'valid': z['valid'], 'excluded': z['excluded']}


def summary(per, values):
    out = {}
    for v in values:
        xs = [p[str(v)] for p in per.values() if p[str(v)] is not None]
        out[str(v)] = {'median_v3': float(np.median(xs)) if xs else None, 'n': len(xs)}
    return out


def evaluate():
    import val_a_amend4_tune as T
    von = SPEC['rule']['v_on_m_s']; robust = [von]+SPEC['rule']['robustness_values_m_s']
    walk, jump = T.development()
    dev = {}
    for cls, rows in (('walk', walk), ('jump', jump)):
        for r, work in rows:
            lm = T.landmarks(r, work); meas = Path(r['sources']['kin30'])/'measured.npz'
            dev[r['id']] = {'class': cls, **{str(v): v3(lm, meas, v) for v in GRID}}
    out = {'v_on': von, 'development': {cls: summary({k: p for k, p in dev.items() if p['class'] == cls}, GRID) for cls in ('walk', 'jump')}}
    diffs = {}
    for r, _ in walk:
        a, b = v3(*jointcenters(r['id']), von), dev[r['id']][str(von)]
        if a is not None and b is not None:
            diffs[r['id']] = a-b
    out['bridge'] = {'trials': len(diffs), 'median_abs_diff': float(np.median(np.abs(list(diffs.values())))),
                     'median_diff': float(np.median(list(diffs.values())))}
    rows = json.loads((LOG/'amend5_heldout_selection.json').read_text())['trials']
    held = {r['id']: {str(v): v3(*jointcenters(r['id']), v) for v in robust} for r in rows}
    out['heldout'] = {'trials': len(rows), 'subjects': len({r['subject'] for r in rows}), 'per_value': summary(held, robust),
                      'per_subject': {s: summary({r['id']: held[r['id']] for r in rows if r['subject'] == s}, robust)
                                      for s in sorted({r['subject'] for r in rows})}}
    sec_rows = json.loads((LOG/'amend4_heldout_selection.json').read_text())['trials']
    sec = {r['id']: {str(v): v3(*jointcenters(r['id']), v) for v in robust} for r in sec_rows}
    out['secondary_amend4_heldout'] = {'trials': len(sec_rows), 'per_value': summary(sec, robust)}
    out['per_trial'] = {'development': dev, 'heldout': held, 'secondary': sec}
    (LOG/'amend5_eval.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_trial'}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--heldout', action='store_true')
    p.add_argument('--evaluate', action='store_true')
    a = p.parse_args()
    if a.heldout:
        heldout()
    if a.evaluate:
        evaluate()
