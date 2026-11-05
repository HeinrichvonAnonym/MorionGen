#!/usr/bin/env python
"""Reference-data audit of the amendment-4/5 held-out walking trials (diagnostic; selects nothing, gates nothing).

In standing frames (body-progress proxy < 0.2 m/s, as in val_a_amend5_standing.py) a foot whose sole landmarks are both
within 3 cm of their 5th percentile and slower than 0.1 m/s is resting on the floor. The audit counts standing frames in
which such a resting foot carries < 20 N measured vertical force (a foot on unmeasured floor, or a full weight shift),
and the measured total vertical force in those frames relative to the trial's double-support standing level and to the
subject's mean total vertical force over walking frames (over whole gait cycles this mean equals body weight).

Writes develop_logs/test_runs/20261105_val_a/reference_audit.json. OpenSim python.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'python/scripts'), str(ROOT/'utils/motiogen')]
import val_a_amend5 as A  # noqa: E402


def trial(lm, m):
    t = lm['times']; dt = float(np.median(np.diff(t)))
    mid = (lm['r_heel'][:, [0, 2]]+lm['l_heel'][:, [0, 2]])/2
    prog = np.linalg.norm(np.gradient(sosfiltfilt(butter(2, 1., fs=1/dt, output='sos'), mid, axis=0), dt, axis=0), axis=1)
    trim = slice(int(.5/dt), len(t)-int(.5/dt)); tt = t[trim]
    near = np.clip(np.round(tt/np.median(np.diff(m['times']))).astype(int), 0, len(m['times'])-1)
    ok = m['valid'][near] & ~m['excluded'][near]
    standing = (prog[trim] < .2) & ok
    rest, force = {}, {}
    for s in 'rl':
        r = np.ones(len(tt), bool)
        for k in ('heel', 'toe'):
            p = lm[f'{s}_{k}']; h = p[:, 1]; v = np.linalg.norm(np.gradient(p[:, [0, 2]], dt, axis=0), axis=1)
            r &= ((h <= np.percentile(h, 5)+.03) & (v <= .1))[trim]
        rest[s] = r; force[s] = np.interp(tt, m['times'], m[f'grf_{s}'][:, 1])
    both = standing & rest['r'] & rest['l']
    unloaded = both & ((force['r'] < 20) | (force['l'] < 20))
    loaded = both & (force['r'] >= 20) & (force['l'] >= 20)
    total = force['r']+force['l']
    level = float(np.median(total[loaded])) if loaded.sum() >= 10 else None
    walking = (prog[trim] >= .2) & ok
    return {'walking_total_sum_n': float(total[walking].sum()), 'walking_frames': int(walking.sum()),
            'unloaded_total_n': total[unloaded].tolist(), 'valid_frames': int(ok.sum()), 'standing_share': float(standing.sum()/max(ok.sum(), 1)),
            'standing_both_feet_resting': int(both.sum()), 'resting_foot_unloaded': int(unloaded.sum()),
            'total_force_ratio_when_unloaded': float(np.median(total[unloaded])/level) if level and unloaded.any() else None}


def main():
    out = {'status': 'diagnostic reference-data audit; no rule is selected or changed by it', 'sets': {}}
    for fn, name in (('amend4_heldout_selection.json', 'amend4_heldout'), ('amend5_heldout_selection.json', 'amend5_heldout')):
        rows = json.loads((A.LOG/fn).read_text())['trials']
        per = {r['id']: {'subject': r['subject'], **trial(*A.jointcenters(r['id']))} for r in rows}
        bw = {}
        for subj in {p['subject'] for p in per.values()}:
            ps = [p for p in per.values() if p['subject'] == subj]
            bw[subj] = sum(p['walking_total_sum_n'] for p in ps)/max(sum(p['walking_frames'] for p in ps), 1)
        ratio_bw = np.concatenate([np.array(p.pop('unloaded_total_n'))/bw[p['subject']] for p in per.values()])
        both = sum(p['standing_both_feet_resting'] for p in per.values()); un = sum(p['resting_foot_unloaded'] for p in per.values())
        ratios = [p['total_force_ratio_when_unloaded'] for p in per.values() if p['total_force_ratio_when_unloaded'] is not None]
        out['sets'][name] = {'trials': len(per), 'standing_share_median': float(np.median([p['standing_share'] for p in per.values()])),
                             'frames_both_feet_resting': both, 'resting_foot_unloaded_share': un/max(both, 1),
                             'trials_with_unloaded_resting_foot': sum(p['resting_foot_unloaded'] > 0 for p in per.values()),
                             'total_force_ratio_when_unloaded_median': float(np.median(ratios)) if ratios else None,
                             'unloaded_total_over_walking_mean_force': np.percentile(ratio_bw, [10, 25, 50, 75, 90]).tolist() if len(ratio_bw) else None,
                             'walking_mean_force_n_by_subject': bw, 'per_trial': per}
        s = out['sets'][name]
        print(name, {k: v for k, v in s.items() if k != 'per_trial'})
    (A.LOG/'reference_audit.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
