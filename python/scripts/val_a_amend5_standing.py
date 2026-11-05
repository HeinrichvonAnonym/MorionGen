#!/usr/bin/env python
"""Amendment 5 diagnostic, after both held-out gates failed (does not gate or select): standing still inside walking segments. Body-progress proxy = speed of the heel midpoint low-passed at 1 Hz
(independent of the contact rule). Share of false contacts in standing frames (proxy < 0.2 m/s) and V3 on walking frames."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python/scripts'))
sys.path.insert(0, str(ROOT/'utils/motiogen'))
import val_a_amend5 as A, val_a_amend4_tune as T, biomech


def masks(lm, rule, v):
    dt = float(np.median(np.diff(lm['times'])))
    heights = {s: {k: lm[f'{s}_{k}'][:, 1] for k in ('heel', 'toe')} for s in 'rl'}
    hor = {s: {k: lm[f'{s}_{k}'][:, [0, 2]] for k in ('heel', 'toe')} for s in 'rl'}
    kw = {} if rule == 'height' else {'hmax': v} if rule == 'hmax' else {'onset': v}
    return biomech.contact_masks(heights, dt=dt, horizontal=hor, **kw)[1]

RULES = [('height', None), ('hmax', .3), ('hmax', .5), ('hmax', 1.2), ('onset', .3), ('onset', .5), ('onset', .7)]


def trial(lm, m):
    t = lm['times']; dt = float(np.median(np.diff(t)))
    mid = (lm['r_heel'][:, [0, 2]]+lm['l_heel'][:, [0, 2]])/2
    prog = np.linalg.norm(np.gradient(sosfiltfilt(butter(2, 1., fs=1/dt, output='sos'), mid, axis=0), dt, axis=0), axis=1)
    trim = slice(int(.5/dt), len(t)-int(.5/dt)); tt = t[trim]
    near = np.clip(np.round(tt/np.median(np.diff(m['times']))).astype(int), 0, len(m['times'])-1)
    ok = m['valid'][near] & ~m['excluded'][near]
    walking = prog[trim] >= .2
    res = {'standing_share': float(np.mean(~walking[ok])) if ok.any() else None}
    for rule, v in RULES:
        mk = masks(lm, rule, v); fp = fp_stand = 0; acc = {'all': [], 'walking': []}
        for s in 'rl':
            pred = (mk[s]['heel'] | mk[s]['toe'])[trim]
            true = np.interp(tt, m['times'], m[f'grf_{s}'][:, 1]) > 20.
            fp += int((pred & ~true & ok).sum()); fp_stand += int((pred & ~true & ok & ~walking).sum())
            for tag, sel in (('all', ok), ('walking', ok & walking)):
                acc[tag].append(T.balanced(pred[sel], true[sel]))
        for tag in acc:
            a = [x for x in acc[tag] if x is not None]; res[f'{rule} {v} {tag}'] = float(np.mean(a)) if a else None
        res[f'{rule} {v} fp'] = fp; res[f'{rule} {v} fp_stand'] = fp_stand
    return res


def report(name, per):
    st = [p['standing_share'] for p in per if p['standing_share'] is not None]
    print(f"{name} ({len(per)} trials): standing share of valid frames median {np.median(st):.3f} mean {np.mean(st):.3f}")
    for rule, v in RULES:
        a = [p[f'{rule} {v} all'] for p in per if p[f'{rule} {v} all'] is not None]
        w = [p[f'{rule} {v} walking'] for p in per if p[f'{rule} {v} walking'] is not None]
        fs = sum(p[f'{rule} {v} fp_stand'] for p in per)/max(sum(p[f'{rule} {v} fp'] for p in per), 1)
        print(f"   {rule:6s} {str(v):4s} V3 all {np.median(a):.3f} | walking frames {np.median(w):.3f} (n {len(w)}) | false contacts while standing {fs:.2f}")


if __name__ == '__main__':
    walk, _ = T.development()
    report('development', [trial(*A.jointcenters(r['id'])) for r, _ in walk])
    for fn, name in (('amend4_heldout_selection.json', 'amendment-4 held-out'), ('amend5_heldout_selection.json', 'amendment-5 held-out')):
        rows = json.loads((A.LOG/fn).read_text())['trials']
        report(name, [trial(*A.jointcenters(r['id'])) for r in rows])
