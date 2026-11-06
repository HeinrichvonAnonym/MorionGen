#!/usr/bin/env python
"""RULA/REBA legs and activity inputs on real people (AddBiomechanics kin30 trials, config/val_a_addb_20261105.json):
the same frames scored twice by evalkit.ergo, once with the height-only contact of evalkit.residual (floor 0, each
subject's leave-one-trial-out kin30 reference) and once with force-plate contact (vertical GRF > 20 N per foot);
every other input is identical, so class differences are the legs / flight errors of the height-only contact.
Source neck and shoulder inputs are fixed at their minimum (mode excluded), the narrowest ranges. Reuses the
registered kin30 IK (no new IK). OpenSim python.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils/motiogen'), str(ROOT/'python/scripts')]
VAL = ROOT/'develop_logs/test_runs/20261105_val_a'
KIN30 = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/addb_val_20261105/work/kin30')
OUT = ROOT/'develop_logs/test_runs/20261107_evalkit/addb_legs'
PLATE_N = 20.


def job(row):
    out = OUT/'trials'/f"{row['id']}.json"
    if out.exists():
        return json.loads(out.read_text())
    os.nice(19)
    rec = {k: row.get(k) for k in ('id', 'study', 'subject', 'class', 'task', 'mass_kg')}
    try:
        import opensim as osim
        osim.Logger.setLevelString('error')
        import biomech
        import e_v2_floor_audit as audit
        from evalkit import ergo, fit, residual
        work = KIN30/row['id']
        ref, tier = audit.kin30_reference(row['id'])
        if ref is None:
            ref, tier = pooled_reference(row['id'], audit)
        if ref is None or not (work/'coordinates.mot').exists():
            rec['status'] = 'no_reference'
        else:
            times, labels, coords = biomech.read_storage(work/'coordinates.mot')
            kin = biomech.body_kinematics(osim.Model(str(work/'model.osim')), labels, coords)
            res = residual.evaluate(kin, times, row['mass_kg'], 0., ref)
            m = np.load(Path(row['sources']['kin30'])/'measured.npz')
            near = np.clip(np.round(times/np.median(np.diff(m['times']))).astype(int), 0, len(m['times'])-1)
            ok = m['valid'][near] & ~m['excluded'][near]
            plate = {s: np.interp(times, m['times'], m[f'grf_{s}'][:, 1]) > PLATE_N for s in 'rl'}
            sel = fit.scored(times) & ok
            zero = ('excluded', None, None)
            tool = ergo.score(work/'model.osim', times, labels, coords, res['masks'], sel, None, zero, True, None)
            true = ergo.score(work/'model.osim', times, labels, coords, {s: {'heel': plate[s], 'toe': plate[s]} for s in 'rl'}, sel, None, zero, True, None)
            k = tool['index']
            tool_foot = {s: (res['masks'][s]['heel'] | res['masks'][s]['toe'])[k] for s in 'rl'}
            rec.update(status='ok', reference_tier=tier, frames=int(len(k)),
                       both_agree=float(np.mean((tool_foot['r'] & tool_foot['l']) == (plate['r'][k] & plate['l'][k]))),
                       flight_tool=float(tool['flight'].mean()), flight_plate=float(true['flight'].mean()),
                       foot_balanced_accuracy={s: balanced(tool_foot[s], plate[s][k]) for s in 'rl'})
            for meth in ('reba', 'rula'):
                a, b = tool[f'{meth}_class'], true[f'{meth}_class']
                rec[meth] = {'confusion': [[int(((b == i) & (a == j)).sum()) for j in range(4)] for i in range(4)]}
    except Exception as e:
        rec.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    return rec


def pooled_reference(trial, audit, min_frames=10):
    """Fallback for subjects without stand / walk / sts / squat trials (run- or jump-only): median over every other
    subject's trials of the first tier (约定，未经推演; reported as its own tier)."""
    trials = json.loads((audit.OUT/'kin30_trials.json').read_text())
    subject = trials[trial]['subject']
    for classes in audit.REFERENCE_TIERS:
        use = [t for t in trials.values() if t['subject'] != subject and t['class'] in classes]
        ref = {s: {k: (float(np.median(v)) if (v := [t['sides'][s][k] for t in use if t['sides'][s]['frames'] >= min_frames]) else None)
                   for k in audit.LANDMARK_BODY} for s in 'rl'}
        if all(ref[s][k] is not None for s in 'rl' for k in audit.LANDMARK_BODY):
            return ref, 'pooled:'+'+'.join(classes)
    return None, None


def balanced(pred, true):
    if true.all() or (~true).all():
        return None
    return float(.5*(pred[true].mean()+(~pred[~true]).mean()))


def summarize(recs):
    from evalkit import validate
    ok = [r for r in recs if r['status'] == 'ok']
    out = {'trials': len(recs), 'scored': len(ok), 'status': {s: sum(r['status'] == s for r in recs) for s in {r['status'] for r in recs}},
           'by_class': {}}
    for meth in ('reba', 'rula'):
        C = np.sum([r[meth]['confusion'] for r in ok], 0)
        fk, nk = C[3, 1]+C[3, 0], C[3].sum()
        mk, nm = C[1, 3], C[1].sum()
        out[meth] = {'confusion_plate_rows_tool_cols': C.tolist(), 'false_kill': validate.rate(fk, nk), 'missed_kill': validate.rate(mk, nm),
                     'class_disagreement': float(1-np.trace(C)/C.sum())}
    out['by_tier'] = {t: sum(r['reference_tier'] == t for r in ok) for t in {r['reference_tier'] for r in ok}}
    for cls in sorted({r['class'] for r in ok}):
        rs = [r for r in ok if r['class'] == cls and r['frames']]
        out['by_class'][cls] = {'trials': len(rs), 'pooled_reference': sum(r['reference_tier'].startswith('pooled') for r in rs),
                                'both_agree': float(np.mean([r['both_agree'] for r in rs])),
                                'flight_tool': float(np.mean([r['flight_tool'] for r in rs])), 'flight_plate': float(np.mean([r['flight_plate'] for r in rs])),
                                **{m: float(np.mean([1-np.trace(np.array(r[m]['confusion']))/max(np.sum(r[m]['confusion']), 1) for r in rs])) for m in ('reba', 'rula')}}
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('--workers', type=int, default=6)
    args = a.parse_args()
    rows = [r for r in json.loads((VAL/'selection.json').read_text())['trials'] if isinstance(r['sources'].get('kin30'), str)]
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        recs = list(p.map(job, rows))
    s = summarize(recs)
    (OUT/'summary.json').write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1)[:4000])


if __name__ == '__main__':
    main()
