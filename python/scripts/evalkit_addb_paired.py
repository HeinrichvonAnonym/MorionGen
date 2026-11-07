#!/usr/bin/env python
"""Paired real-people check of the evalkit ground reaction forces (config/evalkit_addb_paired_20261107.json):
AddBiomechanics kin30 joint centres through evalkit.fit and evalkit.residual, per-foot GRF against the force plates.
Steps: fit (per trial: fit, loaded-flat landmark heights for the subject references), eval (leave-one-trial-out
reference, residual, GRF metrics), summary. OpenSim python.
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
PROTOCOL = json.loads((ROOT/'config/evalkit_addb_paired_20261107.json').read_text())
VAL = ROOT/'develop_logs/test_runs/20261105_val_a'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/evalkit_20261107/addb_paired')
OUT = ROOT/'develop_logs/test_runs/20261107_evalkit/addb_paired'
LOADED_BW, FLAT_SPEED_M_S, PLATE_N, FILTER_HZ = .2, .1, 20., 6.
TIERS = (('stand',), ('walk', 'sts', 'squat'))


def rows():
    return [r for r in json.loads((VAL/'selection.json').read_text())['trials'] if isinstance(r['sources'].get('kin30'), str)]


def kinematics(row):
    import opensim as osim
    osim.Logger.setLevelString('error')
    import biomech
    from evalkit import fit
    f = fit.fit(row['sources']['kin30'], DATA/row['id'], row['mass_kg'], 0.)
    kin = biomech.body_kinematics(osim.Model(str(DATA/row['id']/'model.osim')), f['labels'], f['coords'])
    return f, kin


def measured(row, times, filtered=True):
    from scipy.signal import butter, sosfiltfilt
    m = np.load(Path(row['sources']['kin30'])/'measured.npz')
    near = np.clip(np.round((times-m['times'][0])/np.median(np.diff(m['times']))).astype(int), 0, len(m['times'])-1)
    ok = m['valid'][near] & ~m['excluded'][near]
    fs = 1/np.median(np.diff(m['times']))
    grf = {}
    for s in 'rl':
        g = m[f'grf_{s}']
        if filtered:
            g = sosfiltfilt(butter(4, FILTER_HZ, fs=fs, output='sos'), g, axis=0)
        grf[s] = np.stack([np.interp(times, m['times'], g[:, k]) for k in range(3)], 1)
    return grf, ok


def fit_job(row):
    out = OUT/'trials'/f"{row['id']}_fit.json"
    if out.exists():
        return json.loads(out.read_text())
    os.nice(19)
    rec = {'id': row['id'], 'subject': row['subject'], 'class': row['class']}
    try:
        import biomech
        f, kin = kinematics(row)
        times = f['times']; dt = float(np.median(np.diff(times))); bw = row['mass_kg']*9.80665
        grf, ok = measured(row, times, filtered=False)
        rec['sides'] = {}
        for s in 'rl':
            _, h = biomech.sole_points(kin, s)
            v = {k: np.abs(np.gradient(x, dt)) for k, x in h.items()}
            sel = ok & (grf[s][:, 1]/bw > LOADED_BW) & (v['heel'] < FLAT_SPEED_M_S) & (v['toe'] < FLAT_SPEED_M_S)
            rec['sides'][s] = {'frames': int(sel.sum()), **{k: (float(np.median(x[sel])) if sel.any() else None) for k, x in h.items()}}
        rec['status'] = 'ok'
    except Exception as e:
        rec.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec, indent=1))
    return rec


def reference(trial, trials, min_frames=10):
    """Leave-one-trial-out subject reference over the tiers; else the pooled median of the other subjects."""
    subject = trials[trial]['subject']
    pick = lambda use: {s: {k: (float(np.median(v)) if (v := [t['sides'][s][k] for t in use if t['sides'][s]['frames'] >= min_frames]) else None)
                            for k in ('heel', 'toe')} for s in 'rl'}
    done = lambda ref: all(ref[s][k] is not None for s in 'rl' for k in ('heel', 'toe'))
    ok = {i: t for i, t in trials.items() if t.get('status') == 'ok'}
    for classes in TIERS:
        ref = pick([t for i, t in ok.items() if i != trial and t['subject'] == subject and t['class'] in classes])
        if done(ref):
            return ref, '+'.join(classes)
    for classes in TIERS:
        ref = pick([t for t in ok.values() if t['subject'] != subject and t['class'] in classes])
        if done(ref):
            return ref, 'pooled:'+'+'.join(classes)
    return None, None


def balanced(pred, true):
    if true.all() or (~true).all():
        return None
    return float(.5*(pred[true].mean()+(~pred[~true]).mean()))


def grf_metrics(est, meas, sel, bw):
    out = {}
    for s in 'rl':
        e = (est[s]-meas[s])[sel]/bw
        out[f'G1_vertical_mae_{s}'] = float(np.abs(e[:, 1]).mean()); out[f'G1_vertical_rmse_{s}'] = float(np.sqrt((e[:, 1]**2).mean()))
        out[f'G2_horizontal_mae_{s}'] = float(np.linalg.norm(e[:, [0, 2]], axis=1).mean())
    te, tm = (est['r'][:, 1]+est['l'][:, 1])[sel]/bw, (meas['r'][:, 1]+meas['l'][:, 1])[sel]/bw
    out.update(G3_total_vertical_mae=float(np.abs(te-tm).mean()), G3_total_vertical_rmse=float(np.sqrt(((te-tm)**2).mean())),
               G3_total_vertical_r=float(np.corrcoef(te, tm)[0, 1]) if te.std() > 0 and tm.std() > 0 else None)
    out['G1_vertical_mae'] = .5*(out['G1_vertical_mae_r']+out['G1_vertical_mae_l'])
    out['G1_vertical_rmse'] = .5*(out['G1_vertical_rmse_r']+out['G1_vertical_rmse_l'])
    out['G2_horizontal_mae'] = .5*(out['G2_horizontal_mae_r']+out['G2_horizontal_mae_l'])
    return out


def eval_job(task):
    row, trials = task
    out = OUT/'trials'/f"{row['id']}_eval.json"
    if out.exists():
        return json.loads(out.read_text())
    os.nice(19)
    rec = {k: row.get(k) for k in ('id', 'study', 'subject', 'class', 'task', 'mass_kg')}
    try:
        from evalkit import fit, residual
        ref, tier = reference(row['id'], trials)
        if ref is None:
            rec['status'] = 'no_reference'
        else:
            f, kin = kinematics(row)
            times = f['times']; bw = row['mass_kg']*9.80665
            res = residual.evaluate(kin, times, row['mass_kg'], 0., ref)
            est = {s: res['grf'][s]['force'] for s in 'rl'}
            meas_f, ok = measured(row, times, True); meas_r, _ = measured(row, times, False)
            sel = fit.scored(times) & ok
            plate = {s: meas_r[s][:, 1] > PLATE_N for s in 'rl'}
            foot = {s: res['masks'][s]['heel'] | res['masks'][s]['toe'] for s in 'rl'}
            rec.update(status='ok', reference_tier=tier, frames=int(sel.sum()),
                       filtered=grf_metrics(est, meas_f, sel, bw), unfiltered=grf_metrics(est, meas_r, sel, bw),
                       G4_balanced_accuracy={s: balanced(foot[s][sel], plate[s][sel]) for s in 'rl'},
                       G5_residual_rms_bw=float(np.sqrt((res['residual_bw'][sel]**2).mean())),
                       measured_vertical_mean_bw=float(((meas_r['r'][:, 1]+meas_r['l'][:, 1])[sel]/bw).mean()))
    except Exception as e:
        rec.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(rec, indent=1))
    return rec


def stats(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return {'median': float(np.median(v)), 'q25': float(np.percentile(v, 25)), 'q75': float(np.percentile(v, 75)), 'n': len(v)} if v else None


def summary(recs):
    ok = [r for r in recs if r['status'] == 'ok']
    keys = ('G1_vertical_mae', 'G1_vertical_rmse', 'G2_horizontal_mae', 'G3_total_vertical_mae', 'G3_total_vertical_rmse', 'G3_total_vertical_r')
    out = {'protocol': PROTOCOL['protocol'], 'trials': len(recs), 'status': {s: sum(r['status'] == s for r in recs) for s in {r['status'] for r in recs}},
           'tiers': {t: sum(r['reference_tier'] == t for r in ok) for t in {r['reference_tier'] for r in ok}}, 'by_class': {}}
    for cls in sorted({r['class'] for r in ok}):
        rs = [r for r in ok if r['class'] == cls]
        c = {'trials': len(rs)}
        for variant in ('filtered', 'unfiltered'):
            c[variant] = {k: stats([r[variant][k] for r in rs]) for k in keys}
        c['G4_balanced_accuracy'] = stats([x for r in rs for x in r['G4_balanced_accuracy'].values()])
        c['G5_residual_rms_bw'] = stats([r['G5_residual_rms_bw'] for r in rs])
        c['measured_vertical_mean_bw'] = stats([r['measured_vertical_mean_bw'] for r in rs])
        out['by_class'][cls] = c
    out['mean_over_classes_filtered'] = {k: float(np.mean([c['filtered'][k]['median'] for c in out['by_class'].values() if c['filtered'][k]]))
                                         for k in keys}
    out['comparison'] = PROTOCOL['comparison']
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('step', choices=('fit', 'eval', 'summary'))
    a.add_argument('--workers', type=int, default=8); a.add_argument('--limit', type=int)
    args = a.parse_args()
    rs = rows()[:args.limit]
    ctx = mp.get_context('spawn')
    if args.step == 'fit':
        with ProcessPoolExecutor(args.workers, mp_context=ctx, max_tasks_per_child=1) as p:
            recs = list(p.map(fit_job, rs))
        return print(json.dumps({s: sum(r['status'] == s for r in recs) for s in {r['status'] for r in recs}}))
    trials = {r['id']: json.loads((OUT/'trials'/f"{r['id']}_fit.json").read_text()) for r in rows()
              if (OUT/'trials'/f"{r['id']}_fit.json").exists()}
    if args.step == 'eval':
        with ProcessPoolExecutor(args.workers, mp_context=ctx, max_tasks_per_child=1) as p:
            recs = list(p.map(eval_job, [(r, trials) for r in rs]))
    else:
        recs = [json.loads((OUT/'trials'/f"{r['id']}_eval.json").read_text()) for r in rs if (OUT/'trials'/f"{r['id']}_eval.json").exists()]
    s = summary(recs)
    (OUT/'summary.json').write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if k != 'comparison'}, indent=1)[:6000])


if __name__ == '__main__':
    main()
