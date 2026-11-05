#!/usr/bin/env python
"""E pipeline on AddBiomechanics trials and comparison with the force plates (config/val_a_addb_20261105.json).

Per trial: biomech.prepare + biomech.dynamics on the exported source (heading unclamped), then V1-V4 on the frames
after the 0.5 s trims whose AddB GRF is valid and not excluded. Writes develop_logs/test_runs/20261105_val_a/runs/
<variant>/<id>.json and <id>_frames.npz; work directories go to <DATA>/addb_val_20261105/work/<variant>/<id>.
Runs in the OpenSim environment; --workers processes in parallel. --estimator a1 applies the amended contact rule
(config/val_a_addb_20261105_amend1.json) and writes <variant>_a1, reusing the registered IK when it exists;
--selection heldout runs the held-out walk trials. --estimator a4 applies the horizontal-speed rule with the v_h selected
on the development trials (config/val_a_addb_20261105_amend4.json, amend4_dev_grid.json) and writes <variant>_a4.
"""
import argparse
import json
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
LOG = ROOT/'develop_logs/test_runs/20261105_val_a'
WORK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/addb_val_20261105/work')
G = 9.80665


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 1e-9 and b.std() > 1e-9 else None


def balanced_accuracy(pred, true):
    if true.all() or (~true).all():
        return None
    return float(.5*(pred[true].mean()+(~pred[~true]).mean()))


def compare(work, measured, mass, contact_n=(20., 50.)):
    dyn = np.load(work/'dynamics.npz'); m = np.load(measured)
    t = dyn['times']; dt = float(np.median(np.diff(t))); bw = mass*G
    trim = slice(int(.5/dt), len(t)-int(.5/dt))
    t = t[trim]
    near = np.clip(np.round(t/np.median(np.diff(m['times']))).astype(int), 0, len(m['times'])-1)
    ok = m['valid'][near] & ~m['excluded'][near]
    meas = {s: np.stack([np.interp(t, m['times'], m[f'grf_{s}'][:, a]) for a in range(3)], 1) for s in 'rl'}
    est = {s: dyn[f'grf_{s}'][trim] for s in 'rl'}
    residual = np.linalg.norm(dyn['residual_force'][trim], axis=1)/bw
    tot_m, tot_e = meas['r']+meas['l'], est['r']+est['l']
    out = {'frames': int(len(t)), 'grf_frames': int(ok.sum()), 'residual_frames': int((~m['excluded'][near]).sum())}
    if ok.sum() > 2:
        vm, ve = tot_m[ok, 1]/bw, tot_e[ok, 1]/bw
        hm, he = np.linalg.norm(tot_m[ok][:, [0, 2]], axis=1)/bw, np.linalg.norm(tot_e[ok][:, [0, 2]], axis=1)/bw
        out.update(V1_r=pearson(vm, ve), V1_rmse_bw=float(np.sqrt(np.mean((vm-ve)**2))), V1_bias_bw=float(np.mean(ve-vm)),
                   V2_r=pearson(hm, he), V2_rmse_bw=float(np.sqrt(np.mean((hm-he)**2))),
                   measured_vertical_mean_bw=float(vm.mean()))
        for thr in contact_n:
            acc = []
            for s in 'rl':
                pred = (dyn[f'contact_{s}_heel'][trim] | dyn[f'contact_{s}_toe'][trim])[ok]
                acc.append(balanced_accuracy(pred.astype(bool), meas[s][ok, 1] > thr))
            acc = [a for a in acc if a is not None]
            out[f'V3_balanced_accuracy_{thr:g}N'] = float(np.mean(acc)) if acc else None
    keep = ~m['excluded'][near]
    out.update(V4_residual_rms_bw=float(np.sqrt(np.mean(residual[keep]**2))) if keep.any() else None,
               V4_share_above_5pct=float(np.mean(residual[keep] > .05)) if keep.any() else None,
               V4_residual_p95_bw=float(np.percentile(residual[keep], 95)) if keep.any() else None)
    contact_any = np.any([dyn[f'contact_{s}_{k}'][trim] for s in 'rl' for k in ('heel', 'toe')], axis=0)
    frames = {'times': t, 'residual_bw': residual, 'keep': keep, 'grf_ok': ok, 'contact_any': contact_any, 'vertical_measured_bw': tot_m[:, 1]/bw,
              'vertical_estimated_bw': tot_e[:, 1]/bw}
    return out, frames


def a4_hmax():
    grid = LOG/'amend4_dev_grid.json'
    return json.loads(grid.read_text())['selected_v_h'] if grid.exists() else None


A5_ONSET = json.loads((ROOT/'config/val_a_addb_20261105_amend5.json').read_text())['rule']['v_on_m_s']
ESTIMATORS = {'registered': {}, 'a1': {'contact_vmax': .2}, 'a4': {'contact_hmax': a4_hmax()}, 'a5': {'contact_onset': A5_ONSET}}


def job(args):
    variant, row, estimator = args
    import biomech
    name = variant if estimator == 'registered' else f'{variant}_{estimator}'
    src = Path(row['sources'][variant]); work = WORK/name/row['id']; dest = LOG/'runs'/name
    dest.mkdir(parents=True, exist_ok=True)
    if (dest/f"{row['id']}.json").exists():
        return 'skip'
    result = {k: row[k] for k in ('id', 'study', 'subject', 'trial', 'task', 'class', 'seconds', 'mass_kg', 'mass_header_kg', 'mass_source')}
    result.update(variant=variant, estimator=estimator)
    try:
        base = next((b for b in (WORK/variant/row['id'], WORK/f'{variant}_a1'/row['id']) if (b/'ik/ik.mot').exists()), WORK/variant/row['id'])
        if estimator != 'registered' and (base/'receipt.json').exists() and (base/'ik/ik.mot').exists():
            (work/'ik').mkdir(parents=True, exist_ok=True)
            for f in ('model.osim', 'receipt.json', 'ik/ik.mot'):
                shutil.copy(base/f, work/f)
            for f in base.glob('ik/*_ik_marker_errors.sto'):
                shutil.copy(f, work/'ik'/f.name)
            receipt = json.loads((work/'receipt.json').read_text()); result['ik_reused'] = True
        else:
            receipt = biomech.prepare(src, work, mass=row['mass_kg'], free_yaw=True)
        metrics = biomech.dynamics(work, row['mass_kg'], **ESTIMATORS[estimator])
        cmp, frames = compare(work, src/'measured.npz', row['mass_kg'])
        result.update(status='success', ik_marker_error=receipt['ik_marker_error'], e_metrics=metrics, **cmp)
        np.savez_compressed(dest/f"{row['id']}_frames.npz", **frames)
    except Exception as e:
        result.update(status='failed', error=f'{type(e).__name__}: {e}', traceback=traceback.format_exc()[-2000:])
    (dest/f"{row['id']}.json").write_text(json.dumps(result, indent=1))
    return result['status']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', nargs='+', default=['kin30', 'dyn30', 'kin_native'])
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--only', nargs='*')
    p.add_argument('--estimator', default='registered', choices=list(ESTIMATORS))
    p.add_argument('--selection', default='registered', choices=['registered', 'heldout'])
    args = p.parse_args()
    rows = json.loads((LOG/('selection.json' if args.selection == 'registered' else 'selection_heldout.json')).read_text())['trials']
    jobs = [(v, r, args.estimator) for v in args.variants for r in rows
            if isinstance(r['sources'].get(v), str) and (not args.only or r['id'] in args.only)]
    counts = {}
    with ProcessPoolExecutor(args.workers) as pool:
        for status in pool.map(job, jobs):
            counts[status] = counts.get(status, 0)+1
            print(json.dumps(counts), flush=True)
    print(json.dumps({'jobs': len(jobs), **counts}))


if __name__ == '__main__':
    main()
