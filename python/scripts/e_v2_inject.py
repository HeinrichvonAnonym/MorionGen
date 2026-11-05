#!/usr/bin/env python
"""E v2 defect injection (config/e_v2_protocol_20261106.json, validation.injection): 12 Moco test references and 12 new
human kin30 clips, each unmodified and with the v1 defects (plus whole-clip hover / penetration sizes), through the full
pipeline; per-frame v2 residual, envelope verdict, depth and skate under R0 / R4 / R5.

  --prepare  build the modified motions, scaling / IK / coordinate filtering only (no E quantity); Moco bases need their
             solved reference (python/scripts/e_v2_moco_refs.py)
  --score    v2 evaluation with the frozen tau -> develop_logs/test_runs/20261106_e_v2/inject/<id>.json and .npz
OpenSim python.
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
PROTOCOL = json.loads((ROOT/'config/e_v2_protocol_20261106.json').read_text())
REFS = json.loads((ROOT/'config/e_v2_moco_refs_20261106.json').read_text())
OUT = ROOT/'develop_logs/test_runs/20261106_e_v2/inject'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_v2_20261106/inject')
MOCO = ROOT/'develop_logs/test_runs/20261106_e_v2/moco_refs'
VAL = ROOT/'develop_logs/test_runs/20261105_val_a'
V1_JOBS = ROOT/'develop_logs/test_runs/20261105_e_domain/inject_jobs.json'
DEFECTS = {'float_window_m': [.02, .05, .1], 'float_whole_m': [.02, .05, .1], 'penetrate_window_m': [.02, .05, .1],
           'penetrate_whole_m': [.02, .05, .1], 'lean_deg': [5, 10, 20], 'slide_m_s': [.1, .3, .6], 'time_compression': [1.25, 1.5, 2.]}
FPS_MOCO, DEPTH = 30., .03


def human_bases():
    import e_v2_floor_audit as audit
    rng = np.random.default_rng(PROTOCOL['validation']['seed'])
    v1 = {j['base'] for j in json.loads(V1_JOBS.read_text())}
    moco = {r['trial'] for r in REFS['refs']}
    trials = json.loads((audit.OUT/'kin30_trials.json').read_text())
    rows = json.loads((VAL/'selection.json').read_text())['trials']
    out = []
    for c in ('walk', 'sts'):
        rs = [r for r in rows if r['class'] == c and r['id'] not in v1 and r['id'] not in moco and r['id'] in trials
              and audit.kin30_reference(r['id'], trials)[0] is not None]
        out += [{'base': rs[k]['id'], 'kind': 'human', 'class': c, 'run': rs[k]['sources']['kin30'], 'mass_kg': rs[k]['mass_kg']}
                for k in sorted(rng.choice(len(rs), 6, replace=False))]
    return out


def moco_bases():
    return [{'base': r['trial'], 'kind': 'moco', 'class': r['class'], 'mass_kg': r['mass_kg']} for r in REFS['refs'] if r['split'] == 'test']


def jobs():
    out = []
    for b in human_bases()+moco_bases():
        out.append({**b, 'id': f"{b['base']}__control", 'defect': 'control', 'size': 0})
        for defect, sizes in DEFECTS.items():
            out += [{**b, 'id': f"{b['base']}__{defect}_{s:g}", 'defect': defect, 'size': s} for s in sizes]
    return out


def moco_dir(trial):
    status = MOCO/'status'/f'{trial}.json'
    if not status.exists():
        return None
    rec = json.loads(status.read_text())
    return ROOT/rec['dir'] if rec['status'] == 'ok' else None


def base_motion(p):
    """OpenSim Y-up joints, source names, fps, reference_m, world_floor_m of the base."""
    if p['kind'] == 'human':
        import biomech_sources
        import e_v2_floor_audit as audit
        from e_domain_inject import CROP_S
        joints, names, fps = biomech_sources.load(p['run'])
        keep = int(min(len(joints), CROP_S*fps)); c0 = (len(joints)-keep)//2
        return joints[c0:c0+keep], names, fps, audit.kin30_reference(p['base'])[0], 0.
    from scipy.interpolate import CubicSpline
    import e_domain_moco
    out = moco_dir(p['base'])
    if out is None:
        raise FileNotFoundError(f"no solved Moco reference for {p['base']}")
    times, P, names, _, _ = e_domain_moco.moco_outputs(out)
    t = np.arange(times[0], times[-1]+1e-9, 1/FPS_MOCO)
    floor = json.loads((out/'solve.json').read_text())['floor_m']
    return CubicSpline(times, P, axis=0)(t), names, FPS_MOCO, {s: {k: floor for k in ('heel', 'toe')} for s in 'rl'}, floor-e_domain_moco.SPHERE_R


def modify(joints, names, fps, defect, size):
    from e_domain_inject import inject
    if defect == 'penetrate_whole_m':
        j = joints.copy(); j[..., 1] -= size
        return j, fps, None
    return inject(joints, names, fps, defect, size)


def prepare(p):
    os.nice(19)
    import biomech
    work = DISK/'work'/p['id']
    if (work/'coordinates.mot').exists() or (work/'skipped.json').exists():
        return 'skip'
    work.mkdir(parents=True, exist_ok=True)
    try:
        joints, names, fps, _, _ = base_motion(p)
        mod, fps2, window = modify(joints, names, fps, p['defect'], p['size'])
        if mod is None:
            (work/'skipped.json').write_text(json.dumps({'reason': 'no right-foot stance of >= 0.3 s'}))
            return 'skipped'
        biomech.load = lambda _run: (mod, names, fps2)
        receipt = biomech.prepare(work, work, p['mass_kg'], free_yaw=True)
        biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
        (work/'inject_receipt.json').write_text(json.dumps({'fps': fps2, 'frames': len(mod), 'window_s': window,
                                                            'ik': receipt['ik_marker_error']}))
        return 'prepared'
    except FileNotFoundError as e:
        return 'waiting'
    except Exception as e:
        (work/'prepare_error.txt').write_text(f'{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}')
        return 'failed'


def score(p):
    os.nice(19)
    import e_rules
    out = OUT/f"{p['id']}.json"
    if out.exists():
        return 'skip'
    row = dict(p); work = DISK/'work'/p['id']
    try:
        if (work/'skipped.json').exists():
            row.update(status='skipped', **json.loads((work/'skipped.json').read_text()))
            out.write_text(json.dumps(row, indent=1)); return 'skipped'
        if not (work/'coordinates.mot').exists():
            raise RuntimeError((work/'prepare_error.txt').read_text()[:500] if (work/'prepare_error.txt').exists() else 'not prepared')
        _, _, _, ref, floor = base_motion(p)
        row['floor_reference'] = {'reference_m': ref, 'world_floor_m': floor}
        row.update(json.loads((work/'inject_receipt.json').read_text()))
        times, _, _, kin = e_rules.kinematics(work)
        tau = PROTOCOL['tau_envelope_bw']; arrays = {'times': times}
        for rule in e_rules.RULES:
            res = e_rules.evaluate(times, kin, p['mass_kg'], rule, reference=ref, world_floor=floor)
            bad, st = e_rules.infeasible(res, rule, tau[rule])
            contact = np.stack([res['masks'][s][k] for s in ('r', 'l') for k in ('heel', 'toe')], 1)
            fast = np.stack([res['speed'][s][k] > e_rules.SKATE_M_S for s in ('r', 'l') for k in ('heel', 'toe')], 1)
            low = res['lowest_elevation']
            arrays.update({f'{rule}_residual_bw': res['residual_bw'], f'{rule}_infeasible': bad, f'{rule}_contact': contact,
                           f'{rule}_skate': fast & contact, f'{rule}_lowest_elevation': low, f'{rule}_hover': low > DEPTH,
                           f'{rule}_penetration': low < -DEPTH, **{f'{rule}_state_{k}': v for k, v in st.items()}})
        np.savez_compressed(OUT/f"{p['id']}.npz", **arrays)
        row['status'] = 'success'
    except Exception as e:
        row.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    out.write_text(json.dumps(row, indent=1))
    return row['status']


def main():
    a = argparse.ArgumentParser(); a.add_argument('--workers', type=int, default=4)
    a.add_argument('--prepare', action='store_true'); a.add_argument('--score', action='store_true')
    a.add_argument('--kind', choices=('human', 'moco'))
    args = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    job_file = OUT.parent/'inject_jobs.json'
    if not job_file.exists():
        job_file.write_text(json.dumps(jobs(), indent=1))
    todo = [j for j in json.loads(job_file.read_text()) if not args.kind or j['kind'] == args.kind]
    if args.score and not PROTOCOL['tau_envelope_bw']:
        raise SystemExit('tau is not frozen in the protocol yet')
    fn = prepare if args.prepare else score
    counts = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        for s in pool.map(fn, todo):
            counts[s] = counts.get(s, 0)+1
    print(json.dumps({'jobs': len(todo), **counts}))


if __name__ == '__main__':
    main()
