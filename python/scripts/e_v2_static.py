#!/usr/bin/env python
"""E v2 static validation (config/e_v2_protocol_20261106.json, validation.static): 216 newly drawn held poses (seed
20261106, none from the v1 draw) through the full pipeline; truth = COM projection against the support polygon of the
foot vertices within the contact margin of the method's floor.

  --prepare  draw the poses, run scaling / IK / coordinate filtering only (no E quantity)
  --score    contact_v2 truth and the three rules with the frozen tau -> develop_logs/test_runs/20261106_e_v2/static/<id>.json
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
OUT = ROOT/'develop_logs/test_runs/20261106_e_v2/static'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_v2_20261106/static')
VAL = ROOT/'develop_logs/test_runs/20261105_val_a'
E4 = ROOT/'develop_logs/test_runs/20261103_e4_cross_method/runs'
V1_POSES = ROOT/'develop_logs/test_runs/20261105_e_domain/static_poses.json'
HUMAN_CLASSES = ('stand', 'squat', 'sts', 'wholebody', 'walk', 'jump')
FPS, HOLD_S, MASS = 30., 2., 75.
HEEL = (-.03, .035); TOE = (.05, .04)
SUPPORT_TOL, SUPPORT_TOL_SENS, BAND, DEPTH = .03, .01, .02, .03


def load_joints(run):
    import biomech_sources
    joints, names, _ = biomech_sources.load(run)
    return joints, names


def poses():
    import e_v2_floor_audit as audit
    rng = np.random.default_rng(PROTOCOL['validation']['seed']); out = []
    v1 = json.loads(V1_POSES.read_text())
    v1_runs = {p['run'] for p in v1}
    trials = json.loads((audit.OUT/'kin30_trials.json').read_text())
    rows = json.loads((VAL/'selection.json').read_text())['trials']
    for c in HUMAN_CLASSES:
        rs = [r for r in rows if r['class'] == c and r['sources']['kin30'] not in v1_runs and r['id'] in trials
              and audit.kin30_reference(r['id'], trials)[0] is not None]
        for k in sorted(rng.choice(len(rs), min(8, len(rs)), replace=False)):
            r = rs[k]; joints, _ = load_joints(r['sources']['kin30'])
            for f in rng.integers(int(.25*len(joints)), int(.75*len(joints)), 2):
                out.append({'id': f"h_{c}_{r['id']}_f{int(f)}", 'source': 'human', 'class': c, 'trial': r['id'],
                            'run': r['sources']['kin30'], 'frame': int(f)})
    by = {}
    for m in sorted(E4.glob('*/metrics.json')):
        rec = json.loads(m.read_text())
        run = json.loads((Path(rec['work'])/'receipt.json').read_text())['run'] if rec.get('valid') else None
        if run and run not in v1_runs:
            by.setdefault(rec['method'], []).append((str(m), run))
    for method, runs in sorted(by.items()):
        for k in sorted(rng.choice(len(runs), min(6, len(runs)), replace=False)):
            metrics, run = runs[k]; joints, _ = load_joints(run)
            for f in rng.integers(int(.25*len(joints)), int(.75*len(joints)), 2):
                out.append({'id': f"g_{method}_{Path(run).name}_f{int(f)}", 'source': 'generated', 'class': method, 'metrics': metrics,
                            'run': run, 'frame': int(f)})
    picks = rng.choice(len(out), 24, replace=False)
    for i, k in enumerate(picks):
        dz = .05 if i < 12 else -.05
        out.append({**out[k], 'id': out[k]['id']+('_hover05' if dz > 0 else '_sink05'), 'offset_m': dz})
    return out


def reference(p):
    """(reference_m, world_floor_m) of the pose's source."""
    import e_v2_floor_audit as audit
    if p['source'] == 'human':
        return audit.kin30_reference(p['trial'])[0], 0.
    import e_v2_rescore as rescore
    metrics = json.loads(Path(p['metrics']).read_text())
    row = {k: metrics.get(k) for k in ('method', 'arm')}
    _, ref, floor = rescore.floor_reference('e4', metrics, row)
    return ref, floor


def prepare(p):
    os.nice(19)
    import biomech
    work = DISK/'work'/p['id']
    if (work/'coordinates.mot').exists():
        return 'skip'
    try:
        joints, names = load_joints(p['run'])
        held = np.repeat(joints[p['frame']][None], int(HOLD_S*FPS), 0)
        if p.get('offset_m'):
            held[..., 1] += p['offset_m']
        biomech.load = lambda _run: (held, names, FPS)
        receipt = biomech.prepare(work, work, MASS, free_yaw=True)
        biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
        (work/'ik_receipt.json').write_text(json.dumps(receipt['ik_marker_error']))
        return 'prepared'
    except Exception as e:
        (work/'prepare_error.txt').write_text(f'{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}')
        return 'failed'


def hull_margin(verts, com, tol):
    from e_domain_static import hull, signed_margin
    sup = [v for v in verts if v[3] <= tol]
    poly = hull(np.array([[v[1], v[2]] for v in sup])) if sup else np.zeros((0, 2))
    return signed_margin(com, poly), sorted({v[0] for v in sup})


def truth(kin, com, sel, ref):
    names = kin['names']; out = {}
    for tol in (SUPPORT_TOL, SUPPORT_TOL_SENS):
        margins, supports, lowest = [], [], []
        for t in np.flatnonzero(sel):
            verts = []
            for s in 'rl':
                heel = kin['origin'][t, names.index(f'calcn_{s}')]; toe = kin['origin'][t, names.index(f'toes_{s}')]
                fwd = toe-heel; fwd[1] = 0; fwd /= np.linalg.norm(fwd)+1e-9; lat = np.cross(fwd, [0., 1., 0.])
                for base, (f, w), e in ((heel, HEEL, heel[1]-ref[s]['heel']), (toe, TOE, toe[1]-ref[s]['toe'])):
                    for sgn in (1, -1):
                        q = base+f*fwd+sgn*w*lat; verts.append((s, q[0], q[2], e))
            m, feet = hull_margin(verts, np.array([com[t, 0], com[t, 2]]), tol)
            margins.append(m); supports.append(feet); lowest.append(min(v[3] for v in verts))
        m = float(np.mean(margins)) if np.all(np.isfinite(margins)) else -np.inf
        feet = max(supports, key=len) if supports else []
        low = float(np.median(lowest))
        label = 'infeasible' if (not np.isfinite(m) or m <= -BAND) else 'feasible' if m >= BAND else 'indeterminate'
        category = 'penetrating' if low < -DEPTH else 'unsupported' if not feet else 'balance'
        out['primary' if tol == SUPPORT_TOL else 'support_tol_0.01'] = {
            'margin_m': m if np.isfinite(m) else None, 'support_feet': feet, 'label': label, 'category': category, 'lowest_elevation_m': low}
    return out


def score(p):
    os.nice(19)
    import e_rules
    out = OUT/f"{p['id']}.json"
    if out.exists():
        return 'skip'
    row = dict(p); work = DISK/'work'/p['id']
    try:
        if not (work/'coordinates.mot').exists():
            raise RuntimeError((work/'prepare_error.txt').read_text()[:500] if (work/'prepare_error.txt').exists() else 'not prepared')
        ref, floor = reference(p)
        row['floor_reference'] = {'reference_m': ref, 'world_floor_m': floor}
        times, _, _, kin = e_rules.kinematics(work)
        sel = e_rules.trim(times)
        tau = PROTOCOL['tau_envelope_bw']
        res = {r: e_rules.evaluate(times, kin, MASS, r, reference=ref, world_floor=floor) for r in e_rules.RULES}
        row['ik'] = json.loads((work/'ik_receipt.json').read_text())
        row['truth'] = truth(kin, res['R0']['com'], sel, ref)
        row['rules'] = {}
        for r, x in res.items():
            s = e_rules.summary(x, r, sel, res['R0']['masks'], tau=tau[r])
            low = x['lowest_elevation'][sel]
            s['hover_share'] = float(np.mean(low > DEPTH)); s['penetration_share'] = float(np.mean(low < -DEPTH))
            row['rules'][r] = s
        row['status'] = 'success'
    except Exception as e:
        row.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    out.write_text(json.dumps(row, indent=1))
    return row['status']


def main():
    a = argparse.ArgumentParser(); a.add_argument('--workers', type=int, default=4)
    a.add_argument('--prepare', action='store_true'); a.add_argument('--score', action='store_true')
    args = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pose_file = OUT.parent/'static_poses.json'
    if not pose_file.exists():
        pose_file.write_text(json.dumps(poses(), indent=1))
    todo = json.loads(pose_file.read_text())
    if args.score and not PROTOCOL['tau_envelope_bw']:
        raise SystemExit('tau is not frozen in the protocol yet')
    fn = prepare if args.prepare else score
    counts = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        for s in pool.map(fn, todo):
            counts[s] = counts.get(s, 0)+1
    print(json.dumps({'poses': len(todo), **counts}))


if __name__ == '__main__':
    main()
