#!/usr/bin/env python
"""Quasi-static ground truth (config/e_criterion_domain_20261105.json, 'static'): held postures through the full E
pipeline; truth = signed distance of the COM ground projection to the support polygon in the same scaled model.

Writes develop_logs/test_runs/20261105_e_domain/static/<id>.json; work directories under <DISK>/e_domain_20261105/static.
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
sys.path.insert(0, str(ROOT/'utils/motiogen'))
SPEC = json.loads((ROOT/'config/e_criterion_domain_20261105.json').read_text())['static']
OUT = ROOT/'develop_logs/test_runs/20261105_e_domain/static'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_domain_20261105/static')
VAL = ROOT/'develop_logs/test_runs/20261105_val_a'
E4 = ROOT/'develop_logs/test_runs/20261103_e4_cross_method/runs'
HUMAN_CLASSES = ('stand', 'squat', 'sts', 'wholebody', 'walk', 'jump')
FPS, HOLD_S, MASS = 30., 2., 75.
HEEL = (-.03, .035); TOE = (.05, .04); SUPPORT_TOL = .01; BAND = .02


def load_joints(run):
    """OpenSim Y-up joints and marker source names, via the E4 multi-source loader."""
    import biomech_sources
    joints, names, _ = biomech_sources.load(run)
    return joints, names


def poses():
    rng = np.random.default_rng(20261105); out = []
    rows = json.loads((VAL/'selection.json').read_text())['trials']
    for c in HUMAN_CLASSES:
        rs = [r for r in rows if r['class'] == c]
        for k in rng.choice(len(rs), min(8, len(rs)), replace=False):
            r = rs[k]; joints, _ = load_joints(r['sources']['kin30'])
            for f in rng.integers(int(.25*len(joints)), int(.75*len(joints)), 2):
                out.append({'id': f"h_{c}_{r['id']}_f{int(f)}", 'source': 'human', 'class': c, 'run': r['sources']['kin30'], 'frame': int(f)})
    by = {}
    for m in sorted(E4.glob('*/metrics.json')):
        rec = json.loads(m.read_text())
        if rec.get('valid'):
            by.setdefault(rec['method'], []).append(json.loads((Path(rec['work'])/'receipt.json').read_text())['run'])
    for method, runs in sorted(by.items()):
        for k in rng.choice(len(runs), min(6, len(runs)), replace=False):
            joints, _ = load_joints(runs[k])
            for f in rng.integers(int(.25*len(joints)), int(.75*len(joints)), 2):
                out.append({'id': f"g_{method}_{Path(runs[k]).name}_f{int(f)}", 'source': 'generated', 'class': method, 'run': runs[k], 'frame': int(f)})
    for k in rng.choice(len(out), 24, replace=False):
        out.append({**out[k], 'id': out[k]['id']+'_hover10', 'hover_m': .10})
    return out


def hull(p):
    p = sorted(map(tuple, p))
    if len(p) <= 2:
        return np.array(p)
    cross = lambda o, a, b: (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lo, hi = [], []
    for q in p:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
            lo.pop()
        lo.append(q)
    for q in reversed(p):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0:
            hi.pop()
        hi.append(q)
    return np.array(lo[:-1]+hi[:-1])


def seg_dist(x, a, b):
    ab = b-a; t = np.clip(np.dot(x-a, ab)/max(np.dot(ab, ab), 1e-12), 0, 1)
    return np.linalg.norm(x-(a+t*ab))


def signed_margin(x, poly):
    if len(poly) == 0:
        return -np.inf
    if len(poly) < 3:
        return -min(seg_dist(x, poly[i], poly[(i+1) % len(poly)]) for i in range(len(poly))) if len(poly) == 2 else -np.linalg.norm(x-poly[0])
    edges = [(poly[i], poly[(i+1) % len(poly)]) for i in range(len(poly))]
    d = min(seg_dist(x, a, b) for a, b in edges)
    inside = all((b[0]-a[0])*(x[1]-a[1])-(b[1]-a[1])*(x[0]-a[0]) >= 0 for a, b in edges)
    return d if inside else -d


def truth(work, kin, com, sel, hover):
    import opensim as osim
    model = osim.Model(str(Path(work)/'model.osim')); state = model.initSystem(); bodies = model.getBodySet()
    pos = lambda n: np.array([bodies.get(n).getPositionInGround(state).get(i) for i in range(3)])
    flat = {s: pos(f'toes_{s}')[1]-pos(f'calcn_{s}')[1] for s in 'rl'}
    names = kin['names']; margins, supports, pitch, lowest = [], [], [], []
    for t in np.flatnonzero(sel):
        verts = []
        pitch.append([kin['origin'][t, names.index(f'calcn_{s}')][1]-(kin['origin'][t, names.index(f'toes_{s}')][1]-flat[s]) for s in 'rl'])
        for s in 'rl':
            heel = kin['origin'][t, names.index(f'calcn_{s}')]; toe = kin['origin'][t, names.index(f'toes_{s}')]
            fwd = toe-heel; fwd[1] = 0; fwd /= np.linalg.norm(fwd)+1e-9; lat = np.cross(fwd, [0., 1., 0.])
            for base, (f, w), h in ((heel, HEEL, heel[1]), (toe, TOE, toe[1]-flat[s])):
                for sgn in (1, -1):
                    p = base+f*fwd+sgn*w*lat; verts.append((s, p[0], p[2], h))
        low = min(v[3] for v in verts)-(hover or 0.)
        lowest.append([min(v[3] for v in verts if v[0] == s)-low for s in 'rl'])
        sup = [v for v in verts if v[3]-low <= SUPPORT_TOL]
        supports.append(sorted({v[0] for v in sup}))
        poly = hull(np.array([[v[1], v[2]] for v in sup])) if sup else np.zeros((0, 2))
        margins.append(signed_margin(np.array([com[t, 0], com[t, 2]]), poly))
    m = float(np.mean(margins)) if np.all(np.isfinite(margins)) else -np.inf
    feet = max(supports, key=len) if supports else []
    label = 'infeasible' if (not np.isfinite(m) or m <= -BAND) else 'feasible' if m >= BAND else 'indeterminate'
    return {'margin_m': m if np.isfinite(m) else None, 'support_feet': feet, 'support': 'none' if not feet else 'two' if len(feet) == 2 else 'one',
            'label': label, 'heel_minus_toe_m': dict(zip('rl', np.mean(pitch, 0).tolist())),
            'foot_clearance_m': dict(zip('rl', np.mean(lowest, 0).tolist()))}


def job(p):
    os.nice(19)
    import biomech
    import e_rules
    out = OUT/f"{p['id']}.json"
    if out.exists():
        return 'skip'
    row = dict(p)
    try:
        joints, names = load_joints(p['run'])
        held = np.repeat(joints[p['frame']][None], int(HOLD_S*FPS), 0)
        if p.get('hover_m'):
            held[..., 1] += p['hover_m']
        run = DISK/'motions'/p['id']; run.mkdir(parents=True, exist_ok=True)
        np.savez(run/'motion_yup.npz', joints=held, joint_names=np.array(names), fps=FPS)
        biomech.load = lambda _run: (held, names, FPS)
        work = DISK/'work'/p['id']
        receipt = biomech.prepare(run, work, MASS, free_yaw=True)
        biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
        times, _, _, kin = e_rules.kinematics(work)
        sel = e_rules.trim(times)
        res = {r: e_rules.evaluate(times, kin, MASS, r) for r in e_rules.RULES}
        row['ik'] = receipt['ik_marker_error']
        row['truth'] = truth(work, kin, res['R0']['com'], sel, p.get('hover_m'))
        row['rules'] = {r: e_rules.summary(x, r, sel, res['R0']['masks']) for r, x in res.items()}
        row['status'] = 'success'
    except Exception as e:
        row.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    out.write_text(json.dumps(row, indent=1))
    return row['status']


def main():
    a = argparse.ArgumentParser(); a.add_argument('--workers', type=int, default=6); a.add_argument('--only', nargs='*')
    args = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    todo = poses()
    (OUT.parent/'static_poses.json').write_text(json.dumps(todo, indent=1))
    if args.only:
        todo = [p for p in todo if p['id'] in args.only]
    counts = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as pool:
        for s in pool.map(job, todo):
            counts[s] = counts.get(s, 0)+1
            print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
