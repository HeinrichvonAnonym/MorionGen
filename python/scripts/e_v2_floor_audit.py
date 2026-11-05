"""E v2 floor references (plan: E 准则修正版, section 2): for each method, the heights of E's four foot landmarks (calcn and
toes origins of the scaled model, left and right) when the method's own body model or skeleton stands flat-footed on the
method's own floor, taken through the same scaling, IK and coordinate filtering as every scored motion.

Writes config/e_v2_floor_reference_20261106.json and develop_logs/test_runs/20261106_e_v2/floor_audit/. OpenSim python;
pose builders that need a method's body model run in that method's environment and hand over a Z-up joints npz.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils/motiogen'), str(ROOT/'python/scripts')]
import biomech  # noqa: E402

OUT = ROOT/'develop_logs/test_runs/20261106_e_v2/floor_audit'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_v2_20261106/floor_audit')
CONFIG = ROOT/'config/e_v2_floor_reference_20261106.json'
HOLD_S, FPS, MASS = 2., 30., 75.
LANDMARK_BODY = {'heel': 'calcn', 'toe': 'toes'}


def reference_from_joints(joints_zup, names, tag, mass=MASS):
    """joints_zup: (J, 3) one standing frame in the method's canonical Z-up world (floor z per its convention)."""
    import e_rules
    joints = biomech.to_opensim(np.repeat(np.asarray(joints_zup, float)[None], int(HOLD_S*FPS), 0))
    biomech.load = lambda _run: (joints, list(names), FPS)
    work = DISK/tag
    rec = biomech.prepare(work, work, mass, free_yaw=True)
    biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
    times, _, _, kin = e_rules.kinematics(work)
    sel = e_rules.trim(times)
    ref = {s: {k: float(np.median(kin['origin'][sel, kin['names'].index(f'{b}_{s}'), 1])) for k, b in LANDMARK_BODY.items()}
           for s in 'rl'}
    src = {n: joints[0, names.index(n)] for n in names}
    ankle_minus_foot = {s: float(src[f'{side}_ankle'][1]-src[f'{side}_foot'][1]) for s, side in (('r', 'right'), ('l', 'left'))}
    return {'reference_m': ref, 'source_lowest_joint_m': float(joints[0, :, 1].min()),
            'source_ankle_minus_foot_m': ankle_minus_foot, 'ik_marker_error': rec['ik_marker_error']}


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--pose', nargs=2, action='append', metavar=('TAG', 'NPZ'),
                   help='standing pose npz (joints (J,3) Z-up, joint_names, method, note) to reference')
    args = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, path in args.pose or []:
        with np.load(path, allow_pickle=False) as m:
            joints, names = m['joints'], [str(n) for n in m['joint_names']]
            meta = json.loads(str(m['meta'])) if 'meta' in m.files else {}
        rec = {'tag': tag, 'pose_file': str(path), 'meta': meta, 'created': time.strftime('%Y-%m-%d %H:%M'),
               **reference_from_joints(joints, names, tag)}
        (OUT/f'{tag}.json').write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: rec[k] for k in ('tag', 'reference_m', 'source_ankle_minus_foot_m')}), flush=True)


if __name__ == '__main__':
    main()
