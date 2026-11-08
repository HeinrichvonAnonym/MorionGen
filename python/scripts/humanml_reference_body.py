#!/usr/bin/env python
"""HumanML3D reference body for the task reachability premise check (no method output is read).

HumanML3D preprocessing retargets every motion to one skeleton (bone lengths of sample 000021); its rest offsets are
axis-aligned and do not give a standing pose, so the standing geometry is measured on quiet upright frames of the
official test set. Run with the motiogen-motionlcm python (torch, upstream motion_process); the OpenSim side
(task_reachability.py --overhead-limit) scales the Rajagopal model to these numbers.

Output: develop_logs/test_runs/20261109_task_reachability/humanml_reference_body.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.config import data_root, load_config  # noqa: E402

DATA = data_root(load_config())
sys.path.insert(0, str(DATA/'baselines/OmniControl'))
import data_loaders.humanml.scripts.motion_process as process  # noqa: E402

T2M_GT = DATA/'outputs/assets/closd-official/data/humanml3d/t2m_test.npy'
OUT = ROOT/'develop_logs/test_runs/20261109_task_reachability'
J = {'pelvis': 0, 'hip_l': 1, 'hip_r': 2, 'knee_l': 4, 'knee_r': 5, 'ankle_l': 7, 'ankle_r': 8, 'foot_l': 10, 'foot_r': 11,
     'neck': 12, 'head': 15, 'shoulder_l': 16, 'shoulder_r': 17, 'elbow_l': 18, 'elbow_r': 19, 'wrist_l': 20, 'wrist_r': 21}
BONES = {f'{name}_{s}': (J[f'{a}_{s}'], J[f'{b}_{s}']) for s in 'lr'
         for name, a, b in (('thigh', 'hip', 'knee'), ('shank', 'knee', 'ankle'), ('upper_arm', 'shoulder', 'elbow'),
                            ('forearm', 'elbow', 'wrist'))}
# quiet upright standing frames (conventions, not derived)
SELECT = {'foot_joint_max_height_m': .03, 'ankle_max_height_m': .12, 'knee_min_angle_deg': 172.,
          'trunk_max_tilt_deg': 4., 'pelvis_max_speed_m_s': .1}
FPS = 20.


def angle(a, b, c):
    u, v = a-b, c-b
    return np.degrees(np.arccos(np.clip((u*v).sum(-1)/np.linalg.norm(u, axis=-1)/np.linalg.norm(v, axis=-1), -1, 1)))


def main():
    data = np.load(T2M_GT, allow_pickle=True).item()
    names = list(data['name_list'])
    bones, rows, clips = [], [], set()
    for n in names:
        j = process.recover_from_ric(torch.from_numpy(data['data_dict'][n]['motion'].astype(np.float32)), 22).numpy()
        bones.append([np.linalg.norm(j[0, a]-j[0, b]) for a, b in BONES.values()])
        y = j[..., 1]
        trunk = j[:, J['neck']]-j[:, J['pelvis']]
        speed = np.r_[0., np.linalg.norm(np.diff(j[:, 0], axis=0), axis=-1)*FPS]
        keep = ((y[:, J['foot_l']] < SELECT['foot_joint_max_height_m']) & (y[:, J['foot_r']] < SELECT['foot_joint_max_height_m'])
                & (y[:, J['ankle_l']] < SELECT['ankle_max_height_m']) & (y[:, J['ankle_r']] < SELECT['ankle_max_height_m'])
                & (angle(j[:, J['hip_l']], j[:, J['knee_l']], j[:, J['ankle_l']]) > SELECT['knee_min_angle_deg'])
                & (angle(j[:, J['hip_r']], j[:, J['knee_r']], j[:, J['ankle_r']]) > SELECT['knee_min_angle_deg'])
                & (np.degrees(np.arccos(trunk[:, 1]/np.linalg.norm(trunk, axis=-1))) < SELECT['trunk_max_tilt_deg'])
                & (speed < SELECT['pelvis_max_speed_m_s']))
        for p in j[keep]:
            left = p[J['hip_l']]-p[J['hip_r']]; left[1] = 0; left /= np.linalg.norm(left)
            fwd = np.cross(left, [0., 1., 0.])
            o = p[J['pelvis']]*[1, 0, 1]
            rows.append({k: [(p[i]-o)@fwd, -(p[i]-o)@left, p[i][1]] for k, i in J.items()})
            clips.add(n)
    bones = np.array(bones)
    standing = {}
    for k in J:
        a = np.array([r[k] for r in rows])
        standing[k] = {'median_body_m': [round(float(v), 4) for v in np.median(a, 0)],
                       'p10_body_m': [round(float(v), 4) for v in np.percentile(a, 10, 0)],
                       'p90_body_m': [round(float(v), 4) for v in np.percentile(a, 90, 0)]}
    out = {'created': time.strftime('%Y-%m-%d %H:%M'), 'source': str(T2M_GT.relative_to(DATA)), 'clips': len(names),
           'skeleton': 'HumanML3D 22-joint target skeleton (bone lengths of 000021, applied to every motion by the preprocessing)',
           'bone_lengths_m': {k: round(float(v), 4) for k, v in zip(BONES, bones.mean(0))},
           'bone_length_max_spread_across_clips_m': float(np.round((bones.max(0)-bones.min(0)).max(), 6)),
           'standing': {'frame': 'body frame [forward, right, up] in metres; origin: pelvis joint projected on the dataset floor (y = 0, lowest joint of each clip)',
                        'selection_conventions_not_derived': SELECT, 'frames': len(rows), 'clips': len(clips), 'joints': standing}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'humanml_reference_body.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ('bone_lengths_m', 'bone_length_max_spread_across_clips_m')}))
    print({k: v['median_body_m'] for k, v in standing.items()}, len(rows), len(clips))


if __name__ == '__main__':
    main()
