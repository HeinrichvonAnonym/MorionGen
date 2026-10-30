#!/usr/bin/env python
"""Compare a fresh MotionCorrection build with the pre-built runtime extension.

run:     execute correct_motion from --package-dir on fixed-seed inputs, save outputs.
compare: element-wise comparison of two run outputs.
The runtime extension is never replaced; each build is imported in its own process.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

CASES = [(seed, mask) for seed in range(8) for mask in ('none', 'root', 'hands', 'feet', 'fullbody')]


class Joint:
    def __init__(self, name, parent, t_pose_translation, t_pose_rotation, retarget_tag=''):
        self.name, self.parent = name, parent
        self.t_pose_translation, self.t_pose_rotation, self.retarget_tag = t_pose_translation, t_pose_rotation, retarget_tag


def rig():
    """The upstream run_test.py test rig."""
    identity = [0., 0., 0., 1.]
    return [Joint('Hips', None, [0., 1., 0.], identity, 'Root'), Joint('Spine', 'Hips', [0., .1, 0.], identity),
            Joint('LeftUpLeg', 'Hips', [-.1, -.05, 0.], identity), Joint('LeftLeg', 'LeftUpLeg', [0., -.4, 0.], identity),
            Joint('LeftFoot', 'LeftLeg', [0., -.4, 0.], identity, 'LeftFoot'),
            Joint('RightUpLeg', 'Hips', [.1, -.05, 0.], identity), Joint('RightLeg', 'RightUpLeg', [0., -.4, 0.], identity),
            Joint('RightFoot', 'RightLeg', [0., -.4, 0.], identity, 'RightFoot'),
            Joint('LeftArm', 'Spine', [-.3, .3, 0.], identity), Joint('LeftHand', 'LeftArm', [-.3, 0., 0.], identity, 'LeftHand'),
            Joint('RightArm', 'Spine', [.3, .3, 0.], identity), Joint('RightHand', 'RightArm', [.3, 0., 0.], identity, 'RightHand')]


def run(package_dir, out):
    sys.path.insert(0, str(package_dir))
    import torch
    import motion_correction
    from motion_correction.motion_postprocess import correct_motion
    frames, joints = 60, 12
    arrays = {}
    for seed, mask in CASES:
        g = torch.Generator().manual_seed(seed)
        hips = torch.randn(1, frames, 3, generator=g)
        rotations = torch.randn(1, frames, joints, 4, generator=g)
        rotations = rotations/rotations.norm(dim=-1, keepdim=True)
        contacts = torch.rand(1, frames, 4, generator=g)
        masks = {k: torch.zeros(frames) for k in ('Root', 'FullBody', 'LeftHand', 'RightHand', 'LeftFoot', 'RightFoot')}
        keys = {'none': [], 'root': ['Root'], 'hands': ['LeftHand', 'RightHand'], 'feet': ['LeftFoot', 'RightFoot'], 'fullbody': ['FullBody']}[mask]
        for key in keys:
            masks[key][::10] = 1.
        hips_in, rotations_in = hips.clone(), rotations.clone()
        correct_motion(hipTranslations=hips, jointRotations=rotations, contacts=contacts, hipTranslationsInput=hips_in,
                       rotationsInput=rotations_in, constraint_masks=masks, contact_threshold=.5, root_margin=.01, working_rig=rig())
        arrays[f'{seed}_{mask}_hips'] = hips.numpy(); arrays[f'{seed}_{mask}_rotations'] = rotations.numpy()
        arrays[f'{seed}_{mask}_changed'] = np.array(float((hips-hips_in).abs().max()+(rotations-rotations_in).abs().max()))
    np.savez(out, **arrays)
    print(json.dumps({'package': motion_correction.__file__, 'cases': len(CASES), 'out': str(out)}))


def compare(a, b, out):
    a, b = np.load(a), np.load(b)
    rows = {k: {'max_abs_diff': float(np.abs(a[k]-b[k]).max()), 'bit_identical': bool(np.array_equal(a[k], b[k]))} for k in a.files}
    changed = [float(a[k]) for k in a.files if k.endswith('_changed')]
    result = {'arrays': len(rows), 'bit_identical': all(r['bit_identical'] for r in rows.values()),
              'max_abs_diff': max(r['max_abs_diff'] for r in rows.values()),
              'cases_where_correction_changed_input': int(sum(c > 0 for c in changed)), 'rows': rows}
    Path(out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('run'); r.add_argument('--package-dir', required=True, type=Path); r.add_argument('--out', required=True, type=Path)
    c = sub.add_parser('compare'); c.add_argument('a'); c.add_argument('b'); c.add_argument('--out', required=True)
    args = parser.parse_args()
    run(args.package_dir, args.out) if args.mode == 'run' else compare(args.a, args.b, args.out)


if __name__ == '__main__':
    main()
