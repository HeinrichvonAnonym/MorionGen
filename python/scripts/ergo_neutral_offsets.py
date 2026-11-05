#!/usr/bin/env python
"""Skeleton-neutral neck and shoulder-elevation offsets for the source-based inputs of RULA/REBA (config/ergo_rula_reba_20261105.json, revision 3).

Neck flexion and side bend are measured from source joints as the neck-to-head line against the pelvis-to-neck line
(ergo_frames.neck_angles). Every skeleton's rest pose (head assumed neutral) gives a nonzero reading; that reading is
subtracted per skeleton. Shoulder elevation (ergo_frames.shoulder_elevation) uses the same rest pose as its zero. Writes config/ergo_neutral_offsets_20261105.json. Methods whose joints are predicted
directly (no rigid skeleton) are listed as excluded: their neck is not scored. Runs in the OmniControl environment.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
from ergo_frames import rest_reading
from motion import SOMA77_NAMES

DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen')
SMPL22 = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3',
          'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
          'right_elbow', 'left_wrist', 'right_wrist']
ARDY_RUN = ROOT/'develop_logs/test_runs/20261025_132001_ardy_demo_597d87ef/motion.npz'
METHODS = {'closd': 'smpl', 'dart': 'smplx_male', 'primal': 'smplx_neutral', 'chois': 'per_run_fk', 'ardy': 'cskel27',
           'kimodo': 'somaskel77'}
EXCLUDED = {'motionlcm': 'joint positions are predicted directly (HumanML3D representation): no rigid skeleton, so no verifiable zero; '
                         '18.5% of E4 frames put the head point less than 3 cm above the neck point',
            'omnicontrol': 'joint positions are predicted directly (HumanML3D representation): neck-to-head length 0.12-0.14 m against '
                           '0.083 m of SMPL, so the SMPL zero does not apply; with it nearly every frame reads 14-18 deg extension'}
SHOULDER_CHECK = ('clavicle-to-glenohumeral length matches the zero skeleton: CLoSD 0.102 / SMPL 0.102, PRIMAL 0.116 / SMPL-X neutral 0.116, '
                  'DART 0.143 / SMPL-X male 0.144, ARDY 0.159 / cskel27 0.159, Kimodo 0.160 / SOMA77 0.160 m; MotionLCM and OmniControl '
                  '0.12 m (+20% against SMPL) with within-clip variation up to 5.4%. The rest pose has arms out, so arms-down frames read '
                  'below zero (SMPL family 11-24 deg, ARDY 3 deg, Kimodo 8.5 deg): the zero and the 10 deg threshold are conventions')
NOTES = {'dart': 'zero of SMPL-X male, betas 0 (dart_feedback FK); the E4 path exports DART environment joints with a 2.4% within-clip '
                 'neck-to-head length variation and a constant 22 deg side-bend median: E4 DART neck readings are flagged',
         'chois': 'per run: rest pose rebuilt from raw/native_prediction.npz (bone offsets in the parent frame are constant to 3e-7 m), '
                  'fallback SMPL-X neutral',
         'kimodo': 'skeleton joints.p; the skin bind pose reads 20.5 deg (1.6 deg higher)'}


def regress(J, v):
    J = J.todense() if hasattr(J, 'todense') else J
    return np.asarray(J, float) @ np.asarray(v, float)


def rest_poses():
    smpl = pickle.load(open(DATA/'outputs/assets/omnicontrol/smpl/SMPL_NEUTRAL.pkl', 'rb'), encoding='latin1')
    smplx = {g: np.load(DATA/f'outputs/assets/chois/processed_data/smpl_all_models/smplx/SMPLX_{g.upper()}.npz', allow_pickle=True)
             for g in ('neutral', 'male', 'female')}
    with np.load(ARDY_RUN) as m:
        ardy_names = [str(n) for n in m['joint_names']]
    sk = DATA/'baselines'
    return {'smpl': (regress(smpl['J_regressor'], smpl['v_template'])[:22], SMPL22),
            **{f'smplx_{g}': (regress(m['J_regressor'], m['v_template'])[:22], SMPL22) for g, m in smplx.items()},
            'cskel27': (torch.load(sk/'ardy/ardy/assets/skeletons/cskel27/joints.p').numpy(), ardy_names),
            'somaskel77': (torch.load(sk/'kimodo/kimodo/assets/skeletons/somaskel77/joints.p').numpy(), list(SOMA77_NAMES))}


def main():
    out = {'definition': 'rest-pose readings (degrees): neck flexion and side bend of neck-to-head against pelvis-to-neck; '
                         'shoulder elevation [left, right] of clavicle-to-glenohumeral above the plane normal to pelvis-to-neck',
           'assumption': 'the rest pose head is neutral (rendered: SMPL, SMPL-X, cskel27 and SOMA77 rest poses are upright and face forward)',
           'skeletons': {}, 'methods': METHODS, 'excluded': EXCLUDED, 'notes': NOTES,
           'shoulder_check': SHOULDER_CHECK}
    for name, (joints, names) in rest_poses().items():
        (flex, side), shoulder = rest_reading(joints, names)
        out['skeletons'][name] = {'neck_flexion_deg': float(flex), 'neck_side_bend_deg': float(side),
                                  'shoulder_elevation_deg': [float(x) for x in shoulder]}
    (ROOT/'config/ergo_neutral_offsets_20261105.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
