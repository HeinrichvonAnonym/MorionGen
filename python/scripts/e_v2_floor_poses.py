"""Flat-foot standing poses for the E v2 floor audit (config/e_v2_floor_reference_20261106.json), one per method variant,
each built with the method's own body model or skeleton and placed on the method's own floor convention.

  --group smpl   (motiogen-dart env: torch + smplx)  DART, PRIMAL, CHOIS per window, HumanML3D (MotionLCM, OmniControl)
  --group rig    (any numpy env)                     ARDY cskel27 and Kimodo SOMA77 bind skeletons
  --group closd  (motiogen-smplsim env: mujoco)      CLoSD humanoid XML
  --group humanml (motiogen-dart env; SMPL neutral v1.1.0, chumpy-free)  HumanML3D t2m skeleton (MotionLCM, OmniControl)

Writes <DISK>/poses/<tag>.npz: joints (J, 3) canonical Z-up metres, joint_names as biomech_sources returns them, meta (json).
Floor conventions and code locations are recorded in meta['convention'].
"""
import argparse, json, pickle, sys
from pathlib import Path

import numpy as np

DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen')
ROOT = Path(__file__).resolve().parents[2]
OUT = DATA/'outputs/e_v2_20261106/floor_audit/poses'
Y_TO_Z = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])
HUMANML = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3',
           'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
           'right_elbow', 'left_wrist', 'right_wrist']
RENAME = {'spine1': 'Spine', 'spine3': 'Spine3'}
FEET = ('left_foot', 'right_foot')


def save(tag, joints, names, meta, rule, value=0., vertices=None, side_bottoms=None):
    """Write one pose per side, translated along z so that this side's foot meets the convention; both feet of a real
    flat-foot stance are on the floor, while the body models and the t2m skeleton have unequal leg heights.
    rule: 'mesh' lowest vertex of this foot, 'toe' this side's toe joint, 'geom' this foot's collision-geometry bottom."""
    OUT.mkdir(parents=True, exist_ok=True)
    names = [RENAME.get(n, n) for n in names] if 'Spine' not in names else list(names)
    joints = np.asarray(joints, float)
    j = {n: joints[i] for i, n in enumerate(names)}
    left_sign = np.sign(j['left_foot'][0]-j['pelvis'][0])
    rows = {}
    for side, toe, other in (('l', 'left_foot', 'right_foot'), ('r', 'right_foot', 'left_foot')):
        if rule == 'mesh':
            v = vertices[vertices[:, 2] < vertices[:, 2].min()+.15]
            low = v[np.sign(v[:, 0]-j['pelvis'][0]) == (left_sign if side == 'l' else -left_sign), 2].min()
        elif rule == 'toe':
            low = j[toe][2]
        else:
            low = side_bottoms[side]
        shift = float(value-low)
        out = joints.copy(); out[:, 2] += shift
        m = dict(meta, side=side, placement_rule=rule, placement_value_m=value, shift_m=shift)
        np.savez(OUT/f'{tag}@{side}.npz', joints=out, joint_names=np.array(names), meta=json.dumps(m))
        rows[side] = {'toe_z': round(float(j[toe][2]+shift), 4), 'ankle_z': round(float(j[toe.replace('foot', 'ankle')][2]+shift), 4),
                      'other_toe_z': round(float(j[other][2]+shift), 4)}
    print(json.dumps({'tag': tag, **rows}), flush=True)


def smplx_body(path, betas=None, num_betas=16):
    import smplx, torch
    model = smplx.SMPLX(model_path=str(path), num_betas=num_betas, use_pca=False, flat_hand_mean=True)
    b = torch.zeros(1, num_betas) if betas is None else torch.tensor(np.asarray(betas, np.float32).reshape(1, -1)[:, :num_betas])
    with torch.no_grad():
        o = model(betas=b, return_verts=True)
    return o.joints[0, :22].numpy() @ Y_TO_Z.T, o.vertices[0].numpy() @ Y_TO_Z.T


def group_smpl():
    # DART: male locked-head SMPL-X, betas 0 (control/train_reach_location_mld.py:201-202); reset puts the lowest joint of
    # frame 0 at z = 0 (control/env/env_reach_location_mld.py:131,144).
    J, V = smplx_body(DATA/'outputs/assets/smplx-locked-head-compatible/SMPLX_MALE.npz')
    for rule, tag in (('toe', 'dart'), ('mesh', 'dart__mesh')):
        save(tag, J, HUMANML, {'method': 'dart', 'body': 'SMPL-X locked-head male, betas 0, zero pose',
                               'convention': 'lowest predicted joint of the reset frame at z=0 (DART control/env/env_reach_location_mld.py:131,144)'
                               if rule == 'toe' else 'sensitivity: mesh sole at z=0'}, rule, vertices=V)
    # PRIMAL: SMPL-X neutral, betas 0 (python/scripts/primal_feedback_smoke.py:92-102); ground snap puts the lowest of the 22
    # joints at 0.01 m (primal/models/motion_diffuser.py:533-554, 971-986).
    J, V = smplx_body(DATA/'outputs/assets/primal-registry/models/SMPLX/neutral/SMPLX_neutral.npz')
    for rule, value, tag in (('toe', .01, 'primal'), ('mesh', 0., 'primal__mesh')):
        save(tag, J, HUMANML, {'method': 'primal', 'body': 'SMPL-X neutral, betas 0, zero pose',
                               'convention': 'ground snap: lowest of 22 joints at 0.01 m (PRIMAL motion_diffuser.py:533-554, 971-986)'
                               if rule == 'toe' else 'sensitivity: mesh sole at z=0'}, rule, value, V)
    for b0 in (-2., 2.):
        betas = np.zeros(16); betas[0] = b0
        J, V = smplx_body(DATA/'outputs/assets/primal-registry/models/SMPLX/neutral/SMPLX_neutral.npz', betas)
        save(f'smplx_neutral_beta0_{b0:+g}__mesh', J, HUMANML, {'method': 'sensitivity', 'body': f'SMPL-X neutral, betas[0]={b0:+g}',
                                                               'convention': 'shape sensitivity: mesh sole at z=0'}, 'mesh', vertices=V)
    # CHOIS: gendered SMPL-X with the window's 16 betas (trainer_chois.py:2608-2631; manip/data/cano_traj_dataset.py:242-269);
    # z is the OMOMO mocap world, the mesh sole on z=0 (verified on source windows).
    calib = json.loads((ROOT/'develop_logs/test_runs/20261028_supplement/chois_skeleton_calibration.json').read_text())['windows']
    for w, rec in sorted(calib.items(), key=lambda x: int(x[0])):
        path = DATA/f"outputs/assets/chois/processed_data/smpl_all_models/smplx/SMPLX_{rec['gender'].upper()}.npz"
        J, V = smplx_body(path, rec['shape'])
        save(f'chois_w{w}', J, HUMANML, {'method': 'chois', 'window': int(w), 'sequence': rec['sequence'], 'gender': rec['gender'],
                                         'body': 'gendered SMPL-X, window betas, zero pose',
                                         'convention': 'mesh sole at z=0 (OMOMO world; CHOIS canonicalisation rotates about z and zeroes xy only)'},
             'mesh', vertices=V)


def smpl_rest_joints(path):
    """J_regressor @ v_template of an SMPL pkl; chumpy arrays are read as their stored ndarray (no chumpy needed)."""
    class Stub:
        def __init__(self, *a, **k):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {'state': state})

    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            return Stub if module.startswith('chumpy') else super().find_class(module, name)

    def array(x):
        if isinstance(x, Stub):
            x = x.__dict__.get('x', next(v for v in x.__dict__.values() if isinstance(v, np.ndarray)))
        return x.toarray() if hasattr(x, 'toarray') else np.asarray(x, float)
    with open(path, 'rb') as fh:
        d = Reader(fh, encoding='latin1').load()
    return array(d['J_regressor'])@array(d['v_template'])


def group_humanml():
    # HumanML3D (MotionLCM, OmniControl): t2m skeleton (bone lengths of the uniform skeleton, constant in every processed file),
    # bone directions of the SMPL neutral rest pose; each sequence floored so its lowest joint is 0
    # (HumanML3D motion_process.py:169-179) -> standing toe joints at z=0.
    lengths, spread = t2m_lengths()
    rest = smpl_rest_joints(DATA/'outputs/assets/closd-official/data/smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl')[:22] @ Y_TO_Z.T
    for tag, foot in (('humanml3d', None), ('humanml3d__foot0.111', .111), ('humanml3d__foot0.146', .146)):
        L = dict(lengths)
        if foot:
            L[(7, 10)] = L[(8, 11)] = foot
        save(tag, retarget(rest, L), HUMANML, {'method': 'motionlcm/omnicontrol', 'body': 't2m bone lengths on SMPL neutral rest directions'
                               + (f'; ankle-foot {foot} m (OmniControl per-run range)' if foot else ''),
                               't2m_lengths_m': {f'{a}-{b}': v for (a, b), v in L.items()}, 't2m_length_spread_m': spread,
                               'convention': 'HumanML3D floors each sequence at its lowest joint (motion_process.py:169-179); standing: toe joints at z=0'}, 'toe')


CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]


def t2m_lengths():
    """Bone lengths from HumanML3D new_joint_vecs (ric positions are in the root-facing frame, so lengths are exact)."""
    per = []
    for f in sorted((DATA/'outputs/assets/motionlcm/humanml3d/new_joint_vecs').glob('*.npy')):
        x = np.load(f); root = np.zeros((len(x), 1, 3)); root[:, 0, 1] = x[:, 3]
        pos = np.concatenate([root, x[:, 4:4+63].reshape(-1, 21, 3)], 1)
        per.append({(a, b): np.linalg.norm(pos[:, b]-pos[:, a], axis=-1) for c in CHAIN for a, b in zip(c[:-1], c[1:])})
    lengths = {k: float(np.median(np.concatenate([p[k] for p in per]))) for k in per[0]}
    spread = float(max(np.ptp(np.concatenate([p[k] for p in per])) for k in per[0]))
    return lengths, spread


def retarget(rest, lengths):
    out = rest.copy()
    for c in CHAIN:
        for a, b in zip(c[:-1], c[1:]):
            d = rest[b]-rest[a]; out[b] = out[a]+lengths[(a, b)]*d/np.linalg.norm(d)
    return out


def group_rig():
    sys.path.insert(0, str(ROOT/'utils/motiogen'))
    aliases = {'Hips': 'pelvis', 'Pelvis': 'pelvis', 'Neck': 'neck', 'Head': 'head', 'RightArm': 'right_shoulder', 'LeftArm': 'left_shoulder',
               'RightForeArm': 'right_elbow', 'LeftForeArm': 'left_elbow', 'RightHand': 'right_wrist', 'LeftHand': 'left_wrist',
               'RightUpLeg': 'right_hip', 'LeftUpLeg': 'left_hip', 'RightLeg': 'right_knee', 'LeftLeg': 'left_knee',
               'RightFoot': 'right_ankle', 'LeftFoot': 'left_ankle', 'RightToeBase': 'right_foot', 'LeftToeBase': 'left_foot',
               'Neck1': 'neck', 'LeftUpperLeg': 'left_hip', 'RightUpperLeg': 'right_hip', 'Spine1': 'spine1', 'Spine2': 'spine2', 'Chest': 'spine3'}
    soma = dict(aliases, LeftLeg='left_hip', RightLeg='right_hip', LeftShin='left_knee', RightShin='right_knee')
    specs = [('ardy', 'cskel27', aliases, [('ardy', None, 'mesh sole at z=0: raw output follows the training floor (bind skin flat on y=0)'),
                                           ('ardy__correction', .007, 'correction=1: contact frames pin the toe base at the rig height 0.007 m '
                                            '(ardy postprocess.py:117,156-161; BindingsPython.cpp:194-203; Utility.cpp:841-852)')]),
             ('kimodo', 'somaskel77', soma, [('kimodo', None, 'mesh sole at z=0: raw output and benchmark ground truth keep the BVH floor'),
                                             ('kimodo__postprocess', .020, 'post-processing: above_ground_offset 0.02 for SOMA skeletons '
                                              '(kimodo/postprocess.py:276); post-processed toe base observed at 0.0202 m')])]
    for method, skel, alias, variants in specs:
        m = np.load(DATA/f'baselines/{method}/{method}/assets/skeletons/{skel}/skin_standard.npz')
        names = [alias.get(str(n), str(n)) for n in m['rig_joint_names']]
        J = m['bind_rig_transform'][:, :3, 3].astype(float) @ Y_TO_Z.T
        V = m['bind_vertices'].astype(float) @ Y_TO_Z.T
        for tag, toe_height, note in variants:
            meta = {'method': method, 'body': f'{skel} bind skeleton (skin_standard.npz)', 'convention': note}
            if toe_height is None:
                save(tag, J, names, meta, 'mesh', vertices=V)
            else:
                save(tag, J, names, meta, 'toe', toe_height)


def group_closd():
    import mujoco
    xml = DATA/'baselines/CLoSD/closd/data/robot_cache/smpl_humanoid_0.xml'
    model = mujoco.MjModel.from_xml_path(str(xml)); data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0; data.qpos[2] = 2.; mujoco.mj_forward(model, data)
    mapping = {'pelvis': 'Pelvis', 'left_hip': 'L_Hip', 'right_hip': 'R_Hip', 'spine1': 'Torso', 'spine2': 'Spine', 'spine3': 'Chest',
               'left_knee': 'L_Knee', 'right_knee': 'R_Knee', 'left_ankle': 'L_Ankle', 'right_ankle': 'R_Ankle', 'left_foot': 'L_Toe',
               'right_foot': 'R_Toe', 'neck': 'Neck', 'head': 'Head', 'left_collar': 'L_Thorax', 'right_collar': 'R_Thorax',
               'left_shoulder': 'L_Shoulder', 'right_shoulder': 'R_Shoulder', 'left_elbow': 'L_Elbow', 'right_elbow': 'R_Elbow',
               'left_wrist': 'L_Wrist', 'right_wrist': 'R_Wrist'}
    J = np.array([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mapping[n])] for n in HUMANML])
    bottoms = {}
    for g in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g])
        c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])*h+c
        world = data.geom_xpos[g]+corners@data.geom_xmat[g].reshape(3, 3).T
        bottoms[body] = min(bottoms.get(body, np.inf), float(world[:, 2].min()))
    sides = {s: min(bottoms[f'{s.upper()}_Ankle'], bottoms[f'{s.upper()}_Toe']) for s in 'lr'}
    save('closd', J, HUMANML, {'method': 'closd', 'body': 'smpl_humanoid_0.xml (fixed shape, box feet), qpos0',
                               'foot_geom_bottoms_m': {b: bottoms[b] for b in ('L_Ankle', 'R_Ankle', 'L_Toe', 'R_Toe')},
                               'convention': 'IsaacGym plane z=0 (humanoid.py:589-599), rest_offset 0; box feet bottoms at z=0'},
         'geom', side_bottoms=sides)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--group', required=True, choices=('smpl', 'rig', 'closd', 'humanml'))
    {'smpl': group_smpl, 'rig': group_rig, 'closd': group_closd, 'humanml': group_humanml}[a.parse_args().group]()
