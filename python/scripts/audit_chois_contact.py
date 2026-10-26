"""Reconstruct GT SMPL-X hands and full-mesh contact confusion counts, CPU only.

Uses released body decoding, but independently counts contact with SciPy/NumPy.
Does not run diffusion or claim independent SDF/penetration validation.
"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from pytorch3d import transforms
from human_body_prior.body_model.body_model import BodyModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'utils'))
from motiogen.runtime import checksum, json_write
from motiogen.chois_compat import install_chamfer_fallback

DATA = Path(json.loads((ROOT / 'config/baselines.json').read_text())['data_root'])
repo = DATA / 'baselines/chois_release'
sys.path.insert(0, str(repo))
install_chamfer_fallback()  # Import-only dependency; no BPS/Chamfer call in this audit.
import manip.data.cano_traj_dataset as dataset
dataset.SMPLH_PATH = str(DATA / 'outputs/assets/chois/processed_data/smpl_all_models/smplh_amass')
torch.set_num_threads(2)
base = ROOT / 'develop_logs/test_runs'
data = DATA / 'outputs/assets/chois/processed_data'
source = data / 'cano_test_diffusion_manip_window_120_joints24.p'
plan = json.loads((base / '20261024_native_followups/plan.json').read_text())
assert checksum(source) == plan['source_windows_sha256']
windows = [v for v in joblib.load(source).values()
           if (data / 'omomo_text_anno_json_data' / (v['seq_name'] + '.json')).exists()
           and len(v['motion']) >= 120 and v['start_t_idx'] == 0]
stats = joblib.load(data / 'cano_min_max_mean_std_data_window_120_joints24.p')
jmin = torch.as_tensor(stats['global_jpos_min'], dtype=torch.float32).reshape(24, 3)
jmax = torch.as_tensor(stats['global_jpos_max'], dtype=torch.float32).reshape(24, 3)
analysis = json.loads((base / '20261024_native_matrix_analysis/analysis.json').read_text())
cache = {}; models = {}; checks = []; identities = {}

def projected(rot):
    return transforms.quaternion_to_matrix(transforms.matrix_to_quaternion(torch.from_numpy(rot).float())).numpy()

def distances(joints, rotations, centers, tree):
    # Query every original vertex through an exact nearest-neighbour tree.
    # Rotation projection matches trainer_chois.py:2618-2620.
    local = np.einsum('tji,thj->thi', rotations, joints[:, 22:24] - centers[:, None])
    return tree.query(local.reshape(-1, 3), k=1)[0].reshape(-1, 2)

for row in analysis['rows']:
    if row['method'] != 'chois':
        continue
    idx = row['task_spec'].get('native_window_index', 0)
    if idx not in cache:
        w = windows[idx]; gender = str(np.asarray(w['gender']).item()); seq = w['seq_name']
        if gender not in models:
            path = data / 'smpl_all_models/smplx' / ('SMPLX_' + gender.upper() + '.npz')
            identities[str(path)] = checksum(path)
            models[gender] = BodyModel(bm_fname=str(path), num_betas=16).eval().requires_grad_(False)
        motion = torch.as_tensor(w['motion'], dtype=torch.float32)
        joints = motion[:, :72].reshape(-1, 24, 3)
        # Match native normalization/de-normalization before mesh reconstruction.
        normalized = ((joints - jmin) / (jmax - jmin)) * 2 - 1
        root = ((normalized + 1) * .5 * (jmax - jmin) + jmin)[:, 0]
        rot = transforms.rotation_6d_to_matrix(motion[:, 144:].reshape(-1, 22, 6))
        aa = transforms.matrix_to_axis_angle(dataset.quat_ik_torch(rot))
        with torch.no_grad():
            body = models[gender](root_orient=aa[:, 0], pose_body=aa[:, 1:].reshape(-1, 63),
                pose_hand=torch.zeros(len(aa), 90),
                betas=torch.as_tensor(w['betas'], dtype=torch.float32).expand(len(aa), -1),
                trans=root + torch.as_tensor(w['trans2joint'], dtype=torch.float32))
        gt_joints = body.Jtr[:, list(range(22)) + [28, 43]].numpy()
        mesh_path = data / 'rest_object_geo' / (seq.split('_')[1] + '.ply')
        identities[str(mesh_path)] = checksum(mesh_path)
        tree = cKDTree(np.asarray(trimesh.load_mesh(mesh_path).vertices, dtype=np.float32))
        gt_distance = distances(gt_joints, projected(np.asarray(w['obj_rot_mat'], dtype=np.float32)),
            np.asarray(w['window_obj_com_pos'], dtype=np.float32), tree)
        cache[idx] = (tree, (gt_distance < .05).any(axis=1), gt_distance, seq)
        print(json.dumps({'stage': 'gt_body_and_full_mesh', 'sequence': seq, 'vertices': tree.n}), flush=True)
    tree, truth, gt_distance, seq = cache[idx]
    with np.load(ROOT / row['run'] / 'raw/native_prediction.npz') as pred:
        d = distances(pred['joints'], projected(pred['object_rotations']), pred['object_positions'], tree)
        error = float(np.max(np.abs(d - pred['hand_object_distances_m'])))
        labels = (d < .05).any(axis=1)
        saved_labels = (pred['hand_object_distances_m'] < .05).any(axis=1)
        assert error < 2e-6 and np.array_equal(labels, saved_labels), (row['run'], error)
    tp = int((truth & labels).sum()); fp = int((~truth & labels).sum())
    tn = int((~truth & ~labels).sum()); fn = int((truth & ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    expected = {'mean_gt_contact_percent': float(truth.mean()), 'mean_contact_percent': float(labels.mean()),
        'mean_contact_precision': precision, 'mean_contact_recall': recall,
        'mean_contact_acc': (tp + tn) / len(truth),
        'mean_contact_f1_score': 2 * precision * recall / (precision + recall) if precision + recall else 0.}
    diffs = {key: abs(value - row['values']['native_object_' + key]) for key, value in expected.items()}
    checks.append({'run': row['run'], 'task': row['task'], 'seed': row['seed'], 'sequence': seq,
        'confusion': {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}, 'recomputed': expected,
        'absolute_differences': diffs, 'pred_distance_max_error_m': error,
        'gt_min_threshold_margin_m': float(np.min(np.abs(gt_distance - .05))),
        'passed': all(v < 1e-12 for v in diffs.values())})
summary = [{'metric': key, 'checked_runs': len(checks),
    'passed_runs': sum(c['absolute_differences'][key] < 1e-12 for c in checks),
    'max_absolute_difference': max(c['absolute_differences'][key] for c in checks)} for key in expected]
receipt = {'analysis_cutoff': analysis['updated_at'], 'source_sha256': checksum(source),
    'asset_sha256': identities, 'checks': checks, 'summary': summary,
    'passed': bool(checks) and all(c['passed'] for c in checks),
    'scope': 'CPU native SMPL-X GT reconstruction from registered source root/rotations/betas; independent SciPy nearest-vertex queries on full object meshes and NumPy confusion counts. Predicted geometry is checked against saved native distances. Strict 0.05 m, either hand. No diffusion, no GPU, no SDF penetration reconstruction.'}
json_write(base / '20261024_chois_contact_audit/audit.json', receipt)
print(json.dumps({'passed': receipt['passed'], 'runs': len(checks), 'summary': summary,
    'failures': [c for c in checks if not c['passed']]}))
if not receipt['passed']:
    raise SystemExit(1)
