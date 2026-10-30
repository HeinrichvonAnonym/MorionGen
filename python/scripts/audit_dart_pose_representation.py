"""Compare DART's native predicted joint channel with its saved SMPL-X pose channel."""
import json
from pathlib import Path
import sys

import numpy as np
import torch
import smplx

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
DATA=Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
state=json.loads((BASE/'dart_matrix/queue/state.json').read_text())
run=Path(next(iter(state['jobs'].values()))['run_dir'])
models=DATA/'outputs/runtime/dart/data/smplx_lockedhead_20230207/models_lockedhead'
torch.set_num_threads(2)
body=smplx.build_layer(models,model_type='smplx',gender='male',ext='npz',num_pca_comps=12).eval()
rows=[]
for path in [run/'primitive_000.npz',run/'primitive_037.npz',run/'primitive_074.npz']:
    saved=np.load(path)
    # Official native run forces zero male shape; fail rather than silently use
    # a mismatched shape. Native global transforms are already applied.
    assert np.all(saved['betas']==0)
    with torch.no_grad():
        result=body(betas=torch.from_numpy(saved['betas'][0]),
                    global_orient=torch.from_numpy(saved['global_orient'][0])[:,None],
                    body_pose=torch.from_numpy(saved['body_pose'][0]),
                    transl=torch.from_numpy(saved['transl'][0]))
    actual=result.joints[:,:22].numpy()
    difference=np.linalg.norm(actual-saved['joints'][0],axis=-1)
    rows.append(dict(primitive=str(path),sha256=checksum(path),frames=len(actual),
        mean_joint_channel_difference_m=float(difference.mean()),max_joint_channel_difference_m=float(difference.max()),
        same_positions_at_2e4_tolerance=bool(difference.max()<2e-4)))
    np.savez_compressed(BASE/f'dart_pose_channel_{path.stem}.npz',smplx_joints=actual,predicted_joints=saved['joints'][0])
json_write(BASE/'dart_pose_representation_audit.json',dict(rows=rows,
    body_model_sha256=checksum(models/'smplx/SMPLX_MALE.npz'),
    source='DART utils/smpl_utils.py feature_dict_to_smpl_dict explicitly preserves network-predicted joints alongside independently decoded pose rotations.',
    scope='Three primitives from one fixed pilot; channel agreement audit, not population accuracy. If channels differ, pose-derived anatomy must be labeled separately from predicted-joint task metrics; no silent substitution.'))
print(json.dumps(rows))
