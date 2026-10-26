"""CPU forward equivalence of raw locked-head geometry and schema adaptation."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import smplx
from smplx.lbs import lbs, batch_rodrigues

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(json.loads((ROOT / 'config/baselines.json').read_text())['data_root'])
torch.set_num_threads(2)
torch.manual_seed(20261024)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

checks = []
for gender in ['male', 'female', 'neutral']:
    raw = DATA / f'outputs/assets/smplx-locked-head/{gender}/model.npz'
    adapted = DATA / f'outputs/assets/smplx-locked-head-compatible/SMPLX_{gender.upper()}.npz'
    with np.load(raw, allow_pickle=True) as a, np.load(adapted, allow_pickle=True) as b:
        keys = ['v_template', 'shapedirs', 'posedirs', 'weights', 'J_regressor', 'f', 'kintree_table']
        identical = {key: bool(np.array_equal(a[key], b[key])) for key in keys}
        tensor = lambda value: torch.tensor(np.asarray(value), dtype=torch.float32)
        model = smplx.build_layer(str(adapted), model_type='smplx', gender=gender,
                                 ext='npz', num_pca_comps=12)
        batch = 5
        angles = torch.randn(batch, 22, 3) * .2
        angles[0] = 0
        pose = torch.eye(3).repeat(batch, 55, 1, 1)
        pose[:, :22] = batch_rodrigues(angles.reshape(-1, 3)).reshape(batch, 22, 3, 3)
        betas = torch.randn(batch, 10) * .5
        betas[0] = 0
        expression = torch.zeros(batch, model.num_expression_coeffs)
        # Explicit raw-array implementation; no supplemental facial/PCA arrays.
        expr_start = 300 if a['shapedirs'].shape[-1] >= 400 else 10
        dirs = np.concatenate([a['shapedirs'][:, :, :10],
                               a['shapedirs'][:, :, expr_start:expr_start+model.num_expression_coeffs]], axis=-1)
        parents = torch.tensor(a['kintree_table'][0].astype(np.int64))
        parents[0] = -1
        with torch.no_grad():
            vertices, joints = lbs(torch.cat([betas, expression], dim=-1), pose,
                tensor(a['v_template']), tensor(dirs),
                tensor(a['posedirs'].reshape(-1, a['posedirs'].shape[-1]).T),
                tensor(a['J_regressor']), parents, tensor(a['weights']), pose2rot=False)
            result = model(betas=betas, global_orient=pose[:, :1], body_pose=pose[:, 1:22],
                           expression=expression)
        errors = {'vertices_max_abs_m': float((vertices-result.vertices).abs().max()),
                  'body22_joints_max_abs_m': float((joints[:, :22]-result.joints[:, :22]).abs().max())}
        checks.append({'gender': gender, 'raw': str(raw), 'adapted': str(adapted),
            'raw_sha256': digest(raw), 'adapted_sha256': digest(adapted),
            'core_arrays_identical': identical, 'test_poses': batch, **errors,
            'passed': all(identical.values()) and max(errors.values()) <= 1e-7})

output = ROOT / 'develop_logs/test_runs/20261024_dart_body_compatibility/audit.json'
output.parent.mkdir(parents=True, exist_ok=True)
receipt = {'checks': checks, 'passed': all(c['passed'] for c in checks),
    'scope': 'User-provided locked-head core geometry retained; ordinary SMPL-X contributes missing hand-PCA and facial landmark metadata only. SMPLXLayer identity hands and first 22 body joints tested. This does not establish byte identity to the separately named official 20230207 archive, or validate facial landmark outputs.',
    'script_sha256': digest(Path(__file__)), 'torch': torch.__version__, 'device': 'cpu'}
output.write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
assert receipt['passed']
