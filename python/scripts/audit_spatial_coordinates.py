"""Audit MotionLCM input packing and raw/native-to-report coordinates on CPU.

This executes the real batch builder, not the generator. CUDA transfer alone is
replaced by identity in this isolated diagnostic process; no inference is claimed.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'utils'))
from motiogen.adapters import MotionLCM, require
from motiogen.config import ROOT, load_config, repo_root
from motiogen.motion import HUMANML_NAMES
from motiogen.runtime import checksum, json_write

cfg = load_config()
upstream=repo_root(cfg,'motionlcm')
sys.path.insert(0,str(upstream))
from mld.data.humanml.dataset import Text2MotionDataset
base = ROOT / 'develop_logs/test_runs'
plan = json.loads((base / '20261024_control_matrix/plan.json').read_text())
analysis = json.loads((base / '20261024_native_matrix_analysis/analysis.json').read_text())
adapter = MotionLCM.__new__(MotionLCM)
adapter.model = SimpleNamespace(is_controlnet=True)
adapter.spec = cfg['baselines']['motionlcm']
adapter.names = HUMANML_NAMES
adapter.fps = 20
adapter.raw_mean = np.load(require(cfg, 'motionlcm', 'Mean_raw.npy')).reshape(22, 3)
adapter.raw_std = np.load(require(cfg, 'motionlcm', 'Std_raw.npy')).reshape(22, 3)
assert adapter.spec['axes'] == 'Y-up'
packing = []
for task in plan['tasks']:
    scene = task['scene']
    with patch.object(torch.Tensor, 'cuda', lambda self, *a, **k: self):
        batch = adapter.batch(scene['prompt'], scene['duration_seconds'], scene['constraints'])
    hint, mask = batch['hint'].numpy(), batch['hint_mask'].numpy()
    expected = np.zeros_like(hint)
    errors = []
    for c in scene['constraints']:
        t = round(c['time_seconds'] * 20)
        j = HUMANML_NAMES.index(c['joint'])
        x, y, z = c['position_m']
        # Independent explicit inverse of native (x,y,z) -> canonical (x,-z,y).
        native_target = np.array([x, z, -y])
        expected[0, t, j] = (native_target - adapter.raw_mean[j]) / adapter.raw_std[j]
        reconstructed = hint[0, t, j] * adapter.raw_std[j] + adapter.raw_mean[j]
        errors.append(float(np.max(np.abs(reconstructed - native_target))))
    # Execute the official sparse-hint constructor with the same selected
    # joint/frame. Only its random choice is fixed; its normalization and mask
    # construction remain the repository implementation.
    if len(scene['constraints'])!=1:raise ValueError('This registered audit expects one sparse target per task')
    dataset=Text2MotionDataset.__new__(Text2MotionDataset)
    dataset.testing_control_joints=np.array([j]);dataset.testing_density=1
    dataset.t_ctrl=False;dataset.njoints=22
    dataset.raw_mean=adapter.raw_mean;dataset.raw_std=adapter.raw_std
    joints=np.zeros(hint.shape[1:],dtype=np.float64);joints[t,j]=native_target
    with patch.object(np.random,'choice',return_value=np.array([t])):
        official_hint,official_mask=dataset.random_mask(joints)
    official_exact=bool(np.array_equal(official_hint.astype(np.float32),hint[0])
                        and np.array_equal(official_mask.astype(np.float32),mask[0]))
    packing.append({'task': task['id'], 'normalized_hint_exact': bool(np.array_equal(hint, expected)),
                    'official_sparse_constructor_exact':official_exact,
                    'unconstrained_entries_zero': bool((hint[mask == 0] == 0).all()),
                    'mask_scalar_count': int(mask.sum()),
                    'expected_mask_scalar_count': 3 * len(scene['constraints']),
                    'max_target_roundtrip_error_m': max(errors)})
runs = []
for row in analysis['rows']:
    if row['method'] != 'motionlcm':
        continue
    path = ROOT / row['run']
    native = np.load(path / 'raw/joints.npy')
    canonical = np.load(path / 'motion.npz')['joints']
    explicit = np.stack([native[..., 0], -native[..., 2], native[..., 1]], axis=-1)
    errors = []
    for c in row['task_spec']['constraints']:
        t = round(c['time_seconds'] * 20)
        j = HUMANML_NAMES.index(c['joint'])
        x, y, z = c['position_m']
        errors.append(float(np.linalg.norm(native[t, j] - [x, z, -y])))
    saved = json.loads((path / 'metrics.json').read_text())['metrics']['hand_position_error']['value']
    runs.append({'run': row['run'], 'raw_to_canonical_exact': bool(np.array_equal(explicit, canonical)),
                 'native_space_error_m': float(np.mean(errors)), 'saved_canonical_error_m': saved,
                 'metric_difference_m': abs(float(np.mean(errors)) - saved)})
passed = all(r['normalized_hint_exact'] and r['official_sparse_constructor_exact'] and r['unconstrained_entries_zero']
             and r['mask_scalar_count'] == r['expected_mask_scalar_count']
             and r['max_target_roundtrip_error_m'] < 1e-6 for r in packing)
passed = passed and bool(runs) and all(r['raw_to_canonical_exact'] and r['metric_difference_m'] < 1e-6 for r in runs)
receipt = {'passed': passed, 'analysis_cutoff': analysis['updated_at'], 'packing': packing, 'runs': runs,
           'batch_source_sha256': checksum(ROOT / 'utils/motiogen/adapters.py'),
           'official_sparse_source_sha256':checksum(upstream/'mld/data/humanml/dataset.py'),
           'scope': 'CPU execution of real MotionLCM.batch with only CUDA transfer disabled; all 12 registered tasks. Compare against official Text2MotionDataset.random_mask with the same selected joint/frame and only random selection fixed. Independent explicit axis conversion and native-space Euclidean error on saved raw model outputs. Does not measure reachable-target feasibility or model generalization.'}
json_write(base / '20261024_spatial_coordinates/audit.json', receipt)
print(json.dumps({'passed': passed, 'tasks': len(packing), 'raw_runs': len(runs),
                  'max_roundtrip_error_m': max(r['max_target_roundtrip_error_m'] for r in packing),
                  'max_metric_difference_m': max(r['metric_difference_m'] for r in runs)}))
if not passed:
    raise SystemExit(1)
