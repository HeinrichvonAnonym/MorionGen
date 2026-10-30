"""Audit official cached references without changing frozen generation inputs."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'utils'))
from motiogen.config import data_root, load_config
from motiogen.runtime import checksum, json_write

out = ROOT / 'develop_logs/test_runs/20261028_supplement/reference_recovery'
out.mkdir(parents=True, exist_ok=True)
assets = data_root(load_config()) / 'outputs/assets'
path = assets / 'closd-official/data/humanml3d/t2m_test.npy'
cache = np.load(path, allow_pickle=True).item()
motions = cache['data_dict']
requests = json.loads((out.parent / 'generation_inputs.json').read_text())['requests']
selected = {r['source_id']: r for r in requests}
rows, recovered, overlap = [], [], []
for key, record in motions.items():
    motion = np.asarray(record['motion'])
    valid = motion.ndim == 2 and motion.shape[1] == 263 and np.isfinite(motion).all()
    row = {'key': key, 'shape': list(motion.shape), 'finite_263_features': bool(valid),
           'declared_length': int(record['length']), 'text_count': len(record['text'])}
    rows.append(row)
    for folder in [assets / 'motionlcm/humanml3d/new_joint_vecs',
                   assets / 'closd-official/data/dataset/HumanML3D/new_joint_vecs']:
        original = folder / f'{key}.npy'
        if original.is_file():
            raw = np.load(original)
            overlap.append({'key': key, 'path': str(original), 'same_shape': raw.shape == motion.shape,
                            'max_abs_difference': float(np.max(np.abs(raw - motion))) if raw.shape == motion.shape else None})
    if key not in selected:
        continue
    request = selected[key]
    exact = [t for t in record['text'] if t['caption'] == request['text']]
    destination = out / f'{key}.npy'
    if valid and exact:
        np.save(destination, motion)
    recovered.append(dict(row, exact_caption_match=bool(exact),
                          reference_path=str(destination) if valid and exact else None,
                          reference_sha256=checksum(destination) if valid and exact else None,
                          requested_frames_in_frozen_pilot=request['requested_frames'],
                          tokens_match=any(t['tokens'] == request['tokens'].split() for t in exact)))
names = list(cache['name_list'])
summary = {'source': str(path), 'source_sha256': checksum(path),
           'source_receipt': str(path.with_suffix('.npy.verified.json')),
           'records': len(motions), 'name_list_count': len(names),
           'unique_names': len(set(names)), 'name_keys_match': set(names) == set(motions),
           'finite_263_records': sum(r['finite_263_features'] for r in rows),
           'length_mismatches': [r['key'] for r in rows if r['shape'][0] != r['declared_length']],
           'selected_references': recovered, 'selected_absent_keys': sorted(set(selected) - set(motions)),
           'raw_file_overlap': overlap, 'inventory': rows,
           'scope': 'Official processed test cache; not full raw AMASS or full HumanML3D. Frozen generation lengths and outputs unchanged. Mirror/subclip entries are not independent source motions.'}
json_write(out / 'audit.json', summary)
print(json.dumps({k: v for k, v in summary.items() if k != 'inventory'}, indent=2))
