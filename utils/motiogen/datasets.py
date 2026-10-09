"""Read-only dataset receipts and deterministic benchmark subset preparation."""
import hashlib
import json
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import ROOT, data_root, read_json
from .motion import canonicalize

STRATIFIED_SEED = 20261008
KIMODO_GROUP_COUNT = 58
KIT21_NAMES = (
    'pelvis', 'spine1', 'spine2', 'neck', 'head',
    'right_shoulder', 'right_elbow', 'right_wrist',
    'left_shoulder', 'left_elbow', 'left_wrist',
    'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe',
    'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',
)


def archive_format(path):
    """Validate the container signature without consuming a multi-GB archive.

    Full stream integrity is subsequently exercised by the checked extraction
    command.  This inexpensive check makes truncated/wrong-format inputs fail
    before preparation starts.
    """
    path=Path(path)
    with path.open('rb') as handle:
        header=handle.read(8)
    if header.startswith(b'\x1f\x8b'):
        return 'gzip'
    if header.startswith(b'Rar!\x1a\x07\x00'):
        return 'rar4'
    if header.startswith(b'Rar!\x1a\x07\x01\x00'):
        return 'rar5'
    raise ValueError(f'unsupported or corrupt archive header: {path}')


def parquet_rows(path):
    """Read only Parquet footer metadata; return None when pyarrow is absent."""
    try:
        import pyarrow.parquet as parquet
    except ImportError:
        return None
    return int(parquet.ParquetFile(path).metadata.num_rows)


def local_datasets():
    return read_json(ROOT / 'config/local_assets.json').get('datasets', {})


def _required_file(root, relative, label):
    path = (Path(root) / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'{label} missing: {path}')
    return path


def source_receipts():
    specs = local_datasets()
    bones = specs['bones_seed']; bones_root = Path(bones['source_root']).resolve()
    kit = specs['kit_ml']; kit_root = Path(kit['source_root']).resolve()
    result = {
        'bones_seed': {
            'source_root': str(bones_root),
            'archive': str(_required_file(bones_root, bones['soma_uniform_archive'], 'BONES-SEED archive')),
            'metadata': str(_required_file(bones_root, bones['metadata'], 'BONES-SEED metadata')),
            'license': str(_required_file(bones_root, bones['license'], 'BONES-SEED license')),
            'mode': bones['mode'],
        },
        'kit_ml': {
            'source_root': str(kit_root),
            'archives': [str(_required_file(kit_root, name, 'KIT-ML archive')) for name in kit['archives']],
            'split': str(_required_file(kit_root, kit['split'], 'KIT-ML test split')),
            'license': str(_required_file(kit_root,kit['license'],'KIT-ML license')) if kit.get('license') else None,
            'license_status': 'present' if kit.get('license') else 'not_in_download',
            'official_terms_url': kit.get('official_terms_url'),
            'mode': kit['mode'],
        },
    }
    for dataset in result.values():
        paths = [dataset[k] for k in ('archive','metadata','license') if k in dataset]
        paths = [path for path in paths if path]
        paths += dataset.get('archives', []) + ([dataset['split']] if 'split' in dataset else [])
        dataset['files'] = [{'path': p, 'bytes': Path(p).stat().st_size,
                             'mtime_ns': Path(p).stat().st_mtime_ns} for p in paths]
    result['bones_seed']['archive_format']=archive_format(result['bones_seed']['archive'])
    result['kit_ml']['archive_formats']=[archive_format(path) for path in result['kit_ml']['archives']]
    return result


def benchmark_metadata_root(cfg):
    return data_root(cfg) / 'outputs/assets/kimodo/benchmark_metadata/testsuite'


def prepared_root(cfg):
    return data_root(cfg) / 'datasets'


def stratified_root(cfg):
    return data_root(cfg) / 'benchmarks/kimodo-stratified-20261008'


def discover_groups(testsuite):
    testsuite = Path(testsuite)
    groups = defaultdict(list)
    for meta in sorted(testsuite.rglob('meta.json')):
        case = meta.parent
        if not (case / 'seed_motion.json').is_file():
            continue
        groups[str(case.parent.relative_to(testsuite))].append(case)
    return dict(groups)


def select_stratified(testsuite, per_group=32, seed=STRATIFIED_SEED):
    groups = discover_groups(testsuite)
    if len(groups) != KIMODO_GROUP_COUNT:
        raise ValueError(f'expected {KIMODO_GROUP_COUNT} Kimodo leaf groups, found {len(groups)}')
    selected = []
    for group, cases in sorted(groups.items()):
        if len(cases) < per_group:
            raise ValueError(f'{group} has only {len(cases)} cases, needs {per_group}')
        chooser = random.Random(f'{seed}:{group}')
        for case in sorted(chooser.sample(cases, per_group)):
            selected.append({'group': group, 'case': str(case.relative_to(testsuite))})
    if len({row['case'] for row in selected}) != len(selected):
        raise ValueError('duplicate Kimodo benchmark case selection')
    return selected


def selection_sha(rows):
    payload = json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def materialize_subset(source, target, rows):
    source, target = Path(source), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    allowed = ('meta.json','seed_motion.json','seed_constraints.json')
    for row in rows:
        src = source / row['case']; dst = target / row['case']
        dst.mkdir(parents=True, exist_ok=True)
        for name in allowed:
            if (src / name).is_file():
                shutil.copy2(src / name, dst / name)


def bvh_members(subset):
    members = set()
    for seed_file in Path(subset).rglob('seed_motion.json'):
        seed = json.loads(seed_file.read_text())
        relative = seed['bvh_path'].replace('BVH/', 'bvh/')
        if relative.startswith('/') or '..' in Path(relative).parts:
            raise ValueError(f'unsafe BVH path in {seed_file}: {relative}')
        members.add('soma_uniform/' + relative)
    return sorted(members)


def extract_members(archive, destination, members, list_file):
    destination, list_file = Path(destination), Path(list_file)
    destination.mkdir(parents=True, exist_ok=True); list_file.parent.mkdir(parents=True, exist_ok=True)
    list_file.write_text('\n'.join(members) + '\n')
    subprocess.run(['bsdtar','-xzf',str(archive),'-C',str(destination),'-T',str(list_file)], check=True)


def extract_kit(archives, destination):
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        subprocess.run(['bsdtar','-xf',str(archive),'-C',str(destination)], check=True)


def validate_kit(prepared, split_file):
    prepared = Path(prepared)
    ids = [line.strip() for line in Path(split_file).read_text().splitlines() if line.strip()]
    missing = []
    for item in ids:
        for folder in ('new_joint_vecs','new_joints','texts'):
            suffix = '.txt' if folder == 'texts' else '.npy'
            if not (prepared / folder / (item + suffix)).is_file():
                missing.append(f'{folder}/{item}{suffix}')
    return {'status':'ready' if not missing else 'missing_resources','test_ids':len(ids),
            'missing_count':len(missing),'missing':missing[:100]}


def kit21_motion(joints, fps=12.5):
    """Convert official KIT-ML 21-joint arrays to the shared motion contract.

    KIT's published ``new_joints`` values use millimetres and Y-up axes.  The
    shared evaluators consume metres in a right-handed Z-up frame.  Keeping
    this conversion here makes both the reference analysis and synthetic
    contract tests use the same named-joint convention.
    """
    joints = np.asarray(joints)
    if joints.ndim != 3 or joints.shape[1:] != (len(KIT21_NAMES), 3):
        raise ValueError(f'KIT-ML motion must have shape [T,21,3], got {joints.shape}')
    if not np.isfinite(joints).all():
        raise ValueError('KIT-ML motion contains non-finite coordinates')
    if not np.isfinite(fps) or float(fps) <= 0:
        raise ValueError(f'KIT-ML fps must be positive, got {fps}')
    canonical, _ = canonicalize(joints, 'Y-up', units='mm')
    return {
        'joints': canonical,
        'joint_names': np.asarray(KIT21_NAMES),
        'fps': np.asarray(float(fps)),
        'metadata': {
            'source': 'KIT-ML official 21-joint recovery',
            'source_units': 'mm',
            'source_axes': 'right-handed Y-up',
            'units': 'm',
            'axes': 'right-handed Z-up',
        },
    }
