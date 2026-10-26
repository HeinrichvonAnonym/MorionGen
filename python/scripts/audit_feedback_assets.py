"""Record local asset integrity, compatibility invariants and reproducible versions."""
from pathlib import Path
import hashlib
import json
import subprocess
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(json.loads((ROOT / 'config/baselines.json').read_text())['data_root'])


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


paths = [DATA / 'outputs/assets/dart-official' / name for name in [
    'mld_denoiser/mld_fps_clip_euler/checkpoint_300000.pt',
    'mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt',
    'mvae/mvae_fps_clip/checkpoint_200000.pt',
    'policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100/iter_5000.pth',
    'policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100/args.yaml',
    'data/stand.pkl']]
paths += list((DATA / 'outputs/assets/primal-official/motion_diffuser_ar/runs/silu_anchor/tensorboard/version_0').glob('**/*.ckpt'))
paths += list((DATA / 'outputs/assets/primal-official/motion_diffuser_ar/runs/silu_anchor/tensorboard/version_0').glob('hparams.yaml'))
assets = []
for path in paths:
    row = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            assert bad is None, (path, bad)
            row['zip_crc'] = 'all entries passed'
    assets.append(row)
body = []
for gender in ['male', 'female', 'neutral']:
    original = DATA / f'outputs/assets/smplx-locked-head/{gender}/model.npz'
    compatible = DATA / f'outputs/assets/smplx-locked-head-compatible/SMPLX_{gender.upper()}.npz'
    a, b = np.load(original, allow_pickle=True), np.load(compatible, allow_pickle=True)
    keys = ['v_template', 'shapedirs', 'posedirs', 'weights', 'J_regressor', 'f', 'kintree_table']
    for key in keys:
        assert np.array_equal(a[key], b[key]), (gender, key)
    body.append({'gender': gender, 'original_sha256': digest(original),
                 'compatible_sha256': digest(compatible), 'bit_identical_keys': keys,
                 'auxiliary_added_keys': sorted(set(b.files) - set(a.files))})
sources = {}
for method in ['DART', 'PRIMAL', 'CLoSD']:
    repo = DATA / 'baselines' / method
    sources[method] = {'commit': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
                       'status': subprocess.check_output(['git', '-C', str(repo), 'status', '--short'], text=True)}
code = {str(p.relative_to(ROOT)): digest(p) for p in [
    ROOT / 'python/scripts/dart_feedback_smoke.py', ROOT / 'python/scripts/primal_feedback_smoke.py',
    ROOT / 'utils/motiogen/feedback.py', ROOT / 'utils/motiogen/dart_feedback.py',
    ROOT / 'python/scripts/report_online_feedback.py', ROOT / 'python/scripts/report_primal_feedback.py']}
environment = subprocess.check_output([str(DATA / 'envs/motiogen-dart/bin/python'), '-m', 'pip', 'freeze'], text=True)
receipt = {'assets': assets, 'body_compatibility': body, 'sources': sources, 'code_sha256': code,
           'environment_pip_freeze': environment.splitlines(),
           'integrity_scope': 'local SHA256 and ZIP CRC; no claim of publisher-signed hashes',
           'inference_scope': 'official model loading and forward passes completed; no training or force simulation',
           'commands': ['PYTHONPATH=utils:python /home/heinrich/anaconda3/bin/python -m pytest -q python/tests',
                        'PYTHONPATH=utils /home/heinrich/anaconda3/bin/python python/scripts/report_online_feedback.py',
                        'python python/scripts/primal_feedback_smoke.py --online --frames 300 --clock realtime --goal-feedback 1 --seed 0 --handoff committed',
                        'python python/scripts/dart_feedback_smoke.py --online --frames 300 --clock realtime --goal-feedback 1 --seed 0 --handoff committed']}
out = ROOT / 'develop_logs/test_runs/20261024_feedback_reproducibility.json'
out.write_text(json.dumps(receipt, indent=2))
print({'receipt': str(out), 'assets': len(assets), 'body_models': len(body), 'sources': sources})
