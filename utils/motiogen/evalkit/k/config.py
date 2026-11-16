import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / 'config/evalkit_k_v2.json'
CONFIG = json.loads(PATH.read_text())
VERSION = CONFIG['version']


def resolve_config(config=None):
    """Explicit per-call configuration; never mutate module globals in workers."""
    if config is None:
        return CONFIG
    if isinstance(config, dict):
        return config
    path = Path(config)
    return json.loads((path if path.is_absolute() else ROOT/path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def implementation_key():
    paths = [PATH, *sorted(Path(__file__).parent.glob('*.py')), *sorted((ROOT/'config/evalkit_k_calibration').glob('*.json'))]
    paths += [ROOT/'utils/motion_retarget/bvh.py', ROOT/'utils/motiogen/motion.py',
              ROOT/'python/scripts/evalkit_k_humanml_codec.py', ROOT/'python/scripts/evalkit_k_validate.py']
    evidence = json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())
    upstream = Path(evidence['local_roots']['andy']).parents[2]/'baselines/OmniControl/data_loaders/humanml'
    external = {str(p): digest(p) for folder in ('common', 'scripts', 'utils')
                for p in sorted((upstream/folder).glob('*.py'))}
    return {**{str(p.relative_to(ROOT)): digest(p) for p in paths}, **external}
