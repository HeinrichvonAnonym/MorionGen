import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text())


def load_config(path=None):
    path = Path(path).resolve() if path else ROOT / 'config/baselines.json'
    cfg = read_json(path)
    if cfg.get('schema_version') != 1:
        raise ValueError('Unsupported config schema')
    cfg['_source'] = str(path)
    return cfg


def scenario(name):
    path = Path(name)
    if not path.is_file():
        path = ROOT / 'config/scenarios' / (name + '.json')
    result = read_json(path)
    if result['duration_seconds'] <= 0:
        raise ValueError('Motion duration must be positive')
    return result


def asset_root(cfg):
    return ROOT / cfg['asset_root']
