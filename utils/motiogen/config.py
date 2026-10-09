import json
import os
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
    cfg['_data_root'] = str(data_root(cfg))
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
    path = Path(cfg['asset_root'])
    return path if path.is_absolute() else data_root(cfg) / path


def data_root(cfg=None):
    """Large, relocatable state root; source/config/logs remain in the repo."""
    configured = os.environ.get('MORIONGEN_DATA_ROOT')
    if not configured and cfg:
        configured = cfg.get('data_root')
    return Path(configured).expanduser().resolve() if configured else ROOT


def repo_root(cfg, baseline):
    path = Path(cfg['baselines'][baseline]['repo'])
    return path if path.is_absolute() else data_root(cfg) / path


def environment_root(cfg, baseline):
    return data_root(cfg) / 'envs' / cfg['baselines'][baseline]['environment']
