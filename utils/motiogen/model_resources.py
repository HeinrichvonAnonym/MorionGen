"""Shared relocatable resources for the legacy and evalkit upper-body pipelines."""
from pathlib import Path
if __package__:
    from .config import ROOT, data_root, read_json
else:  # Historical E3 entry points put utils/motiogen directly on sys.path.
    from config import ROOT, data_root, read_json

CONFIG_PATH = ROOT / 'config/biomechanics_models.json'


def model_root(config=None):
    config = read_json(CONFIG_PATH) if config is None else config
    path = Path(config['model_root']).expanduser()
    return path.resolve() if path.is_absolute() else data_root(config) / path


def model_path(name, config=None):
    config = read_json(CONFIG_PATH) if config is None else config
    path = Path(config['models'][name]).expanduser()
    return path.resolve() if path.is_absolute() else model_root(config) / path
