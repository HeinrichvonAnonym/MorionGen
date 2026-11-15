from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .schema import RetargetProfile


@dataclass(frozen=True)
class RetargetConfig:
    path: Path
    raw: Mapping[str, Any]

    @property
    def dataset_root(self) -> Path:
        return Path(self.raw["paths"]["dataset_root"]).expanduser().resolve()

    @property
    def data_root(self) -> Path:
        override = os.environ.get("MORIONGEN_DATA_ROOT")
        return Path(override or self.raw["paths"]["data_root"]).expanduser().resolve()

    @property
    def output_root(self) -> Path:
        override = os.environ.get("MOTION_RETARGET_OUTPUT_ROOT")
        if override:
            return Path(override).expanduser().resolve()
        configured = Path(self.raw["paths"]["output_root"])
        default_data = Path(self.raw["paths"]["data_root"])
        if self.data_root != default_data and configured.is_relative_to(default_data):
            configured = self.data_root / configured.relative_to(default_data)
        return configured.expanduser().resolve()

    @property
    def smplx_model_dir(self) -> Path:
        return Path(self.raw["paths"]["smplx_model_dir"]).expanduser().resolve()

    @property
    def profiles(self) -> dict[str, RetargetProfile]:
        return {
            name: RetargetProfile(
                name=name,
                gender=str(item["gender"]),
                height_m=float(item["height_m"]),
                betas=tuple(float(v) for v in item["betas"]),
            )
            for name, item in self.raw["profiles"].items()
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def dataset_path(self, name: str) -> Path:
        return (self.dataset_root / self.raw["datasets"][name]["root"]).resolve()


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path, seen: frozenset[Path] = frozenset()) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"Recursive motion-retarget config inheritance: {path}")
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"Motion-retarget config must be a mapping: {path}")
    parent = raw.get("extends")
    if parent is None:
        return dict(raw)
    parent_path = (path.parent / str(parent)).expanduser().resolve()
    return _deep_merge(_load_raw_config(parent_path, seen | {path}), raw)


def load_config(path: str | Path) -> RetargetConfig:
    config_path = Path(path).expanduser().resolve()
    raw = _load_raw_config(config_path)
    if raw.get("schema_version") != 1:
        raise ValueError(f"Unsupported motion-retarget config schema: {raw.get('schema_version')}")
    return RetargetConfig(config_path, raw)

