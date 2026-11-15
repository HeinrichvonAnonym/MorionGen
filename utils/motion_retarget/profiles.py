from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .schema import RetargetProfile
from .schema import SequenceRef
from .config import RetargetConfig
from .gpu import assigned_gpu_index
from .smplx_model import create_smplx_model


@lru_cache(maxsize=32)
def _unscaled_geometry(
    model_dir: str, gender: str, betas: tuple[float, ...],
) -> tuple[float, tuple[float, float, float]]:
    import torch
    model = create_smplx_model(model_dir, gender, len(betas)).eval()
    with torch.no_grad():
        output = model(betas=torch.tensor(betas, dtype=torch.float32).reshape(1, -1))
    vertices = output.vertices[0].detach().cpu().numpy()
    # The licensed SMPL-X assets use Y as the canonical vertical direction.
    height = float(vertices[:, 1].max() - vertices[:, 1].min())
    pivot = tuple(float(value) for value in output.joints[0, 0].detach().cpu().numpy())
    return height, pivot


def profile_scale(model_dir: Path, profile: RetargetProfile) -> tuple[float, float]:
    base_height, _ = _unscaled_geometry(str(model_dir), profile.gender, profile.betas)
    if not np.isfinite(base_height) or base_height <= 0:
        raise ValueError(f"Invalid SMPL-X base height for {profile.name}: {base_height}")
    scale = profile.height_m / base_height
    return float(scale), float(base_height)


def model_root_pivot(
    model_dir: Path, gender: str, betas: tuple[float, ...], scale: float = 1.0,
) -> np.ndarray:
    _, pivot = _unscaled_geometry(str(model_dir), gender, betas)
    return np.asarray(pivot, dtype=np.float64) * float(scale)


def profile_root_pivot(model_dir: Path, profile: RetargetProfile, scale: float) -> np.ndarray:
    return model_root_pivot(model_dir, profile.gender, profile.betas, scale)


SOURCE_NATIVE_PROFILE = "source_native"


def normalize_gender(value: object) -> str | None:
    normalized = str(value or "").strip().lower()
    if normalized in {"m", "male", "man"}:
        return "male"
    if normalized in {"f", "female", "woman"}:
        return "female"
    if normalized in {"n", "neutral"}:
        return "neutral"
    return None


def _native_profile(reference: SequenceRef) -> RetargetProfile | None:
    metadata: Mapping[str, object] = reference.metadata
    actor_id = str(metadata.get("actor_id") or metadata.get("participant") or "") or None
    smplx_metadata = metadata.get("smplx")
    if isinstance(smplx_metadata, Mapping):
        gender = normalize_gender(smplx_metadata.get("gender"))
        try:
            height_m = float(smplx_metadata.get("height_m", 0.0))
            betas = tuple(float(value) for value in smplx_metadata.get("betas", ()))
        except (TypeError, ValueError):
            gender, height_m, betas = None, 0.0, ()
        if gender is not None and height_m > 0 and len(betas) in {10, 16} and np.isfinite(betas).all():
            return RetargetProfile(
                SOURCE_NATIVE_PROFILE, gender, height_m, betas,
                profile_source="source_smplx", actor_id=actor_id,
                betas_source="source_smplx",
            )
    gender = normalize_gender(metadata.get("gender"))
    try:
        height_m = float(metadata.get("height_m", 0.0))
    except (TypeError, ValueError):
        height_m = 0.0
    if gender is None or not np.isfinite(height_m) or height_m <= 0:
        return None
    return RetargetProfile(
        SOURCE_NATIVE_PROFILE, gender, height_m, (0.0,) * 16,
        profile_source="actor_metadata", actor_id=actor_id,
        betas_source="zero_default_no_source_smplx_betas",
    )


def validate_profile_requests(config: RetargetConfig, requests: Iterable[str] | None) -> list[str]:
    selected = list(requests) if requests else list(config.profiles)
    unknown = set(selected) - set(config.profiles) - {SOURCE_NATIVE_PROFILE}
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")
    return list(dict.fromkeys(selected))


def profiles_for_reference(
    config: RetargetConfig, reference: SequenceRef, requests: Iterable[str] | None,
) -> list[RetargetProfile]:
    resolved: list[RetargetProfile] = []
    for request in validate_profile_requests(config, requests):
        if request != SOURCE_NATIVE_PROFILE:
            resolved.append(config.profiles[request])
            continue
        native = _native_profile(reference)
        if native is not None:
            resolved.append(native)
        else:
            # Locked batch policy: if even actor gender/height are unavailable,
            # retain the two configured reference profiles instead of guessing.
            resolved.extend(config.profiles[name] for name in ("male_178", "female_165"))
    by_name: dict[str, RetargetProfile] = {}
    for profile in resolved:
        by_name.setdefault(profile.name, profile)
    return list(by_name.values())


def floor_align_translation(
    config: RetargetConfig, profile: RetargetProfile, scale: float, fps: float,
    global_orient: np.ndarray, body_pose: np.ndarray, left_hand_pose: np.ndarray,
    right_hand_pose: np.ndarray, translation: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Move the first second's lowest SMPL-X vertex exactly onto Z=0."""
    import torch
    from .fitting import _device_lock
    count = min(len(translation), max(1, int(round(fps))))
    requested = str(config.raw["fitting"].get("device", "auto"))
    use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda"}
    device_index = assigned_gpu_index(torch.cuda.device_count()) if use_cuda else 0
    device = torch.device(f"cuda:{device_index}" if use_cuda else "cpu")
    with _device_lock(config, use_cuda, device_index):
        model = create_smplx_model(
            config.smplx_model_dir, profile.gender, len(profile.betas), device=device,
        ).eval()
        zeros3 = torch.zeros((count, 3), device=device)
        with torch.no_grad():
            output = model(
                global_orient=torch.tensor(global_orient[:count], device=device),
                body_pose=torch.tensor(body_pose[:count], device=device),
                left_hand_pose=torch.tensor(left_hand_pose[:count], device=device),
                right_hand_pose=torch.tensor(right_hand_pose[:count], device=device),
                betas=torch.tensor(profile.betas, device=device).reshape(1, -1).expand(count, -1),
                jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                expression=torch.zeros((count, model.num_expression_coeffs), device=device),
            )
        vertices = output.vertices.cpu().numpy() * scale + translation[:count, None]
    minimum = float(vertices[:, :, 2].min())
    aligned = np.asarray(translation, dtype=np.float32).copy()
    aligned[:, 2] -= minimum
    return aligned, -minimum, minimum

