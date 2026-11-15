from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from .adapters import (
    bvh_pose, inventory, load_andy_heights, load_comfi_pose, load_hiphi_metadata,
)
from .canonical import canonicalize_bvh_root
from .config import RetargetConfig
from .hands import HandTrack, coverage_quality, empty_hand_track, load_wilor_cache, observed_hand_track
from .fitting import fit_smplx_observations
from .gvhmr import ensure_gvhmr_cache, load_gvhmr_result
from .ledger import Ledger
from .jsonutil import json_safe
from .observations import (
    COMFI_EXCLUDED_JCP, COMFI_TO_SMPLX, load_andy_glove, load_andy_markers,
    load_comfi_joint_centers, load_ssm2_vertices, select_comfi_joints,
)
from .profiles import (
    floor_align_translation, model_root_pivot, profile_root_pivot, profile_scale,
    profiles_for_reference,
)
from .quality import native_hand_fk_quality, retarget_native_hands
from .rotations import slerp_rotvec
from .schema import MotionRecord, RetargetProfile, SequenceRef
from .storage import write_json_atomic, write_motion_npz
from .trajectory import c3_archive_payload, c3_smooth_rotvec, c3_smooth_translation
from .video import ensure_wilor_cache, find_comfi_video_archive


def _safe_id(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def output_path(config: RetargetConfig, reference: SequenceRef, profile: RetargetProfile) -> Path:
    return config.output_root / "motions" / reference.dataset / profile.name / f"{_safe_id(reference.sequence_id)}.smplx.npz"


def run_config_hash(
    config: RetargetConfig, reference: SequenceRef, input_mode: str | None,
) -> str:
    default_mode = str(config.raw["datasets"][reference.dataset]["input_mode"])
    if input_mode is None or input_mode == default_mode:
        return config.digest
    return hashlib.sha256(
        f"{config.digest}\0input_mode={input_mode}".encode()
    ).hexdigest()


def model_dependency_hash(
    config: RetargetConfig, ledger: Ledger, reference: SequenceRef, profile: RetargetProfile,
    input_mode: str | None = None,
) -> str:
    """Hash every model asset whose availability/content can change this output."""
    project_root = Path(__file__).resolve().parents[2]
    dependencies = [
        config.smplx_model_dir / f"SMPLX_{profile.gender.upper()}.npz",
        *sorted((project_root / "utils/motion_retarget").glob("*.py")),
    ]
    mode = input_mode or str(config.raw["datasets"][reference.dataset]["input_mode"])
    if reference.dataset == "andy" or (reference.dataset == "comfi" and mode != "mocap_raw"):
        wilor = Path(config.raw["paths"]["wilor_repo"])
        dependencies.extend([
            wilor / "pretrained_models/wilor_final.ckpt",
            wilor / "pretrained_models/detector.pt",
            wilor / "mano_data/MANO_RIGHT.pkl",
            project_root / "python/scripts/extract_wilor_mano.py",
        ])
    if (reference.dataset == "andy" or mode == "gvhmr_video" or
            (reference.dataset == "comfi" and mode == "auto")):
        gvhmr = Path(config.raw["paths"]["gvhmr_repo"])
        dependencies.extend([
            gvhmr / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
            gvhmr / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
            gvhmr / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
            gvhmr / "inputs/checkpoints/yolo/yolov8x.pt",
            project_root / "python/scripts/extract_gvhmr_body.py",
        ])
    digest = hashlib.sha256()
    for dependency in dependencies:
        resolved = dependency.resolve(strict=False)
        digest.update(str(resolved).encode())
        digest.update(b"\0")
        digest.update((ledger.sha256(resolved) if resolved.is_file() else "MISSING").encode())
        digest.update(b"\0")
    return digest.hexdigest()


def input_dependency_paths(
    config: RetargetConfig, reference: SequenceRef, input_mode: str | None = None,
) -> list[Path]:
    mode = input_mode or str(config.raw["datasets"][reference.dataset]["input_mode"])
    if reference.dataset == "andy":
        root = config.dataset_path("andy") / "data/raw"
        return [
            root / "xsens_bvh.zip", root / "qualisys_csv.zip", root / "glove_csv.zip",
            root / "videos_camera1_mp4.zip", root / "videos_camera2_mp4.zip",
        ]
    if reference.dataset == "comfi":
        root = config.dataset_path("comfi") / "data/raw"
        paths = [root / "mocap.zip", root / "metadata.zip"]
        if mode != "mocap_raw":
            paths.extend([root / "cam_params.zip", find_comfi_video_archive(config, reference)])
        return paths
    if reference.dataset == "hiphi":
        return [Path(str(reference.metadata.get("origin_archive", reference.source)))]
    raise ValueError(reference.dataset)


def input_dependency_hash(
    config: RetargetConfig, ledger: Ledger, reference: SequenceRef,
    input_mode: str | None = None,
) -> str:
    digest = hashlib.sha256()
    for dependency in sorted(input_dependency_paths(config, reference, input_mode)):
        resolved = dependency.resolve()
        digest.update(str(resolved).encode())
        digest.update(b"\0")
        digest.update(ledger.sha256(resolved).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def cache_dependency_hash(
    config: RetargetConfig, ledger: Ledger, reference: SequenceRef,
    input_hash: str, kind: str,
) -> str:
    root = Path(__file__).resolve().parents[2]
    if kind == "wilor":
        repo = Path(config.raw["paths"]["wilor_repo"])
        dependencies = [
            repo / "pretrained_models/wilor_final.ckpt",
            repo / "pretrained_models/detector.pt",
            repo / "mano_data/MANO_RIGHT.pkl",
            root / "python/scripts/extract_wilor_mano.py",
            root / "utils/motion_retarget/video.py",
        ]
    elif kind == "gvhmr":
        repo = Path(config.raw["paths"]["gvhmr_repo"])
        dependencies = [
            repo / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
            repo / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
            repo / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
            repo / "inputs/checkpoints/yolo/yolov8x.pt",
            config.smplx_model_dir / "SMPLX_NEUTRAL.npz",
            root / "python/scripts/extract_gvhmr_body.py",
            root / "utils/motion_retarget/gvhmr.py",
        ]
    else:
        raise ValueError(f"Unknown cache dependency kind: {kind}")
    digest = hashlib.sha256()
    digest.update(input_hash.encode())
    digest.update(b"\0")
    digest.update(str(config.raw["upstreams"].get(kind, {})).encode())
    digest.update(b"\0")
    if kind == "wilor":
        digest.update(json.dumps(config.raw["hands"], sort_keys=True).encode())
        digest.update(b"\0")
    for dependency in dependencies:
        resolved = dependency.resolve(strict=False)
        digest.update(str(resolved).encode())
        digest.update(b"\0")
        digest.update((ledger.sha256(resolved) if resolved.is_file() else "MISSING").encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _disk_guard(config: RetargetConfig) -> None:
    config.output_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(config.output_root).free
    reserve = int(float(config.raw["quality"]["disk_reserve_gib"]) * 1024 ** 3)
    if free < reserve:
        raise RuntimeError(f"Free space {free / 1024**3:.1f} GiB is below the configured {reserve / 1024**3:.1f} GiB reserve")


def _canonicalize_z_up(
    root: np.ndarray, translation: np.ndarray, height_ratio: float,
    root_pivot: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    translation = np.asarray(translation, dtype=np.float64) * height_ratio
    pivot = np.zeros(3, dtype=np.float64) if root_pivot is None else np.asarray(root_pivot, dtype=np.float64)
    count = max(1, min(len(root), 20))
    matrices = Rotation.from_rotvec(root[:count]).as_matrix()
    forward = matrices[:, :, 2].mean(axis=0)
    yaw = np.arctan2(forward[1], forward[0])
    heading = Rotation.from_euler("z", -yaw).as_matrix()
    converted_root = Rotation.from_matrix(heading @ Rotation.from_rotvec(root).as_matrix()).as_rotvec().astype(np.float32)
    # SMPL-X global_orient rotates around its non-zero pelvis joint. Rotate the
    # world-space pelvis (pivot + transl), then convert back to SMPL-X transl.
    # Treating transl itself as the root point introduces a profile-dependent
    # rigid offset in source_to_canonical.
    converted_translation = (translation + pivot) @ heading.T - pivot
    converted_root_position = converted_translation + pivot
    offset = converted_root_position[0, :2].copy()
    converted_translation[:, :2] -= offset
    transform = np.eye(4)
    transform[:3, :3] = heading
    transform[:2, 3] = -offset
    return converted_root, converted_translation.astype(np.float32), transform


def _comfi_height(config: RetargetConfig, reference: SequenceRef) -> tuple[float, str]:
    archive = config.dataset_path("comfi") / "data/raw/metadata.zip"
    member = f"metadata/{reference.metadata['subject']}.yaml"
    try:
        import yaml
        with zipfile.ZipFile(archive) as bundle:
            data = yaml.safe_load(bundle.read(member))
        for key in ("height", "height_m", "body_height"):
            if key in data:
                value = float(data[key])
                return (value / 100.0 if value > 3 else value), f"metadata:{key}"
    except (KeyError, OSError, ValueError, TypeError):
        pass
    try:
        _, names, points, valid = load_comfi_joint_centers(reference, "mocap_aligned")
        index = {name: position for position, name in enumerate(names)}

        def bone(a: str, b: str) -> float:
            keep = valid[:, index[a]] & valid[:, index[b]]
            values = np.linalg.norm(points[keep, index[a]] - points[keep, index[b]], axis=1)
            return float(np.median(values))

        left_leg = bone("Left_Hip", "Left_Knee") + bone("Left_Knee", "Left_Ankle")
        right_leg = bone("Right_Hip", "Right_Knee") + bone("Right_Knee", "Right_Ankle")
        trunk = bone("Mid_Hip", "Neck")
        # COMFI has no head landmark.  The conventional 0.24 m neck-to-crown
        # allowance is explicit provenance rather than an unlabelled default.
        estimate = 0.5 * (left_leg + right_leg) + trunk + 0.24
        if 1.2 <= estimate <= 2.2:
            return estimate, "static_skeleton_chain_plus_0.24m_neck_to_crown"
    except (KeyError, ValueError, IndexError):
        pass
    return 1.70, "fallback_population_height_after_skeleton_estimate_failed"


def _video_cache_path(
    config: RetargetConfig, reference: SequenceRef, cache_key: str | None = None,
) -> Path:
    suffix = f".{cache_key[:16]}" if cache_key else ""
    return (config.output_root / config.raw["batch"]["cache_subdir"] / "wilor" /
            reference.dataset / f"{_safe_id(reference.sequence_id)}{suffix}.npz")


def _gvhmr_cache_path(
    config: RetargetConfig, reference: SequenceRef, cache_key: str | None = None,
) -> Path:
    suffix = f".{cache_key[:16]}" if cache_key else ""
    return (config.output_root / config.raw["batch"]["cache_subdir"] / "gvhmr" /
            reference.dataset / f"{_safe_id(reference.sequence_id)}{suffix}.pt")


def _load_gvhmr_body(
    config: RetargetConfig, reference: SequenceRef, profile: RetargetProfile, source_height: float,
    target_root_pivot: np.ndarray, cache_key: str | None = None,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    cache = _gvhmr_cache_path(config, reference, cache_key)
    receipt = ensure_gvhmr_cache(config, reference, cache)
    if not cache.is_file():
        raise RuntimeError(json.dumps(receipt, sort_keys=True))
    prediction = load_gvhmr_result(cache)
    root = np.asarray(prediction["global_orient"], dtype=np.float32).reshape(-1, 3)
    body = np.asarray(prediction["body_pose"], dtype=np.float32).reshape(len(root), 63)
    translation = np.asarray(prediction["transl"], dtype=np.float32).reshape(len(root), 3)
    source_betas_value = prediction.get("betas")
    if source_betas_value is None:
        source_betas = (0.0,) * 10
    else:
        source_betas_array = np.asarray(source_betas_value, dtype=np.float64).reshape(-1, np.asarray(source_betas_value).shape[-1])
        source_betas = tuple(np.median(source_betas_array, axis=0).astype(float).tolist())
    source_root_pivot = model_root_pivot(
        config.smplx_model_dir, "neutral", source_betas,
    )
    source_root_position = translation + source_root_pivot
    root, translation, transform = canonicalize_bvh_root(
        root, source_root_position * 100.0, profile.height_m, source_height,
    )
    translation -= np.asarray(target_root_pivot, dtype=np.float32)
    metadata = {
        "receipt": receipt, "cache": str(cache), "source_model": "GVHMR neutral SMPL-X",
        "target_profile_betas": list(profile.betas), "translation_height_scaled": True,
        "temporal_basis": "official GVHMR 30 Hz demo",
        "source_root_pivot_m": source_root_pivot.astype(float).tolist(),
        "target_root_pivot_m": np.asarray(target_root_pivot).astype(float).tolist(),
    }
    return 30.0, root, body, translation, transform, metadata


def _load_video_hands(
    config: RetargetConfig, reference: SequenceRef, timestamps: np.ndarray,
    video_to_body_offset_s: float = 0.0, cache_key: str | None = None,
) -> tuple[HandTrack, HandTrack, dict]:
    cache = _video_cache_path(config, reference, cache_key)
    if not cache.is_file():
        receipt = ensure_wilor_cache(config, reference, cache)
        if not cache.is_file():
            empty = empty_hand_track(len(timestamps))
            return empty, empty, {"mode": "wilor_video", **receipt, "cache": str(cache)}
    hand_cfg = config.raw["hands"]
    left, right, metadata = load_wilor_cache(
        cache, timestamps - float(video_to_body_offset_s),
        1.0 / np.median(np.diff(timestamps)),
        float(hand_cfg["short_gap_seconds"]), float(hand_cfg["min_detection_confidence"]),
        float(hand_cfg["max_normalized_reprojection_error"]),
        float(hand_cfg["max_joint_angular_speed_rad_s"]),
        config.raw["temporal_continuity"],
    )
    return left, right, {
        "mode": "wilor_video", "status": "loaded", "cache": str(cache),
        "video_to_body_offset_s": float(video_to_body_offset_s), **metadata,
    }


def _rotation_motion_energy(
    root: np.ndarray, body: np.ndarray, fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    pose = np.concatenate([
        np.asarray(root, dtype=np.float64).reshape(-1, 1, 3),
        np.asarray(body, dtype=np.float64).reshape(-1, 21, 3),
    ], axis=1)
    if len(pose) < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    matrices = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(len(pose), 22, 3, 3)
    relative = matrices[:-1].transpose(0, 1, 3, 2) @ matrices[1:]
    speeds = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(-1, 22) * float(fps)
    # Average the four most active joints. This still detects one-limb actions,
    # while diluting a single noisy joint instead of letting it set the signal.
    energy = np.sort(speeds, axis=1)[:, -4:].mean(axis=1)
    times = (np.arange(len(energy), dtype=np.float64) + 0.5) / float(fps)
    return times, energy


def _estimate_motion_time_offset(
    body_root: np.ndarray, body_pose: np.ndarray, body_fps: float,
    video_root: np.ndarray, video_pose: np.ndarray, video_fps: float,
    max_abs_offset_s: float, min_overlap_s: float,
) -> tuple[float, float, int]:
    body_times, body_energy = _rotation_motion_energy(body_root, body_pose, body_fps)
    video_times, video_energy = _rotation_motion_energy(video_root, video_pose, video_fps)
    if not len(body_times) or not len(video_times):
        return 0.0, float("nan"), 0
    step = 1.0 / max(1.0, float(video_fps))
    candidates = np.arange(-max_abs_offset_s, max_abs_offset_s + 0.5 * step, step)
    best = (float("-inf"), 0.0, 0)
    minimum = max(3, int(np.ceil(min_overlap_s * video_fps)))
    for offset in candidates:
        query = video_times + offset
        keep = (query >= body_times[0]) & (query <= body_times[-1])
        overlap = int(keep.sum())
        if overlap < minimum:
            continue
        reference = np.interp(query[keep], body_times, body_energy)
        observed = video_energy[keep]
        if np.std(reference) < 1e-8 or np.std(observed) < 1e-8:
            continue
        correlation = float(np.corrcoef(reference, observed)[0, 1])
        if np.isfinite(correlation) and correlation > best[0]:
            best = (correlation, float(offset), overlap)
    if not np.isfinite(best[0]):
        return 0.0, float("nan"), best[2]
    return best[1], best[0], best[2]


def _andy_video_sync(
    config: RetargetConfig, reference: SequenceRef,
    body_root: np.ndarray, body_pose: np.ndarray, body_fps: float,
    cache_key: str | None = None,
) -> tuple[float, dict]:
    cache = _gvhmr_cache_path(config, reference, cache_key)
    receipt = ensure_gvhmr_cache(config, reference, cache)
    if not cache.is_file():
        return 0.0, {
            "status": "unavailable_zero_offset", "applied": False,
            "offset_s": 0.0, "receipt": receipt,
        }
    prediction = load_gvhmr_result(cache)
    video_root = np.asarray(prediction["global_orient"], dtype=np.float32).reshape(-1, 3)
    video_pose = np.asarray(prediction["body_pose"], dtype=np.float32).reshape(len(video_root), 63)
    settings = config.raw["datasets"]["andy"]["video_sync"]
    offset, correlation, overlap = _estimate_motion_time_offset(
        body_root, body_pose, body_fps, video_root, video_pose, 30.0,
        float(settings["max_abs_offset_seconds"]), float(settings["min_overlap_seconds"]),
    )
    accepted = bool(np.isfinite(correlation) and correlation >= float(settings["min_correlation"]))
    return (offset if accepted else 0.0), {
        "status": "aligned" if accepted else "low_confidence_zero_offset",
        "applied": accepted, "offset_s": float(offset if accepted else 0.0),
        "candidate_offset_s": float(offset),
        "correlation": float(correlation) if np.isfinite(correlation) else None,
        "overlap_frames_at_30hz": overlap, "receipt": receipt,
        "method": "22_joint_SO3_top4_speed_cross_correlation",
    }


def _resample_pose(values: np.ndarray, source_fps: float, target_times: np.ndarray) -> np.ndarray:
    source_times = np.arange(len(values), dtype=np.float64) / source_fps
    result = np.zeros((len(target_times), values.shape[1]), dtype=np.float32)
    for joint in range(values.shape[1] // 3):
        sl = slice(3 * joint, 3 * joint + 3)
        result[:, sl] = slerp_rotvec(source_times, values[:, sl], target_times)
    return result


def _orientation_from_landmarks(
    points: np.ndarray, valid: np.ndarray, names: list[str], left_names: tuple[str, ...], right_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    left_ids = [names.index(name) for name in left_names if name in names]
    right_ids = [names.index(name) for name in right_names if name in names]
    if not left_ids or not right_ids:
        raise ValueError("Cannot estimate pelvis orientation: left/right landmarks are missing")
    left = np.nanmean(np.where(valid[:, left_ids, None], points[:, left_ids], np.nan), axis=1)
    right = np.nanmean(np.where(valid[:, right_ids, None], points[:, right_ids], np.nan), axis=1)
    pelvis = (left + right) / 2.0
    x_axis = left - right
    x_axis /= np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1e-8)
    y_axis = np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), x_axis.shape).copy()
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.maximum(np.linalg.norm(z_axis, axis=1, keepdims=True), 1e-8)
    x_axis = np.cross(y_axis, z_axis)
    matrices = np.stack([x_axis, y_axis, z_axis], axis=2)
    bad = ~np.isfinite(matrices).all(axis=(1, 2)) | ~np.isfinite(pelvis).all(axis=1)
    for frame in range(len(matrices)):
        if bad[frame]:
            matrices[frame] = matrices[frame - 1] if frame else np.eye(3)
            pelvis[frame] = pelvis[frame - 1] if frame else 0.0
    return matrices, pelvis


def _fit_andy(
    config: RetargetConfig, reference: SequenceRef, profile: RetargetProfile,
    motion, source_root: np.ndarray, source_body: np.ndarray, source_height: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    marker_times, names, markers, marker_valid = load_andy_markers(config, reference)
    mapping = load_ssm2_vertices(config)
    selected = [index for index, name in enumerate(names) if name in mapping]
    if len(selected) < 20:
        raise ValueError(f"Only {len(selected)} AnDy markers map to SSM2")
    target = markers[:, selected]
    valid = marker_valid[:, selected]
    indices = np.asarray([mapping[names[index]] for index in selected], dtype=np.int64)
    orientation, pelvis = _orientation_from_landmarks(
        markers, marker_valid, names, ("LASI", "LPSI"), ("RASI", "RPSI"),
    )
    ratio = profile.height_m / source_height
    origin = pelvis[0].copy()
    target = target * ratio
    pelvis = pelvis * ratio
    target[:, :, :2] -= origin[None, None, :2] * ratio
    pelvis[:, :2] -= origin[None, :2] * ratio
    duration = min(marker_times[-1], (motion.frame_count - 1) / motion.fps)
    keep = marker_times <= duration
    marker_times, target, valid, pelvis = marker_times[keep], target[keep], valid[keep], pelvis[keep]
    root_resampled = _resample_pose(source_root, motion.fps, marker_times)
    body_resampled = _resample_pose(source_body, motion.fps, marker_times)
    selected_names = [names[index] for index in selected]
    head_names = ("RBHD", "RFHD", "LBHD", "LFHD")
    head_selected = [selected_names.index(name) for name in head_names if name in selected_names]
    head_reliable = np.zeros(len(marker_times), dtype=bool)
    if len(head_selected) >= 3:
        head_points = target[:, head_selected]
        head_valid = valid[:, head_selected]
        for frame in range(len(marker_times)):
            observed = head_points[frame, head_valid[frame]]
            if len(observed) < int(config.raw["fitting"]["andy_head_min_markers"]):
                continue
            maximum_area = 0.0
            for first in range(len(observed) - 2):
                for second in range(first + 1, len(observed) - 1):
                    for third in range(second + 1, len(observed)):
                        area = 0.5 * np.linalg.norm(np.cross(
                            observed[second] - observed[first], observed[third] - observed[first],
                        ))
                        maximum_area = max(maximum_area, float(area))
            head_reliable[frame] = maximum_area >= float(
                config.raw["fitting"]["andy_head_min_triangle_area_m2"]
            )
    fallback_mask = np.zeros((len(marker_times), 21), dtype=bool)
    fallback_mask[~head_reliable, 11] = True  # neck
    fallback_mask[~head_reliable, 14] = True  # head
    source_matrices = Rotation.from_rotvec(root_resampled).as_matrix()
    aligned_root = orientation[0] @ source_matrices[0].T @ source_matrices
    root_initial = Rotation.from_matrix(aligned_root).as_rotvec().astype(np.float32)
    root, body, translation, metrics = fit_smplx_observations(
        config, profile, target, valid, indices, root_initial, body_resampled, pelvis,
        "vertices", 120.0, fallback_body=body_resampled,
        fallback_body_mask=fallback_mask,
    )
    metrics["head_marker_reliable_frames"] = int(head_reliable.sum())
    metrics["head_marker_reliable_fraction"] = float(head_reliable.mean())
    metrics["head_marker_fallback"] = "Xsens neck/head when fewer than three non-collinear head markers"
    metrics["target_names"] = selected_names
    metrics["excluded_marker_names"] = [
        selected_names[index] for index in metrics["excluded_target_indices"]
    ]
    root, translation, transform = _canonicalize_z_up(
        root, translation, 1.0, np.asarray(metrics["root_pivot_m"]),
    )
    transform[:3, 3] -= transform[:3, :3] @ np.asarray(
        [origin[0] * ratio, origin[1] * ratio, 0.0], dtype=np.float64,
    )
    return 120.0, root, body, translation, transform, metrics


def _fit_comfi(
    config: RetargetConfig, reference: SequenceRef, profile: RetargetProfile, mode: str,
    initial_root: np.ndarray, initial_body: np.ndarray, source_height: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    fps, names, points, valid = load_comfi_joint_centers(reference, mode)
    indices, target, selected_valid = select_comfi_joints(names, points, valid)
    orientation, pelvis = _orientation_from_landmarks(
        points, valid, names, ("Left_Hip",), ("Right_Hip",),
    )
    ratio = profile.height_m / source_height
    origin = pelvis[0].copy()
    target = target * ratio
    pelvis = pelvis * ratio
    target[:, :, :2] -= origin[None, None, :2] * ratio
    pelvis[:, :2] -= origin[None, :2] * ratio
    count = min(len(target), len(initial_root))
    target, selected_valid, pelvis = target[:count], selected_valid[:count], pelvis[:count]
    # COMFI freeflyer already contains the full pelvis orientation, including
    # sagittal tilt and lateral lean. The hip-only landmark frame forces world
    # up and must not replace it.
    root_initial = np.asarray(initial_root[:count], dtype=np.float32)
    root, body, translation, metrics = fit_smplx_observations(
        config, profile, target, selected_valid, indices, root_initial,
        initial_body[:count], pelvis, "joints", fps,
    )
    selected_names = [
        name for name in COMFI_TO_SMPLX if name in names and name not in COMFI_EXCLUDED_JCP
    ]
    metrics["target_names"] = selected_names
    metrics["preexcluded_joint_tracks"] = dict(COMFI_EXCLUDED_JCP)
    metrics["excluded_joint_names"] = [
        selected_names[index] for index in metrics["excluded_target_indices"]
    ]
    root, translation, transform = _canonicalize_z_up(
        root, translation, 1.0, np.asarray(metrics["root_pivot_m"]),
    )
    transform[:3, 3] -= transform[:3, :3] @ np.asarray(
        [origin[0] * ratio, origin[1] * ratio, 0.0], dtype=np.float64,
    )
    return fps, root, body, translation, transform, metrics


def _andy_glove_quality(
    config: RetargetConfig, reference: SequenceRef, timestamps: np.ndarray, right_hand: HandTrack,
) -> dict:
    from scipy.stats import spearmanr
    if not right_hand.valid.any():
        return {"status": "no_valid_video_hand"}
    try:
        glove_times, glove_angles = load_andy_glove(config, reference)
    except (KeyError, FileNotFoundError) as exc:
        return {"status": "glove_missing", "error": str(exc)}
    finger_slices = (slice(0, 9), slice(9, 18), slice(36, 45))
    correlations = {}
    for name, angle, pose_slice in zip(("index", "middle", "thumb"), glove_angles.T, finger_slices):
        interpolated = np.interp(timestamps, glove_times, angle, left=np.nan, right=np.nan)
        pose = right_hand.pose[:, pose_slice].reshape(len(timestamps), 3, 3)
        flexion = np.linalg.norm(pose, axis=2).sum(axis=1)
        keep = right_hand.valid & np.isfinite(interpolated)
        correlations[name] = float(spearmanr(interpolated[keep], flexion[keep]).statistic) if keep.sum() >= 10 else None
    return {"status": "auxiliary_only", "assumed_side": "right", "spearman": correlations,
            "note": "Three glove bend channels are not promoted to 15-joint MANO observations."}


def _ensure_body_c3(
    config: RetargetConfig, fps: float, root: np.ndarray, body: np.ndarray,
    translation: np.ndarray, quality: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project native/fallback trajectories; fitted IK already performs this before its quality gate."""
    fit = quality.get("body_fit", {})
    required = int(config.raw["temporal_continuity"]["required_order"])
    if (str(quality.get("body_mode", "")).endswith("_ik") and
            int(fit.get("c3_continuity_order", -1)) >= required):
        quality["body_temporal_continuity"] = {
            "source": "fit_smplx_observations", "continuity_order": int(fit["c3_continuity_order"]),
        }
        return root, body, translation
    settings = config.raw["temporal_continuity"]
    times = np.arange(len(root), dtype=np.float64) / float(fps)
    combined, rotation_metadata = c3_smooth_rotvec(
        times, np.concatenate([root, body], axis=1), float(settings["body_knot_seconds"]),
        int(settings["spline_degree"]), int(settings["robust_iterations"]),
        float(settings["rotation_huber_degrees"]),
    )
    translation, translation_metadata = c3_smooth_translation(
        times, translation, float(settings["translation_knot_seconds"]),
        int(settings["spline_degree"]), int(settings["robust_iterations"]),
        float(settings["translation_huber_m"]),
    )
    quality["body_temporal_continuity"] = {
        "source": "post_retarget_projection", "continuity_order": int(settings["spline_degree"]) - 1,
        "rotation": rotation_metadata, "translation": translation_metadata,
    }
    return combined[:, :3], combined[:, 3:], translation


def build_motion(
    config: RetargetConfig, reference: SequenceRef, profile: RetargetProfile,
    input_mode: str | None = None, cache_keys: dict[str, str] | None = None,
) -> MotionRecord:
    cache_keys = cache_keys or {}
    scale, base_height = profile_scale(config.smplx_model_dir, profile)
    target_root_pivot = profile_root_pivot(config.smplx_model_dir, profile, scale)
    quality: dict = {
        "config_hash": config.digest,
        "profile_base_height_m": base_height,
        "profile_height_error_m": abs(base_height * scale - profile.height_m),
        "target_root_pivot_m": target_root_pivot.astype(float).tolist(),
        "input_mode_requested": input_mode or config.raw["datasets"][reference.dataset]["input_mode"],
        "actor_id": profile.actor_id,
        "profile_source": profile.profile_source,
        "betas_source": profile.betas_source,
        "target_gender": profile.gender,
        "target_height_m": profile.height_m,
        "target_betas": list(profile.betas),
    }
    mode = input_mode or str(config.raw["datasets"][reference.dataset]["input_mode"])
    allowed_modes = {
        "andy": {"auto", "qualisys", "xsens_bvh", "gvhmr_video"},
        "comfi": {"auto", "mocap_aligned", "mocap_raw", "gvhmr_video"},
        "hiphi": {"auto", "bvh"},
    }
    if mode not in allowed_modes[reference.dataset]:
        raise ValueError(f"input_mode={mode} is not valid for {reference.dataset}")
    if reference.dataset in {"andy", "hiphi"}:
        motion, root, body, left_pose, right_pose = bvh_pose(reference)
        fps = motion.fps
        source_translation = motion.local_positions()[:, 0]
        if reference.dataset == "andy":
            source_height, height_source = load_andy_heights(config)[reference.metadata["participant"]], "participant_specifications"
            fitted = None
            if mode in {"auto", "qualisys"}:
                try:
                    fitted = _fit_andy(config, reference, profile, motion, root, body, source_height)
                except Exception as exc:
                    quality["qualisys_failure"] = f"{type(exc).__name__}: {exc}"
            if fitted is not None:
                fit_fps, fit_root, fit_body, fit_translation, fit_transform, fit_metrics = fitted
                quality["body_fit"] = fit_metrics
                good = (fit_metrics["median_error_m"] <= float(config.raw["quality"]["body_median_error_m"]) and
                        fit_metrics["p95_error_m"] <= float(config.raw["quality"]["body_p95_error_m"]))
                if good:
                    fps, root, body, translation, transform = fit_fps, fit_root, fit_body, fit_translation, fit_transform
                    quality["body_mode"] = "qualisys_marker_ik"
                    quality["body_quality"] = "good"
                    timestamps = np.arange(len(root), dtype=np.float64) / fps
                else:
                    fitted = None
                    quality["qualisys_failure"] = "quality_gate"
            if fitted is None and mode in {"auto", "gvhmr_video"}:
                try:
                    fps, root, body, translation, transform, gvhmr_metadata = _load_gvhmr_body(
                        config, reference, profile, source_height, target_root_pivot,
                        cache_keys.get("gvhmr"),
                    )
                    timestamps = np.arange(len(root), dtype=np.float64) / fps
                    quality["body_mode"] = "gvhmr_video_fallback"
                    quality["body_quality"] = "video_fallback"
                    quality["gvhmr"] = gvhmr_metadata
                    fitted = True
                except Exception as exc:
                    quality["gvhmr_failure"] = f"{type(exc).__name__}: {exc}"
                    if mode == "gvhmr_video":
                        raise RuntimeError(quality["gvhmr_failure"]) from exc
            if fitted is None:
                quality["body_mode"] = "xsens_bvh_fallback"
                quality["body_quality"] = "native_fallback"
            left = right = empty_hand_track(motion.frame_count)
        else:
            metadata = load_hiphi_metadata(reference)
            source_height = float(metadata["actor_metadata"]["height_cm"]) / 100.0
            height_source = "metadata.actor_metadata.height_cm"
            quality.update({
                "body_mode": "native_bvh", "is_hoi": bool(metadata.get("is_hoi")),
                "actor_id": metadata.get("actor_id"),
                "object_metadata_reference": metadata.get("objects", metadata.get("object_metadata")),
                "has_object_metadata": bool(metadata.get("objects", metadata.get("object_metadata"))),
            })
            quality["body_quality"] = "good"
        if reference.dataset == "hiphi" or fitted is None:
            root, translation, transform = canonicalize_bvh_root(root, source_translation, profile.height_m, source_height)
            translation -= target_root_pivot.astype(np.float32)
            timestamps = np.arange(motion.frame_count, dtype=np.float64) / fps
        root, body, translation = _ensure_body_c3(
            config, fps, root, body, translation, quality,
        )
        if reference.dataset == "hiphi":
            left_pose, right_pose, hand_retarget = retarget_native_hands(
                config, profile, motion, left_pose, right_pose, source_height,
                root, body, transform,
            )
            quality["native_hand_retarget"] = hand_retarget
            left, right = observed_hand_track(left_pose), observed_hand_track(right_pose)
            quality["native_hand_fk"] = native_hand_fk_quality(
                config, profile, motion, root, body, left.pose, right.pose,
                translation, transform, source_height,
            )
    elif reference.dataset == "comfi":
        comfi_mode = "mocap_raw" if mode == "mocap_raw" else "mocap_aligned"
        fps, root, body, source_translation, _ = load_comfi_pose(reference, comfi_mode)
        source_height, height_source = _comfi_height(config, reference)
        fitted = None
        if mode in {"auto", "mocap_aligned", "mocap_raw"}:
            try:
                fitted = _fit_comfi(config, reference, profile, comfi_mode, root, body, source_height)
            except Exception as exc:
                quality["mocap_ik_failure"] = f"{type(exc).__name__}: {exc}"
        if fitted is not None:
            fit_fps, fit_root, fit_body, fit_translation, fit_transform, fit_metrics = fitted
            quality["body_fit"] = fit_metrics
            good = (fit_metrics["median_error_m"] <= float(config.raw["quality"]["body_median_error_m"]) and
                    fit_metrics["p95_error_m"] <= float(config.raw["quality"]["body_p95_error_m"]))
            if good:
                fps, root, body, translation, transform = fit_fps, fit_root, fit_body, fit_translation, fit_transform
                quality["body_mode"] = f"{comfi_mode}_joint_ik"
                quality["body_quality"] = "good"
            else:
                fitted = None
                quality["mocap_ik_failure"] = "quality_gate"
        if fitted is None and mode in {"auto", "gvhmr_video"}:
            try:
                fps, root, body, translation, transform, gvhmr_metadata = _load_gvhmr_body(
                    config, reference, profile, source_height, target_root_pivot,
                    cache_keys.get("gvhmr"),
                )
                quality["body_mode"] = "gvhmr_video_fallback"
                quality["body_quality"] = "video_fallback"
                quality["gvhmr"] = gvhmr_metadata
                fitted = True
            except Exception as exc:
                quality["gvhmr_failure"] = f"{type(exc).__name__}: {exc}"
                if mode == "gvhmr_video":
                    raise RuntimeError(quality["gvhmr_failure"]) from exc
        if fitted is None:
            root, translation, transform = _canonicalize_z_up(root, source_translation, profile.height_m / source_height)
            translation -= target_root_pivot.astype(np.float32)
            quality["body_mode"] = f"{comfi_mode}_joint_angles_fallback"
            quality["body_quality"] = "unusable"
        timestamps = np.arange(len(root), dtype=np.float64) / fps
        root, body, translation = _ensure_body_c3(
            config, fps, root, body, translation, quality,
        )
        if comfi_mode == "mocap_raw":
            left = right = empty_hand_track(len(timestamps))
            hand_provenance = {"mode": "body_only", "status": "disabled_for_mocap_raw"}
        else:
            left, right, hand_provenance = _load_video_hands(
                config, reference, timestamps, cache_key=cache_keys.get("wilor"),
            )
        quality["hand_provenance"] = hand_provenance
    else:
        raise ValueError(reference.dataset)

    if reference.dataset == "andy":
        video_offset, sync_provenance = _andy_video_sync(
            config, reference, root, body, fps, cache_keys.get("gvhmr"),
        )
        left, right, hand_provenance = _load_video_hands(
            config, reference, timestamps, video_offset, cache_keys.get("wilor"),
        )
        hand_provenance["time_sync"] = sync_provenance
        quality["hand_provenance"] = hand_provenance
        quality["glove_quality"] = _andy_glove_quality(config, reference, timestamps, right)
    translation, floor_shift, pre_floor_minimum = floor_align_translation(
        config, profile, scale, fps, root, body,
        left.pose, right.pose, translation,
    )
    transform = np.asarray(transform).copy()
    transform[2, 3] += floor_shift
    quality["floor_alignment"] = {
        "first_second_minimum_before_m": pre_floor_minimum,
        "translation_shift_m": floor_shift,
    }
    label, coverage = coverage_quality(
        left, right, float(config.raw["hands"]["good_coverage"]), float(config.raw["hands"]["partial_coverage"]),
    )
    quality.update({
        "source_height_m": source_height, "source_height_basis": height_source,
        "source_motion_scale": profile.height_m / source_height,
        "source_transform_convention": (
            "p_canonical = source_to_canonical.R @ (p_source_m * source_motion_scale) "
            "+ source_to_canonical.t"
        ),
        "hand_quality": label, "hand_coverage": coverage,
        "source_archive": str(reference.metadata.get("origin_archive", reference.source)),
        "source_member": reference.metadata.get("origin_member", reference.member),
        "kinematic_only": True,
        "temporal_continuity": {
            "required_order": int(config.raw["temporal_continuity"]["required_order"]),
            "provided_order": int(config.raw["temporal_continuity"]["spline_degree"]) - 1,
            "representation": "quintic B-spline (C4 on the open time domain)",
        },
    })
    continuous_spline = c3_archive_payload(
        timestamps, root, body, left.pose, right.pose, translation,
        int(config.raw["temporal_continuity"]["spline_degree"]),
    )
    if not continuous_spline:
        raise ValueError("Sequence is too short to construct the required C4 trajectory")
    record = MotionRecord(
        reference, profile, fps, timestamps, root, body,
        left.pose, right.pose, translation, scale,
        left.valid, right.valid, left.confidence, right.confidence,
        left.imputed, right.imputed, transform, quality, continuous_spline,
    )
    record.validate()
    return record


def extract_one(
    config: RetargetConfig, ledger: Ledger, reference: SequenceRef, profile: RetargetProfile,
    resume: bool = True, input_mode: str | None = None,
) -> dict:
    target = output_path(config, reference, profile)
    key = (reference.dataset, reference.sequence_id, profile.name)
    task_config_hash = run_config_hash(config, reference, input_mode)
    try:
        _disk_guard(config)
        input_hash = input_dependency_hash(config, ledger, reference, input_mode)
        model_hash = model_dependency_hash(config, ledger, reference, profile, input_mode)
        if resume and ledger.completed(*key, task_config_hash, input_hash, model_hash):
            return {
                "status": "skipped", "output": str(target),
                "sequence_id": reference.sequence_id, "profile": profile.name,
            }
        ledger.upsert(*key, "running", task_config_hash, increment_attempts=True,
                      input_hash=input_hash, model_hash=model_hash)
        mode = input_mode or str(config.raw["datasets"][reference.dataset]["input_mode"])
        cache_keys = {}
        if reference.dataset == "andy" or (reference.dataset == "comfi" and mode != "mocap_raw"):
            cache_keys["wilor"] = cache_dependency_hash(
                config, ledger, reference, input_hash, "wilor",
            )
        if (reference.dataset == "andy" or mode == "gvhmr_video" or
                (reference.dataset == "comfi" and mode == "auto")):
            cache_keys["gvhmr"] = cache_dependency_hash(
                config, ledger, reference, input_hash, "gvhmr",
            )
        record = build_motion(
            config, reference, profile, input_mode=input_mode, cache_keys=cache_keys,
        )
        record.quality["config_hash"] = task_config_hash
        record.quality["input_hash"] = input_hash
        record.quality["model_dependency_hash"] = model_hash
        write_motion_npz(target, record)
        metrics = dict(json_safe(record.quality))
        metrics.update({"frames": len(record.timestamps), "fps": record.fps, "output_bytes": target.stat().st_size})
        ledger.upsert(*key, "extracted", task_config_hash, str(target), metrics,
                      input_hash=input_hash, model_hash=model_hash)
        return {"status": "extracted", "output": str(target), "sequence_id": reference.sequence_id, "profile": profile.name, "metrics": metrics}
    except Exception as exc:
        error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        ledger.upsert(*key, "failed", task_config_hash, str(target), error=error)
        return {"status": "failed", "sequence_id": reference.sequence_id, "profile": profile.name, "error": error}


def extract_reference(
    config_path: str | Path, reference: SequenceRef, profile_requests: list[str],
    resume: bool = True, retries: int = 0, input_mode: str | None = None,
) -> list[dict]:
    """Extract all profiles for one source in one process so decoded BVH is reused."""
    from .config import load_config
    config = load_config(config_path)
    ledger = Ledger(config.output_root / "ledger.sqlite3")
    results: list[dict] = []
    try:
        # External video stacks must be spawned before this long-lived worker
        # creates a CUDA context for SMPL-X IK. Forking from a CUDA-initialized
        # parent can make upstream preprocessors ignore device visibility.
        prewarm_reference(config.path, reference, input_mode)
        for profile in profiles_for_reference(config, reference, profile_requests):
            result: dict = {}
            for _ in range(retries + 1):
                result = extract_one(config, ledger, reference, profile,
                                     resume=resume, input_mode=input_mode)
                if result["status"] != "failed":
                    break
            results.append(result)
    finally:
        ledger.close()
    return results


def prewarm_reference(
    config_path: str | Path, reference: SequenceRef, input_mode: str | None = None,
) -> dict:
    """Build visual caches before any SMPL-X torch/CUDA context exists."""
    from .config import load_config
    config = load_config(config_path)
    ledger = Ledger(config.output_root / "ledger.sqlite3")
    receipts = {}
    try:
        mode = input_mode or str(config.raw["datasets"][reference.dataset]["input_mode"])
        input_hash = input_dependency_hash(config, ledger, reference, input_mode)
        if reference.dataset == "andy" or (reference.dataset == "comfi" and mode != "mocap_raw"):
            wilor_key = cache_dependency_hash(config, ledger, reference, input_hash, "wilor")
            receipts["wilor"] = ensure_wilor_cache(
                config, reference, _video_cache_path(config, reference, wilor_key),
            )
        if (reference.dataset == "andy" or mode == "gvhmr_video" or
                (reference.dataset == "comfi" and mode == "auto")):
            gvhmr_key = cache_dependency_hash(config, ledger, reference, input_hash, "gvhmr")
            receipts["gvhmr"] = ensure_gvhmr_cache(
                config, reference, _gvhmr_cache_path(config, reference, gvhmr_key),
            )
        failed = {
            name: receipt for name, receipt in receipts.items()
            if receipt.get("status") not in {"cached", "created"}
        }
        return {
            "status": "failed" if failed else "complete", "dataset": reference.dataset,
            "sequence_id": reference.sequence_id, "receipts": receipts,
            "error": json.dumps(failed, sort_keys=True) if failed else None,
        }
    finally:
        ledger.close()


def write_manifest(config: RetargetConfig, ledger: Ledger) -> Path:
    path = config.output_root / "manifest.jsonl"
    temporary = path.with_suffix(".jsonl.partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w") as stream:
        for row in ledger.rows():
            stream.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path


def inventory_payload(
    config: RetargetConfig, datasets: Iterable[str], allow_incomplete: bool = False,
    profile_requests: Iterable[str] | None = None,
) -> dict:
    result = {name: inventory(config, name, allow_incomplete=allow_incomplete) for name in datasets}
    requests = list(profile_requests) if profile_requests is not None else list(config.profiles)
    payload = {
        "config_hash": config.digest,
        "datasets": {
            name: {
                "sequence_count": len(rows),
                "expected_outputs": sum(
                    len(profiles_for_reference(config, row, requests)) for row in rows
                ),
                "resolved_profile_counts": {
                    profile_name: sum(
                        profile.name == profile_name
                        for row in rows
                        for profile in profiles_for_reference(config, row, requests)
                    )
                    for profile_name in sorted({
                        profile.name
                        for row in rows
                        for profile in profiles_for_reference(config, row, requests)
                    })
                },
                "sequences": [
                    {
                        "sequence_id": row.sequence_id,
                        "source": str(row.source),
                        "member": row.member,
                        "metadata": dict(row.metadata),
                    }
                    for row in rows
                ],
            }
            for name, rows in result.items()
        },
        "profile_requests": requests,
    }
    write_json_atomic(config.output_root / "inventory.json", payload)
    return payload
