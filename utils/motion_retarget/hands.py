from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .rotations import interpolate_pose_gaps, slerp_rotvec
from .trajectory import c3_smooth_rotvec


@dataclass(frozen=True)
class HandTrack:
    pose: np.ndarray
    valid: np.ndarray
    confidence: np.ndarray
    imputed: np.ndarray


def _resample_pose(source_times: np.ndarray, pose: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    result = np.zeros((len(target_times), 45), dtype=np.float32)
    for joint in range(15):
        sl = slice(3 * joint, 3 * joint + 3)
        result[:, sl] = slerp_rotvec(source_times, pose[:, sl], target_times)
    return result


def load_wilor_cache(
    path: Path, target_times: np.ndarray, fps: float, short_gap_seconds: float,
    min_confidence: float, max_reprojection_error: float,
    max_joint_angular_speed: float = 30.0,
    c3_settings: dict | None = None,
) -> tuple[HandTrack, HandTrack, dict]:
    """Load the stable interchange emitted by the isolated WiLoR worker.

    Cache pose values are MANO local axis-angle in SMPL-X's 15-joint order.  A
    cache never makes missing observations valid: interpolation only changes the
    playable pose and sets ``imputed``.
    """
    with np.load(path, allow_pickle=False) as data:
        source_times = np.asarray(data["timestamps"], dtype=np.float64)
        if len(source_times) == 0 or (len(source_times) > 1 and not np.all(np.diff(source_times) > 0)):
            raise ValueError(f"WiLoR cache has an invalid time axis: {path}")
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        metadata = dict(metadata)
        reprojection_available = bool(metadata.get("reprojection_available", False))
        tracks: list[HandTrack] = []
        quality: dict[str, dict] = {}
        for side in ("left", "right"):
            pose = np.asarray(data[f"{side}_hand_pose"], dtype=np.float32)
            confidence = np.asarray(data[f"{side}_confidence"], dtype=np.float32)
            reprojection = np.asarray(data[f"{side}_reprojection_error"], dtype=np.float32)
            observed = np.asarray(data[f"{side}_valid"], dtype=bool)
            valid = observed & (confidence >= min_confidence)
            if reprojection_available:
                valid &= reprojection <= max_reprojection_error
            if len(pose) > 1:
                matrices = Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix().reshape(len(pose), 15, 3, 3)
                relative = matrices[:-1].transpose(0, 1, 3, 2) @ matrices[1:]
                angles = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(len(pose) - 1, 15)
                dt = np.maximum(np.diff(source_times), 1e-6)
                temporal_ok = np.max(angles, axis=1) / dt <= max_joint_angular_speed
                valid[1:] &= temporal_ok
            source_dt = float(np.median(np.diff(source_times))) if len(source_times) > 1 else 1.0 / fps
            pose, imputed = interpolate_pose_gaps(pose, valid, 1.0 / source_dt, short_gap_seconds)
            resampled = _resample_pose(source_times, pose, target_times)
            temporal_metadata = None
            if c3_settings is not None:
                resampled, temporal_metadata = c3_smooth_rotvec(
                    target_times, resampled, float(c3_settings["hand_knot_seconds"]),
                    int(c3_settings["spline_degree"]), int(c3_settings["robust_iterations"]),
                    float(c3_settings["rotation_huber_degrees"]),
                )
            upper = np.clip(np.searchsorted(source_times, target_times), 0, len(source_times) - 1)
            lower = np.maximum(0, upper - 1)
            choose_lower = np.abs(target_times - source_times[lower]) <= np.abs(source_times[upper] - target_times)
            indices = np.where(choose_lower, lower, upper)
            temporally_observed = (
                (target_times >= source_times[0]) & (target_times <= source_times[-1]) &
                (np.abs(target_times - source_times[indices]) <= 0.75 * source_dt)
            )
            target_valid = valid[indices] & temporally_observed
            tracks.append(HandTrack(
                resampled,
                target_valid,
                confidence[indices],
                imputed[indices] | ~target_valid,
            ))
            accepted_confidence = confidence[valid]
            quality[side] = {
                "source_observed_frames": int(observed.sum()),
                "source_accepted_frames": int(valid.sum()),
                "target_valid_frames": int(target_valid.sum()),
                "target_coverage": float(target_valid.mean()),
                "accepted_confidence_median": (
                    float(np.median(accepted_confidence)) if len(accepted_confidence) else None
                ),
                "c3_projection": temporal_metadata,
            }
        metadata["quality"] = quality
    return tracks[0], tracks[1], metadata


def empty_hand_track(frame_count: int) -> HandTrack:
    return HandTrack(
        pose=np.zeros((frame_count, 45), dtype=np.float32),
        valid=np.zeros(frame_count, dtype=bool),
        confidence=np.zeros(frame_count, dtype=np.float32),
        imputed=np.ones(frame_count, dtype=bool),
    )


def observed_hand_track(pose: np.ndarray) -> HandTrack:
    count = len(pose)
    return HandTrack(
        pose=np.asarray(pose, dtype=np.float32),
        valid=np.ones(count, dtype=bool),
        confidence=np.ones(count, dtype=np.float32),
        imputed=np.zeros(count, dtype=bool),
    )


def coverage_quality(left: HandTrack, right: HandTrack, good: float, partial: float) -> tuple[str, float]:
    coverage = float(np.mean(np.concatenate([left.valid, right.valid])))
    label = "good" if coverage >= good else "partial" if coverage >= partial else "unusable"
    return label, coverage
