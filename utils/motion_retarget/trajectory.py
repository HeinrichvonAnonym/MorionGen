from __future__ import annotations

import json

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline, make_lsq_spline
from scipy.spatial.transform import Rotation


def _continuous_quaternions(pose: np.ndarray) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(len(pose), -1, 3)
    quaternions = Rotation.from_rotvec(values.reshape(-1, 3)).as_quat().reshape(len(values), -1, 4)
    for frame in range(1, len(quaternions)):
        flip = np.sum(quaternions[frame - 1] * quaternions[frame], axis=1) < 0
        quaternions[frame, flip] *= -1.0
    return quaternions


def _lsq_knots(times: np.ndarray, interval_seconds: float, degree: int) -> np.ndarray:
    start, end = float(times[0]), float(times[-1])
    internal = np.arange(start + interval_seconds, end - 0.25 * interval_seconds, interval_seconds)
    while len(internal) + degree + 1 > len(times):
        internal = internal[::2]
    return np.concatenate([
        np.full(degree + 1, start), internal, np.full(degree + 1, end),
    ])


def c4_bspline_basis(
    times: np.ndarray, interval_seconds: float, degree: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense design matrix for an open degree-5 spline control trajectory."""
    times = np.asarray(times, dtype=np.float64)
    knots = _lsq_knots(times, interval_seconds, degree)
    basis = BSpline.design_matrix(times, knots, degree, extrapolate=False).toarray()
    return basis.astype(np.float32), knots.astype(np.float64)


def c3_smooth_rotvec(
    times: np.ndarray, pose: np.ndarray, knot_seconds: float,
    degree: int = 5, robust_iterations: int = 3, huber_degrees: float = 8.0,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Robustly project local rotations onto a degree-5 (C4, hence C3) spline."""
    times = np.asarray(times, dtype=np.float64)
    pose = np.asarray(pose, dtype=np.float64)
    if len(times) < degree + 1 or float(times[-1] - times[0]) <= knot_seconds:
        return pose.astype(np.float32), {
            "applied": False, "reason": "sequence_too_short", "degree": degree,
        }
    quaternions = _continuous_quaternions(pose)
    filtered = np.empty_like(quaternions)
    knots = _lsq_knots(times, knot_seconds, degree)
    delta = np.deg2rad(huber_degrees)
    base_weights = (
        np.ones(len(times), dtype=np.float64) if sample_weights is None
        else np.asarray(sample_weights, dtype=np.float64)
    )
    if base_weights.shape != (len(times),) or not np.isfinite(base_weights).all() or np.any(base_weights <= 0):
        raise ValueError("Rotation spline sample_weights must be finite positive per-frame values")
    residuals = []
    for joint in range(quaternions.shape[1]):
        source = quaternions[:, joint]
        weights = base_weights.copy()
        estimate = source
        for _ in range(max(1, robust_iterations)):
            spline = make_lsq_spline(times, source, knots, k=degree, w=weights, axis=0)
            estimate = spline(times)
            estimate /= np.maximum(np.linalg.norm(estimate, axis=1, keepdims=True), 1e-8)
            dot = np.abs(np.sum(estimate * source, axis=1))
            angular_residual = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
            weights = base_weights * np.minimum(1.0, delta / np.maximum(angular_residual, 1e-8))
        filtered[:, joint] = estimate
        residuals.append(angular_residual)
    rotvec = Rotation.from_quat(filtered.reshape(-1, 4)).as_rotvec().reshape(pose.shape)
    residual = np.stack(residuals, axis=1)
    return rotvec.astype(np.float32), {
        "applied": True, "degree": degree, "continuity_order": degree - 1,
        "knot_seconds": float(knot_seconds), "control_count": int(len(knots) - degree - 1),
        "robust_iterations": int(robust_iterations), "huber_degrees": float(huber_degrees),
        "weighted_samples": bool(sample_weights is not None),
        "median_input_residual_degrees": float(np.degrees(np.median(residual))),
        "p95_input_residual_degrees": float(np.degrees(np.percentile(residual, 95))),
    }


def c3_smooth_relative_rotvec(
    times: np.ndarray, baseline: np.ndarray, optimized: np.ndarray, knot_seconds: float,
    degree: int = 5, robust_iterations: int = 3, huber_degrees: float = 8.0,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """C4-project only the optimized SO(3) delta, then compose with the baseline."""
    baseline = np.asarray(baseline, dtype=np.float64)
    optimized = np.asarray(optimized, dtype=np.float64)
    if baseline.shape != optimized.shape or baseline.ndim < 2 or baseline.shape[-1] % 3:
        raise ValueError("Relative rotation smoothing requires matching poses with 3D joint blocks")
    base_matrix = Rotation.from_rotvec(baseline.reshape(-1, 3)).as_matrix()
    optimized_matrix = Rotation.from_rotvec(optimized.reshape(-1, 3)).as_matrix()
    relative = Rotation.from_matrix(
        base_matrix.transpose(0, 2, 1) @ optimized_matrix
    ).as_rotvec().reshape(baseline.shape)
    smoothed_relative, metadata = c3_smooth_rotvec(
        times, relative, knot_seconds, degree, robust_iterations, huber_degrees,
        sample_weights,
    )
    relative_matrix = Rotation.from_rotvec(smoothed_relative.reshape(-1, 3)).as_matrix()
    composed = Rotation.from_matrix(base_matrix @ relative_matrix).as_rotvec().reshape(baseline.shape)
    metadata = dict(metadata)
    metadata["representation"] = "baseline_composed_relative_SO3_delta"
    return composed.astype(np.float32), metadata


def c3_smooth_translation(
    times: np.ndarray, values: np.ndarray, knot_seconds: float,
    degree: int = 5, robust_iterations: int = 3, huber_m: float = 0.02,
) -> tuple[np.ndarray, dict]:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if len(times) < degree + 1 or float(times[-1] - times[0]) <= knot_seconds:
        return values.astype(np.float32), {
            "applied": False, "reason": "sequence_too_short", "degree": degree,
        }
    knots = _lsq_knots(times, knot_seconds, degree)
    weights = np.ones(len(times), dtype=np.float64)
    estimate = values
    for _ in range(max(1, robust_iterations)):
        spline = make_lsq_spline(times, values, knots, k=degree, w=weights, axis=0)
        estimate = spline(times)
        residual = np.linalg.norm(estimate - values, axis=1)
        weights = np.minimum(1.0, huber_m / np.maximum(residual, 1e-8))
    return estimate.astype(np.float32), {
        "applied": True, "degree": degree, "continuity_order": degree - 1,
        "knot_seconds": float(knot_seconds), "control_count": int(len(knots) - degree - 1),
        "robust_iterations": int(robust_iterations), "huber_m": float(huber_m),
        "median_input_residual_m": float(np.median(residual)),
        "p95_input_residual_m": float(np.percentile(residual, 95)),
    }


def _rotation_interpolation_payload(
    prefix: str, times: np.ndarray, pose: np.ndarray, degree: int,
) -> dict[str, np.ndarray]:
    quaternions = _continuous_quaternions(pose)
    spline = make_interp_spline(
        times, quaternions.reshape(len(times), -1), k=degree, axis=0,
    )
    coefficients = spline.c.reshape(len(spline.c), quaternions.shape[1], 4)
    return {
        f"c3_{prefix}_knots": spline.t.astype(np.float64),
        f"c3_{prefix}_quat_coeffs": coefficients.astype(np.float32),
    }


def c3_archive_payload(
    times: np.ndarray, global_orient: np.ndarray, body_pose: np.ndarray,
    left_hand_pose: np.ndarray, right_hand_pose: np.ndarray, translation: np.ndarray,
    degree: int = 5,
) -> dict[str, np.ndarray]:
    """Serialize an exact continuous representation through the delivered samples.

    Quaternion splines are evaluated component-wise and normalized. Continuous
    normalization preserves C4 wherever the quaternion norm is non-zero; all
    delivered sample quaternions are unit length and sign-continuous.
    """
    times = np.asarray(times, dtype=np.float64)
    if len(times) < degree + 1:
        return {}
    relative_times = times - times[0]
    body = np.concatenate([global_orient, body_pose], axis=1)
    payload: dict[str, np.ndarray] = {
        "trajectory_continuity_order": np.int32(degree - 1),
        "trajectory_spline_degree": np.int32(degree),
        "trajectory_time_origin": np.float64(times[0]),
        "trajectory_representation_json": np.asarray(json.dumps({
            "basis": "degree-5 B-spline",
            "translation": "Euclidean B-spline",
            "rotation": "sign-continuous quaternion B-spline, normalized at evaluation",
            "continuity": "C4 on the open time domain; satisfies required C3",
        }, sort_keys=True)),
    }
    payload.update(_rotation_interpolation_payload("body", relative_times, body, degree))
    payload.update(_rotation_interpolation_payload("left_hand", relative_times, left_hand_pose, degree))
    payload.update(_rotation_interpolation_payload("right_hand", relative_times, right_hand_pose, degree))
    translation_spline = make_interp_spline(relative_times, translation, k=degree, axis=0)
    payload["c3_translation_knots"] = translation_spline.t.astype(np.float64)
    payload["c3_translation_coeffs"] = translation_spline.c.astype(np.float32)
    return payload
