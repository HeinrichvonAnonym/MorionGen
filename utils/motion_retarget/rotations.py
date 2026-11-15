from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def euler_degrees_to_rotvec(values: np.ndarray, order: str) -> np.ndarray:
    # BVH rotation channels describe intrinsic rotations in declaration order.
    return Rotation.from_euler(order.upper(), values, degrees=True).as_rotvec()


def rotvec_to_matrix(values: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(np.asarray(values).reshape(-1, 3)).as_matrix().reshape(*values.shape[:-1], 3, 3)


def matrix_to_rotvec(values: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(values).reshape(-1, 3, 3)).as_rotvec().reshape(*values.shape[:-2], 3)


def slerp_rotvec(times: np.ndarray, values: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    target_times = np.asarray(target_times, dtype=np.float64)
    if len(times) == 1:
        return np.broadcast_to(values[0], (len(target_times), 3)).copy()
    clipped = np.clip(target_times, times[0], times[-1])
    return Slerp(times, Rotation.from_rotvec(values))(clipped).as_rotvec()


def interpolate_pose_gaps(
    poses: np.ndarray, valid: np.ndarray, fps: float, max_gap_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(poses, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    imputed = ~valid.copy()
    max_gap = max(1, int(round(max_gap_seconds * fps)))
    good = np.flatnonzero(valid)
    if len(good) == 0:
        return np.zeros_like(poses, dtype=np.float32), imputed
    poses[: good[0]] = poses[good[0]]
    poses[good[-1] + 1 :] = poses[good[-1]]
    for start, end in zip(good[:-1], good[1:]):
        gap = end - start - 1
        if gap <= 0:
            continue
        if gap <= max_gap:
            ts = np.array([start, end], dtype=np.float64) / fps
            target = np.arange(start + 1, end, dtype=np.float64) / fps
        else:
            # Long missing spans use an ease-in/ease-out bridge as a playable
            # prior; unlike the short-gap interpolation it approaches both
            # observed endpoints with zero scalar slope.
            fraction = np.arange(1, gap + 1, dtype=np.float64) / (gap + 1)
            target = fraction * fraction * (3.0 - 2.0 * fraction)
            ts = np.asarray([0.0, 1.0])
        for joint in range(poses.shape[1] // 3):
            sl = slice(joint * 3, joint * 3 + 3)
            poses[start + 1 : end, sl] = slerp_rotvec(ts, poses[[start, end], sl], target)
        # Long bridges remain imputed and downstream validity is never
        # promoted; smooth playback is not reported as recovered observation.
    return poses.astype(np.float32), imputed
