from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


Y_UP_TO_Z_UP = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def change_basis_rotvec(rotvec: np.ndarray, basis: np.ndarray = Y_UP_TO_Z_UP) -> np.ndarray:
    matrices = Rotation.from_rotvec(np.asarray(rotvec).reshape(-1, 3)).as_matrix()
    converted = basis @ matrices @ basis.T
    return Rotation.from_matrix(converted).as_rotvec().reshape(np.asarray(rotvec).shape).astype(np.float32)


def canonicalize_bvh_root(
    root_orient: np.ndarray, root_position_cm: np.ndarray,
    target_height_m: float, source_height_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # The SMPL-X template remains in its native Y-up coordinates. The root
    # therefore maps template vectors into the Z-up world by left-multiplying
    # the world basis. Conjugation would rotate the template domain as well.
    source_root = Rotation.from_rotvec(np.asarray(root_orient).reshape(-1, 3)).as_matrix()
    root = Rotation.from_matrix(Y_UP_TO_Z_UP @ source_root).as_rotvec().astype(np.float32)
    translation = (np.asarray(root_position_cm, dtype=np.float64) / 100.0) @ Y_UP_TO_Z_UP.T
    translation *= target_height_m / source_height_m
    transform = np.eye(4, dtype=np.float64)
    # Canonical heading uses the median facing direction during the first 0.5 seconds.
    count = max(1, min(len(root), 15))
    matrices = Rotation.from_rotvec(root[:count]).as_matrix()
    forward = matrices[:, :, 2].mean(axis=0)
    yaw = np.arctan2(forward[1], forward[0])
    heading = Rotation.from_euler("z", -yaw).as_matrix()
    root_matrices = heading @ Rotation.from_rotvec(root).as_matrix()
    root = Rotation.from_matrix(root_matrices).as_rotvec().astype(np.float32)
    translation = translation @ heading.T
    initial_xy = translation[0, :2].copy()
    translation[:, :2] -= initial_xy
    transform[:3, :3] = heading @ Y_UP_TO_Z_UP
    transform[:2, 3] = -initial_xy
    return root, translation.astype(np.float32), transform
