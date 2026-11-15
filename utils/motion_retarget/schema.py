from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .smplx_model import HAND_POSE_CONVENTION


@dataclass(frozen=True)
class RetargetProfile:
    name: str
    gender: str
    height_m: float
    betas: tuple[float, ...]
    profile_source: str = "configured"
    actor_id: str | None = None
    betas_source: str = "configured"

    def __post_init__(self) -> None:
        if self.gender not in {"male", "female", "neutral"}:
            raise ValueError(f"Unsupported gender: {self.gender}")
        if self.height_m <= 0:
            raise ValueError("height_m must be positive")
        if len(self.betas) not in {10, 16}:
            raise ValueError("SMPL-X betas must contain 10 or 16 values")


@dataclass(frozen=True)
class SequenceRef:
    dataset: str
    sequence_id: str
    source: Path
    member: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ObjectTrackRecord:
    """Fixed HiPHI rigid-object observations attached to one human motion."""

    ids: np.ndarray
    categories: np.ndarray
    transl: np.ndarray
    quat_xyzw: np.ndarray
    valid: np.ndarray
    source_transl: np.ndarray
    source_quat_xyzw: np.ndarray
    source_timestamps: np.ndarray
    canonical_timestamps: np.ndarray
    mesh_paths: np.ndarray
    mesh_sha256: np.ndarray
    mesh_scale_to_m: np.ndarray
    left_contact: np.ndarray
    right_contact: np.ndarray
    body_contact: np.ndarray
    left_distance_m: np.ndarray
    right_distance_m: np.ndarray
    body_distance_m: np.ndarray
    penetration_depth_m: np.ndarray
    left_anchor_local: np.ndarray
    right_anchor_local: np.ndarray
    left_vertex_index: np.ndarray
    right_vertex_index: np.ndarray
    source_hand_landmark_names: np.ndarray | None = None
    source_hand_region_names: np.ndarray | None = None
    source_hand_segment_names: np.ndarray | None = None
    source_hand_landmarks_local: np.ndarray | None = None
    source_hand_landmark_velocity_local: np.ndarray | None = None
    target_hand_landmarks_local: np.ndarray | None = None
    target_reference_hand_landmarks_local: np.ndarray | None = None
    source_hand_observed: np.ndarray | None = None
    source_hand_contact: np.ndarray | None = None
    annotation_assisted: np.ndarray | None = None
    source_hand_proxy_gap_m: np.ndarray | None = None
    contact_source_local: np.ndarray | None = None
    contact_anchor_local: np.ndarray | None = None
    contact_face: np.ndarray | None = None
    contact_barycentric: np.ndarray | None = None
    contact_normal_local: np.ndarray | None = None
    contact_region: np.ndarray | None = None
    contact_segment: np.ndarray | None = None
    contact_proxy_index: np.ndarray | None = None
    contact_valid: np.ndarray | None = None
    contact_constraint_valid: np.ndarray | None = None
    contact_normal_constraint_valid: np.ndarray | None = None
    contact_target_vertex: np.ndarray | None = None
    source_hand_proxy_radii_m: np.ndarray | None = None
    solver_version: str | None = None
    solver_pilot_windows: bool | None = None
    solver_pilot_frame_mask: np.ndarray | None = None
    source_palm_position_local: np.ndarray | None = None
    source_palm_quat_xyzw_local: np.ndarray | None = None
    target_palm_position_local: np.ndarray | None = None
    target_palm_quat_xyzw_local: np.ndarray | None = None
    remapped_contact_anchor_local: np.ndarray | None = None
    remapped_contact_face: np.ndarray | None = None
    remapped_contact_barycentric: np.ndarray | None = None
    remapped_contact_geodesic_m: np.ndarray | None = None
    palm_constraint_multiplier: np.ndarray | None = None
    contact_constraint_multiplier: np.ndarray | None = None
    solver_layer_residual_p95: np.ndarray | None = None
    solver_layer_penalty: np.ndarray | None = None
    solver_layer_multiplier_norm: np.ndarray | None = None
    solver_layer_converged: np.ndarray | None = None
    palm_jacobian_rank: np.ndarray | None = None
    palm_jacobian_condition: np.ndarray | None = None
    full_body_fallback: np.ndarray | None = None
    support_foot: np.ndarray | None = None
    support_foot_baseline_world: np.ndarray | None = None
    support_foot_target_world: np.ndarray | None = None
    root_fallback_translation: np.ndarray | None = None
    root_fallback_rotvec: np.ndarray | None = None
    solver_spline_knots: np.ndarray | None = None
    solver_spline_degree: int | None = None
    solver_spline_body_delta_coeff: np.ndarray | None = None
    solver_spline_root_delta_coeff: np.ndarray | None = None
    solver_spline_translation_delta_coeff: np.ndarray | None = None
    solver_spline_left_hand_delta_coeff: np.ndarray | None = None
    solver_spline_right_hand_delta_coeff: np.ndarray | None = None
    solver_spline_rotation_basis: str | None = None

    def validate(self, frames: int) -> None:
        n = len(self.ids)
        expected = {
            "categories": (n,), "transl": (frames, n, 3),
            "quat_xyzw": (frames, n, 4), "valid": (frames, n),
            "source_transl": (frames, n, 3), "source_quat_xyzw": (frames, n, 4),
            "source_timestamps": (frames,), "canonical_timestamps": (frames,),
            "mesh_paths": (n,), "mesh_sha256": (n,), "mesh_scale_to_m": (n,),
            "left_contact": (frames, n), "right_contact": (frames, n),
            "body_contact": (frames, n), "left_distance_m": (frames, n),
            "right_distance_m": (frames, n), "body_distance_m": (frames, n),
            "penetration_depth_m": (frames, n),
            "left_anchor_local": (frames, n, 3), "right_anchor_local": (frames, n, 3),
            "left_vertex_index": (frames, n), "right_vertex_index": (frames, n),
        }
        if n == 0:
            raise ValueError("object record must contain at least one object")
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"object {name} shape {value.shape}, expected {shape}")
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise ValueError(f"object {name} contains non-finite values")
        norms = np.linalg.norm(np.asarray(self.quat_xyzw), axis=2)
        if not np.allclose(norms[np.asarray(self.valid, dtype=bool)], 1.0, atol=1e-4):
            raise ValueError("object quaternions are not normalized")
        if frames > 1 and not np.all(np.diff(self.source_timestamps) > 0):
            raise ValueError("object source timestamps must be strictly increasing")
        if self.source_hand_landmarks_local is not None:
            landmark_count = len(np.asarray(self.source_hand_landmark_names))
            contact_count = np.asarray(self.contact_valid).shape[-1]
            dynamic_expected = {
                "source_hand_region_names": (6,),
                "source_hand_segment_names": (21,),
                "source_hand_landmarks_local": (frames, n, 2, landmark_count, 3),
                "source_hand_landmark_velocity_local": (frames, n, 2, landmark_count, 3),
                "target_hand_landmarks_local": (frames, n, 2, landmark_count, 3),
                "target_reference_hand_landmarks_local": (frames, n, 2, landmark_count, 3),
                "source_hand_observed": (frames, n, 2),
                "source_hand_contact": (frames, n, 2),
                "annotation_assisted": (frames, n, 2),
                "source_hand_proxy_gap_m": (frames, n, 2),
                "contact_source_local": (frames, n, 2, contact_count, 3),
                "contact_anchor_local": (frames, n, 2, contact_count, 3),
                "contact_face": (frames, n, 2, contact_count),
                "contact_barycentric": (frames, n, 2, contact_count, 3),
                "contact_normal_local": (frames, n, 2, contact_count, 3),
                "contact_region": (frames, n, 2, contact_count),
                "contact_segment": (frames, n, 2, contact_count),
                "contact_proxy_index": (frames, n, 2, contact_count),
                "contact_valid": (frames, n, 2, contact_count),
                "contact_constraint_valid": (frames, n, 2, contact_count),
                "contact_normal_constraint_valid": (frames, n, 2, contact_count),
                "contact_target_vertex": (frames, n, 2, contact_count),
                "source_hand_proxy_radii_m": (2, 6),
            }
            if landmark_count < 22:
                raise ValueError("dynamic source hand record must include wrist, palm, joints and tips")
            for name, shape in dynamic_expected.items():
                raw_value = getattr(self, name)
                if name == "contact_normal_constraint_valid" and raw_value is None:
                    raw_value = self.contact_constraint_valid
                value = np.asarray(raw_value)
                if value.shape != shape:
                    raise ValueError(f"object {name} shape {value.shape}, expected {shape}")
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError(f"object {name} contains non-finite values")
        if self.solver_version == "constrained_se3_v4":
            contact_count = np.asarray(self.contact_valid).shape[-1]
            spline_degree = int(self.solver_spline_degree)
            spline_knots = np.asarray(self.solver_spline_knots)
            control_count = len(spline_knots) - spline_degree - 1
            if spline_degree != 5 or control_count <= 0:
                raise ValueError("v5 solver trajectory must use a non-empty quintic B-spline")
            v5_expected = {
                "source_palm_position_local": (frames, n, 2, 3),
                "source_palm_quat_xyzw_local": (frames, n, 2, 4),
                "target_palm_position_local": (frames, n, 2, 3),
                "target_palm_quat_xyzw_local": (frames, n, 2, 4),
                "remapped_contact_anchor_local": (frames, n, 2, contact_count, 3),
                "remapped_contact_face": (frames, n, 2, contact_count),
                "remapped_contact_barycentric": (frames, n, 2, contact_count, 3),
                "remapped_contact_geodesic_m": (frames, n, 2, contact_count),
                "palm_constraint_multiplier": (frames, n, 2, 6),
                "contact_constraint_multiplier": (frames, n, 2, contact_count, 3),
                "solver_layer_residual_p95": (5, 2),
                "solver_layer_penalty": (5, 2),
                "solver_layer_multiplier_norm": (5, 2),
                "solver_layer_converged": (5, 2),
                "palm_jacobian_rank": (frames, 2),
                "palm_jacobian_condition": (frames, 2),
                "full_body_fallback": (frames,),
                "support_foot": (frames, 2),
                "support_foot_baseline_world": (frames, 2, 2, 3),
                "support_foot_target_world": (frames, 2, 2, 3),
                "root_fallback_translation": (frames, 3),
                "root_fallback_rotvec": (frames, 3),
                "solver_pilot_frame_mask": (frames,),
                "solver_spline_knots": (control_count + spline_degree + 1,),
                "solver_spline_body_delta_coeff": (control_count, 21, 3),
                "solver_spline_root_delta_coeff": (control_count, 3),
                "solver_spline_translation_delta_coeff": (control_count, 3),
                "solver_spline_left_hand_delta_coeff": (control_count, 15, 3),
                "solver_spline_right_hand_delta_coeff": (control_count, 15, 3),
            }
            for name, shape in v5_expected.items():
                value = np.asarray(getattr(self, name))
                if value.shape != shape:
                    raise ValueError(f"object {name} shape {value.shape}, expected {shape}")
                if value.dtype.kind in "fc" and not np.isfinite(value).all():
                    raise ValueError(f"object {name} contains non-finite values")
            quaternion_norm = np.linalg.norm(self.source_palm_quat_xyzw_local, axis=-1)
            if not np.allclose(quaternion_norm, 1.0, atol=1e-4):
                raise ValueError("source palm quaternions are not normalized")
            quaternion_norm = np.linalg.norm(self.target_palm_quat_xyzw_local, axis=-1)
            if not np.allclose(quaternion_norm, 1.0, atol=1e-4):
                raise ValueError("target palm quaternions are not normalized")
            if self.solver_spline_rotation_basis != "baseline_composed_right_SO3_rotvec_delta":
                raise ValueError("v5 solver rotation basis is missing or unsupported")
            if self.solver_pilot_windows is None:
                raise ValueError("v5 solver must declare whether it is a short-window pilot")


@dataclass
class MotionRecord:
    sequence: SequenceRef
    profile: RetargetProfile
    fps: float
    timestamps: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    transl: np.ndarray
    scale: float
    left_hand_valid: np.ndarray
    right_hand_valid: np.ndarray
    left_hand_confidence: np.ndarray
    right_hand_confidence: np.ndarray
    left_hand_imputed: np.ndarray
    right_hand_imputed: np.ndarray
    source_to_canonical: np.ndarray
    quality: dict[str, Any] = field(default_factory=dict)
    continuous_spline: dict[str, np.ndarray] = field(default_factory=dict)
    objects: ObjectTrackRecord | None = None
    hand_pose_convention: str = HAND_POSE_CONVENTION

    def validate(self) -> None:
        t = len(self.timestamps)
        expected = {
            "global_orient": (t, 3),
            "body_pose": (t, 63),
            "left_hand_pose": (t, 45),
            "right_hand_pose": (t, 45),
            "transl": (t, 3),
            "source_to_canonical": (4, 4),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape}, expected {shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")
        for name in (
            "left_hand_valid", "right_hand_valid", "left_hand_confidence",
            "right_hand_confidence", "left_hand_imputed", "right_hand_imputed",
        ):
            if np.asarray(getattr(self, name)).shape != (t,):
                raise ValueError(f"{name} must contain one value per frame")
        if t == 0 or self.fps <= 0:
            raise ValueError("motion must have frames and a positive fps")
        if t > 1 and not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("timestamps must be strictly increasing")
        if self.continuous_spline:
            order = int(np.asarray(self.continuous_spline["trajectory_continuity_order"]))
            degree = int(np.asarray(self.continuous_spline["trajectory_spline_degree"]))
            if order < 4 or degree < 5:
                raise ValueError("continuous trajectory must provide quintic C4 continuity")
        if self.objects is not None:
            self.objects.validate(t)
        if self.hand_pose_convention != HAND_POSE_CONVENTION:
            raise ValueError(f"Unsupported hand pose convention: {self.hand_pose_convention}")

    @property
    def poses(self) -> np.ndarray:
        zeros = np.zeros((len(self.timestamps), 9), dtype=np.float32)
        return np.concatenate(
            [self.global_orient, self.body_pose, zeros, self.left_hand_pose, self.right_hand_pose],
            axis=1,
        ).astype(np.float32, copy=False)

    @property
    def pose_aa_smplsim(self) -> np.ndarray:
        return np.concatenate(
            [self.global_orient, self.body_pose, self.left_hand_pose, self.right_hand_pose], axis=1
        ).astype(np.float32, copy=False)

