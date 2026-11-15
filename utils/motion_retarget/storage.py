from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .schema import MotionRecord
from .jsonutil import json_safe


def write_motion_npz(path: Path, motion: MotionRecord) -> None:
    motion.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as stream:
            payload = dict(
                schema_version=np.int32(
                    5 if motion.objects is not None and motion.objects.solver_version == "constrained_se3_v4"
                    else (4 if motion.objects is not None and motion.objects.source_hand_landmarks_local is not None
                    else (3 if motion.objects is not None else (2 if motion.continuous_spline else 1))
                    )
                ),
                poses=motion.poses,
                pose_aa_smplsim=motion.pose_aa_smplsim,
                global_orient=motion.global_orient.astype(np.float32),
                body_pose=motion.body_pose.astype(np.float32),
                left_hand_pose=motion.left_hand_pose.astype(np.float32),
                right_hand_pose=motion.right_hand_pose.astype(np.float32),
                jaw_pose=np.zeros((len(motion.timestamps), 3), dtype=np.float32),
                leye_pose=np.zeros((len(motion.timestamps), 3), dtype=np.float32),
                reye_pose=np.zeros((len(motion.timestamps), 3), dtype=np.float32),
                face_valid=np.zeros(len(motion.timestamps), dtype=bool),
                transl=motion.transl.astype(np.float32),
                betas=np.asarray(motion.profile.betas, dtype=np.float32),
                scale=np.float32(motion.scale),
                timestamps=motion.timestamps.astype(np.float64),
                fps=np.float32(motion.fps),
                gender=np.asarray(motion.profile.gender),
                height_m=np.float32(motion.profile.height_m),
                left_hand_valid=motion.left_hand_valid.astype(bool),
                right_hand_valid=motion.right_hand_valid.astype(bool),
                left_hand_confidence=motion.left_hand_confidence.astype(np.float32),
                right_hand_confidence=motion.right_hand_confidence.astype(np.float32),
                left_hand_imputed=motion.left_hand_imputed.astype(bool),
                right_hand_imputed=motion.right_hand_imputed.astype(bool),
                source_to_canonical=motion.source_to_canonical.astype(np.float64),
                source_motion_scale=np.float32(motion.quality["source_motion_scale"]),
                config_hash=np.asarray(motion.quality["config_hash"]),
                input_hash=np.asarray(motion.quality["input_hash"]),
                model_dependency_hash=np.asarray(motion.quality["model_dependency_hash"]),
                dataset=np.asarray(motion.sequence.dataset),
                sequence_id=np.asarray(motion.sequence.sequence_id),
                profile=np.asarray(motion.profile.name),
                actor_id=np.asarray(motion.profile.actor_id or ""),
                profile_source=np.asarray(motion.profile.profile_source),
                betas_source=np.asarray(motion.profile.betas_source),
                hand_pose_convention=np.asarray(motion.hand_pose_convention),
                smplx_model_signature=np.asarray(motion.quality.get("smplx_model_signature", "")),
                quality_json=np.asarray(json.dumps(json_safe(motion.quality), sort_keys=True, allow_nan=False)),
            )
            payload.update(motion.continuous_spline)
            if motion.objects is not None:
                objects = motion.objects
                payload.update(
                    object_ids=np.asarray(objects.ids),
                    object_categories=np.asarray(objects.categories),
                    object_transl=np.asarray(objects.transl, dtype=np.float32),
                    object_quat_xyzw=np.asarray(objects.quat_xyzw, dtype=np.float32),
                    object_valid=np.asarray(objects.valid, dtype=bool),
                    object_source_transl=np.asarray(objects.source_transl, dtype=np.float32),
                    object_source_quat_xyzw=np.asarray(objects.source_quat_xyzw, dtype=np.float32),
                    object_source_timestamps=np.asarray(objects.source_timestamps, dtype=np.float64),
                    object_canonical_timestamps=np.asarray(objects.canonical_timestamps, dtype=np.float64),
                    object_mesh_paths=np.asarray(objects.mesh_paths),
                    object_mesh_sha256=np.asarray(objects.mesh_sha256),
                    object_mesh_scale_to_m=np.asarray(objects.mesh_scale_to_m, dtype=np.float32),
                    left_hand_object_contact=np.asarray(objects.left_contact, dtype=bool),
                    right_hand_object_contact=np.asarray(objects.right_contact, dtype=bool),
                    body_object_contact=np.asarray(objects.body_contact, dtype=bool),
                    left_hand_object_distance_m=np.asarray(objects.left_distance_m, dtype=np.float32),
                    right_hand_object_distance_m=np.asarray(objects.right_distance_m, dtype=np.float32),
                    body_object_distance_m=np.asarray(objects.body_distance_m, dtype=np.float32),
                    body_object_penetration_depth_m=np.asarray(objects.penetration_depth_m, dtype=np.float32),
                    left_hand_object_anchor_local=np.asarray(objects.left_anchor_local, dtype=np.float32),
                    right_hand_object_anchor_local=np.asarray(objects.right_anchor_local, dtype=np.float32),
                    left_hand_contact_vertex_index=np.asarray(objects.left_vertex_index, dtype=np.int32),
                    right_hand_contact_vertex_index=np.asarray(objects.right_vertex_index, dtype=np.int32),
                )
                if objects.source_hand_landmarks_local is not None:
                    payload.update(
                        source_hand_landmark_names=np.asarray(objects.source_hand_landmark_names),
                        source_hand_region_names=np.asarray(objects.source_hand_region_names),
                        source_hand_segment_names=np.asarray(objects.source_hand_segment_names),
                        source_hand_landmarks_local=np.asarray(objects.source_hand_landmarks_local, dtype=np.float32),
                        source_hand_landmark_velocity_local=np.asarray(objects.source_hand_landmark_velocity_local, dtype=np.float32),
                        target_hand_landmarks_local=np.asarray(objects.target_hand_landmarks_local, dtype=np.float32),
                        target_reference_hand_landmarks_local=np.asarray(objects.target_reference_hand_landmarks_local, dtype=np.float32),
                        source_hand_observed=np.asarray(objects.source_hand_observed, dtype=bool),
                        source_hand_contact=np.asarray(objects.source_hand_contact, dtype=bool),
                        annotation_assisted=np.asarray(objects.annotation_assisted, dtype=bool),
                        source_hand_proxy_gap_m=np.asarray(objects.source_hand_proxy_gap_m, dtype=np.float32),
                        dynamic_contact_source_local=np.asarray(objects.contact_source_local, dtype=np.float32),
                        dynamic_contact_anchor_local=np.asarray(objects.contact_anchor_local, dtype=np.float32),
                        dynamic_contact_face=np.asarray(objects.contact_face, dtype=np.int32),
                        dynamic_contact_barycentric=np.asarray(objects.contact_barycentric, dtype=np.float32),
                        dynamic_contact_normal_local=np.asarray(objects.contact_normal_local, dtype=np.float32),
                        dynamic_contact_region=np.asarray(objects.contact_region, dtype=np.int16),
                        dynamic_contact_segment=np.asarray(objects.contact_segment, dtype=np.int16),
                        dynamic_contact_proxy_index=np.asarray(objects.contact_proxy_index, dtype=np.int16),
                        dynamic_contact_valid=np.asarray(objects.contact_valid, dtype=bool),
                        dynamic_contact_constraint_valid=np.asarray(objects.contact_constraint_valid, dtype=bool),
                        dynamic_contact_normal_constraint_valid=np.asarray(
                            objects.contact_normal_constraint_valid
                            if objects.contact_normal_constraint_valid is not None
                            else objects.contact_constraint_valid,
                            dtype=bool,
                        ),
                        dynamic_contact_target_vertex=np.asarray(objects.contact_target_vertex, dtype=np.int32),
                        source_hand_proxy_radii_m=np.asarray(objects.source_hand_proxy_radii_m, dtype=np.float32),
                    )
                if objects.solver_version == "constrained_se3_v4":
                    payload.update(
                        object_solver_version=np.asarray(objects.solver_version),
                        solver_pilot_windows=np.asarray(bool(objects.solver_pilot_windows)),
                        solver_pilot_frame_mask=np.asarray(objects.solver_pilot_frame_mask, dtype=bool),
                        source_palm_position_local=np.asarray(objects.source_palm_position_local, dtype=np.float32),
                        source_palm_quat_xyzw_local=np.asarray(objects.source_palm_quat_xyzw_local, dtype=np.float32),
                        target_palm_position_local=np.asarray(objects.target_palm_position_local, dtype=np.float32),
                        target_palm_quat_xyzw_local=np.asarray(objects.target_palm_quat_xyzw_local, dtype=np.float32),
                        remapped_contact_anchor_local=np.asarray(objects.remapped_contact_anchor_local, dtype=np.float32),
                        remapped_contact_face=np.asarray(objects.remapped_contact_face, dtype=np.int32),
                        remapped_contact_barycentric=np.asarray(objects.remapped_contact_barycentric, dtype=np.float32),
                        remapped_contact_geodesic_m=np.asarray(objects.remapped_contact_geodesic_m, dtype=np.float32),
                        palm_constraint_multiplier=np.asarray(objects.palm_constraint_multiplier, dtype=np.float32),
                        contact_constraint_multiplier=np.asarray(objects.contact_constraint_multiplier, dtype=np.float32),
                        solver_layer_residual_p95=np.asarray(objects.solver_layer_residual_p95, dtype=np.float32),
                        solver_layer_penalty=np.asarray(objects.solver_layer_penalty, dtype=np.float32),
                        solver_layer_multiplier_norm=np.asarray(objects.solver_layer_multiplier_norm, dtype=np.float32),
                        solver_layer_converged=np.asarray(objects.solver_layer_converged, dtype=bool),
                        palm_jacobian_rank=np.asarray(objects.palm_jacobian_rank, dtype=np.int16),
                        palm_jacobian_condition=np.asarray(objects.palm_jacobian_condition, dtype=np.float32),
                        full_body_fallback=np.asarray(objects.full_body_fallback, dtype=bool),
                        support_foot=np.asarray(objects.support_foot, dtype=bool),
                        support_foot_baseline_world=np.asarray(objects.support_foot_baseline_world, dtype=np.float32),
                        support_foot_target_world=np.asarray(objects.support_foot_target_world, dtype=np.float32),
                        root_fallback_translation=np.asarray(objects.root_fallback_translation, dtype=np.float32),
                        root_fallback_rotvec=np.asarray(objects.root_fallback_rotvec, dtype=np.float32),
                        solver_spline_knots=np.asarray(objects.solver_spline_knots, dtype=np.float64),
                        solver_spline_degree=np.int32(objects.solver_spline_degree),
                        solver_spline_body_delta_coeff=np.asarray(objects.solver_spline_body_delta_coeff, dtype=np.float32),
                        solver_spline_root_delta_coeff=np.asarray(objects.solver_spline_root_delta_coeff, dtype=np.float32),
                        solver_spline_translation_delta_coeff=np.asarray(objects.solver_spline_translation_delta_coeff, dtype=np.float32),
                        solver_spline_left_hand_delta_coeff=np.asarray(objects.solver_spline_left_hand_delta_coeff, dtype=np.float32),
                        solver_spline_right_hand_delta_coeff=np.asarray(objects.solver_spline_right_hand_delta_coeff, dtype=np.float32),
                        solver_spline_rotation_basis=np.asarray(objects.solver_spline_rotation_basis),
                    )
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_motion_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}

