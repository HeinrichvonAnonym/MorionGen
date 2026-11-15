from __future__ import annotations

import json
import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import RetargetConfig
from .storage import read_motion_npz, write_json_atomic
from .smplx_model import HAND_POSE_CONVENTION


def validate_v5_source_constraints(
    config: RetargetConfig, path: Path, reference,
) -> dict:
    """Rebuild v5 source hands from the released BVH/track/OBJ, not the NPZ masks."""
    from .adapters import bvh_pose, load_hiphi_metadata
    from .objects import load_repaired_mesh, parse_object_track_csv, resolve_object_sources
    from .source_hands import recover_source_hand_contacts

    data = read_motion_npz(path)
    if int(np.asarray(data.get("schema_version", 0))) < 5:
        return {"applicable": False, "passed": True}
    metadata = load_hiphi_metadata(reference)
    native_motion = bvh_pose(reference)[0]
    sources = resolve_object_sources(config.dataset_path("hiphi"), reference, metadata)
    translations, quaternions, meshes = [], [], []
    for source in sources:
        _, translation, quaternion = parse_object_track_csv(
            source.trajectory_path.read_bytes(), expected_frames=len(data["timestamps"]),
        )
        translations.append(translation)
        quaternions.append(quaternion)
        meshes.append(load_repaired_mesh(
            source.mesh_path, float(config.raw["object_pilot"]["mesh_scale_to_m"]),
        ))
    recovered = recover_source_hand_contacts(
        native_motion, np.stack(translations, axis=1), np.stack(quaternions, axis=1),
        meshes, float(data["fps"]), str(metadata.get("text_annotation", "")),
        dict(config.raw["object_pilot"]),
    )
    archived_landmarks = np.asarray(data["source_hand_landmarks_local"], dtype=np.float64)
    archived_contact = np.asarray(data["source_hand_contact"], dtype=bool)
    archived_valid = np.asarray(data["dynamic_contact_valid"], dtype=bool)
    archived_face = np.asarray(data["dynamic_contact_face"], dtype=np.int64)
    archived_source = np.asarray(data["dynamic_contact_source_local"], dtype=np.float64)
    archived_anchor = np.asarray(data["dynamic_contact_anchor_local"], dtype=np.float64)
    archived_barycentric = np.asarray(data["dynamic_contact_barycentric"], dtype=np.float64)
    archived_normal = np.asarray(data["dynamic_contact_normal_local"], dtype=np.float64)
    checks = {
        "source_landmarks_recomputed": bool(
            archived_landmarks.shape == recovered.landmarks_local.shape and
            np.allclose(archived_landmarks, recovered.landmarks_local, atol=1e-6)
        ),
        "source_contact_recomputed": bool(np.array_equal(archived_contact, recovered.contact)),
        "source_contact_valid_recomputed": bool(np.array_equal(archived_valid, recovered.contact_valid)),
        "source_contact_face_recomputed": bool(np.array_equal(
            archived_face[archived_valid], recovered.contact_face[recovered.contact_valid],
        )),
        "source_contact_geometry_recomputed": bool(
            np.allclose(
                archived_source[archived_valid],
                recovered.contact_source_local[recovered.contact_valid], atol=1e-6,
            ) and np.allclose(
                archived_anchor[archived_valid],
                recovered.contact_anchor_local[recovered.contact_valid], atol=1e-6,
            ) and np.allclose(
                archived_barycentric[archived_valid],
                recovered.contact_barycentric[recovered.contact_valid], atol=1e-6,
            ) and np.allclose(
                archived_normal[archived_valid],
                recovered.contact_normal_local[recovered.contact_valid], atol=1e-6,
            )
        ),
    }
    return {
        "applicable": True,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "landmark_max_error_m": float(np.max(np.abs(
            archived_landmarks - recovered.landmarks_local
        ), initial=0.0)),
        "left_contact_frames": int(recovered.contact[:, :, 0].sum()),
        "right_contact_frames": int(recovered.contact[:, :, 1].sum()),
        "source_archive": str(reference.metadata.get("origin_archive", reference.source)),
        "source_member": str(reference.metadata.get("origin_member", reference.member or "")),
    }


def validate_archive(
    path: Path, expected_config_hash: str | None = None,
    max_height_error_m: float | None = None,
) -> dict:
    data = read_motion_npz(path)
    required = {
        "poses": 165, "pose_aa_smplsim": 156, "global_orient": 3,
        "body_pose": 63, "left_hand_pose": 45, "right_hand_pose": 45, "transl": 3,
    }
    frames = len(data["timestamps"])
    checks: dict[str, bool] = {}
    checks["finite"] = all(np.isfinite(data[key]).all() for key in required)
    checks["shapes"] = all(data[key].shape == (frames, width) for key, width in required.items())
    checks["timestamps"] = frames > 0 and (frames == 1 or bool(np.all(np.diff(data["timestamps"]) > 0)))
    checks["masks"] = all(data[key].shape == (frames,) for key in (
        "left_hand_valid", "right_hand_valid", "left_hand_confidence", "right_hand_confidence",
        "left_hand_imputed", "right_hand_imputed", "face_valid",
    ))
    checks["profile_metadata"] = (
        np.asarray(data.get("betas", [])).shape in {(10,), (16,)} and
        float(data.get("scale", 0.0)) > 0 and float(data.get("height_m", 0.0)) > 0 and
        str(data.get("gender", np.asarray("")).item()) in {"male", "female", "neutral"}
    )
    checks["profile_provenance"] = all(
        key in data and bool(str(data[key].item()))
        for key in ("profile_source", "betas_source")
    ) and "actor_id" in data
    checks["hand_confidence"] = all(
        np.isfinite(data[key]).all() and np.all((data[key] >= 0) & (data[key] <= 1))
        for key in ("left_hand_confidence", "right_hand_confidence")
    )
    checks["face_unavailable_consistent"] = (
        not np.asarray(data["face_valid"], dtype=bool).any() and
        all(np.allclose(data[key], 0.0) for key in ("jaw_pose", "leye_pose", "reye_pose"))
    )
    source_transform = np.asarray(data.get("source_to_canonical", np.empty((0, 0))), dtype=np.float64)
    checks["source_transform_rigid"] = bool(
        source_transform.shape == (4, 4) and
        np.allclose(source_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8) and
        np.allclose(source_transform[:3, :3] @ source_transform[:3, :3].T, np.eye(3), atol=1e-6) and
        abs(np.linalg.det(source_transform[:3, :3]) - 1.0) < 1e-6
    )
    if expected_config_hash is not None:
        checks["config_hash"] = (
            "config_hash" in data and str(data["config_hash"].item()) == expected_config_hash
        )
    checks["provenance_hashes"] = all(
        key in data and len(str(data[key].item())) == 64
        for key in ("input_hash", "model_dependency_hash")
    )
    schema_version = int(np.asarray(data.get("schema_version", 0)))
    if schema_version >= 4:
        checks["hand_pose_convention"] = bool(
            "hand_pose_convention" in data and
            str(data["hand_pose_convention"].item()) == HAND_POSE_CONVENTION
        )
        checks["smplx_model_signature"] = bool(
            "smplx_model_signature" in data and
            len(str(data["smplx_model_signature"].item())) == 64
        )
    spline_degree = int(np.asarray(data.get("trajectory_spline_degree", 0)))
    continuity_order = int(np.asarray(data.get("trajectory_continuity_order", -1)))
    spline_pairs = (
        ("c3_body_knots", "c3_body_quat_coeffs", 22, 4),
        ("c3_left_hand_knots", "c3_left_hand_quat_coeffs", 15, 4),
        ("c3_right_hand_knots", "c3_right_hand_quat_coeffs", 15, 4),
        ("c3_translation_knots", "c3_translation_coeffs", 3, None),
    )
    spline_valid = schema_version >= 2 and spline_degree >= 5 and continuity_order >= 4
    for knot_key, coefficient_key, width, tail in spline_pairs:
        if knot_key not in data or coefficient_key not in data:
            spline_valid = False
            continue
        knots = np.asarray(data[knot_key], dtype=np.float64)
        coefficients = np.asarray(data[coefficient_key], dtype=np.float64)
        expected_count = len(knots) - spline_degree - 1
        spline_valid &= bool(
            knots.ndim == 1 and len(knots) >= 2 * (spline_degree + 1) and
            np.all(np.diff(knots) >= 0) and np.isfinite(knots).all() and
            len(coefficients) == expected_count and np.isfinite(coefficients).all()
        )
        if tail is None:
            spline_valid &= coefficients.ndim == 2 and coefficients.shape[1] == width
        else:
            spline_valid &= coefficients.ndim == 3 and coefficients.shape[1:] == (width, tail)
    checks["c4_continuous_trajectory"] = bool(spline_valid)
    if schema_version >= 5 and spline_valid:
        from .constrained_object_v4 import derivative_jump_metrics

        derivative_checks = []
        derivative_metrics = {}
        for knot_key, coefficient_key, _, _ in spline_pairs:
            metrics = derivative_jump_metrics(
                np.asarray(data[knot_key], dtype=np.float64),
                np.asarray(data[coefficient_key], dtype=np.float64),
                spline_degree,
            )
            derivative_metrics[coefficient_key] = metrics
            derivative_checks.append(
                np.isfinite(list(metrics.values())).all() and
                float(metrics["order_3_maximum_jump"]) <= 1e-2
            )
        checks["c3_derivative_continuity"] = bool(all(derivative_checks))
    matrices = Rotation.from_rotvec(data["pose_aa_smplsim"].reshape(-1, 3)).as_matrix()
    determinant_error = float(np.max(np.abs(np.linalg.det(matrices) - 1.0)))
    orthogonality_error = float(np.max(np.abs(matrices @ matrices.transpose(0, 2, 1) - np.eye(3))))
    checks["rotations"] = determinant_error < 1e-5 and orthogonality_error < 1e-5
    quality = json.loads(str(data["quality_json"].item())) if "quality_json" in data else {}
    dataset = str(data.get("dataset", np.asarray("")).item())
    if max_height_error_m is not None:
        checks["profile_height"] = (
            "profile_height_error_m" in quality and
            float(quality["profile_height_error_m"]) <= max_height_error_m
        )
    checks["root_pivot_metadata"] = bool(
        np.asarray(quality.get("target_root_pivot_m", [])).shape == (3,) and
        np.isfinite(np.asarray(quality.get("target_root_pivot_m", []), dtype=np.float64)).all()
    )
    if "body_quality" in quality:
        checks["body_quality"] = quality["body_quality"] != "unusable"
    if "native_hand_fk" in quality:
        checks["native_hand_fk"] = bool(quality["native_hand_fk"].get("passed"))
    if dataset == "hiphi":
        checks["native_hands_fully_observed"] = bool(
            np.asarray(data["left_hand_valid"], dtype=bool).all() and
            np.asarray(data["right_hand_valid"], dtype=bool).all()
        )
    object_metrics = None
    if schema_version >= 3:
        object_fields = (
            "object_ids", "object_categories", "object_transl", "object_quat_xyzw",
            "object_valid", "object_source_transl", "object_source_quat_xyzw",
            "object_source_timestamps", "object_canonical_timestamps", "object_mesh_paths",
            "object_mesh_sha256", "object_mesh_scale_to_m", "left_hand_object_contact",
            "right_hand_object_contact", "body_object_contact", "left_hand_object_distance_m",
            "right_hand_object_distance_m", "body_object_distance_m",
            "body_object_penetration_depth_m",
        )
        checks["object_schema_fields"] = all(key in data for key in object_fields)
        if checks["object_schema_fields"]:
            object_count = len(data["object_ids"])
            checks["object_shapes"] = bool(
                object_count > 0 and data["object_transl"].shape == (frames, object_count, 3) and
                data["object_quat_xyzw"].shape == (frames, object_count, 4) and
                data["object_valid"].shape == (frames, object_count)
            )
            checks["object_coverage"] = bool(np.asarray(data["object_valid"], dtype=bool).all())
            checks["object_timestamps"] = bool(
                np.allclose(data["object_canonical_timestamps"], data["timestamps"], atol=2e-6) and
                (frames == 1 or np.all(np.diff(data["object_source_timestamps"]) > 0))
            )
            source_translation = np.asarray(data["object_source_transl"], dtype=np.float64)
            source_quaternion = np.asarray(data["object_source_quat_xyzw"], dtype=np.float64)
            expected_translation = source_translation @ source_transform[:3, :3].T + source_transform[:3, 3]
            expected_rotation = source_transform[:3, :3] @ Rotation.from_quat(
                source_quaternion.reshape(-1, 4)
            ).as_matrix()
            actual_rotation = Rotation.from_quat(
                np.asarray(data["object_quat_xyzw"], dtype=np.float64).reshape(-1, 4)
            ).as_matrix()
            translation_error = np.linalg.norm(
                np.asarray(data["object_transl"], dtype=np.float64) - expected_translation, axis=2,
            )
            rotation_error = np.degrees(Rotation.from_matrix(
                expected_rotation.transpose(0, 2, 1) @ actual_rotation
            ).magnitude())
            acceptance = quality.get("object_quality", {})
            thresholds = quality.get("object_acceptance_thresholds")
            if thresholds is None:
                thresholds = {
                    "object_translation_m": 0.0001, "object_rotation_deg": 0.01,
                    "contact_median_m": 0.02, "contact_p95_m": 0.05,
                    "penetration_p95_m": 0.01, "penetration_max_m": 0.03,
                    "noncontact_joint_median_m": 0.03, "noncontact_joint_p95_m": 0.08,
                }
            checks["fixed_object_translation"] = float(translation_error.max(initial=0.0)) <= float(thresholds["object_translation_m"])
            checks["fixed_object_rotation"] = float(rotation_error.max(initial=0.0)) <= float(thresholds["object_rotation_deg"])
            contact = np.concatenate([
                np.asarray(data["left_hand_object_distance_m"])[np.asarray(data["left_hand_object_contact"], dtype=bool)],
                np.asarray(data["right_hand_object_distance_m"])[np.asarray(data["right_hand_object_contact"], dtype=bool)],
            ])
            penetration = np.asarray(data["body_object_penetration_depth_m"], dtype=np.float64)
            checks["object_contact_detected"] = bool(len(contact))
            checks["object_contact_distance"] = bool(
                len(contact) and np.median(contact) <= float(thresholds["contact_median_m"]) and
                np.percentile(contact, 95) <= float(thresholds["contact_p95_m"])
            )
            checks["object_nonpenetration"] = bool(
                float(acceptance.get("penetration_p95_m", np.inf)) <=
                float(thresholds["penetration_p95_m"]) and
                float(acceptance.get("penetration_max_m", np.inf)) <=
                float(thresholds["penetration_max_m"])
            )
            checks["noncontact_human_preservation"] = bool(
                acceptance and acceptance.get("noncontact_joint_deviation_median_m") is not None and
                acceptance.get("noncontact_joint_deviation_p95_m") is not None and
                float(acceptance["noncontact_joint_deviation_median_m"]) <= float(thresholds["noncontact_joint_median_m"]) and
                float(acceptance["noncontact_joint_deviation_p95_m"]) <= float(thresholds["noncontact_joint_p95_m"])
            )
            if "normal_angle_p95_deg" in thresholds:
                checks["contact_surface_normal"] = bool(
                    acceptance and
                    float(acceptance.get("normal_angle_deg_median", np.inf)) <= float(thresholds["normal_angle_median_deg"]) and
                    float(acceptance.get("normal_angle_deg_p95", np.inf)) <= float(thresholds["normal_angle_p95_deg"])
                )
                checks["contact_point_to_plane"] = bool(
                    float(acceptance.get("normal_gap_m_p95", np.inf)) <= float(thresholds["normal_gap_p95_m"])
                )
                checks["contact_tangent_anchor"] = bool(
                    float(acceptance.get("tangent_error_m_p95", np.inf)) <= float(thresholds["tangent_error_p95_m"])
                )
                checks["contact_tangent_slip"] = bool(
                    float(acceptance.get("tangent_slip_m_s_p95", np.inf)) <= float(thresholds["tangent_slip_p95_m_s"])
                )
                if "contact_patch_frame_coverage" in thresholds:
                    checks["contact_patch_coverage"] = bool(
                        float(acceptance.get("contact_patch_frame_coverage", -np.inf))
                        >= float(thresholds["contact_patch_frame_coverage"]) and
                        float(acceptance.get("contact_patch_active_vertices_p05", -np.inf))
                        >= float(thresholds["contact_patch_active_vertices_p05"])
                    )
                checks["contact_arm_pose_preservation"] = bool(
                    float(acceptance.get("arm_delta_frame_p95_deg", np.inf)) <= float(thresholds["arm_delta_p95_deg"]) and
                    float(acceptance.get("arm_delta_frame_max_deg", np.inf)) <= float(thresholds["arm_delta_max_deg"])
                )
                checks["contact_mano_pose_preservation"] = bool(
                    float(acceptance.get("mano_delta_frame_p95_deg", np.inf)) <= float(thresholds["mano_delta_p95_deg"]) and
                    float(acceptance.get("mano_delta_frame_max_deg", np.inf)) <= float(thresholds["mano_delta_max_deg"])
                )
            mesh_checks = []
            for relative, expected_hash in zip(data["object_mesh_paths"], data["object_mesh_sha256"]):
                relative_path = Path(str(relative))
                resolved = next((parent / relative_path for parent in path.parents if (parent / relative_path).is_file()), None)
                if resolved is None:
                    mesh_checks.append(False)
                    continue
                digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
                mesh_checks.append(digest == str(expected_hash))
            checks["object_mesh_hashes"] = bool(mesh_checks and all(mesh_checks))
            object_metrics = {
                "translation_max_error_m": float(translation_error.max(initial=0.0)),
                "rotation_max_error_deg": float(rotation_error.max(initial=0.0)),
                "contact_frames": int(len(contact)),
                "contact_median_m": float(np.median(contact)) if len(contact) else None,
                "contact_p95_m": float(np.percentile(contact, 95)) if len(contact) else None,
                "penetration_p95_m": acceptance.get("penetration_p95_m"),
                "penetration_frame_max_p95_m": acceptance.get("penetration_frame_max_p95_m"),
                "penetration_max_m": float(penetration.max(initial=0.0)),
                "normal_angle_median_deg": acceptance.get("normal_angle_deg_median"),
                "normal_angle_p95_deg": acceptance.get("normal_angle_deg_p95"),
                "tangent_slip_p95_m_s": acceptance.get("tangent_slip_m_s_p95"),
                "arm_delta_p95_deg": acceptance.get("arm_delta_frame_p95_deg"),
                "mano_delta_p95_deg": acceptance.get("mano_delta_frame_p95_deg"),
            }
            if schema_version >= 4:
                dynamic_fields = (
                    "source_hand_landmark_names", "source_hand_region_names",
                    "source_hand_segment_names",
                    "source_hand_landmarks_local", "source_hand_landmark_velocity_local",
                    "target_hand_landmarks_local", "source_hand_observed",
                    "target_reference_hand_landmarks_local",
                    "source_hand_contact", "annotation_assisted",
                    "source_hand_proxy_gap_m", "dynamic_contact_source_local",
                    "dynamic_contact_anchor_local", "dynamic_contact_face",
                    "dynamic_contact_barycentric", "dynamic_contact_normal_local",
                    "dynamic_contact_region", "dynamic_contact_valid",
                    "dynamic_contact_segment", "dynamic_contact_proxy_index",
                    "dynamic_contact_constraint_valid",
                    "dynamic_contact_target_vertex",
                    "source_hand_proxy_radii_m",
                )
                if schema_version >= 5:
                    dynamic_fields = (*dynamic_fields, "dynamic_contact_normal_constraint_valid")
                checks["dynamic_hand_schema"] = all(key in data for key in dynamic_fields)
                if checks["dynamic_hand_schema"]:
                    landmarks = len(data["source_hand_landmark_names"])
                    contacts = data["dynamic_contact_valid"].shape[-1]
                    checks["dynamic_hand_shapes"] = bool(
                        landmarks >= 22 and
                        data["source_hand_landmarks_local"].shape == (frames, object_count, 2, landmarks, 3) and
                        data["target_hand_landmarks_local"].shape == (frames, object_count, 2, landmarks, 3) and
                        data["source_hand_contact"].shape == (frames, object_count, 2) and
                        data["dynamic_contact_valid"].shape == (frames, object_count, 2, contacts) and
                        data["dynamic_contact_constraint_valid"].shape == (frames, object_count, 2, contacts) and
                        (
                            schema_version < 5 or
                            data["dynamic_contact_normal_constraint_valid"].shape ==
                            (frames, object_count, 2, contacts)
                        ) and
                        data["dynamic_contact_barycentric"].shape == (frames, object_count, 2, contacts, 3)
                    )
                    barycentric = np.asarray(data["dynamic_contact_barycentric"], dtype=np.float64)
                    dynamic_valid = np.asarray(data["dynamic_contact_valid"], dtype=bool)
                    checks["dynamic_contact_barycentric"] = bool(
                        dynamic_valid.any() and
                        np.allclose(barycentric[dynamic_valid].sum(axis=1), 1.0, atol=1e-4)
                    )
                dynamic = quality.get("dynamic_hand_object_quality", {})
                side_values = dynamic.get("sides", {})
                checks["source_contact_recall"] = bool(
                    side_values and all(
                        int(values.get("source_contact_frames", 0)) > 0 and
                        float(values.get("source_contact_recall", -np.inf)) >= float(thresholds["source_contact_recall"])
                        for values in side_values.values()
                    )
                )
                checks["contact_transition_timing"] = bool(
                    float(dynamic.get("contact_transition_median_frames", np.inf)) <=
                    float(thresholds["contact_transition_median_frames"])
                )
                checks["dynamic_wrist_palm_trajectory"] = bool(
                    float(dynamic.get("wrist_palm_object_local_p95_m", np.inf)) <=
                    float(thresholds["wrist_palm_object_local_p95_m"])
                )
                checks["dynamic_finger_trajectory"] = bool(
                    float(dynamic.get("finger_object_local_p95_m", np.inf)) <=
                    float(thresholds["finger_object_local_p95_m"])
                )
                checks["dynamic_grasp_aperture"] = bool(
                    float(dynamic.get("grasp_aperture_p95_m", np.inf)) <=
                    float(thresholds["grasp_aperture_p95_m"])
                )
                checks["dynamic_hand_rotation"] = bool(
                    side_values and all(
                        float(values.get("bone_direction_p95_deg", np.inf)) <=
                        float(thresholds["hand_rotation_p95_deg"])
                        for values in side_values.values()
                    )
                )
                checks["dynamic_finger_bend"] = bool(
                    side_values and all(
                        float(values.get("finger_bend_p95_deg", np.inf)) <=
                        float(thresholds["finger_bend_p95_deg"])
                        for values in side_values.values()
                    )
                )
                checks["dynamic_thumb_opposition"] = bool(
                    side_values and all(
                        float(values.get("thumb_opposition_p95_m", np.inf)) <=
                        float(thresholds["thumb_opposition_p95_m"])
                        for values in side_values.values()
                    )
                )
                checks["dynamic_multiregion_contact"] = bool(
                    float(dynamic.get("active_contact_regions_p05", -np.inf)) >=
                    float(thresholds["contact_patch_minimum_regions"])
                )
                checks["dynamic_surface_contact"] = bool(
                    float(dynamic.get("contact_surface_gap_median", np.inf)) <=
                    float(thresholds["contact_median_m"]) and
                    float(dynamic.get("contact_surface_gap_p95", np.inf)) <=
                    float(thresholds["contact_p95_m"])
                )
                object_metrics["dynamic_hand_object_quality"] = dynamic
            if schema_version >= 5:
                solver_windows = quality.get("object_joint_retarget", {}).get("windows", [])
                checks["surface_correspondence_converged"] = bool(
                    solver_windows and all(
                        bool(window.get("correspondence_converged", False))
                        for window in solver_windows
                    )
                )
                checks["solver_hierarchy_converged"] = bool(
                    solver_windows and all(
                        window.get("tiers") and all(
                            bool(tier.get("converged", False)) for tier in window["tiers"]
                        )
                        for window in solver_windows
                    )
                )
                v5_fields = (
                    "object_solver_version", "source_palm_position_local",
                    "solver_pilot_windows", "solver_pilot_frame_mask",
                    "source_palm_quat_xyzw_local", "target_palm_position_local",
                    "target_palm_quat_xyzw_local", "remapped_contact_anchor_local",
                    "remapped_contact_face", "remapped_contact_barycentric",
                    "remapped_contact_geodesic_m", "palm_constraint_multiplier",
                    "contact_constraint_multiplier", "solver_layer_residual_p95",
                    "solver_layer_penalty", "solver_layer_multiplier_norm",
                    "solver_layer_converged", "palm_jacobian_rank",
                    "palm_jacobian_condition", "full_body_fallback", "support_foot",
                    "support_foot_baseline_world", "support_foot_target_world",
                    "root_fallback_translation", "root_fallback_rotvec",
                    "solver_spline_knots", "solver_spline_degree",
                    "solver_spline_body_delta_coeff", "solver_spline_root_delta_coeff",
                    "solver_spline_translation_delta_coeff",
                    "solver_spline_left_hand_delta_coeff",
                    "solver_spline_right_hand_delta_coeff", "solver_spline_rotation_basis",
                )
                checks["constrained_v5_schema"] = bool(
                    all(key in data for key in v5_fields) and
                    str(data.get("object_solver_version", np.asarray("")).item()) ==
                    "constrained_se3_v4"
                )
                if checks["constrained_v5_schema"]:
                    contacts = data["dynamic_contact_valid"].shape[-1]
                    solver_degree = int(np.asarray(data["solver_spline_degree"]))
                    solver_knots = np.asarray(data["solver_spline_knots"], dtype=np.float64)
                    solver_controls = len(solver_knots) - solver_degree - 1
                    v5_shapes = {
                        "source_palm_position_local": (frames, object_count, 2, 3),
                        "solver_pilot_frame_mask": (frames,),
                        "source_palm_quat_xyzw_local": (frames, object_count, 2, 4),
                        "target_palm_position_local": (frames, object_count, 2, 3),
                        "target_palm_quat_xyzw_local": (frames, object_count, 2, 4),
                        "remapped_contact_anchor_local": (frames, object_count, 2, contacts, 3),
                        "remapped_contact_face": (frames, object_count, 2, contacts),
                        "remapped_contact_barycentric": (frames, object_count, 2, contacts, 3),
                        "remapped_contact_geodesic_m": (frames, object_count, 2, contacts),
                        "palm_constraint_multiplier": (frames, object_count, 2, 6),
                        "contact_constraint_multiplier": (frames, object_count, 2, contacts, 3),
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
                        "solver_spline_knots": (solver_controls + solver_degree + 1,),
                        "solver_spline_body_delta_coeff": (solver_controls, 21, 3),
                        "solver_spline_root_delta_coeff": (solver_controls, 3),
                        "solver_spline_translation_delta_coeff": (solver_controls, 3),
                        "solver_spline_left_hand_delta_coeff": (solver_controls, 15, 3),
                        "solver_spline_right_hand_delta_coeff": (solver_controls, 15, 3),
                    }
                    checks["constrained_v5_shapes"] = bool(all(
                        np.asarray(data[key]).shape == shape and
                        (np.asarray(data[key]).dtype.kind not in "fc" or np.isfinite(data[key]).all())
                        for key, shape in v5_shapes.items()
                    ) and solver_degree == 5 and solver_controls > 0 and
                        str(data["solver_spline_rotation_basis"].item()) ==
                        "baseline_composed_right_SO3_rotvec_delta")
                    solver_derivative_metrics = {}
                    solver_derivative_ok = True
                    for key in (
                        "solver_spline_body_delta_coeff", "solver_spline_root_delta_coeff",
                        "solver_spline_translation_delta_coeff",
                        "solver_spline_left_hand_delta_coeff",
                        "solver_spline_right_hand_delta_coeff",
                    ):
                        metrics = derivative_jump_metrics(
                            solver_knots, np.asarray(data[key], dtype=np.float64), solver_degree,
                        )
                        solver_derivative_metrics[key] = metrics
                        solver_derivative_ok &= bool(
                            np.isfinite(list(metrics.values())).all() and
                            float(metrics["order_3_maximum_jump"]) <= 1e-2
                        )
                    checks["solver_control_c3_continuity"] = bool(solver_derivative_ok)
                    active = np.asarray(data["source_hand_contact"], dtype=bool)
                    if bool(np.asarray(data["solver_pilot_windows"])):
                        active &= np.asarray(
                            data["solver_pilot_frame_mask"], dtype=bool,
                        )[:, None, None]
                    source_position = np.asarray(data["source_palm_position_local"], dtype=np.float64)
                    target_position = np.asarray(data["target_palm_position_local"], dtype=np.float64)
                    source_rotation = Rotation.from_quat(
                        np.asarray(data["source_palm_quat_xyzw_local"], dtype=np.float64).reshape(-1, 4)
                    ).as_matrix().reshape(frames, object_count, 2, 3, 3)
                    target_rotation = Rotation.from_quat(
                        np.asarray(data["target_palm_quat_xyzw_local"], dtype=np.float64).reshape(-1, 4)
                    ).as_matrix().reshape(frames, object_count, 2, 3, 3)
                    position_error = np.linalg.norm(target_position - source_position, axis=-1)
                    orientation_error = np.degrees(Rotation.from_matrix(
                        (source_rotation.transpose(0, 1, 2, 4, 3) @ target_rotation).reshape(-1, 3, 3)
                    ).magnitude()).reshape(frames, object_count, 2)
                    palm_sides = {}
                    for side, label in enumerate(("left", "right")):
                        mask = active[:, :, side]
                        palm_sides[label] = {
                            "position_p95_m": float(np.percentile(position_error[:, :, side][mask], 95)) if mask.any() else float("inf"),
                            "orientation_p95_deg": float(np.percentile(orientation_error[:, :, side][mask], 95)) if mask.any() else float("inf"),
                        }
                    checks["palm_se3_per_side"] = bool(all(
                        value["position_p95_m"] <= float(thresholds["palm_position_p95_m"]) and
                        value["orientation_p95_deg"] <= float(thresholds["palm_orientation_p95_deg"])
                        for value in palm_sides.values()
                    ))
                    remap = np.asarray(data["remapped_contact_geodesic_m"], dtype=np.float64)
                    remap_valid = np.asarray(data["dynamic_contact_constraint_valid"], dtype=bool)
                    checks["surface_remap_bound"] = bool(
                        remap_valid.any() and np.all(remap[remap_valid] >= 0.0) and
                        float(remap[remap_valid].max(initial=0.0)) <=
                        float(thresholds["surface_remap_max_geodesic_m"]) + 1e-7
                    )
                    barycentric = np.asarray(data["remapped_contact_barycentric"], dtype=np.float64)
                    checks["surface_remap_barycentric"] = bool(
                        np.all(barycentric[remap_valid] >= -1e-5) and
                        np.allclose(barycentric[remap_valid].sum(axis=1), 1.0, atol=1e-4)
                    )
                    root_translation = np.linalg.norm(
                        np.asarray(data["root_fallback_translation"], dtype=np.float64), axis=1,
                    )
                    root_rotation = np.degrees(np.linalg.norm(
                        np.asarray(data["root_fallback_rotvec"], dtype=np.float64), axis=1,
                    ))
                    fallback = np.asarray(data["full_body_fallback"], dtype=bool)
                    checks["fallback_root_bounds"] = bool(
                        (not fallback.any()) or (
                            root_translation[fallback].max(initial=0.0) <=
                            float(thresholds["fallback_root_translation_m"]) + 1e-6 and
                            root_rotation[fallback].max(initial=0.0) <=
                            float(thresholds["fallback_root_rotation_deg"]) + 1e-4
                        )
                    )
                    support = np.asarray(data["support_foot"], dtype=bool)
                    support_active = support & fallback[:, None]
                    baseline_feet = np.asarray(data["support_foot_baseline_world"], dtype=np.float64)
                    target_feet = np.asarray(data["support_foot_target_world"], dtype=np.float64)
                    support_position = np.linalg.norm(target_feet - baseline_feet, axis=-1).max(axis=-1)
                    baseline_direction = baseline_feet[:, :, 1] - baseline_feet[:, :, 0]
                    target_direction = target_feet[:, :, 1] - target_feet[:, :, 0]
                    baseline_direction /= np.maximum(
                        np.linalg.norm(baseline_direction, axis=-1, keepdims=True), 1e-8,
                    )
                    target_direction /= np.maximum(
                        np.linalg.norm(target_direction, axis=-1, keepdims=True), 1e-8,
                    )
                    support_orientation = np.degrees(np.arccos(np.clip(
                        np.sum(baseline_direction * target_direction, axis=-1), -1.0, 1.0,
                    )))
                    support_position_p95 = (
                        float(np.percentile(support_position[support_active], 95))
                        if support_active.any() else 0.0
                    )
                    support_orientation_p95 = (
                        float(np.percentile(support_orientation[support_active], 95))
                        if support_active.any() else 0.0
                    )
                    checks["fallback_support_feet"] = bool(
                        support_position_p95 <= float(thresholds["support_foot_position_p95_m"]) and
                        support_orientation_p95 <= float(thresholds["support_foot_orientation_p95_deg"])
                    )
                    object_metrics["palm_se3_sides"] = palm_sides
                    object_metrics["surface_remap_max_geodesic_m"] = float(
                        remap[remap_valid].max(initial=0.0)
                    )
                    object_metrics["support_foot_position_p95_m"] = support_position_p95
                    object_metrics["support_foot_orientation_p95_deg"] = support_orientation_p95
                    object_metrics["c3_derivative_metrics"] = derivative_metrics
                    object_metrics["solver_control_c3_derivative_metrics"] = solver_derivative_metrics
    if dataset == "andy":
        checks["andy_body_source_traced"] = quality.get("body_mode") in {
            "qualisys_marker_ik", "gvhmr_video_fallback", "xsens_bvh_fallback",
        }
    return {
        "path": str(path), "frames": frames, "fps": float(data["fps"]),
        "schema_version": schema_version, "trajectory_continuity_order": continuity_order,
        "determinant_error": determinant_error, "orthogonality_error": orthogonality_error,
        "checks": checks, "passed": all(checks.values()), "quality": quality,
        "objects": object_metrics,
    }


def run_mujoco_validation(config: RetargetConfig, path: Path, report: Path, render_path: Path | None = None) -> dict:
    python = Path(config.raw["environments"]["mujoco_python"])
    worker = Path(__file__).resolve().parents[2] / "python/scripts/validate_motion_mujoco.py"
    command = [str(python), str(worker), "--config", str(config.path), "--input", str(path), "--report", str(report)]
    if render_path is not None:
        command += ["--render", str(render_path)]
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    if render_path is not None:
        environment.setdefault("MUJOCO_GL", "glfw")
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment,
    )
    # The worker uses exit status 1 for a completed kinematic replay that did
    # not pass a quality gate.  Keep that structured report so the ledger and
    # manifest expose the actual FK/contact/limit metrics.  A non-zero status
    # without a report is an execution failure.
    if report.is_file():
        payload = json.loads(report.read_text())
        payload["worker_returncode"] = int(result.returncode)
        return payload
    if result.returncode:
        raise RuntimeError(f"MuJoCo worker failed ({result.returncode}):\n{result.stdout[-8000:]}")
    raise RuntimeError(f"MuJoCo worker produced no report: {report}\n{result.stdout[-8000:]}")


def run_smplx_mesh_render(config: RetargetConfig, path: Path, render_path: Path) -> dict:
    """Render the actual SMPL-X surface; MuJoCo collision geoms are not a visual result."""
    python = Path(config.raw["environments"]["mujoco_python"])
    worker = Path(__file__).resolve().parents[2] / "python/scripts/render_smplx_mesh.py"
    command = [
        str(python), str(worker), "--config", str(config.path),
        "--input", str(path), "--output", str(render_path),
    ]
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    environment.setdefault("PYOPENGL_PLATFORM", "egl")
    result = subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=environment,
    )
    if result.returncode:
        raise RuntimeError(f"SMPL-X mesh renderer failed ({result.returncode}):\n{result.stdout[-8000:]}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not render_path.is_file() or not lines:
        raise RuntimeError(f"SMPL-X mesh renderer produced no video: {render_path}\n{result.stdout[-8000:]}")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"SMPL-X mesh renderer returned no receipt:\n{result.stdout[-8000:]}") from exc


def run_penetration_region_audit(config: RetargetConfig, path: Path, report: Path) -> dict:
    """Separate contact-hand soft compression from unexplained penetration."""
    python = Path(config.raw["environments"]["mujoco_python"])
    worker = Path(__file__).resolve().parents[2] / "python/scripts/audit_object_penetration_regions.py"
    result = subprocess.run(
        [str(python), str(worker), "--config", str(config.path),
         "--input", str(path), "--report", str(report)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    if report.is_file():
        payload = json.loads(report.read_text())
        payload["worker_returncode"] = int(result.returncode)
        return payload
    raise RuntimeError(
        f"Penetration-region audit failed ({result.returncode}):\n{result.stdout[-8000:]}"
    )


def validate_motion(
    config: RetargetConfig, path: Path, with_mujoco: bool = True,
    render_path: Path | None = None, expected_config_hash: str | None = None,
    source_reference=None,
) -> dict:
    archive = validate_archive(
        path, expected_config_hash=expected_config_hash or config.digest,
        max_height_error_m=float(config.raw["quality"]["max_height_error_m"]),
    )
    report_path = path.with_suffix(".validation.json")
    report: dict = {"archive": archive, "scope": "Kinematic replay and physical sanity only; no controller tracking claim."}
    if int(archive.get("schema_version", 0)) >= 5:
        if source_reference is None:
            report["source_recompute"] = {
                "applicable": True, "passed": False,
                "error": "v5 validation requires the staged released BVH/object reference",
            }
        else:
            report["source_recompute"] = validate_v5_source_constraints(
                config, path, source_reference,
            )
        archive["checks"]["source_constraints_recomputed"] = bool(
            report["source_recompute"].get("passed")
        )
        archive["passed"] = all(archive["checks"].values())
    if int(archive.get("schema_version", 0)) >= 4 and archive.get("objects") is not None:
        penetration_audit = run_penetration_region_audit(
            config, path, report_path.with_suffix(".penetration_regions.json"),
        )
        report["penetration_region_audit"] = penetration_audit
        # Preserve the strict result inside the audit, but allow the explicit
        # soft-tissue interpretation only when *all* residual overlap is
        # limited to contact hands.  Non-hand and non-contact-hand overlap
        # remain capped by the original 3 mm geometric threshold.
        archive["checks"]["object_nonpenetration"] = bool(penetration_audit["passed"])
        archive["passed"] = all(archive["checks"].values())
        archive["objects"]["penetration_acceptance_mode"] = penetration_audit["acceptance_mode"]
        archive["objects"]["strict_geometric_pass"] = penetration_audit["strict_geometric_pass"]
        archive["objects"]["soft_tissue_tolerance_pass"] = penetration_audit["soft_tissue_tolerance_pass"]
    # A failed archive is exactly the case for which an independent MuJoCo
    # replay is most useful.  Do not short-circuit physical/FK diagnostics on
    # an archive quality-gate failure.
    if with_mujoco:
        report["mujoco"] = run_mujoco_validation(config, path, report_path.with_suffix(".mujoco.json"))
    report["passed"] = archive["passed"] and (not with_mujoco or bool(report.get("mujoco", {}).get("passed")))
    # Rendering is evidence, not a reward for passing.  Failed samples need
    # the same visual artifact so a reviewer can inspect the actual defect.
    if render_path is not None:
        report["visualization"] = run_smplx_mesh_render(config, path, render_path)
    write_json_atomic(report_path, report)
    return report
