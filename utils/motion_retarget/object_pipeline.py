from __future__ import annotations

import json
import hashlib
import os
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .adapters import bvh_pose, load_hiphi_metadata
from .config import RetargetConfig, load_config
from .fitting import _device_lock
from .ledger import Ledger
from .objects import (
    SignedDistanceGrid, build_signed_distance_grid, canonicalize_object_track,
    contact_hysteresis, copy_hashed_asset, load_repaired_mesh, parse_object_track_csv,
    resolve_object_sources, sha256_file,
)
from .pipeline import (
    build_motion, input_dependency_hash, model_dependency_hash, output_path, run_config_hash,
)
from .profiles import profiles_for_reference
from .quality import native_hand_fk_quality
from .schema import MotionRecord, ObjectTrackRecord, RetargetProfile, SequenceRef
from .smplx_model import create_smplx_model, smplx_hand_landmarks
from .smplx_model import smplx_model_signature
from .source_hands import (
    SourceHandContacts, hand_vertex_segments, read_source_hand_contacts,
    recover_source_hand_contacts, write_source_hand_contacts,
)
from .storage import write_motion_npz
from .trajectory import (
    c3_archive_payload, c3_smooth_relative_rotvec, c3_smooth_translation,
    c4_bspline_basis,
)


def _device(config: RetargetConfig):
    import torch
    from .gpu import assigned_gpu_index

    requested = str(config.raw["fitting"].get("device", "auto"))
    use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda"}
    index = assigned_gpu_index(torch.cuda.device_count()) if use_cuda else 0
    return torch.device(f"cuda:{index}" if use_cuda else "cpu"), use_cuda, index


def _model(config: RetargetConfig, profile: RetargetProfile, device):
    return create_smplx_model(
        config.smplx_model_dir, profile.gender, len(profile.betas), device=device,
    ).eval()


SOURCE_CONTACT_SETTING_KEYS = (
    "mesh_scale_to_m",
    "source_contact_points_per_hand",
    "source_contact_relative_speed_m_s",
    "source_contact_enter_m",
    "source_contact_exit_m",
    "source_contact_minimum_frames",
    "source_contact_switch_hysteresis_m",
    "annotation_max_gap_m",
    "annotation_max_speed_m_s",
    "annotation_max_bridge_s",
)


def source_contact_dependency_hash(
    reference: SequenceRef, sources, text_annotation: str, settings: dict[str, Any],
) -> str:
    """Hash only immutable source-hand/contact dependencies.

    HiPHI members are streamed into a new temporary directory on every run, so
    absolute paths and the global solver config digest are deliberately absent.
    This lets expensive raw-BVH/OBJ contact recovery survive changes to IK
    weights, correspondence policy, validation, and rendering.
    """
    payload = {
        # v4_2 orients thin/open OBJ normal axes toward the capsule centre,
        # not a possibly penetrating surface sample that can cross the face.
        "schema": "hiphi_source_hand_contacts_v4_2",
        "motion_sha256": sha256_file(Path(reference.source)),
        "text_annotation": str(text_annotation),
        "settings": {
            key: settings.get(key) for key in SOURCE_CONTACT_SETTING_KEYS
        },
        "objects": [
            {
                "object_id": str(source.object_id),
                "category": str(source.category),
                "mesh_sha256": sha256_file(Path(source.mesh_path)),
                "trajectory_sha256": sha256_file(Path(source.trajectory_path)),
            }
            for source in sources
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _target_hand_landmarks(joints, vertices):
    """Compatibility wrapper around the canonical true-fingertip mapping."""
    return smplx_hand_landmarks(joints, vertices)


def _contact_segment_landmarks(segment: int) -> tuple[int, int]:
    """Return the endpoint landmarks for one of the 21 hand-proxy segments."""
    value = int(segment)
    if value == 0:
        return 0, 1
    if 1 <= value <= 5:
        return 1, 2 + 4 * (value - 1)
    if 6 <= value < 21:
        finger, phalange = divmod(value - 6, 3)
        start = 2 + 4 * finger + phalange
        return start, start + 1
    raise ValueError(f"Invalid hand proxy segment {segment}")


def _segment_longitudinal_coordinate(
    points: np.ndarray, start: np.ndarray, end: np.ndarray,
) -> np.ndarray:
    """Project surface points onto a proxy bone and return clipped [0,1] position."""
    values = np.asarray(points, dtype=np.float64)
    first = np.asarray(start, dtype=np.float64)
    last = np.asarray(end, dtype=np.float64)
    axis = last - first
    denominator = float(np.dot(axis, axis))
    if denominator <= 1e-12:
        raise ValueError("Degenerate hand proxy segment")
    return np.clip(
        np.einsum("...i,i->...", values - first, axis) / denominator,
        0.0, 1.0,
    )


def _transport_segment_surface_to_reference(
    vertices: np.ndarray,
    current_start: np.ndarray, current_end: np.ndarray,
    current_palm_matrix: np.ndarray,
    reference_start: np.ndarray, reference_end: np.ndarray,
    reference_palm_matrix: np.ndarray,
) -> np.ndarray:
    """Move target-hand surface samples onto the dynamic reference skeleton.

    Initial object-contact binding must not use the distance from the coarse
    human-only baseline to the object: before reach IK that distance is mostly
    an arm-placement error and can select the wrong side/part of a phalanx.
    Preserve each SMPL-X vertex's longitudinal coordinate and radial vector in
    the anatomical palm frame, then place that surface identity around the
    source-driven target-shape reference bone.
    """
    values = np.asarray(vertices, dtype=np.float64)
    current_first = np.asarray(current_start, dtype=np.float64)
    current_last = np.asarray(current_end, dtype=np.float64)
    reference_first = np.asarray(reference_start, dtype=np.float64)
    reference_last = np.asarray(reference_end, dtype=np.float64)
    alpha = _segment_longitudinal_coordinate(
        values, current_first, current_last,
    )
    current_axis = current_last - current_first
    current_axis_points = current_first + alpha[:, None] * current_axis
    radial_signature = (
        values - current_axis_points
    ) @ np.asarray(current_palm_matrix, dtype=np.float64)
    reference_axis = reference_last - reference_first
    return (
        reference_first + alpha[:, None] * reference_axis +
        radial_signature @ np.asarray(reference_palm_matrix, dtype=np.float64).T
    ).astype(np.float32)


def _proxy_identity_candidate_order(
    source_surface: np.ndarray,
    source_segment_start: np.ndarray, source_segment_end: np.ndarray,
    source_palm_matrix: np.ndarray, target_vertices: np.ndarray,
    target_segment_start: np.ndarray, target_segment_end: np.ndarray,
    target_palm_matrix: np.ndarray,
    angle_tolerance_degrees: float, longitudinal_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank same-segment vertices by a pose-independent proxy identity.

    The source proxy supplies an outward radial direction in the anatomical
    palm frame and a position along its capsule axis. Candidates inside both
    physical tolerances form a hard feasible set. Object distance is applied
    by the caller only inside that set, so an unconverged arm pose cannot pick
    the opposite side of a finger.
    """
    source_surface = np.asarray(source_surface, dtype=np.float64)
    source_start = np.asarray(source_segment_start, dtype=np.float64)
    source_end = np.asarray(source_segment_end, dtype=np.float64)
    source_alpha = float(_segment_longitudinal_coordinate(
        source_surface[None], source_start, source_end,
    )[0])
    source_axis_point = source_start + source_alpha * (source_end - source_start)
    source_radial = source_surface - source_axis_point
    length = float(np.linalg.norm(source_radial))
    if length <= 1e-8:
        raise ValueError("Source hand proxy has no radial contact direction")
    source_signature = (source_radial / length) @ np.asarray(
        source_palm_matrix, dtype=np.float64,
    )
    target_vertices = np.asarray(target_vertices, dtype=np.float64)
    target_start = np.asarray(target_segment_start, dtype=np.float64)
    target_end = np.asarray(target_segment_end, dtype=np.float64)
    target_alpha = _segment_longitudinal_coordinate(
        target_vertices, target_start, target_end,
    )
    target_axis_points = (
        target_start[None] + target_alpha[:, None] *
        (target_end - target_start)[None]
    )
    target_radial = target_vertices - target_axis_points
    target_radial_length = np.linalg.norm(target_radial, axis=1, keepdims=True)
    candidate_signature = (
        target_radial / np.maximum(target_radial_length, 1e-8)
    ) @ np.asarray(target_palm_matrix, dtype=np.float64)
    cosine = np.clip(candidate_signature @ source_signature, -1.0, 1.0)
    angular = np.arccos(cosine)
    angular_normalized = angular / max(
        np.deg2rad(float(angle_tolerance_degrees)), 1e-8,
    )
    longitudinal_normalized = np.abs(target_alpha - source_alpha) / max(
        float(longitudinal_tolerance), 1e-8,
    )
    violation = np.maximum(angular_normalized, longitudinal_normalized)
    feasible = violation <= 1.0
    order = np.r_[
        np.flatnonzero(feasible),
        np.flatnonzero(~feasible)[np.argsort(violation[~feasible], kind="stable")],
    ]
    return order.astype(np.int64), feasible, violation.astype(np.float32)


def _assign_source_contacts_to_target(
    config: RetargetConfig, motion: MotionRecord, source: SourceHandContacts,
    object_translation: np.ndarray, object_quaternion: np.ndarray,
    target_reference_landmarks_local: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Bind source proxy identities to the homologous target-hand surface."""
    import torch
    from .constrained_object_v4 import anatomical_palm_frames

    device, use_cuda, device_index = _device(config)
    model = _model(config, motion.profile, device)
    segments = {
        side: hand_vertex_segments(model, side) for side in ("left", "right")
    }
    frames, objects, _, maximum = source.contact_valid.shape
    target_vertex = np.full((frames, objects, 2, maximum), -1, dtype=np.int32)
    constraint_valid = np.zeros((frames, objects, 2, maximum), dtype=bool)
    selection_score = np.full((frames, objects, 2, maximum), np.inf, dtype=np.float32)
    hysteresis = float(config.raw["object_pilot"].get(
        "target_vertex_switch_hysteresis_m", 0.003,
    ))
    previous: dict[tuple[int, int, int], int] = {}
    source_palm_matrix = np.zeros((frames, objects, 2, 3, 3), dtype=np.float32)
    for object_index in range(objects):
        _, source_palm_matrix[:, object_index] = anatomical_palm_frames(
            source.landmarks_local[:, object_index]
        )
    reference_landmarks = (
        None if target_reference_landmarks_local is None
        else np.asarray(target_reference_landmarks_local, dtype=np.float32)
    )
    reference_palm_matrix = None
    if reference_landmarks is not None:
        if reference_landmarks.shape != source.landmarks_local.shape:
            raise ValueError("Target reference hand landmarks do not match source shape")
        reference_palm_matrix = np.zeros(
            (frames, objects, 2, 3, 3), dtype=np.float32,
        )
        for object_index in range(objects):
            _, reference_palm_matrix[:, object_index] = anatomical_palm_frames(
                reference_landmarks[:, object_index]
            )
    angular_tolerance = float(config.raw["object_pilot"].get(
        "target_proxy_radial_tolerance_degrees", 30.0,
    ))
    longitudinal_tolerance = float(config.raw["object_pilot"].get(
        "target_proxy_longitudinal_tolerance", 0.34,
    ))
    batch = int(config.raw["object_pilot"]["forward_batch_frames"])
    with _device_lock(config, use_cuda, device_index), torch.no_grad():
        for start in range(0, frames, batch):
            end = min(frames, start + batch)
            output = _forward(model, motion, slice(start, end), device)
            vertices = output.vertices.cpu().numpy() * motion.scale + motion.transl[start:end, None]
            target_landmarks = (
                _target_hand_landmarks(output.joints, output.vertices).cpu().numpy() *
                motion.scale + motion.transl[start:end, None, None]
            )
            _, target_palm_matrix = anatomical_palm_frames(target_landmarks)
            for local_frame, frame in enumerate(range(start, end)):
                for object_index in range(objects):
                    rotation = Rotation.from_quat(
                        object_quaternion[frame, object_index],
                    ).as_matrix()
                    center = object_translation[frame, object_index]
                    for side_index, side in enumerate(("left", "right")):
                        if not source.contact[frame, object_index, side_index]:
                            for key in [
                                key for key in previous
                                if key[0] == object_index and key[1] == side_index
                            ]:
                                previous.pop(key)
                            continue
                        used: set[int] = set()
                        for contact_index in np.flatnonzero(
                            source.contact_valid[frame, object_index, side_index]
                        ):
                            proxy_index = int(source.contact_proxy_index[
                                frame, object_index, side_index, contact_index
                            ])
                            segment = int(source.contact_segment[
                                frame, object_index, side_index, contact_index
                            ])
                            candidates = segments[side][segment]
                            anchor_local = source.contact_anchor_local[
                                frame, object_index, side_index, contact_index
                            ]
                            anchor_world = anchor_local @ rotation.T + center
                            first, last = _contact_segment_landmarks(segment)
                            if reference_landmarks is None:
                                candidate_geometry = vertices[local_frame, candidates]
                                candidate_start = target_landmarks[
                                    local_frame, side_index, first
                                ]
                                candidate_end = target_landmarks[
                                    local_frame, side_index, last
                                ]
                                candidate_palm_matrix = target_palm_matrix[
                                    local_frame, side_index
                                ]
                                distance = np.linalg.norm(
                                    candidate_geometry - anchor_world, axis=1,
                                )
                            else:
                                reference = reference_landmarks[
                                    frame, object_index, side_index
                                ]
                                candidate_geometry = _transport_segment_surface_to_reference(
                                    vertices[local_frame, candidates],
                                    target_landmarks[local_frame, side_index, first],
                                    target_landmarks[local_frame, side_index, last],
                                    target_palm_matrix[local_frame, side_index],
                                    reference[first], reference[last],
                                    reference_palm_matrix[
                                        frame, object_index, side_index
                                    ],
                                )
                                candidate_start = reference[first]
                                candidate_end = reference[last]
                                candidate_palm_matrix = reference_palm_matrix[
                                    frame, object_index, side_index
                                ]
                                distance = np.linalg.norm(
                                    candidate_geometry - anchor_local, axis=1,
                                )
                            identity_order, identity_feasible, identity_violation = (
                                _proxy_identity_candidate_order(
                                    source.contact_source_local[
                                        frame, object_index, side_index, contact_index
                                    ],
                                    source.landmarks_local[
                                        frame, object_index, side_index, first
                                    ],
                                    source.landmarks_local[
                                        frame, object_index, side_index, last
                                    ],
                                    source_palm_matrix[
                                        frame, object_index, side_index
                                    ],
                                    candidate_geometry,
                                    candidate_start,
                                    candidate_end,
                                    candidate_palm_matrix,
                                    angular_tolerance,
                                    longitudinal_tolerance,
                                )
                            )
                            key = (object_index, side_index, proxy_index)
                            feasible_order = identity_order[identity_feasible[identity_order]]
                            if len(feasible_order):
                                order = feasible_order[np.argsort(
                                    distance[feasible_order], kind="stable",
                                )]
                            else:
                                order = identity_order
                            chosen = None
                            retained = previous.get(key)
                            if retained is not None and retained not in used:
                                retained_index = np.flatnonzero(candidates == retained)
                                if (
                                    len(retained_index) and
                                    identity_feasible[int(retained_index[0])] and
                                    distance[int(retained_index[0])] <=
                                    distance[int(order[0])] + hysteresis
                                ):
                                    chosen = int(retained)
                            if chosen is None:
                                chosen = next((
                                    int(candidates[index]) for index in order
                                    if int(candidates[index]) not in used
                                ), None)
                            if chosen is None:
                                # A narrow segment can expose fewer distinct
                                # SMPL-X surface vertices than released proxy
                                # samples. Never steal another proxy's vertex
                                # or cross anatomy to manufacture a constraint.
                                continue
                            used.add(chosen)
                            previous[key] = chosen
                            target_vertex[frame, object_index, side_index, contact_index] = chosen
                            chosen_index = int(np.flatnonzero(candidates == chosen)[0])
                            selection_score[frame, object_index, side_index, contact_index] = float(
                                distance[chosen_index] + 1e-3 * identity_violation[chosen_index]
                            )
    maximum_constraints = int(config.raw["object_pilot"].get(
        "target_contact_points_per_hand", 4,
    ))
    minimum_regions = int(config.raw["object_pilot"]["acceptance"].get(
        "contact_patch_minimum_regions", 2,
    ))
    for frame in range(frames):
        for object_index in range(objects):
            for side_index in range(2):
                slots = np.flatnonzero(
                    source.contact_valid[frame, object_index, side_index] &
                    (target_vertex[frame, object_index, side_index] >= 0) &
                    np.isfinite(selection_score[frame, object_index, side_index])
                )
                if not len(slots):
                    continue
                order = slots[np.argsort(selection_score[frame, object_index, side_index, slots])]
                chosen_slots: list[int] = []
                chosen_regions: set[int] = set()
                for slot in order:
                    region = int(source.contact_region[frame, object_index, side_index, slot])
                    if region not in chosen_regions:
                        chosen_slots.append(int(slot)); chosen_regions.add(region)
                    if len(chosen_regions) >= minimum_regions:
                        break
                for slot in order:
                    if int(slot) not in chosen_slots:
                        chosen_slots.append(int(slot))
                    if len(chosen_slots) >= maximum_constraints:
                        break
                constraint_valid[frame, object_index, side_index, chosen_slots] = True
    result: dict[str, np.ndarray] = {
        "left_contact": source.contact[:, :, 0],
        "right_contact": source.contact[:, :, 1],
        "source_hand_landmarks_local": source.landmarks_local,
        "target_reference_hand_landmarks_local": (
            source.landmarks_local if target_reference_landmarks_local is None
            else target_reference_landmarks_local
        ),
        "source_hand_landmark_velocity_local": source.landmark_velocity_local,
        "source_hand_observed": source.observed,
        "annotation_assisted": source.annotation_assisted,
        "source_hand_proxy_gap_m": source.proxy_gap_m,
        "dynamic_contact_source_local": source.contact_source_local,
        "dynamic_contact_face": source.contact_face,
        "dynamic_contact_barycentric": source.contact_barycentric,
        "dynamic_contact_region": source.contact_region,
        "dynamic_contact_segment": source.contact_segment,
        "dynamic_contact_proxy_index": source.contact_proxy_index,
        "dynamic_contact_valid": source.contact_valid,
        "dynamic_contact_constraint_valid": constraint_valid,
        "dynamic_contact_target_vertex": target_vertex,
    }
    for side_index, side in enumerate(("left", "right")):
        result[f"{side}_patch_vertex"] = target_vertex[:, :, side_index]
        result[f"{side}_patch_anchor"] = source.contact_anchor_local[:, :, side_index]
        result[f"{side}_patch_normal"] = source.contact_normal_local[:, :, side_index]
        result[f"{side}_patch_valid"] = (
            constraint_valid[:, :, side_index] &
            (target_vertex[:, :, side_index] >= 0)
        )
    return result


def _measure_target_hand_landmarks(
    config: RetargetConfig, motion: MotionRecord,
    object_translation: np.ndarray, object_quaternion: np.ndarray,
) -> np.ndarray:
    import torch

    device, use_cuda, device_index = _device(config)
    model = _model(config, motion.profile, device)
    frames, objects = object_translation.shape[:2]
    result = np.zeros((frames, objects, 2, 22, 3), dtype=np.float32)
    batch = int(config.raw["object_pilot"]["forward_batch_frames"])
    with _device_lock(config, use_cuda, device_index), torch.no_grad():
        for start in range(0, frames, batch):
            end = min(frames, start + batch)
            output = _forward(model, motion, slice(start, end), device)
            landmarks = _target_hand_landmarks(output.joints, output.vertices) * motion.scale
            landmarks = landmarks + torch.tensor(
                motion.transl[start:end, None, None], device=device,
            )
            world = landmarks.cpu().numpy()
            for object_index in range(objects):
                rotation = Rotation.from_quat(
                    object_quaternion[start:end, object_index],
                ).as_matrix()
                result[start:end, object_index] = np.einsum(
                    "tslj,tjk->tslk",
                    world - object_translation[start:end, object_index, None, None],
                    rotation,
                )
    return result


def _adapt_hand_landmark_lengths(
    raw_landmarks: np.ndarray, coarse_target: np.ndarray, contact: np.ndarray,
    source_palm_matrix: np.ndarray | None = None,
    desired_palm_matrix: np.ndarray | None = None,
    coarse_palm_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Apply target bone lengths along the released per-frame finger directions.

    The earlier implementation translated ``coarse_target`` to the raw palm
    but retained its coarse MANO finger directions. That made the reference
    self-consistent while visibly opening the right hand away from the released
    BVH grasp. MCP placement is fixed by the target palm shape and therefore
    stays coarse-target-relative; every controllable phalange direction from
    MCP through the true fingertip comes from ``raw_landmarks``.
    """
    raw = np.asarray(raw_landmarks, dtype=np.float64)
    expected = np.asarray(coarse_target, dtype=np.float64).copy()
    if expected.shape != raw.shape or contact.shape != raw.shape[:3]:
        raise ValueError("Target hand reference shapes disagree")
    del contact
    calibrated = any(
        value is not None for value in (
            source_palm_matrix, desired_palm_matrix, coarse_palm_matrix,
        )
    )
    if calibrated:
        if any(value is None for value in (
            source_palm_matrix, desired_palm_matrix, coarse_palm_matrix,
        )):
            raise ValueError("Source, desired, and coarse palm frames must be supplied together")
        source_frame = np.asarray(source_palm_matrix, dtype=np.float64)
        desired_frame = np.asarray(desired_palm_matrix, dtype=np.float64)
        coarse_frame = np.asarray(coarse_palm_matrix, dtype=np.float64)
        expected_frame_shape = (*raw.shape[:-2], 3, 3)
        if not (
            source_frame.shape == desired_frame.shape == coarse_frame.shape ==
            expected_frame_shape
        ):
            raise ValueError("Calibrated palm frame shapes disagree")
        # Palm translation comes from the released motion. Wrist and MCP
        # locations retain target SMPL-X hand shape, expressed in the same
        # calibrated dynamic palm frame used by the palm 6D constraint.
        raw_palm = raw[..., 1, :]
        coarse_palm = expected[..., 1, :].copy()
        expected[..., 1, :] = raw_palm
        for landmark in (0, 2, 6, 10, 14, 18):
            offset_local = np.einsum(
                "...i,...ij->...j",
                coarse_target[..., landmark, :] - coarse_palm,
                coarse_frame,
            )
            expected[..., landmark, :] = raw_palm + np.einsum(
                "...i,...ji->...j", offset_local, desired_frame,
            )
    else:
        offset = raw[:, :, :, 1] - expected[:, :, :, 1]
        expected += offset[:, :, :, None]
        expected[:, :, :, 0] = raw[:, :, :, 0]
        expected[:, :, :, 1] = raw[:, :, :, 1]
    for base in (2, 6, 10, 14, 18):
        raw_chain = raw[..., base:base + 4, :]
        raw_bones = np.diff(raw_chain, axis=-2)
        raw_unit = raw_bones / np.maximum(
            np.linalg.norm(raw_bones, axis=-1, keepdims=True), 1e-12,
        )
        if calibrated:
            raw_local = np.einsum(
                "...si,...ij->...sj", raw_unit, source_frame,
            )
            raw_unit = np.einsum(
                "...si,...ji->...sj", raw_local, desired_frame,
            )
        target_lengths = np.linalg.norm(
            np.diff(coarse_target[..., base:base + 4, :], axis=-2), axis=-1,
        )
        for segment in range(3):
            expected[..., base + segment + 1, :] = (
                expected[..., base + segment, :] +
                raw_unit[..., segment, :] * target_lengths[..., segment, None]
            )
    return expected.astype(np.float32)


def _target_shape_hand_reference(
    config: RetargetConfig, motion: MotionRecord, source: SourceHandContacts,
    object_translation: np.ndarray, object_quaternion: np.ndarray,
) -> np.ndarray:
    """Retain released joint directions while adapting immutable target bone lengths."""
    from .constrained_object_v4 import (
        anatomical_palm_frames, robust_palm_frame_calibration,
    )

    coarse_target = _measure_target_hand_landmarks(
        config, motion, object_translation, object_quaternion,
    )
    source_matrix = np.zeros((*source.landmarks_local.shape[:-2], 3, 3), dtype=np.float32)
    coarse_matrix = np.zeros_like(source_matrix)
    desired_matrix = np.zeros_like(source_matrix)
    for object_index in range(source.landmarks_local.shape[1]):
        _, source_matrix[:, object_index] = anatomical_palm_frames(
            source.landmarks_local[:, object_index],
        )
        _, coarse_matrix[:, object_index] = anatomical_palm_frames(
            coarse_target[:, object_index],
        )
        calibration = robust_palm_frame_calibration(
            source_matrix[:, object_index], coarse_matrix[:, object_index],
            eligible=~np.asarray(source.contact[:, object_index], dtype=bool),
        )
        desired_matrix[:, object_index] = (
            source_matrix[:, object_index] @ calibration[None]
        )
    return _adapt_hand_landmark_lengths(
        source.landmarks_local, coarse_target, source.contact,
        source_matrix, desired_matrix, coarse_matrix,
    )


def _hand_dynamic_shape_errors(
    source: np.ndarray, target: np.ndarray, active: np.ndarray,
) -> dict[str, float]:
    """Shape-independent BVH-to-SMPL-X finger motion errors.

    Bone directions and inter-bone bend angles do not reward a solver merely
    for hitting a MANO delta clamp, and true fingertip landmarks make the DIP
    rotation observable.  Thumb opposition is retained in metres because it
    is the grasp dimension that was visibly wrong in the rejected gold sample.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    active = np.asarray(active, dtype=bool)
    if source.shape != target.shape or source.shape[-2:] != (22, 3):
        raise ValueError("Hand dynamic landmark shapes disagree")

    def unit(vectors: np.ndarray) -> np.ndarray:
        return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)

    directions, bends = [], []
    for base in (2, 6, 10, 14, 18):
        # The palm-to-MCP vector is governed by actor/target hand shape and
        # wrist pose, not by a MANO finger joint.  Absolute palm and MCP
        # trajectories are already gated separately.  Dynamic articulation
        # therefore starts at MCP and includes the true fingertip so all three
        # MANO rotations remain observable.
        chain = [base, base + 1, base + 2, base + 3]
        source_bones = np.diff(source[..., chain, :], axis=-2)
        target_bones = np.diff(target[..., chain, :], axis=-2)
        source_unit = unit(source_bones)
        target_unit = unit(target_bones)
        directions.append(np.degrees(np.arccos(np.clip(
            np.sum(source_unit * target_unit, axis=-1), -1.0, 1.0,
        ))))
        source_bend = np.degrees(np.arccos(np.clip(
            np.sum(source_unit[..., :-1, :] * source_unit[..., 1:, :], axis=-1), -1.0, 1.0,
        )))
        target_bend = np.degrees(np.arccos(np.clip(
            np.sum(target_unit[..., :-1, :] * target_unit[..., 1:, :], axis=-1), -1.0, 1.0,
        )))
        bends.append(np.abs(source_bend - target_bend))
    direction = np.concatenate(directions, axis=-1)[active]
    bend = np.concatenate(bends, axis=-1)[active]
    source_opposition = np.stack([
        np.linalg.norm(source[..., 21, :] - source[..., 2, :], axis=-1),
        np.linalg.norm(source[..., 21, :] - source[..., 5, :], axis=-1),
    ], axis=-1)
    target_opposition = np.stack([
        np.linalg.norm(target[..., 21, :] - target[..., 2, :], axis=-1),
        np.linalg.norm(target[..., 21, :] - target[..., 5, :], axis=-1),
    ], axis=-1)
    opposition = np.abs(target_opposition - source_opposition)[active]
    return {
        "bone_direction_median_deg": float(np.median(direction)) if len(direction) else float("inf"),
        "bone_direction_p95_deg": float(np.percentile(direction, 95)) if len(direction) else float("inf"),
        "finger_bend_median_deg": float(np.median(bend)) if len(bend) else float("inf"),
        "finger_bend_p95_deg": float(np.percentile(bend, 95)) if len(bend) else float("inf"),
        "thumb_opposition_median_m": float(np.median(opposition)) if len(opposition) else float("inf"),
        "thumb_opposition_p95_m": float(np.percentile(opposition, 95)) if len(opposition) else float("inf"),
    }


def _dynamic_hand_quality(
    config: RetargetConfig, baseline: MotionRecord, result: MotionRecord,
    geometry: dict[str, np.ndarray], target_landmarks_local: np.ndarray,
    object_translation: np.ndarray, object_quaternion: np.ndarray,
    evaluation_frame_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    import torch

    settings = config.raw["object_pilot"]
    acceptance = settings["acceptance"]
    device, use_cuda, device_index = _device(config)
    model = _model(config, result.profile, device)
    frames, objects, _, maximum = geometry["dynamic_contact_valid"].shape
    distance = np.full((frames, objects, 2, maximum), np.inf, dtype=np.float32)
    batch = int(settings["forward_batch_frames"])
    with _device_lock(config, use_cuda, device_index), torch.no_grad():
        for start in range(0, frames, batch):
            end = min(frames, start + batch)
            output = _forward(model, result, slice(start, end), device)
            vertices = output.vertices.cpu().numpy() * result.scale + result.transl[start:end, None]
            for local_frame, frame in enumerate(range(start, end)):
                for object_index in range(objects):
                    rotation = Rotation.from_quat(object_quaternion[frame, object_index]).as_matrix()
                    center = object_translation[frame, object_index]
                    for side in range(2):
                        valid = geometry["dynamic_contact_valid"][frame, object_index, side]
                        ids = geometry["dynamic_contact_target_vertex"][frame, object_index, side]
                        if not valid.any():
                            continue
                        local = (
                            vertices[local_frame, np.clip(ids, 0, None)] - center
                        ) @ rotation
                        anchor = geometry[
                            "left_patch_anchor" if side == 0 else "right_patch_anchor"
                        ][frame, object_index]
                        distance[frame, object_index, side, valid] = np.linalg.norm(
                            local[valid] - anchor[valid], axis=1,
                        )
    source_contact = np.stack([
        geometry["left_contact"], geometry["right_contact"],
    ], axis=2)
    region = geometry["dynamic_contact_region"]
    valid = geometry.get(
        "dynamic_contact_constraint_valid", geometry["dynamic_contact_valid"],
    )
    frame_mask = (
        np.ones(frames, dtype=bool) if evaluation_frame_mask is None
        else np.asarray(evaluation_frame_mask, dtype=bool)
    )
    if frame_mask.shape != (frames,):
        raise ValueError("Dynamic-hand evaluation mask must contain one value per frame")
    source_contact = source_contact & frame_mask[:, None, None]
    valid = valid & frame_mask[:, None, None, None]
    threshold = float(acceptance["contact_p95_m"])
    close = valid & (distance <= threshold)
    minimum_points = int(acceptance["contact_patch_active_vertices_p05"])
    minimum_regions = int(acceptance["contact_patch_minimum_regions"])
    achieved = np.zeros_like(source_contact)
    region_count = np.zeros_like(source_contact, dtype=np.int16)
    for frame in range(frames):
        for object_index in range(objects):
            for side in range(2):
                active_regions = np.unique(region[frame, object_index, side, close[frame, object_index, side]])
                active_regions = active_regions[active_regions >= 0]
                region_count[frame, object_index, side] = len(active_regions)
                achieved[frame, object_index, side] = (
                    close[frame, object_index, side].sum() >= minimum_points and
                    len(active_regions) >= minimum_regions
                )

    def transition_error(source: np.ndarray, target: np.ndarray) -> float:
        source_edges = np.flatnonzero(np.diff(source.astype(np.int8), prepend=0, append=0))
        target_edges = np.flatnonzero(np.diff(target.astype(np.int8), prepend=0, append=0))
        if not len(source_edges):
            return 0.0
        if not len(target_edges):
            return float("inf")
        return float(np.median([np.min(np.abs(target_edges - edge)) for edge in source_edges]))

    source_landmarks = geometry.get(
        "target_reference_hand_landmarks_local", geometry["source_hand_landmarks_local"],
    )
    raw_source_landmarks = geometry["source_hand_landmarks_local"]
    side_metrics = {}
    recalls, transition_errors = [], []
    wrist_errors, finger_errors, aperture_errors = [], [], []
    for side_index, side_name in enumerate(("left", "right")):
        source = source_contact[:, :, side_index]
        hit = achieved[:, :, side_index]
        recall = float((hit & source).sum() / max(1, source.sum()))
        transition = float(np.median([
            transition_error(source[:, object_index], hit[:, object_index])
            for object_index in range(objects)
        ]))
        active = source
        wrist = np.linalg.norm(
            target_landmarks_local[:, :, side_index, :2] -
            source_landmarks[:, :, side_index, :2], axis=-1,
        )
        fingers = np.linalg.norm(
            target_landmarks_local[:, :, side_index, 2:] -
            source_landmarks[:, :, side_index, 2:], axis=-1,
        )
        source_aperture = np.linalg.norm(
            source_landmarks[:, :, side_index, 21] - source_landmarks[:, :, side_index, 5], axis=-1,
        )
        target_aperture = np.linalg.norm(
            target_landmarks_local[:, :, side_index, 21] - target_landmarks_local[:, :, side_index, 5], axis=-1,
        )
        aperture = np.abs(target_aperture - source_aperture)
        active_wrist = wrist[active]
        active_fingers = fingers[active]
        active_aperture = aperture[active]
        raw_fingers = np.linalg.norm(
            target_landmarks_local[:, :, side_index, 2:] -
            raw_source_landmarks[:, :, side_index, 2:], axis=-1,
        )[active]
        dynamic_shape = _hand_dynamic_shape_errors(
            raw_source_landmarks[:, :, side_index],
            target_landmarks_local[:, :, side_index], active,
        )
        side_metrics[side_name] = {
            "source_contact_frames": int(source.sum()),
            "achieved_contact_frames": int((hit & source).sum()),
            "source_contact_recall": recall,
            "contact_transition_median_frames": transition,
            "wrist_palm_object_local_p95_m": float(np.percentile(active_wrist, 95)) if len(active_wrist) else float("inf"),
            "finger_object_local_p95_m": float(np.percentile(active_fingers, 95)) if len(active_fingers) else float("inf"),
            "grasp_aperture_p95_m": float(np.percentile(active_aperture, 95)) if len(active_aperture) else float("inf"),
            "raw_source_finger_object_local_p95_m": float(np.percentile(raw_fingers, 95)) if len(raw_fingers) else float("inf"),
            "active_contact_regions_p05": float(np.percentile(region_count[:, :, side_index][source], 5)) if source.any() else 0.0,
            "annotation_assisted_frames": int(
                geometry["annotation_assisted"][frame_mask, :, side_index].sum()
            ),
            **dynamic_shape,
        }
        recalls.append(recall); transition_errors.append(transition)
        if len(active_wrist): wrist_errors.append(active_wrist.reshape(-1))
        if len(active_fingers): finger_errors.append(active_fingers.reshape(-1))
        if len(active_aperture): aperture_errors.append(active_aperture.reshape(-1))

    before = np.concatenate([
        baseline.left_hand_pose.reshape(frames, 15, 3),
        baseline.right_hand_pose.reshape(frames, 15, 3),
    ], axis=1)
    after = np.concatenate([
        result.left_hand_pose.reshape(frames, 15, 3),
        result.right_hand_pose.reshape(frames, 15, 3),
    ], axis=1)
    relative = (
        Rotation.from_rotvec(before.reshape(-1, 3)).as_matrix().transpose(0, 2, 1) @
        Rotation.from_rotvec(after.reshape(-1, 3)).as_matrix()
    )
    hand_delta = np.degrees(Rotation.from_matrix(relative).magnitude()).reshape(frames, 30)
    any_contact = source_contact.any(axis=(1, 2))
    contact_distance = distance[valid]
    return {
        "sides": side_metrics,
        "source_contact_recall": float(min(recalls)),
        "contact_transition_median_frames": float(max(transition_errors)),
        "wrist_palm_object_local_p95_m": float(np.percentile(np.concatenate(wrist_errors), 95)) if wrist_errors else float("inf"),
        "finger_object_local_p95_m": float(np.percentile(np.concatenate(finger_errors), 95)) if finger_errors else float("inf"),
        "grasp_aperture_p95_m": float(np.percentile(np.concatenate(aperture_errors), 95)) if aperture_errors else float("inf"),
        # This is now the source-relative, shape-independent direction error.
        # The old value was just the optimizer delta and trivially equalled
        # the configured clamp when the solver saturated.
        "hand_rotation_p95_deg": float(max(
            metrics["bone_direction_p95_deg"] for metrics in side_metrics.values()
        )),
        "finger_bend_p95_deg": float(max(
            metrics["finger_bend_p95_deg"] for metrics in side_metrics.values()
        )),
        "thumb_opposition_p95_m": float(max(
            metrics["thumb_opposition_p95_m"] for metrics in side_metrics.values()
        )),
        "mano_adjustment_p95_deg": float(np.percentile(hand_delta[any_contact], 95)) if any_contact.any() else float("inf"),
        "contact_surface_gap_median": float(np.median(contact_distance)) if len(contact_distance) else float("inf"),
        "contact_surface_gap_p95": float(np.percentile(contact_distance, 95)) if len(contact_distance) else float("inf"),
        "active_contact_regions_p05": float(min(
            metrics["active_contact_regions_p05"] for metrics in side_metrics.values()
            if metrics["source_contact_frames"] > 0
        )),
    }


def _vertex_sets(model, maximum_hand: int, maximum_body: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weights = model.lbs_weights.detach().cpu().numpy()
    left_score = weights[:, [20, *range(25, 40)]].sum(axis=1)
    right_score = weights[:, [21, *range(40, 55)]].sum(axis=1)

    def selected(score: np.ndarray, maximum: int) -> np.ndarray:
        candidates = np.flatnonzero(score > 0.15)
        if len(candidates) <= maximum:
            return candidates
        order = candidates[np.argsort(score[candidates])]
        return np.unique(np.r_[order[-maximum // 2:], candidates[np.linspace(0, len(candidates) - 1, maximum // 2).astype(int)]])

    left = selected(left_score, maximum_hand)
    right = selected(right_score, maximum_hand)
    all_left = np.flatnonzero(left_score > 0.01)
    all_right = np.flatnonzero(right_score > 0.01)
    hand = np.zeros(len(weights), dtype=bool)
    hand[all_left] = True
    hand[all_right] = True
    body = np.flatnonzero(~hand)
    if len(body) > maximum_body:
        body = body[np.linspace(0, len(body) - 1, maximum_body).astype(int)]
    # Penetration applies to complete hands as well as a stratified major-body
    # sample; contact fitting alone must not push the rest of a palm/finger
    # through the rigid object.
    body = np.unique(np.concatenate([body, all_left, all_right]))
    return left.astype(np.int64), right.astype(np.int64), body.astype(np.int64)


def _forward(model, motion: MotionRecord, indices: slice, device):
    import torch

    count = len(range(*indices.indices(len(motion.timestamps))))
    zeros3 = torch.zeros((count, 3), device=device)
    return model(
        global_orient=torch.tensor(motion.global_orient[indices], device=device),
        body_pose=torch.tensor(motion.body_pose[indices], device=device),
        left_hand_pose=torch.tensor(motion.left_hand_pose[indices], device=device),
        right_hand_pose=torch.tensor(motion.right_hand_pose[indices], device=device),
        betas=torch.tensor(motion.profile.betas, device=device).reshape(1, -1).expand(count, -1),
        jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
        expression=torch.zeros((count, model.num_expression_coeffs), device=device),
        return_verts=True,
    )


def _object_world_to_local(points: np.ndarray, translation: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(quaternion).as_matrix()
    return np.einsum("...j,...jk->...k", points - translation[..., None, :], rotation)


def _cosine_learning_rate_scale(progress: float, final_scale: float) -> float:
    if not 0.0 <= progress <= 1.0:
        raise ValueError("Learning-rate progress must be in [0, 1]")
    if not 0.0 < final_scale <= 1.0:
        raise ValueError("Final learning-rate scale must be in (0, 1]")
    return float(final_scale + (1.0 - final_scale) * 0.5 * (
        1.0 + np.cos(np.pi * progress)
    ))


def _continuation_scale(progress: float, start: float, end: float) -> float:
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("Continuation interval must satisfy 0 <= start < end <= 1")
    return float(np.clip((progress - start) / (end - start), 0.0, 1.0))


def _contact_runs(contact: np.ndarray) -> np.ndarray:
    padded = np.r_[False, np.asarray(contact, dtype=bool), False]
    return np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)


def _stable_contact_patches(
    contact: np.ndarray, candidate_vertex_ids: np.ndarray,
    candidate_distance: np.ndarray, candidate_surface_index: np.ndarray,
    surface: np.ndarray, surface_normals: np.ndarray, patch_size: int,
    maximum_distance: float,
    candidate_opposition: np.ndarray | None = None,
    minimum_opposition: float = -1.0,
    segment_distance: float = np.inf,
    segment_minimum_frames: int = 15,
    segment_normal_degrees: float = 180.0,
    active_enter_distance: float | None = None,
    active_exit_distance: float | None = None,
    active_minimum_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose persistent patches, but activate only credible per-vertex contact.

    A hand-level contact run means at least one sampled hand vertex is near the
    object.  It does *not* mean every vertex selected for the persistent patch
    touches the object for the entire run.  Keeping those concepts separate
    prevents false multi-finger constraints from contorting MANO and the arm.
    """
    frames, objects, _ = candidate_distance.shape
    vertex = np.full((frames, objects, patch_size), -1, dtype=np.int32)
    anchor = np.zeros((frames, objects, patch_size, 3), dtype=np.float32)
    normal = np.zeros_like(anchor)
    valid = np.zeros((frames, objects, patch_size), dtype=bool)
    active_enter_distance = (
        maximum_distance if active_enter_distance is None else active_enter_distance
    )
    active_exit_distance = (
        maximum_distance if active_exit_distance is None else active_exit_distance
    )
    tree = cKDTree(surface)
    for object_index in range(objects):
        segments = []
        for run_start, run_end in _contact_runs(contact[:, object_index]):
            run_segments = []
            if np.isfinite(segment_distance) and run_end - run_start > segment_minimum_frames:
                run_distance = candidate_distance[run_start:run_end, object_index]
                if candidate_opposition is None:
                    run_opposition = np.ones_like(run_distance)
                else:
                    run_opposition = candidate_opposition[run_start:run_end, object_index]
                candidate_score = run_distance + 0.01 * (1.0 - np.clip(run_opposition, -1.0, 1.0))
                candidate_score[run_opposition < minimum_opposition] = np.inf
                missing = ~np.isfinite(candidate_score).any(axis=1)
                candidate_score[missing] = run_distance[missing]
                closest = np.argmin(candidate_score, axis=1)
                rows = np.arange(run_end - run_start)
                center_indices = candidate_surface_index[run_start:run_end, object_index][rows, closest]
                centers = surface[center_indices]
                center_normals = surface_normals[center_indices]
                segment_start = run_start
                segment_start_offset = 0
                reference = centers[0]
                reference_normal = center_normals[0]
                normal_cosine = float(np.cos(np.deg2rad(segment_normal_degrees)))
                for offset in range(segment_minimum_frames, len(centers)):
                    # Enforce the minimum duration for *every* segment, not
                    # only the first.  Without this guard, once the first
                    # re-anchor happened a sliding grasp could re-anchor on
                    # every following frame and hide discontinuity as
                    # "anchor transitions" rather than measured slip.
                    if offset - segment_start_offset < segment_minimum_frames:
                        continue
                    normal_changed = float(np.dot(
                        center_normals[offset], reference_normal,
                    )) < normal_cosine
                    if (
                        np.linalg.norm(centers[offset] - reference) > segment_distance or
                        normal_changed
                    ):
                        run_segments.append((segment_start, run_start + offset))
                        segment_start = run_start + offset
                        segment_start_offset = offset
                        reference = centers[offset]
                        reference_normal = center_normals[offset]
                if run_end - segment_start < segment_minimum_frames and run_segments:
                    previous_start, _ = run_segments.pop()
                    segment_start = previous_start
                run_segments.append((segment_start, run_end))
            else:
                run_segments.append((run_start, run_end))
            # Never merge the tail of one physical contact run into an earlier
            # run.  Doing so can bind a later grasp to a stale object-local
            # anchor even though contact was broken in between.
            segments.extend(run_segments)
        for start, end in segments:
            segment_distance_values = candidate_distance[start:end, object_index]
            if candidate_opposition is None:
                segment_opposition = np.ones_like(segment_distance_values)
            else:
                segment_opposition = candidate_opposition[start:end, object_index]
            segment_contact = contact[start:end, object_index]

            # Select a *covering* patch.  Ranking vertices by their median
            # distance over the whole segment systematically selects several
            # vertices that all explain the same central part of a sliding
            # grasp, while leaving the beginning and end unconstrained.  The
            # greedy set-cover below instead asks each new vertex to explain
            # contact frames that the already selected vertices do not.  A
            # vertex is eligible on a frame only when both proximity and the
            # human/object normal opposition make physical contact credible.
            active_by_candidate = np.zeros_like(segment_distance_values, dtype=bool)
            for candidate_index in range(segment_distance_values.shape[1]):
                active = contact_hysteresis(
                    segment_distance_values[:, candidate_index],
                    float(active_enter_distance), float(active_exit_distance),
                    int(active_minimum_frames),
                )
                active &= segment_opposition[:, candidate_index] >= minimum_opposition
                active_by_candidate[:, candidate_index] = active & segment_contact

            chosen_list: list[int] = []
            uncovered = segment_contact.copy()
            available = np.ones(segment_distance_values.shape[1], dtype=bool)
            for _ in range(patch_size):
                new_coverage = (active_by_candidate & uncovered[:, None]).sum(axis=0)
                new_coverage[~available] = -1
                best_coverage = int(new_coverage.max(initial=-1))
                if best_coverage <= 0:
                    break
                candidates = np.flatnonzero(new_coverage == best_coverage)
                # Deterministic geometric tie-break: prefer the smaller
                # distance, then the better opposing normal, then vertex id.
                tie_distance = np.asarray([
                    np.median(segment_distance_values[active_by_candidate[:, index], index])
                    for index in candidates
                ])
                tie_opposition = np.asarray([
                    np.median(segment_opposition[active_by_candidate[:, index], index])
                    for index in candidates
                ])
                order = np.lexsort((candidate_vertex_ids[candidates], -tie_opposition, tie_distance))
                selected = int(candidates[order[0]])
                chosen_list.append(selected)
                available[selected] = False
                uncovered &= ~active_by_candidate[:, selected]
            chosen = np.asarray(chosen_list, dtype=np.int64)
            for patch_index, candidate_index in enumerate(chosen):
                nearest = candidate_surface_index[start:end, object_index, candidate_index]
                candidate_active = active_by_candidate[:, candidate_index]
                anchor_frames = candidate_active
                if not anchor_frames.any():
                    anchor_frames = (
                        candidate_distance[start:end, object_index, candidate_index]
                        <= maximum_distance
                    )
                if not anchor_frames.any():
                    continue
                center = np.median(surface[nearest[anchor_frames]], axis=0)
                _, surface_index = tree.query(center)
                vertex[start:end, object_index, patch_index] = int(candidate_vertex_ids[candidate_index])
                anchor[start:end, object_index, patch_index] = surface[int(surface_index)]
                normal[start:end, object_index, patch_index] = surface_normals[int(surface_index)]
                valid[start:end, object_index, patch_index] = candidate_active
    return vertex, anchor, normal, valid


def _incident_face_table(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    incident: list[list[np.ndarray]] = [[] for _ in range(vertex_count)]
    for face in np.asarray(faces, dtype=np.int64):
        for vertex in face:
            incident[int(vertex)].append(face)
    maximum = max(len(rows) for rows in incident)
    table = np.zeros((vertex_count, maximum, 3), dtype=np.int64)
    mask = np.zeros((vertex_count, maximum), dtype=bool)
    for vertex, rows in enumerate(incident):
        if not rows:
            continue
        values = np.asarray(rows, dtype=np.int64)
        table[vertex, :len(values)] = values
        table[vertex, len(values):] = values[0]
        mask[vertex, :len(values)] = True
    return table, mask


def _selected_vertex_normals(vertices, vertex_ids, face_table, face_mask):
    """Differentiable posed-mesh normals for a per-frame vertex-id table."""
    import torch

    ids = torch.clamp(vertex_ids, min=0)
    faces = face_table[ids]
    masks = face_mask[ids]
    batch_shape = [vertices.shape[0]] + [1] * (faces.ndim - 1)
    batch = torch.arange(vertices.shape[0], device=vertices.device).reshape(batch_shape)
    triangles = vertices[batch, faces]
    cross = torch.linalg.cross(
        triangles[..., 1, :] - triangles[..., 0, :],
        triangles[..., 2, :] - triangles[..., 0, :], dim=-1,
    )
    summed = (cross * masks[..., None]).sum(dim=-2)
    return torch.nn.functional.normalize(summed, dim=-1, eps=1e-8)


def _matrix_to_axis_angle(matrix):
    """Differentiable, branch-stable SO(3) matrix to axis-angle conversion.

    The largest quaternion component is selected, following the numerically
    stable construction used by common geometry libraries.  Keeping this local
    avoids introducing a heavyweight geometry dependency into the retarget
    environment.
    """
    import torch

    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Rotation matrices must end in 3x3")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    quaternion_terms = torch.stack([
        1.0 + m00 + m11 + m22,
        1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22,
        1.0 - m00 - m11 + m22,
    ], dim=-1)
    # sqrt(clamp(x, 0)) has an infinite derivative at exactly zero.  Explicit
    # positive indexing gives zero gradient to the unused quaternion branches.
    q_abs = torch.zeros_like(quaternion_terms)
    positive = quaternion_terms > 0.0
    q_abs[positive] = torch.sqrt(quaternion_terms[positive])
    quaternion_candidates = torch.stack([
        torch.stack([q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1].square(), m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2].square(), m12 + m21], dim=-1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3].square()], dim=-1),
    ], dim=-2)
    denominator = 2.0 * torch.clamp(q_abs[..., None], min=0.1)
    candidates = quaternion_candidates / denominator
    choice = torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4).to(matrix.dtype)
    quaternion = (candidates * choice[..., None]).sum(dim=-2)
    quaternion = torch.nn.functional.normalize(quaternion, dim=-1, eps=1e-8)
    quaternion = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1)
    angle = 2.0 * torch.atan2(vector_norm, quaternion[..., 0])
    nonzero = angle.abs() > 1e-6
    regular = torch.sin(0.5 * angle) / torch.where(nonzero, angle, torch.ones_like(angle))
    half_over_angle = torch.where(nonzero, regular, 0.5 - angle.square() / 48.0)
    return vector / torch.clamp(half_over_angle[..., None], min=1e-8)


def _compose_axis_angle(base, local_delta):
    """Compose a local SO(3) perturbation without adding rotation vectors."""
    from smplx.lbs import batch_rodrigues

    shape = base.shape
    base_matrix = batch_rodrigues(base.reshape(-1, 3))
    delta_matrix = batch_rodrigues(local_delta.reshape(-1, 3))
    return _matrix_to_axis_angle(base_matrix @ delta_matrix).reshape(shape)


def _hierarchical_priority_gradients(parameters, gradient_levels):
    """Combine ordered objectives without letting a lower level oppose prior ones.

    Each lower-priority gradient is projected against the cumulative accepted
    gradient whenever their dot product is negative. This is a first-order
    lexicographic update: object-local hand/contact trajectory is accepted
    first, then collision geometry, source finger dynamics, and finally body
    priors. It avoids collapsing those constraints into one weighted loss.
    """
    import torch

    if not gradient_levels:
        return [torch.zeros_like(parameter) for parameter in parameters]
    accepted = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradient_levels[0])
    ]
    for level in gradient_levels[1:]:
        lower = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, level)
        ]
        accepted_norm = sum(gradient.square().sum() for gradient in accepted)
        conflict_dot = sum(
            (high * low).sum() for high, low in zip(accepted, lower)
        )
        coefficient = torch.where(
            (conflict_dot < 0.0) & (accepted_norm > 1e-12),
            conflict_dot / torch.clamp(accepted_norm, min=1e-12),
            torch.zeros_like(conflict_dot),
        )
        accepted = [
            high + low - coefficient * high
            for high, low in zip(accepted, lower)
        ]
    return accepted


def _contact_priority_gradients(parameters, contact_gradients, secondary_gradients):
    """Backward-compatible two-level wrapper used by regression tests."""
    return _hierarchical_priority_gradients(
        parameters, [contact_gradients, secondary_gradients],
    )


def _measure_geometry(
    config: RetargetConfig, motion: MotionRecord, grids: list[SignedDistanceGrid],
    object_translation: np.ndarray, object_quaternion: np.ndarray,
    contact_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    settings = config.raw["object_pilot"]
    device, use_cuda, device_index = _device(config)
    model = _model(config, motion.profile, device)
    face_table_np, face_mask_np = _incident_face_table(np.asarray(model.faces), model.lbs_weights.shape[0])
    face_table = torch.tensor(face_table_np, dtype=torch.long, device=device)
    face_mask = torch.tensor(face_mask_np, dtype=torch.bool, device=device)
    left_ids, right_ids, body_ids = _vertex_sets(
        model, int(settings["hand_vertex_samples"]), int(settings["body_vertex_samples"]),
    )
    frames, objects = object_translation.shape[:2]
    left_distance = np.full((frames, objects), np.inf, dtype=np.float32)
    right_distance = np.full((frames, objects), np.inf, dtype=np.float32)
    body_distance = np.full((frames, objects), np.inf, dtype=np.float32)
    penetration = np.zeros((frames, objects), dtype=np.float32)
    penetration_sample_count = int(settings.get("penetration_topk_vertices", 64))
    penetration_samples = np.zeros(
        (frames, objects, penetration_sample_count), dtype=np.float32,
    )
    penetration_histogram_edges = np.linspace(0.0, 0.05, 501, dtype=np.float64)
    penetration_histogram_counts = np.zeros(
        (objects, len(penetration_histogram_edges) - 1), dtype=np.int64,
    )
    left_vertex = np.full((frames, objects), -1, dtype=np.int32)
    right_vertex = np.full((frames, objects), -1, dtype=np.int32)
    left_anchor = np.zeros((frames, objects, 3), dtype=np.float32)
    right_anchor = np.zeros((frames, objects, 3), dtype=np.float32)
    all_distance = {
        "left": np.full((frames, objects, len(left_ids)), np.inf, dtype=np.float32),
        "right": np.full((frames, objects, len(right_ids)), np.inf, dtype=np.float32),
    }
    all_surface = {
        "left": np.zeros((frames, objects, len(left_ids)), dtype=np.int32),
        "right": np.zeros((frames, objects, len(right_ids)), dtype=np.int32),
    }
    all_opposition = {
        "left": np.zeros((frames, objects, len(left_ids)), dtype=np.float32),
        "right": np.zeros((frames, objects, len(right_ids)), dtype=np.float32),
    }
    batch = int(settings["forward_batch_frames"])
    with _device_lock(config, use_cuda, device_index), torch.no_grad():
        for start in range(0, frames, batch):
            end = min(frames, start + batch)
            output = _forward(model, motion, slice(start, end), device)
            vertices = output.vertices.detach().cpu().numpy() * motion.scale + motion.transl[start:end, None]
            human_normals = {}
            for side, vertex_ids in (("left", left_ids), ("right", right_ids)):
                ids = torch.tensor(
                    np.broadcast_to(vertex_ids, (end - start, len(vertex_ids))).copy(),
                    dtype=torch.long, device=device,
                )
                human_normals[side] = _selected_vertex_normals(
                    output.vertices, ids, face_table, face_mask,
                ).cpu().numpy()
            for object_index, grid in enumerate(grids):
                rotation = Rotation.from_quat(object_quaternion[start:end, object_index]).as_matrix()
                center = object_translation[start:end, object_index]
                for side, target, vertex_ids, distances, vertex_result, anchor_result in (
                    ("left", vertices[:, left_ids], left_ids, left_distance, left_vertex, left_anchor),
                    ("right", vertices[:, right_ids], right_ids, right_distance, right_vertex, right_anchor),
                ):
                    local = np.einsum("tvj,tjk->tvk", target - center[:, None], rotation)
                    nearest_distance, nearest_normal = grid.query_surface(local.reshape(-1, 3))
                    nearest_distance = nearest_distance.reshape(end - start, len(vertex_ids))
                    nearest_normal = nearest_normal.reshape(end - start, len(vertex_ids), 3)
                    all_distance[side][start:end, object_index] = nearest_distance
                    # Dynamic source anchors already carry exact face and
                    # barycentric identity.  This table is only used by the
                    # legacy no-reference path; keep a deterministic nearest
                    # vertex index while all v3 distances/normals come from
                    # the actual triangle query above.
                    _, nearest_surface = cKDTree(grid.surface_points).query(
                        local.reshape(-1, 3), workers=-1,
                    )
                    nearest_surface = nearest_surface.reshape(end - start, len(vertex_ids))
                    all_surface[side][start:end, object_index] = nearest_surface
                    object_normal_local = nearest_normal
                    object_normal_world = np.einsum("tvj,tij->tvi", object_normal_local, rotation)
                    all_opposition[side][start:end, object_index] = -np.einsum(
                        "tvi,tvi->tv", human_normals[side], object_normal_world,
                    )
                    chosen = np.argmin(nearest_distance, axis=1)
                    row = np.arange(end - start)
                    distances[start:end, object_index] = nearest_distance[row, chosen]
                    vertex_result[start:end, object_index] = vertex_ids[chosen]
                    anchor_result[start:end, object_index] = grid.surface_points[nearest_surface[row, chosen]]
                body_local = np.einsum(
                    "tvj,tjk->tvk", vertices[:, body_ids] - center[:, None], rotation,
                )
                sdf = grid.sample(body_local.reshape(-1, 3)).reshape(end - start, len(body_ids))
                body_distance[start:end, object_index] = np.min(np.abs(sdf), axis=1)
                depth = np.maximum(0.0, -sdf)
                positive_depth = depth[depth > 0.0]
                if len(positive_depth):
                    penetration_histogram_counts[object_index] += np.histogram(
                        positive_depth, bins=penetration_histogram_edges,
                    )[0]
                penetration[start:end, object_index] = np.max(depth, axis=1)
                keep_count = min(penetration_sample_count, depth.shape[1])
                largest = np.sort(depth, axis=1)[:, -keep_count:]
                penetration_samples[
                    start:end, object_index, penetration_sample_count - keep_count:
                ] = largest
    left_contact = np.zeros((frames, objects), dtype=bool)
    right_contact = np.zeros((frames, objects), dtype=bool)
    body_contact = np.zeros((frames, objects), dtype=bool)
    for object_index, grid in enumerate(grids):
        if contact_reference is None:
            minimum_opposition = float(settings.get("contact_patch_minimum_opposition", 0.0))
            left_credible_distance = np.min(
                np.where(
                    all_opposition["left"][:, object_index] >= minimum_opposition,
                    all_distance["left"][:, object_index], np.inf,
                ),
                axis=1,
            )
            right_credible_distance = np.min(
                np.where(
                    all_opposition["right"][:, object_index] >= minimum_opposition,
                    all_distance["right"][:, object_index], np.inf,
                ),
                axis=1,
            )
            left_contact[:, object_index] = contact_hysteresis(
                left_credible_distance, float(settings["contact_enter_m"]),
                float(settings["contact_exit_m"]), int(settings["contact_minimum_frames"]),
            )
            right_contact[:, object_index] = contact_hysteresis(
                right_credible_distance, float(settings["contact_enter_m"]),
                float(settings["contact_exit_m"]), int(settings["contact_minimum_frames"]),
            )
        else:
            left_contact[:, object_index] = contact_reference["left_contact"][:, object_index]
            right_contact[:, object_index] = contact_reference["right_contact"][:, object_index]
        body_contact[:, object_index] = contact_hysteresis(
            body_distance[:, object_index], float(settings["body_contact_enter_m"]),
            float(settings["body_contact_exit_m"]), int(settings["contact_minimum_frames"]),
        )
    patch_size = int(settings.get("contact_patch_vertices", 8))
    if contact_reference is None:
        patches = {}
        for side, contact, vertex_ids in (
            ("left", left_contact, left_ids), ("right", right_contact, right_ids),
        ):
            pieces = [[] for _ in range(4)]
            for object_index, grid in enumerate(grids):
                patch = _stable_contact_patches(
                    contact[:, object_index:object_index + 1], vertex_ids,
                    all_distance[side][:, object_index:object_index + 1],
                    all_surface[side][:, object_index:object_index + 1],
                    grid.surface_points, grid.surface_normals, patch_size,
                    float(settings.get("contact_patch_max_distance_m", settings["contact_exit_m"])),
                    all_opposition[side][:, object_index:object_index + 1],
                    float(settings.get("contact_patch_minimum_opposition", 0.0)),
                    float(settings.get("contact_anchor_segment_m", np.inf)),
                    int(settings.get("contact_anchor_segment_minimum_frames", 15)),
                    float(settings.get("contact_anchor_segment_normal_degrees", 180.0)),
                    float(settings.get("contact_patch_active_enter_m", settings["contact_enter_m"])),
                    float(settings.get("contact_patch_active_exit_m", settings["contact_exit_m"])),
                    int(settings.get("contact_patch_active_minimum_frames", settings["contact_minimum_frames"])),
                )
                for index, value in enumerate(patch):
                    pieces[index].append(value)
            patches[side] = tuple(np.concatenate(values, axis=1) for values in pieces)
        left_patch, right_patch = patches["left"], patches["right"]
    else:
        left_patch = tuple(contact_reference[f"left_patch_{name}"] for name in ("vertex", "anchor", "normal", "valid"))
        right_patch = tuple(contact_reference[f"right_patch_{name}"] for name in ("vertex", "anchor", "normal", "valid"))
    for prefix, patch, representative_vertex, representative_anchor in (
        ("left", left_patch, left_vertex, left_anchor),
        ("right", right_patch, right_vertex, right_anchor),
    ):
        valid = patch[3][..., 0]
        representative_vertex[valid] = patch[0][..., 0][valid]
        representative_anchor[valid] = patch[1][..., 0, :][valid]
    result = {
        "left_distance": left_distance, "right_distance": right_distance,
        "body_distance": body_distance, "penetration": penetration,
        "penetration_samples": penetration_samples,
        "penetration_histogram_edges": penetration_histogram_edges,
        "penetration_histogram_counts": penetration_histogram_counts,
        "left_contact": left_contact, "right_contact": right_contact, "body_contact": body_contact,
        "left_vertex": left_vertex, "right_vertex": right_vertex,
        "left_anchor": left_anchor, "right_anchor": right_anchor,
    }
    for prefix, patch in (("left", left_patch), ("right", right_patch)):
        for name, value in zip(("vertex", "anchor", "normal", "valid"), patch):
            result[f"{prefix}_patch_{name}"] = value
    if contact_reference is not None:
        for name in (
            "source_hand_landmarks_local", "source_hand_landmark_velocity_local",
            "target_reference_hand_landmarks_local",
            "source_hand_observed", "annotation_assisted", "source_hand_proxy_gap_m",
            "dynamic_contact_source_local", "dynamic_contact_face",
            "dynamic_contact_barycentric", "dynamic_contact_region",
            "dynamic_contact_segment", "dynamic_contact_proxy_index",
            "dynamic_contact_valid", "dynamic_contact_constraint_valid",
            "dynamic_contact_target_vertex",
        ):
            if name in contact_reference:
                result[name] = contact_reference[name]
    return result


def _torch_grid_sample(points, grid: SignedDistanceGrid, device):
    import torch

    values = torch.as_tensor(grid.values, dtype=points.dtype, device=device)
    origin = torch.as_tensor(grid.origin, dtype=points.dtype, device=device)
    coordinate = (points - origin) / float(grid.pitch)
    lower_float = torch.floor(coordinate)
    fraction = coordinate - lower_float
    shape = torch.as_tensor(grid.values.shape, dtype=torch.long, device=device)
    lower = lower_float.long()
    outside = ((lower < 0) | (lower + 1 >= shape)).any(dim=1)
    lower = torch.minimum(torch.maximum(lower, torch.zeros_like(lower)), shape - 2)
    result = torch.zeros(len(points), dtype=points.dtype, device=device)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                weight = (
                    (fraction[:, 0] if dx else 1.0 - fraction[:, 0]) *
                    (fraction[:, 1] if dy else 1.0 - fraction[:, 1]) *
                    (fraction[:, 2] if dz else 1.0 - fraction[:, 2])
                )
                index = lower + torch.tensor([dx, dy, dz], device=device)
                result = result + weight * values[index[:, 0], index[:, 1], index[:, 2]]
    if outside.any():
        result = torch.where(outside, torch.abs(result), result)
    return result


def _geometry_objectives(
    vertices, geometry: dict[str, np.ndarray], start: int, end: int,
    grids: list[SignedDistanceGrid], object_translation: np.ndarray,
    object_quaternion: np.ndarray, face_table, face_mask, body_vertex_ids,
    settings: dict[str, Any], fps: float,
):
    """Differentiable contact, normal, slip, and object-SDF objectives."""
    import torch

    device = vertices.device
    count = end - start
    losses = {
        name: torch.zeros((), device=device)
        for name in ("plane", "tangent", "normal", "slip", "penetration", "penetration_max")
    }
    for object_index, grid in enumerate(grids):
        rotation = torch.tensor(
            Rotation.from_quat(object_quaternion[start:end, object_index]).as_matrix(),
            dtype=vertices.dtype, device=device,
        )
        center = torch.tensor(object_translation[start:end, object_index], device=device)
        for side in ("left", "right"):
            valid = torch.tensor(
                geometry[f"{side}_patch_valid"][start:end, object_index], device=device,
            )
            if not valid.any():
                continue
            vertex_ids = torch.tensor(
                geometry[f"{side}_patch_vertex"][start:end, object_index],
                dtype=torch.long, device=device,
            )
            row = torch.arange(count, device=device)[:, None]
            predicted = vertices[row, torch.clamp(vertex_ids, min=0)]
            anchor_local = torch.tensor(
                geometry[f"{side}_patch_anchor"][start:end, object_index], device=device,
            )
            normal_local = torch.tensor(
                geometry[f"{side}_patch_normal"][start:end, object_index], device=device,
            )
            anchor_world = torch.einsum("tki,tji->tkj", anchor_local, rotation) + center[:, None]
            normal_world = torch.einsum("tki,tji->tkj", normal_local, rotation)
            difference = predicted - anchor_world
            projection = (difference * normal_world).sum(dim=-1)
            normal_gap = projection - float(settings.get("contact_surface_offset_m", 0.001))
            tangent = difference - projection[..., None] * normal_world
            losses["plane"] = losses["plane"] + normal_gap[valid].square().mean()
            losses["tangent"] = losses["tangent"] + tangent[valid].square().sum(dim=-1).mean()
            human_normal = _selected_vertex_normals(vertices, vertex_ids, face_table, face_mask)
            opposition = (human_normal * normal_world).sum(dim=-1) + 1.0
            losses["normal"] = losses["normal"] + opposition[valid].square().mean()
            if count > 1:
                local_point = torch.einsum(
                    "tki,tij->tkj", predicted - center[:, None], rotation,
                )
                proxy_identity = torch.tensor(
                    geometry["dynamic_contact_proxy_index"][
                        start:end, object_index, 0 if side == "left" else 1,
                    ],
                    dtype=torch.long, device=device,
                )
                pair = (
                    valid[1:] & valid[:-1] &
                    (vertex_ids[1:] == vertex_ids[:-1]) &
                    (proxy_identity[1:] == proxy_identity[:-1])
                )
                if pair.any():
                    velocity = (
                        (local_point[1:] - local_point[:-1]) -
                        (anchor_local[1:] - anchor_local[:-1])
                    ) * float(fps)
                    pair_normal = normal_local[1:]
                    tangent_velocity = velocity - (
                        velocity * pair_normal
                    ).sum(dim=-1, keepdim=True) * pair_normal
                    losses["slip"] = losses["slip"] + tangent_velocity[pair].square().sum(dim=-1).mean()
        sampled = vertices[:, body_vertex_ids]
        local = torch.einsum("tvi,tij->tvj", sampled - center[:, None], rotation)
        sdf = _torch_grid_sample(local.reshape(-1, 3), grid, device).reshape(count, -1)
        violation = torch.relu(float(settings["penetration_margin_m"]) - sdf)
        top_count = min(int(settings["penetration_topk_vertices"]), violation.shape[1])
        losses["penetration"] = losses["penetration"] + torch.topk(
            violation, k=top_count, dim=1, largest=True,
        ).values.square().mean()
        losses["penetration_max"] = losses["penetration_max"] + violation.max(dim=1).values.square().mean()
    return losses


def _coarse_wrist_retarget(
    config: RetargetConfig, motion: MotionRecord, source: SourceHandContacts,
    object_translation: np.ndarray, object_quaternion: np.ndarray,
) -> dict[str, Any]:
    """Stage 1: align pelvis/arms from source wrist and palm trajectories only."""
    import torch

    settings = config.raw["object_pilot"]
    device, use_cuda, device_index = _device(config)
    model = _model(config, motion.profile, device)
    body_joints = np.asarray(settings["optimized_body_joint_indices"], dtype=np.int64)
    body_base = np.asarray(motion.body_pose, dtype=np.float32).reshape(-1, 21, 3).copy()
    root_base = np.asarray(motion.global_orient, dtype=np.float32).copy()
    translation_base = np.asarray(motion.transl, dtype=np.float32).copy()
    body_result = body_base.copy()
    root_result = root_base.copy()
    translation_result = translation_base.copy()
    left_hand = np.asarray(motion.left_hand_pose, dtype=np.float32).reshape(-1, 15, 3)
    right_hand = np.asarray(motion.right_hand_pose, dtype=np.float32).reshape(-1, 15, 3)
    frames = len(body_base)
    window = int(settings["ik_window_frames"])
    iterations = int(settings.get("coarse_ik_iterations", 100))
    learning_rate = float(settings.get("coarse_ik_learning_rate", settings["ik_learning_rate"]))
    body_maximum = torch.deg2rad(torch.tensor(
        settings.get("arm_max_delta_degrees", [12.0] * len(body_joints)), device=device,
    ))
    root_maximum = float(np.deg2rad(settings.get("root_max_delta_degrees", 10.0)))
    translation_maximum = float(settings.get("root_translation_max_delta_m", 0.08))
    active_windows = 0
    with _device_lock(config, use_cuda, device_index):
        for start in range(0, frames, window):
            end = min(frames, start + window)
            contact = source.contact[start:end]
            if not contact.any():
                continue
            active_windows += 1
            count = end - start
            body0 = torch.tensor(body_result[start:end], device=device)
            root0 = torch.tensor(root_result[start:end], device=device)
            translation0 = torch.tensor(translation_result[start:end], device=device)
            left0 = torch.tensor(left_hand[start:end], device=device)
            right0 = torch.tensor(right_hand[start:end], device=device)
            betas = torch.tensor(motion.profile.betas, device=device).reshape(1, -1).expand(count, -1)
            zeros3 = torch.zeros((count, 3), device=device)
            body_delta = torch.nn.Parameter(torch.zeros((count, len(body_joints), 3), device=device))
            root_delta = torch.nn.Parameter(torch.zeros((count, 3), device=device))
            translation_delta = torch.nn.Parameter(torch.zeros((count, 3), device=device))
            parameters = [root_delta, translation_delta, body_delta]
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
            with torch.no_grad():
                initial = model(
                    global_orient=root0, body_pose=body0.reshape(count, 63),
                    left_hand_pose=left0.reshape(count, 45),
                    right_hand_pose=right0.reshape(count, 45), betas=betas,
                    jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                    expression=torch.zeros((count, model.num_expression_coeffs), device=device),
                    return_verts=True,
                )
                foot_indices = torch.tensor(
                    settings.get("foot_joint_indices", [7, 8, 10, 11]),
                    dtype=torch.long, device=device,
                )
                initial_feet = initial.joints[:, foot_indices] * motion.scale + translation0[:, None]
            for iteration in range(iterations):
                progress = iteration / max(1, iterations - 1)
                optimizer.param_groups[0]["lr"] = learning_rate * _cosine_learning_rate_scale(progress, 0.05)
                optimizer.zero_grad(set_to_none=True)
                current_body = body0.clone()
                current_body[:, body_joints] = _compose_axis_angle(
                    body0[:, body_joints], body_delta,
                )
                current_root = _compose_axis_angle(root0, root_delta)
                current_translation = translation0 + translation_delta
                output = model(
                    global_orient=current_root, body_pose=current_body.reshape(count, 63),
                    left_hand_pose=left0.reshape(count, 45),
                    right_hand_pose=right0.reshape(count, 45), betas=betas,
                    jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                    expression=torch.zeros((count, model.num_expression_coeffs), device=device),
                    return_verts=True,
                )
                target_world = _target_hand_landmarks(
                    output.joints, output.vertices,
                )[:, :, :2] * motion.scale
                target_world = target_world + current_translation[:, None, None]
                reach = torch.zeros((), device=device)
                for object_index in range(object_translation.shape[1]):
                    rotation = torch.tensor(
                        Rotation.from_quat(object_quaternion[start:end, object_index]).as_matrix(),
                        dtype=target_world.dtype, device=device,
                    )
                    center = torch.tensor(object_translation[start:end, object_index], device=device)
                    target_local = torch.einsum(
                        "tslj,tjk->tslk", target_world - center[:, None, None], rotation,
                    )
                    source_local = torch.tensor(
                        source.landmarks_local[start:end, object_index, :, :2], device=device,
                    )
                    mask = torch.tensor(contact[:, object_index], device=device)
                    if mask.any():
                        reach = reach + (
                            target_local - source_local
                        ).square().sum(dim=-1)[mask].mean()
                current_feet = output.joints[:, foot_indices] * motion.scale + current_translation[:, None]
                foot = (current_feet - initial_feet).square().sum(dim=-1).mean()
                prior = (
                    body_delta.square().mean() + 0.5 * root_delta.square().mean() +
                    20.0 * translation_delta.square().mean()
                )
                temporal = torch.zeros((), device=device)
                if count > 1:
                    temporal = (
                        (body_delta[1:] - body_delta[:-1]).square().mean() +
                        (root_delta[1:] - root_delta[:-1]).square().mean() +
                        (translation_delta[1:] - translation_delta[:-1]).square().mean()
                    )
                loss = (
                    float(settings.get("coarse_reach_weight", 12000.0)) * reach +
                    float(settings.get("foot_preservation_weight", 1500.0)) * foot +
                    float(settings.get("coarse_pose_prior_weight", 1.0)) * prior +
                    float(settings.get("coarse_temporal_weight", 8.0)) * temporal
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite coarse wrist/palm IK loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                with torch.no_grad():
                    body_norm = torch.linalg.vector_norm(body_delta, dim=-1, keepdim=True)
                    body_delta.mul_(torch.minimum(
                        torch.ones_like(body_norm),
                        body_maximum[None, :, None] / torch.clamp(body_norm, min=1e-8),
                    ))
                    root_norm = torch.linalg.vector_norm(root_delta, dim=-1, keepdim=True)
                    root_delta.mul_(torch.minimum(
                        torch.ones_like(root_norm), root_maximum / torch.clamp(root_norm, min=1e-8),
                    ))
                    translation_norm = torch.linalg.vector_norm(translation_delta, dim=-1, keepdim=True)
                    translation_delta.mul_(torch.minimum(
                        torch.ones_like(translation_norm),
                        translation_maximum / torch.clamp(translation_norm, min=1e-8),
                    ))
            with torch.no_grad():
                body_result[start:end, body_joints] = _compose_axis_angle(
                    body0[:, body_joints], body_delta,
                ).cpu().numpy()
                root_result[start:end] = _compose_axis_angle(root0, root_delta).cpu().numpy()
                translation_result[start:end] = (translation0 + translation_delta).cpu().numpy()

    degree = int(config.raw["temporal_continuity"]["spline_degree"])
    robust = int(config.raw["temporal_continuity"]["robust_iterations"])
    knot_seconds = float(settings["c4_knot_seconds"])
    body_smoothed, body_smoothing = c3_smooth_relative_rotvec(
        motion.timestamps, body_base.reshape(frames, 63), body_result.reshape(frames, 63),
        knot_seconds, degree, robust, float(settings.get("c4_huber_degrees", 45.0)),
    )
    body_smoothed = body_smoothed.reshape(frames, 21, 3)
    keep = np.ones(21, dtype=bool); keep[body_joints] = False
    body_smoothed[:, keep] = body_base[:, keep]
    root_smoothed, root_smoothing = c3_smooth_relative_rotvec(
        motion.timestamps, root_base, root_result, knot_seconds, degree, robust,
        float(settings.get("c4_huber_degrees", 45.0)),
    )
    translation_smoothed, translation_smoothing = c3_smooth_translation(
        motion.timestamps, translation_result, knot_seconds, degree, robust,
        float(settings.get("coarse_translation_huber_m", 0.02)),
    )
    motion.body_pose = body_smoothed.reshape(frames, 63).astype(np.float32)
    motion.global_orient = root_smoothed.astype(np.float32)
    motion.transl = translation_smoothed.astype(np.float32)
    target = _measure_target_hand_landmarks(config, motion, object_translation, object_quaternion)
    source_landmarks = source.landmarks_local[:, :, :, :2]
    errors = np.linalg.norm(target[:, :, :, :2] - source_landmarks, axis=-1)
    active = source.contact
    return {
        "active_windows": active_windows, "iterations": iterations,
        "wrist_palm_error_median_m": float(np.median(errors[active])),
        "wrist_palm_error_p95_m": float(np.percentile(errors[active], 95)),
        "body_c4_initialization": body_smoothing,
        "root_c4_initialization": root_smoothing,
        "translation_c4_initialization": translation_smoothing,
    }


def _joint_retarget(
    config: RetargetConfig, motion: MotionRecord, grids: list[SignedDistanceGrid],
    object_translation: np.ndarray, object_quaternion: np.ndarray, geometry: dict[str, np.ndarray],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from smplx.lbs import batch_rodrigues

    settings = config.raw["object_pilot"]
    device, use_cuda, device_index = _device(config)
    model = _model(config, motion.profile, device)
    face_table_np, face_mask_np = _incident_face_table(np.asarray(model.faces), model.lbs_weights.shape[0])
    face_table = torch.tensor(face_table_np, dtype=torch.long, device=device)
    face_mask = torch.tensor(face_mask_np, dtype=torch.bool, device=device)
    _, _, body_vertex_ids = _vertex_sets(
        model, int(settings["hand_vertex_samples"]), int(settings["body_vertex_samples"]),
    )
    arm_joints = np.asarray(
        settings.get("fine_optimized_body_joint_indices", settings["optimized_body_joint_indices"]),
        dtype=np.int64,
    )
    body = np.asarray(motion.body_pose, dtype=np.float32).reshape(-1, 21, 3).copy()
    baseline = body.copy()
    roots = np.asarray(motion.global_orient, dtype=np.float32).copy()
    root_baseline = roots.copy()
    translations = np.asarray(motion.transl, dtype=np.float32).copy()
    translation_baseline = translations.copy()
    left_hand_result = np.asarray(motion.left_hand_pose, dtype=np.float32).reshape(-1, 15, 3).copy()
    right_hand_result = np.asarray(motion.right_hand_pose, dtype=np.float32).reshape(-1, 15, 3).copy()
    left_hand_baseline = left_hand_result.copy()
    right_hand_baseline = right_hand_result.copy()
    frames = len(body)
    window = int(settings["ik_window_frames"])
    iterations = int(settings["ik_iterations"])
    optimize_hand_pose = bool(settings.get("optimize_hand_pose", True))
    arm_maximum = torch.deg2rad(torch.tensor(
        settings.get(
            "fine_body_max_delta_degrees",
            settings.get("arm_max_delta_degrees", [12.0] * len(arm_joints)),
        ), device=device,
    ))
    if len(arm_maximum) != len(arm_joints):
        raise ValueError("arm_max_delta_degrees must match optimized_body_joint_indices")
    hand_maximum = float(np.deg2rad(settings.get("mano_max_delta_degrees", 30.0)))
    root_rotation_maximum = float(np.deg2rad(settings.get(
        "fine_root_max_delta_degrees", settings.get("root_max_delta_degrees", 10.0),
    )))
    root_translation_maximum = float(settings.get(
        "fine_root_translation_max_delta_m", settings.get("root_translation_max_delta_m", 0.08),
    ))
    direct_c4 = bool(settings.get("direct_c4_optimization", False))
    if direct_c4:
        degree = int(config.raw["temporal_continuity"]["spline_degree"])
        basis_np, c4_knots = c4_bspline_basis(
            motion.timestamps, float(settings["c4_knot_seconds"]), degree,
        )
        control_count = basis_np.shape[1]
        root_coeff = torch.nn.Parameter(torch.zeros((control_count, 3), device=device))
        translation_coeff = torch.nn.Parameter(torch.zeros((control_count, 3), device=device))
        body_coeff = torch.nn.Parameter(torch.zeros((control_count, len(arm_joints), 3), device=device))
        left_coeff = torch.nn.Parameter(torch.zeros((control_count, 15, 3), device=device))
        right_coeff = torch.nn.Parameter(torch.zeros((control_count, 15, 3), device=device))
        learning_rate = float(settings["ik_learning_rate"])
        mano_learning_rate = learning_rate * float(settings.get("mano_learning_rate_scale", 1.0))
        parameters = [root_coeff, translation_coeff, body_coeff, left_coeff, right_coeff]
        optimizer = torch.optim.Adam([
            {"params": [root_coeff, translation_coeff, body_coeff], "lr": learning_rate},
            {"params": [left_coeff, right_coeff], "lr": mano_learning_rate},
        ])
        initial_learning_rates = (learning_rate, mano_learning_rate)
        window_step = max(1, window - degree * 2)
    else:
        basis_np = None
        c4_knots = np.asarray([], dtype=np.float64)
        control_count = 0
        window_step = window
    optimized_windows = 0
    with _device_lock(config, use_cuda, device_index):
        for start in range(0, frames, window_step):
            end = min(frames, start + window)
            active = (
                geometry["left_contact"][start:end].any() or
                geometry["right_contact"][start:end].any() or
                (geometry["penetration"][start:end] > float(settings["penetration_margin_m"])).any()
            )
            if not active:
                continue
            optimized_windows += 1
            count = end - start
            body0 = torch.tensor(
                baseline[start:end] if direct_c4 else body[start:end], device=device,
            )
            if direct_c4:
                basis = torch.tensor(basis_np[start:end], device=device)
                active_control = basis.abs().sum(dim=0) > 0
                delta = torch.einsum("tc,cjd->tjd", basis, body_coeff)
                root_delta = torch.einsum("tc,cd->td", basis, root_coeff)
                translation_delta = torch.einsum("tc,cd->td", basis, translation_coeff)
                left_delta = torch.einsum("tc,cjd->tjd", basis, left_coeff)
                right_delta = torch.einsum("tc,cjd->tjd", basis, right_coeff)
            else:
                delta = torch.nn.Parameter(torch.zeros((count, len(arm_joints), 3), device=device))
                root_delta = torch.nn.Parameter(torch.zeros((count, 3), device=device))
                translation_delta = torch.nn.Parameter(torch.zeros((count, 3), device=device))
                if optimize_hand_pose:
                    left_delta = torch.nn.Parameter(torch.zeros((count, 15, 3), device=device))
                    right_delta = torch.nn.Parameter(torch.zeros((count, 15, 3), device=device))
                    parameters = [root_delta, translation_delta, delta, left_delta, right_delta]
                    learning_rate = float(settings["ik_learning_rate"])
                    mano_learning_rate = (
                        learning_rate * float(settings.get("mano_learning_rate_scale", 1.0))
                    )
                    optimizer = torch.optim.Adam([
                        {"params": [root_delta, translation_delta, delta], "lr": learning_rate},
                        {"params": [left_delta, right_delta], "lr": mano_learning_rate},
                    ])
                    initial_learning_rates = (learning_rate, mano_learning_rate)
                else:
                    left_delta = torch.zeros((count, 15, 3), device=device)
                    right_delta = torch.zeros((count, 15, 3), device=device)
                    parameters = [root_delta, translation_delta, delta]
                    learning_rate = float(settings["ik_learning_rate"])
                    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
                    initial_learning_rates = (learning_rate,)
            root0 = torch.tensor(root_baseline[start:end] if direct_c4 else roots[start:end], device=device)
            left_hand0 = torch.tensor(
                left_hand_baseline[start:end] if direct_c4 else left_hand_result[start:end], device=device,
            )
            right_hand0 = torch.tensor(
                right_hand_baseline[start:end] if direct_c4 else right_hand_result[start:end], device=device,
            )
            translation0 = torch.tensor(
                translation_baseline[start:end] if direct_c4 else translations[start:end], device=device,
            )
            betas = torch.tensor(motion.profile.betas, device=device).reshape(1, -1).expand(count, -1)
            zeros3 = torch.zeros((count, 3), device=device)
            with torch.no_grad():
                baseline_output = model(
                    global_orient=root0, body_pose=body0.reshape(count, 63),
                    left_hand_pose=left_hand0.reshape(count, 45),
                    right_hand_pose=right_hand0.reshape(count, 45), betas=betas,
                    jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                    expression=torch.zeros((count, model.num_expression_coeffs), device=device),
                    return_verts=False,
                )
                foot_indices = torch.tensor(
                    settings.get("foot_joint_indices", [7, 8, 10, 11]),
                    dtype=torch.long, device=device,
                )
                baseline_feet = (
                    baseline_output.joints[:, foot_indices] * motion.scale +
                    translation0[:, None]
                )
            for iteration in range(iterations):
                iteration_progress = iteration / max(1, iterations - 1)
                final_scale = float(settings.get("ik_final_learning_rate_scale", 0.05))
                cosine_scale = _cosine_learning_rate_scale(iteration_progress, final_scale)
                coarse_fraction = float(settings.get("coarse_reach_fraction", 0.20))
                for group_index, (group, initial_learning_rate) in enumerate(zip(
                    optimizer.param_groups, initial_learning_rates,
                )):
                    mano_enabled = not (optimize_hand_pose and group_index == 1 and iteration_progress < coarse_fraction)
                    group["lr"] = initial_learning_rate * cosine_scale * float(mano_enabled)
                optimizer.zero_grad(set_to_none=True)
                if direct_c4:
                    delta = torch.einsum("tc,cjd->tjd", basis, body_coeff)
                    root_delta = torch.einsum("tc,cd->td", basis, root_coeff)
                    translation_delta = torch.einsum("tc,cd->td", basis, translation_coeff)
                    left_delta = torch.einsum("tc,cjd->tjd", basis, left_coeff)
                    right_delta = torch.einsum("tc,cjd->tjd", basis, right_coeff)
                current = body0.clone()
                current[:, arm_joints] = _compose_axis_angle(
                    body0[:, arm_joints], delta,
                )
                current_left_hand = _compose_axis_angle(left_hand0, left_delta)
                current_right_hand = _compose_axis_angle(right_hand0, right_delta)
                current_root = _compose_axis_angle(root0, root_delta)
                current_translation = translation0 + translation_delta
                output = model(
                    global_orient=current_root, body_pose=current.reshape(count, 63),
                    left_hand_pose=current_left_hand.reshape(count, 45),
                    right_hand_pose=current_right_hand.reshape(count, 45), betas=betas,
                    jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                    expression=torch.zeros((count, model.num_expression_coeffs), device=device),
                    return_verts=True,
                )
                vertices = output.vertices * motion.scale + current_translation[:, None]
                contact_plane_loss = torch.zeros((), device=device)
                contact_tangent_loss = torch.zeros((), device=device)
                contact_normal_loss = torch.zeros((), device=device)
                contact_slip_loss = torch.zeros((), device=device)
                penetration_loss = torch.zeros((), device=device)
                penetration_max_loss = torch.zeros((), device=device)
                wrist_palm_landmark_loss = torch.zeros((), device=device)
                finger_landmark_loss = torch.zeros((), device=device)
                hand_direction_loss = torch.zeros((), device=device)
                hand_bend_loss = torch.zeros((), device=device)
                thumb_opposition_loss = torch.zeros((), device=device)
                for object_index, grid in enumerate(grids):
                    rotation = torch.tensor(
                        Rotation.from_quat(object_quaternion[start:end, object_index]).as_matrix(),
                        dtype=vertices.dtype, device=device,
                    )
                    center = torch.tensor(object_translation[start:end, object_index], device=device)
                    target_landmarks = _target_hand_landmarks(
                        output.joints, output.vertices,
                    ) * motion.scale
                    target_landmarks = target_landmarks + current_translation[:, None, None]
                    target_landmarks_local = torch.einsum(
                        "tslj,tjk->tslk",
                        target_landmarks - center[:, None, None], rotation,
                    )
                    source_landmarks_local = torch.tensor(
                        geometry.get(
                            "target_reference_hand_landmarks_local",
                            geometry["source_hand_landmarks_local"],
                        )[start:end, object_index],
                        device=device,
                    )
                    raw_source_landmarks_local = torch.tensor(
                        geometry["source_hand_landmarks_local"][start:end, object_index],
                        device=device,
                    )
                    source_contact = torch.stack([
                        torch.tensor(geometry["left_contact"][start:end, object_index], device=device),
                        torch.tensor(geometry["right_contact"][start:end, object_index], device=device),
                    ], dim=1)
                    if source_contact.any():
                        # Wrist and palm establish the absolute reach.  Finger
                        # landmarks retain the released articulation without
                        # being allowed to move the immutable object.
                        wrist_palm = torch.linalg.vector_norm(
                            target_landmarks_local[:, :, :2] - source_landmarks_local[:, :, :2],
                            dim=-1,
                        )
                        fingers = torch.linalg.vector_norm(
                            target_landmarks_local[:, :, 2:] - source_landmarks_local[:, :, 2:],
                            dim=-1,
                        )
                        wrist_palm_landmark_loss = (
                            wrist_palm_landmark_loss + wrist_palm[source_contact].square().mean()
                        )
                        finger_landmark_loss = (
                            finger_landmark_loss + fingers[source_contact].square().mean()
                        )
                        target_directions = []
                        source_directions = []
                        target_bends = []
                        source_bends = []
                        for base in (2, 6, 10, 14, 18):
                            chain = [base, base + 1, base + 2, base + 3]
                            target_bones = torch.diff(target_landmarks_local[:, :, chain], dim=2)
                            source_bones = torch.diff(raw_source_landmarks_local[:, :, chain], dim=2)
                            target_unit = torch.nn.functional.normalize(target_bones, dim=-1)
                            source_unit = torch.nn.functional.normalize(source_bones, dim=-1)
                            target_directions.append(target_unit)
                            source_directions.append(source_unit)
                            target_bends.append((target_unit[:, :, :-1] * target_unit[:, :, 1:]).sum(-1))
                            source_bends.append((source_unit[:, :, :-1] * source_unit[:, :, 1:]).sum(-1))
                        direction_cosine = (
                            torch.cat(target_directions, dim=2) *
                            torch.cat(source_directions, dim=2)
                        ).sum(-1)
                        direction_error = 1.0 - direction_cosine[source_contact]
                        direction_tail = torch.topk(
                            direction_error.reshape(-1),
                            k=max(1, int(np.ceil(0.10 * direction_error.numel()))),
                        ).values.mean()
                        hand_direction_loss = (
                            hand_direction_loss + direction_error.mean() +
                            float(settings.get("source_hand_tail_multiplier", 2.0)) * direction_tail
                        )
                        bend_difference = (
                            torch.cat(target_bends, dim=2) -
                            torch.cat(source_bends, dim=2)
                        )
                        bend_error = bend_difference[source_contact].square()
                        bend_tail = torch.topk(
                            bend_error.reshape(-1),
                            k=max(1, int(np.ceil(0.10 * bend_error.numel()))),
                        ).values.mean()
                        hand_bend_loss = (
                            hand_bend_loss + bend_error.mean() +
                            float(settings.get("source_hand_tail_multiplier", 2.0)) * bend_tail
                        )
                        target_opposition = torch.stack([
                            torch.linalg.vector_norm(
                                target_landmarks_local[:, :, 21] - target_landmarks_local[:, :, 2], dim=-1,
                            ),
                            torch.linalg.vector_norm(
                                target_landmarks_local[:, :, 21] - target_landmarks_local[:, :, 5], dim=-1,
                            ),
                        ], dim=-1)
                        source_opposition = torch.stack([
                            torch.linalg.vector_norm(
                                raw_source_landmarks_local[:, :, 21] - raw_source_landmarks_local[:, :, 2], dim=-1,
                            ),
                            torch.linalg.vector_norm(
                                raw_source_landmarks_local[:, :, 21] - raw_source_landmarks_local[:, :, 5], dim=-1,
                            ),
                        ], dim=-1)
                        thumb_opposition_loss = thumb_opposition_loss + (
                            target_opposition[source_contact] - source_opposition[source_contact]
                        ).square().mean()
                    for side in ("left", "right"):
                        valid = torch.tensor(
                            geometry[f"{side}_patch_valid"][start:end, object_index], device=device,
                        )
                        if valid.any():
                            vertex_ids = torch.tensor(
                                geometry[f"{side}_patch_vertex"][start:end, object_index],
                                dtype=torch.long, device=device,
                            )
                            row = torch.arange(count, device=device)[:, None]
                            predicted = vertices[row, torch.clamp(vertex_ids, min=0)]
                            anchor_local = torch.tensor(
                                geometry[f"{side}_patch_anchor"][start:end, object_index], device=device,
                            )
                            normal_local = torch.tensor(
                                geometry[f"{side}_patch_normal"][start:end, object_index], device=device,
                            )
                            anchor_world = torch.einsum("tki,tji->tkj", anchor_local, rotation) + center[:, None]
                            normal_world = torch.einsum("tki,tji->tkj", normal_local, rotation)
                            difference = predicted - anchor_world
                            projection = (difference * normal_world).sum(dim=-1)
                            normal_gap = projection - float(settings.get("contact_surface_offset_m", 0.001))
                            tangent = difference - projection[..., None] * normal_world
                            contact_plane_loss = contact_plane_loss + normal_gap[valid].square().mean()
                            contact_tangent_loss = contact_tangent_loss + tangent[valid].square().sum(dim=-1).mean()
                            human_normal = _selected_vertex_normals(
                                vertices, vertex_ids, face_table, face_mask,
                            )
                            opposition = (human_normal * normal_world).sum(dim=-1) + 1.0
                            contact_normal_loss = contact_normal_loss + opposition[valid].square().mean()
                            if count > 1:
                                local_point = torch.einsum(
                                    "tki,tij->tkj", predicted - center[:, None], rotation,
                                )
                                proxy_identity = torch.tensor(
                                    geometry["dynamic_contact_proxy_index"][
                                        start:end, object_index,
                                        0 if side == "left" else 1,
                                    ],
                                    dtype=torch.long, device=device,
                                )
                                pair = (
                                    valid[1:] & valid[:-1] &
                                    (proxy_identity[1:] == proxy_identity[:-1]) &
                                    (vertex_ids[1:] == vertex_ids[:-1])
                                )
                                if pair.any():
                                    # Express slip in m/s, matching the quality
                                    # gate.  Per-frame displacement would make
                                    # this term 90x too small on HiPHI.
                                    target_velocity = (
                                        local_point[1:] - local_point[:-1]
                                    ) * float(motion.fps)
                                    anchor_velocity = (
                                        anchor_local[1:] - anchor_local[:-1]
                                    ) * float(motion.fps)
                                    velocity = target_velocity - anchor_velocity
                                    pair_normal = normal_local[1:]
                                    tangent_velocity = velocity - (
                                        velocity * pair_normal
                                    ).sum(dim=-1, keepdim=True) * pair_normal
                                    contact_slip_loss = contact_slip_loss + tangent_velocity[pair].square().sum(dim=-1).mean()
                    sampled = vertices[:, body_vertex_ids]
                    local = torch.einsum("tvi,tij->tvj", sampled - center[:, None], rotation)
                    sdf = _torch_grid_sample(local.reshape(-1, 3), grid, device).reshape(count, -1)
                    violation = torch.relu(
                        float(settings["penetration_margin_m"]) - sdf
                    )
                    top_count = min(int(settings["penetration_topk_vertices"]), violation.shape[1])
                    penetration_loss = penetration_loss + torch.topk(
                        violation, k=top_count, dim=1, largest=True,
                    ).values.square().mean()
                    penetration_max_loss = penetration_max_loss + violation.max(dim=1).values.square().mean()
                body_relative = batch_rodrigues(delta.reshape(-1, 3))
                body_cosine = torch.clamp((torch.diagonal(body_relative, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0, -1.0, 1.0)
                hand_relative = batch_rodrigues(torch.cat([
                    left_delta.reshape(-1, 3), right_delta.reshape(-1, 3),
                ]))
                hand_cosine = torch.clamp((torch.diagonal(hand_relative, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0, -1.0, 1.0)
                # 2(1-cos(theta)) matches theta^2 near zero without the
                # singular derivative of acos at the identity initialization.
                body_prior = (2.0 * (1.0 - body_cosine)).mean()
                hand_prior = (2.0 * (1.0 - hand_cosine)).mean()
                root_prior = root_delta.square().mean()
                translation_prior = translation_delta.square().mean()
                current_feet = (
                    output.joints[:, foot_indices] * motion.scale + current_translation[:, None]
                )
                foot_preservation = (current_feet - baseline_feet).square().sum(dim=-1).mean()
                temporal = torch.zeros((), device=device)
                acceleration = torch.zeros((), device=device)
                if count > 1:
                    temporal = (
                        (delta[1:] - delta[:-1]).square().mean() +
                        (root_delta[1:] - root_delta[:-1]).square().mean() +
                        (translation_delta[1:] - translation_delta[:-1]).square().mean() +
                        (left_delta[1:] - left_delta[:-1]).square().mean() +
                        (right_delta[1:] - right_delta[:-1]).square().mean()
                    )
                if count > 2:
                    acceleration = (
                        (delta[2:] - 2.0 * delta[1:-1] + delta[:-2]).square().mean() +
                        (root_delta[2:] - 2.0 * root_delta[1:-1] + root_delta[:-2]).square().mean() +
                        (translation_delta[2:] - 2.0 * translation_delta[1:-1] + translation_delta[:-2]).square().mean() +
                        (left_delta[2:] - 2.0 * left_delta[1:-1] + left_delta[:-2]).square().mean() +
                        (right_delta[2:] - 2.0 * right_delta[1:-1] + right_delta[:-2]).square().mean()
                    )
                # Hierarchical continuation: first settle the immutable-object
                # contact manifold, then introduce non-contact penetration.
                # Contact stays active throughout; pose preservation is last.
                penetration_progress = _continuation_scale(iteration_progress, 0.20, 0.50)
                # The maximum-depth barrier must reach full strength while
                # there are still useful optimizer steps left.  Ramping it to
                # 1.0 only on the final iteration made the reported per-frame
                # maximum effectively unoptimized.
                max_penetration_progress = _continuation_scale(iteration_progress, 0.20, 0.50)
                pose_progress = min(max((iteration_progress - 0.20) / 0.40, 0.15), 1.0)
                surface_progress = _continuation_scale(
                    iteration_progress, coarse_fraction,
                    float(settings.get("fine_contact_fraction", 0.45)),
                )
                # Strict optimization order from the v3 contract. The object
                # pose is not an optimization variable. Released contact
                # anchors and object-local hand trajectories are therefore the
                # first active level, ahead of collision cleanup. Keeping the
                # latter separate prevents a large SDF coefficient from
                # pulling the right hand away from its observed grasp.
                contact_anchor_objective = (
                    surface_progress * (
                        float(settings.get("contact_plane_weight", 100.0)) * contact_plane_loss +
                        float(settings.get("contact_tangent_weight", 20.0)) * contact_tangent_loss +
                        float(settings.get("contact_slip_weight", 10.0)) * contact_slip_loss
                    )
                )
                trajectory_objective = (
                    float(settings.get("source_landmark_weight", 8000.0)) * (
                        wrist_palm_landmark_loss + surface_progress *
                        float(settings.get("finger_landmark_relative_weight", 0.35)) *
                        finger_landmark_loss
                    )
                )
                collision_objective = (
                    surface_progress *
                    float(settings.get("contact_normal_weight", 0.05)) * contact_normal_loss +
                    float(settings["penetration_weight"]) *
                    penetration_progress ** 2 * penetration_loss +
                    float(settings.get("penetration_max_weight", 0.0)) *
                    max_penetration_progress ** 2 * penetration_max_loss
                )
                dynamic_objective = (
                    float(settings.get("source_hand_direction_weight", 10.0)) * hand_direction_loss +
                    float(settings.get("source_hand_bend_weight", 10.0)) * hand_bend_loss +
                    float(settings.get("thumb_opposition_weight", 5000.0)) * thumb_opposition_loss
                )
                posture_objective = (
                    pose_progress * float(settings["pose_prior_weight"]) * body_prior +
                    pose_progress * float(settings.get("hand_pose_prior_weight", 0.25)) * hand_prior +
                    pose_progress * float(settings.get("root_pose_prior_weight", 8.0)) * root_prior +
                    pose_progress * float(settings.get("root_translation_prior_weight", 80.0)) * translation_prior +
                    pose_progress * float(settings.get("foot_preservation_weight", 1000.0)) * foot_preservation +
                    pose_progress * float(settings["delta_velocity_weight"]) * temporal +
                    pose_progress * float(settings.get("delta_acceleration_weight", 8.0)) * acceleration
                )
                loss = (
                    contact_anchor_objective + trajectory_objective + collision_objective +
                    dynamic_objective + posture_objective
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite object-contact IK loss or gradient")
                objectives = (
                    contact_anchor_objective, trajectory_objective, collision_objective,
                    dynamic_objective, posture_objective,
                )
                gradient_levels = []
                differentiable = [
                    index for index, objective in enumerate(objectives)
                    if objective.requires_grad
                ]
                last_differentiable = differentiable[-1]
                for index, objective in enumerate(objectives):
                    gradient_levels.append(
                        torch.autograd.grad(
                            objective, parameters,
                            retain_graph=index != last_differentiable,
                            allow_unused=True,
                        ) if objective.requires_grad else [None] * len(parameters)
                    )
                projected_gradients = _hierarchical_priority_gradients(
                    parameters, gradient_levels,
                )
                for parameter, projected_gradient in zip(parameters, projected_gradients):
                    parameter.grad = projected_gradient
                if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
                    raise FloatingPointError("Non-finite object-contact IK loss or gradient")
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                frozen_values = None
                if direct_c4:
                    # Adam updates zero-gradient coefficients through its
                    # momentum.  Restore controls outside this overlapping
                    # window so an unseen part of the trajectory cannot drift.
                    frozen_values = [
                        parameter.detach()[~active_control].clone()
                        for parameter in parameters
                    ]
                optimizer.step()
                with torch.no_grad():
                    if direct_c4:
                        for parameter, frozen in zip(parameters, frozen_values):
                            parameter[~active_control] = frozen
                        body_norm = torch.linalg.vector_norm(body_coeff, dim=-1, keepdim=True)
                        body_coeff.mul_(torch.minimum(
                            torch.ones_like(body_norm),
                            arm_maximum[None, :, None] / torch.clamp(body_norm, min=1e-8),
                        ))
                        for hand_coeff in (left_coeff, right_coeff):
                            hand_norm = torch.linalg.vector_norm(hand_coeff, dim=-1, keepdim=True)
                            hand_coeff.mul_(torch.minimum(
                                torch.ones_like(hand_norm), hand_maximum / torch.clamp(hand_norm, min=1e-8),
                            ))
                        root_norm = torch.linalg.vector_norm(root_coeff, dim=-1, keepdim=True)
                        root_coeff.mul_(torch.minimum(
                            torch.ones_like(root_norm), root_rotation_maximum / torch.clamp(root_norm, min=1e-8),
                        ))
                        translation_norm = torch.linalg.vector_norm(translation_coeff, dim=-1, keepdim=True)
                        translation_coeff.mul_(torch.minimum(
                            torch.ones_like(translation_norm),
                            root_translation_maximum / torch.clamp(translation_norm, min=1e-8),
                        ))
                    else:
                        norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                        delta.mul_(torch.minimum(torch.ones_like(norm), arm_maximum[None, :, None] / torch.clamp(norm, min=1e-8)))
                        if optimize_hand_pose:
                            for hand_delta in (left_delta, right_delta):
                                hand_norm = torch.linalg.vector_norm(hand_delta, dim=-1, keepdim=True)
                                hand_delta.mul_(torch.minimum(
                                    torch.ones_like(hand_norm), hand_maximum / torch.clamp(hand_norm, min=1e-8),
                                ))
                        root_norm = torch.linalg.vector_norm(root_delta, dim=-1, keepdim=True)
                        root_delta.mul_(torch.minimum(
                            torch.ones_like(root_norm), root_rotation_maximum / torch.clamp(root_norm, min=1e-8),
                        ))
                        translation_norm = torch.linalg.vector_norm(translation_delta, dim=-1, keepdim=True)
                        translation_delta.mul_(torch.minimum(
                            torch.ones_like(translation_norm),
                            root_translation_maximum / torch.clamp(translation_norm, min=1e-8),
                        ))
            if not direct_c4:
                with torch.no_grad():
                    body[start:end, arm_joints] = _compose_axis_angle(
                        body0[:, arm_joints], delta,
                    ).cpu().numpy()
                    roots[start:end] = _compose_axis_angle(root0, root_delta).cpu().numpy()
                    translations[start:end] = (translation0 + translation_delta).cpu().numpy()
                if optimize_hand_pose:
                    with torch.no_grad():
                        left_hand_result[start:end] = _compose_axis_angle(
                            left_hand0, left_delta,
                        ).cpu().numpy()
                        right_hand_result[start:end] = _compose_axis_angle(
                            right_hand0, right_delta,
                        ).cpu().numpy()
    if direct_c4:
        body_delta_np = np.einsum("tc,cjd->tjd", basis_np, body_coeff.detach().cpu().numpy())
        root_delta_np = np.einsum("tc,cd->td", basis_np, root_coeff.detach().cpu().numpy())
        translation_delta_np = np.einsum(
            "tc,cd->td", basis_np, translation_coeff.detach().cpu().numpy(),
        )
        left_delta_np = np.einsum("tc,cjd->tjd", basis_np, left_coeff.detach().cpu().numpy())
        right_delta_np = np.einsum("tc,cjd->tjd", basis_np, right_coeff.detach().cpu().numpy())

        def compose_numpy(base: np.ndarray, delta_value: np.ndarray) -> np.ndarray:
            base_matrix = Rotation.from_rotvec(base.reshape(-1, 3)).as_matrix()
            delta_matrix = Rotation.from_rotvec(delta_value.reshape(-1, 3)).as_matrix()
            return Rotation.from_matrix(base_matrix @ delta_matrix).as_rotvec().reshape(base.shape)

        body = baseline.copy()
        body[:, arm_joints] = compose_numpy(baseline[:, arm_joints], body_delta_np)
        roots = compose_numpy(root_baseline, root_delta_np)
        translations = translation_baseline + translation_delta_np
        left_hand_result = compose_numpy(left_hand_baseline, left_delta_np)
        right_hand_result = compose_numpy(right_hand_baseline, right_delta_np)
    # Measure the optimizer result before the mandatory C4 projection.  This
    # separates IK feasibility from spline ringing in the acceptance record.
    motion.body_pose = body.reshape(frames, 63).astype(np.float32)
    motion.global_orient = roots.astype(np.float32)
    motion.transl = translations.astype(np.float32)
    motion.left_hand_pose = left_hand_result.reshape(frames, 45).astype(np.float32)
    motion.right_hand_pose = right_hand_result.reshape(frames, 45).astype(np.float32)
    raw_geometry = _measure_geometry(
        config, motion, grids, object_translation, object_quaternion,
        contact_reference=geometry,
    )
    raw_contact_quality = _contact_patch_quality(
        config, motion, raw_geometry, object_translation, object_quaternion,
    )
    raw_contact_quality.update(_penetration_metrics(raw_geometry))
    if direct_c4:
        smoothing = left_smoothing = right_smoothing = {
            "applied": True,
            "reason": "joint_ik_optimized_directly_in_shared_quintic_bspline_control_space",
            "continuity_order": int(config.raw["temporal_continuity"]["spline_degree"]) - 1,
            "control_count": int(control_count),
            "knot_count": int(len(c4_knots)),
            "window_step_frames": int(window_step),
        }
    elif bool(settings.get("c4_post_projection", True)):
        times = motion.timestamps
        any_contact = geometry["left_contact"].any(axis=1) | geometry["right_contact"].any(axis=1)
        any_penetration = geometry["penetration"].max(axis=1) > float(settings["penetration_margin_m"])
        body_spline_weights = 1.0 + 1.0 * (any_contact | any_penetration)
        smoothed, smoothing = c3_smooth_relative_rotvec(
            times, baseline.reshape(frames, 63), body.reshape(frames, 63),
            float(settings["c4_knot_seconds"]),
            int(config.raw["temporal_continuity"]["spline_degree"]),
            int(config.raw["temporal_continuity"]["robust_iterations"]),
            float(settings.get(
                "c4_huber_degrees", config.raw["temporal_continuity"]["rotation_huber_degrees"],
            )), body_spline_weights,
        )
        smoothed = smoothed.reshape(frames, 21, 3)
        keep = np.ones(21, dtype=bool)
        keep[arm_joints] = False
        smoothed[:, keep] = baseline[:, keep]
        motion.body_pose = smoothed.reshape(frames, 63).astype(np.float32)
        if optimize_hand_pose:
            left_smoothed, left_smoothing = c3_smooth_relative_rotvec(
                times, left_hand_baseline.reshape(frames, 45), left_hand_result.reshape(frames, 45),
                float(settings["c4_knot_seconds"]),
                int(config.raw["temporal_continuity"]["spline_degree"]),
                int(config.raw["temporal_continuity"]["robust_iterations"]),
                float(settings.get(
                    "c4_huber_degrees", config.raw["temporal_continuity"]["rotation_huber_degrees"],
                )), 1.0 + 1.0 * geometry["left_contact"].any(axis=1),
            )
            right_smoothed, right_smoothing = c3_smooth_relative_rotvec(
                times, right_hand_baseline.reshape(frames, 45), right_hand_result.reshape(frames, 45),
                float(settings["c4_knot_seconds"]),
                int(config.raw["temporal_continuity"]["spline_degree"]),
                int(config.raw["temporal_continuity"]["robust_iterations"]),
                float(settings.get(
                    "c4_huber_degrees", config.raw["temporal_continuity"]["rotation_huber_degrees"],
                )), 1.0 + 1.0 * geometry["right_contact"].any(axis=1),
            )
            motion.left_hand_pose = left_smoothed.astype(np.float32)
            motion.right_hand_pose = right_smoothed.astype(np.float32)
        else:
            left_smoothing = right_smoothing = {"applied": False, "reason": "native_mano_frozen"}
    else:
        smoothing = left_smoothing = right_smoothing = {
            "applied": False,
            "reason": "exact_degree5_archive_interpolant_preserves_constrained_samples",
            "continuity_order": int(config.raw["temporal_continuity"]["spline_degree"]) - 1,
        }
    c4_polish = _c4_contact_polish(
        config, motion, grids, object_translation, object_quaternion, geometry,
    )
    motion.continuous_spline = c3_archive_payload(
        motion.timestamps, motion.global_orient, motion.body_pose,
        motion.left_hand_pose, motion.right_hand_pose, motion.transl,
        int(config.raw["temporal_continuity"]["spline_degree"]),
    )
    return {
        "optimized_windows": optimized_windows, "total_windows": int(np.ceil(frames / window)),
        "optimized_body_joint_indices": arm_joints.tolist(), "hand_pose_modified": optimize_hand_pose,
        "root_translation_modified": True, "c4_projection": smoothing,
        "contact_model": "persistent_multi_vertex_point_to_plane_normal_tangent_slip",
        "priority_solver": (
            "five_level_gradient_projection: contact_anchor -> object_local_hand_trajectory -> "
            "normal_and_penetration -> source_finger_dynamics -> body_and_temporal_priors"
        ),
        "direct_c4_control_optimization": direct_c4,
        "optimizer_schedule": {
            "name": "cosine_decay",
            "initial_learning_rate": float(settings["ik_learning_rate"]),
            "final_learning_rate_scale": float(
                settings.get("ik_final_learning_rate_scale", 0.05)
            ),
        },
        "pre_c4_contact_quality": raw_contact_quality,
        "self_collision_enabled": False,
        "left_hand_c4_projection": left_smoothing, "right_hand_c4_projection": right_smoothing,
        "c4_constraint_polish": c4_polish,
    }


def _c4_contact_polish(
    config: RetargetConfig, motion: MotionRecord, grids: list[SignedDistanceGrid],
    object_translation: np.ndarray, object_quaternion: np.ndarray,
    geometry: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Correct contact/collision in spline-control space without post-projection."""
    import torch

    settings = config.raw["object_pilot"]
    iterations = int(settings.get("c4_polish_iterations", 0))
    if iterations <= 0:
        return {"applied": False, "reason": "disabled"}
    degree = int(config.raw["temporal_continuity"]["spline_degree"])
    basis_np, knots = c4_bspline_basis(
        motion.timestamps, float(settings["c4_knot_seconds"]), degree,
    )
    device, use_cuda, device_index = _device(config)
    model = _model(config, motion.profile, device)
    face_table_np, face_mask_np = _incident_face_table(
        np.asarray(model.faces), model.lbs_weights.shape[0],
    )
    face_table = torch.tensor(face_table_np, dtype=torch.long, device=device)
    face_mask = torch.tensor(face_mask_np, dtype=torch.bool, device=device)
    _, _, body_vertex_ids_np = _vertex_sets(
        model, int(settings["hand_vertex_samples"]), int(settings["body_vertex_samples"]),
    )
    body_vertex_ids = torch.tensor(body_vertex_ids_np, dtype=torch.long, device=device)
    arm_joints = np.asarray(
        settings.get("fine_optimized_body_joint_indices", settings["optimized_body_joint_indices"]),
        dtype=np.int64,
    )
    body_base = np.asarray(motion.body_pose, dtype=np.float32).reshape(-1, 21, 3).copy()
    left_base = np.asarray(motion.left_hand_pose, dtype=np.float32).reshape(-1, 15, 3).copy()
    right_base = np.asarray(motion.right_hand_pose, dtype=np.float32).reshape(-1, 15, 3).copy()
    control_count = basis_np.shape[1]
    body_coeff = torch.nn.Parameter(torch.zeros((control_count, len(arm_joints), 3), device=device))
    left_coeff = torch.nn.Parameter(torch.zeros((control_count, 15, 3), device=device))
    right_coeff = torch.nn.Parameter(torch.zeros((control_count, 15, 3), device=device))
    parameters = [body_coeff, left_coeff, right_coeff]
    optimizer = torch.optim.Adam(parameters, lr=float(settings.get("c4_polish_learning_rate", 0.005)))
    coefficient_limit = float(np.deg2rad(settings.get("c4_polish_max_degrees", 2.0)))
    window = int(settings["ik_window_frames"])
    active_windows = 0
    with _device_lock(config, use_cuda, device_index):
        for start in range(0, len(motion.timestamps), window):
            end = min(len(motion.timestamps), start + window)
            active = (
                geometry["left_contact"][start:end].any() or
                geometry["right_contact"][start:end].any() or
                (geometry["penetration"][start:end] > float(settings["penetration_margin_m"])).any()
            )
            if not active:
                continue
            active_windows += 1
            count = end - start
            basis = torch.tensor(basis_np[start:end], device=device)
            active_control = basis.abs().sum(dim=0) > 0
            body0 = torch.tensor(body_base[start:end], device=device)
            left0 = torch.tensor(left_base[start:end], device=device)
            right0 = torch.tensor(right_base[start:end], device=device)
            root = torch.tensor(motion.global_orient[start:end], device=device)
            translation = torch.tensor(motion.transl[start:end], device=device)
            betas = torch.tensor(motion.profile.betas, device=device).reshape(1, -1).expand(count, -1)
            zeros3 = torch.zeros((count, 3), device=device)
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                body_delta = torch.einsum("tc,cjd->tjd", basis, body_coeff)
                left_delta = torch.einsum("tc,cjd->tjd", basis, left_coeff)
                right_delta = torch.einsum("tc,cjd->tjd", basis, right_coeff)
                current_body = body0.clone()
                current_body[:, arm_joints] = _compose_axis_angle(
                    body0[:, arm_joints], body_delta,
                )
                current_left = _compose_axis_angle(left0, left_delta)
                current_right = _compose_axis_angle(right0, right_delta)
                output = model(
                    global_orient=root, body_pose=current_body.reshape(count, 63),
                    left_hand_pose=current_left.reshape(count, 45),
                    right_hand_pose=current_right.reshape(count, 45), betas=betas,
                    jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                    expression=torch.zeros((count, model.num_expression_coeffs), device=device),
                    return_verts=True,
                )
                vertices = output.vertices * motion.scale + translation[:, None]
                losses = _geometry_objectives(
                    vertices, geometry, start, end, grids, object_translation,
                    object_quaternion, face_table, face_mask, body_vertex_ids,
                    settings, motion.fps,
                )
                contact_objective = (
                    float(settings["contact_plane_weight"]) * losses["plane"] +
                    float(settings["contact_tangent_weight"]) * losses["tangent"] +
                    float(settings["contact_slip_weight"]) * losses["slip"]
                )
                collision_objective = (
                    float(settings["contact_normal_weight"]) * losses["normal"] +
                    float(settings["penetration_weight"]) * losses["penetration"] +
                    float(settings["penetration_max_weight"]) * losses["penetration_max"]
                )
                prior_objective = (
                    float(settings.get("c4_polish_prior_weight", 5.0)) * (
                        body_delta.square().mean() + left_delta.square().mean() +
                        right_delta.square().mean()
                    )
                )
                if contact_objective.requires_grad:
                    contact_gradients = torch.autograd.grad(
                        contact_objective, parameters, retain_graph=True, allow_unused=True,
                    )
                else:
                    contact_gradients = [None] * len(parameters)
                collision_gradients = torch.autograd.grad(
                    collision_objective, parameters, retain_graph=True, allow_unused=True,
                )
                prior_gradients = torch.autograd.grad(
                    prior_objective, parameters, allow_unused=True,
                )
                projected = _hierarchical_priority_gradients(
                    parameters, [contact_gradients, collision_gradients, prior_gradients],
                )
                for parameter, gradient in zip(parameters, projected):
                    parameter.grad = gradient
                if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
                    raise FloatingPointError("Non-finite C4 contact-polish gradient")
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                # Adam momentum updates zero-gradient coefficients. Preserve
                # controls outside the current window or a late contact window
                # will silently perturb earlier, already-polished grasps.
                frozen = [
                    parameter.detach()[~active_control].clone()
                    for parameter in parameters
                ]
                optimizer.step()
                with torch.no_grad():
                    for coefficient, frozen_value in zip(parameters, frozen):
                        coefficient[~active_control] = frozen_value
                        norm = torch.linalg.vector_norm(coefficient, dim=-1, keepdim=True)
                        coefficient.mul_(torch.minimum(
                            torch.ones_like(norm),
                            coefficient_limit / torch.clamp(norm, min=1e-8),
                        ))
    body_delta = np.einsum("tc,cjd->tjd", basis_np, body_coeff.detach().cpu().numpy())
    left_delta = np.einsum("tc,cjd->tjd", basis_np, left_coeff.detach().cpu().numpy())
    right_delta = np.einsum("tc,cjd->tjd", basis_np, right_coeff.detach().cpu().numpy())

    def compose_numpy(base: np.ndarray, delta: np.ndarray) -> np.ndarray:
        base_matrix = Rotation.from_rotvec(base.reshape(-1, 3)).as_matrix()
        delta_matrix = Rotation.from_rotvec(delta.reshape(-1, 3)).as_matrix()
        return Rotation.from_matrix(base_matrix @ delta_matrix).as_rotvec().reshape(base.shape)

    body_result = body_base.copy()
    body_result[:, arm_joints] = compose_numpy(body_base[:, arm_joints], body_delta)
    motion.body_pose = body_result.reshape(len(body_result), 63).astype(np.float32)
    motion.left_hand_pose = compose_numpy(left_base, left_delta).reshape(len(left_base), 45).astype(np.float32)
    motion.right_hand_pose = compose_numpy(right_base, right_delta).reshape(len(right_base), 45).astype(np.float32)
    return {
        "applied": True, "degree": degree, "continuity_order": degree - 1,
        "control_count": int(control_count), "knot_seconds": float(settings["c4_knot_seconds"]),
        "iterations_per_window": iterations, "active_windows": active_windows,
        "maximum_control_rotation_degrees": float(settings.get("c4_polish_max_degrees", 2.0)),
        "knots": int(len(knots)),
    }


def _contact_patch_quality(
    config: RetargetConfig, motion: MotionRecord, geometry: dict[str, Any],
    object_translation: np.ndarray, object_quaternion: np.ndarray,
) -> dict[str, float]:
    import torch

    device, use_cuda, device_index = _device(config)
    model = _model(config, motion.profile, device)
    face_table_np, face_mask_np = _incident_face_table(np.asarray(model.faces), model.lbs_weights.shape[0])
    face_table = torch.tensor(face_table_np, dtype=torch.long, device=device)
    face_mask = torch.tensor(face_mask_np, dtype=torch.bool, device=device)
    frames, objects = object_translation.shape[:2]
    patch_size = geometry["left_patch_vertex"].shape[2]
    angles, plane_gaps, tangent_errors = [], [], []
    contact_side_frames = 0
    active_patch_frames = 0
    active_patch_counts = []
    local_points = {
        side: np.zeros((frames, objects, patch_size, 3), dtype=np.float32)
        for side in ("left", "right")
    }
    batch = int(config.raw["object_pilot"]["forward_batch_frames"])
    with _device_lock(config, use_cuda, device_index), torch.no_grad():
        for start in range(0, frames, batch):
            end = min(frames, start + batch)
            output = _forward(model, motion, slice(start, end), device)
            vertices = output.vertices * motion.scale + torch.tensor(
                motion.transl[start:end, None], device=device,
            )
            for object_index in range(objects):
                rotation = torch.tensor(
                    Rotation.from_quat(object_quaternion[start:end, object_index]).as_matrix(),
                    dtype=vertices.dtype, device=device,
                )
                center = torch.tensor(object_translation[start:end, object_index], device=device)
                for side in ("left", "right"):
                    ids = torch.tensor(
                        geometry[f"{side}_patch_vertex"][start:end, object_index],
                        dtype=torch.long, device=device,
                    )
                    valid = torch.tensor(
                        geometry[f"{side}_patch_valid"][start:end, object_index], device=device,
                    )
                    contact = geometry[f"{side}_contact"][start:end, object_index]
                    contact_side_frames += int(contact.sum())
                    active_any = valid.any(dim=-1).cpu().numpy()
                    active_patch_frames += int((active_any & contact).sum())
                    if contact.any():
                        active_patch_counts.append(valid.sum(dim=-1).cpu().numpy()[contact])
                    row = torch.arange(end - start, device=device)[:, None]
                    points = vertices[row, torch.clamp(ids, min=0)]
                    local = torch.einsum("tki,tij->tkj", points - center[:, None], rotation)
                    local_points[side][start:end, object_index] = local.cpu().numpy()
                    if not valid.any():
                        continue
                    anchor = torch.tensor(
                        geometry[f"{side}_patch_anchor"][start:end, object_index], device=device,
                    )
                    object_normal_local = torch.tensor(
                        geometry[f"{side}_patch_normal"][start:end, object_index], device=device,
                    )
                    object_normal_world = torch.einsum("tki,tji->tkj", object_normal_local, rotation)
                    human_normal = _selected_vertex_normals(vertices, ids, face_table, face_mask)
                    opposition = torch.clamp(-(human_normal * object_normal_world).sum(dim=-1), -1.0, 1.0)
                    difference = local - anchor
                    projection = (difference * object_normal_local).sum(dim=-1)
                    tangent = difference - projection[..., None] * object_normal_local
                    angles.append(torch.rad2deg(torch.acos(opposition[valid])).cpu().numpy())
                    plane_gaps.append(torch.abs(projection[valid]).cpu().numpy())
                    tangent_errors.append(torch.linalg.vector_norm(tangent[valid], dim=-1).cpu().numpy())
    slip = []
    transition_frames = np.zeros(max(0, frames - 1), dtype=bool)
    fps = float(motion.fps)
    for side in ("left", "right"):
        valid = geometry[f"{side}_patch_valid"]
        ids = geometry[f"{side}_patch_vertex"]
        normals = geometry[f"{side}_patch_normal"]
        anchors = geometry[f"{side}_patch_anchor"]
        same_identity = (
            geometry.get("dynamic_contact_proxy_index", np.full_like(ids, -1))[1:, :,
                0 if side == "left" else 1] ==
            geometry.get("dynamic_contact_proxy_index", np.full_like(ids, -1))[:-1, :,
                0 if side == "left" else 1]
        ) if "dynamic_contact_proxy_index" in geometry else np.ones_like(valid[1:], dtype=bool)
        transition_frames |= (
            valid[1:] & valid[:-1] &
            ((ids[1:] != ids[:-1]) | ~same_identity)
        ).any(axis=(1, 2))
        pair = valid[1:] & valid[:-1] & same_identity & (ids[1:] == ids[:-1])
        target_velocity = (local_points[side][1:] - local_points[side][:-1]) * fps
        anchor_velocity = (anchors[1:] - anchors[:-1]) * fps
        relative_velocity = target_velocity - anchor_velocity
        tangent_velocity = relative_velocity - (
            relative_velocity * normals[1:]
        ).sum(axis=-1, keepdims=True) * normals[1:]
        if pair.any():
            slip.append(np.linalg.norm(tangent_velocity[pair], axis=-1))

    def summary(values: list[np.ndarray], prefix: str) -> dict[str, float]:
        joined = np.concatenate(values) if values else np.asarray([np.inf])
        return {
            f"{prefix}_median": float(np.median(joined)),
            f"{prefix}_p95": float(np.percentile(joined, 95)),
            f"{prefix}_max": float(np.max(joined)),
        }

    counts = np.concatenate(active_patch_counts) if active_patch_counts else np.asarray([0])
    active_counts = counts[counts > 0]
    return {
        **summary(angles, "normal_angle_deg"),
        **summary(plane_gaps, "normal_gap_m"),
        **summary(tangent_errors, "tangent_error_m"),
        **summary(slip, "tangent_slip_m_s"),
        "contact_patch_frame_coverage": float(active_patch_frames / max(1, contact_side_frames)),
        "contact_patch_active_vertices_median": float(np.median(active_counts)) if len(active_counts) else 0.0,
        "contact_patch_active_vertices_p05": float(np.percentile(active_counts, 5)) if len(active_counts) else 0.0,
        "contact_anchor_transition_frames": int(transition_frames.sum()),
        "contact_anchor_transitions_per_second": float(
            transition_frames.sum() / max(float(motion.timestamps[-1] - motion.timestamps[0]), 1e-8)
        ),
    }


def _contact_pose_deviation(
    baseline: MotionRecord, result: MotionRecord, geometry: dict[str, Any], arm_joints: np.ndarray,
) -> dict[str, float]:
    contact = geometry["left_contact"].any(axis=1) | geometry["right_contact"].any(axis=1)
    if not contact.any():
        return {"contact_frames": 0}

    def rotation_delta(before: np.ndarray, after: np.ndarray) -> np.ndarray:
        rb = Rotation.from_rotvec(before.reshape(-1, 3)).as_matrix()
        ra = Rotation.from_rotvec(after.reshape(-1, 3)).as_matrix()
        return np.degrees(Rotation.from_matrix(rb.transpose(0, 2, 1) @ ra).magnitude()).reshape(before.shape[:-1])

    before_body = baseline.body_pose.reshape(-1, 21, 3)[:, arm_joints]
    after_body = result.body_pose.reshape(-1, 21, 3)[:, arm_joints]
    arm = rotation_delta(before_body, after_body)
    before_hand = np.concatenate([
        baseline.left_hand_pose.reshape(-1, 15, 3), baseline.right_hand_pose.reshape(-1, 15, 3),
    ], axis=1)
    after_hand = np.concatenate([
        result.left_hand_pose.reshape(-1, 15, 3), result.right_hand_pose.reshape(-1, 15, 3),
    ], axis=1)
    hand = rotation_delta(before_hand, after_hand)

    def frame_metrics(values: np.ndarray, prefix: str) -> dict[str, float]:
        maximum = values[contact].max(axis=1)
        return {
            f"{prefix}_frame_median_deg": float(np.median(maximum)),
            f"{prefix}_frame_p95_deg": float(np.percentile(maximum, 95)),
            f"{prefix}_frame_max_deg": float(np.max(maximum)),
        }

    return {
        "contact_frames": int(contact.sum()), "contact_fraction": float(contact.mean()),
        **frame_metrics(arm, "arm_delta"), **frame_metrics(hand, "mano_delta"),
    }


def _joint_deviation(config: RetargetConfig, baseline: MotionRecord, result: MotionRecord) -> np.ndarray:
    import torch

    device, use_cuda, device_index = _device(config)
    model = _model(config, result.profile, device)
    chunks = []
    with _device_lock(config, use_cuda, device_index), torch.no_grad():
        for start in range(0, len(result.timestamps), 128):
            end = min(len(result.timestamps), start + 128)
            before = _forward(model, baseline, slice(start, end), device).joints[:, :55]
            after = _forward(model, result, slice(start, end), device).joints[:, :55]
            chunks.append(torch.linalg.vector_norm((after - before) * result.scale, dim=2).cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _penetration_metrics(geometry: dict[str, np.ndarray]) -> dict[str, float]:
    """Report depth P95 over positive sampled penetrations and frame maxima separately."""
    frame_maximum = np.asarray(geometry["penetration"], dtype=np.float64)
    positive_count = 0
    if "penetration_histogram_counts" in geometry:
        counts = np.asarray(geometry["penetration_histogram_counts"], dtype=np.int64).sum(axis=0)
        edges = np.asarray(geometry["penetration_histogram_edges"], dtype=np.float64)
        positive_count = int(counts.sum())
        if positive_count:
            target = 0.95 * positive_count
            cumulative = np.cumsum(counts)
            index = int(np.searchsorted(cumulative, target, side="left"))
            previous = int(cumulative[index - 1]) if index else 0
            fraction = (target - previous) / max(1, int(counts[index]))
            penetration_p95 = float(edges[index] + fraction * (edges[index + 1] - edges[index]))
        else:
            penetration_p95 = 0.0
    else:
        samples = np.asarray(
            geometry.get("penetration_samples", frame_maximum[..., None]), dtype=np.float64,
        )
        positive = samples[samples > 0.0]
        positive_count = int(len(positive))
        penetration_p95 = float(np.percentile(positive, 95)) if positive_count else 0.0
    return {
        "penetration_p95_m": penetration_p95,
        "penetration_max_m": float(frame_maximum.max(initial=0.0)),
        "penetration_frame_max_p95_m": float(np.percentile(frame_maximum, 95)),
        "penetration_positive_sample_count": positive_count,
    }


def build_object_motion(
    config: RetargetConfig, reference: SequenceRef, profile: RetargetProfile,
    input_hash: str, model_hash: str, pilot_windows: bool = False,
) -> MotionRecord:
    metadata = load_hiphi_metadata(reference)
    native_motion = bvh_pose(reference)[0]
    baseline = build_motion(config, reference, profile, input_mode="bvh")
    if abs(float(baseline.quality["source_motion_scale"]) - 1.0) > 1e-7:
        raise ValueError("Object pilot requires source-native height so object translation remains unscaled")
    baseline.quality.update({
        "config_hash": run_config_hash(config, reference, "bvh"),
        "input_hash": input_hash, "model_dependency_hash": model_hash,
        "baseline_for_object_retarget": True,
    })
    baseline_path = (
        config.output_root / "baselines" / reference.dataset / profile.name /
        f"{reference.sequence_id}.smplx.npz"
    )
    write_motion_npz(baseline_path, baseline)
    # Retain an independent pre-optimization record for joint-deviation metrics.
    baseline_copy = MotionRecord(
        baseline.sequence, baseline.profile, baseline.fps, baseline.timestamps.copy(),
        baseline.global_orient.copy(), baseline.body_pose.copy(), baseline.left_hand_pose.copy(),
        baseline.right_hand_pose.copy(), baseline.transl.copy(), baseline.scale,
        baseline.left_hand_valid.copy(), baseline.right_hand_valid.copy(),
        baseline.left_hand_confidence.copy(), baseline.right_hand_confidence.copy(),
        baseline.left_hand_imputed.copy(), baseline.right_hand_imputed.copy(),
        baseline.source_to_canonical.copy(), dict(baseline.quality), dict(baseline.continuous_spline),
    )
    sources = resolve_object_sources(config.dataset_path("hiphi"), reference, metadata)
    tracks, grids, meshes, asset_paths, asset_hashes = [], [], [], [], []
    for source in sources:
        timestamps, translation, quaternion = parse_object_track_csv(
            source.trajectory_path.read_bytes(), expected_frames=len(baseline.timestamps),
        )
        tolerance = max(2e-6, 0.02 / baseline.fps)
        if np.max(np.abs(timestamps - baseline.timestamps)) > tolerance:
            raise ValueError(f"Object/human timestamps disagree for {source.object_id}")
        canonical_translation, canonical_quaternion = canonicalize_object_track(
            translation, quaternion, baseline.source_to_canonical,
        )
        tracks.append((timestamps, translation, quaternion, canonical_translation, canonical_quaternion))
        mesh = load_repaired_mesh(source.mesh_path, float(config.raw["object_pilot"]["mesh_scale_to_m"]))
        meshes.append(mesh)
        grids.append(build_signed_distance_grid(
            mesh, pitch=float(config.raw["object_pilot"]["sdf_pitch_m"]),
            padding=float(config.raw["object_pilot"]["sdf_padding_m"]),
            sample_count=int(config.raw["object_pilot"]["sdf_surface_samples"]),
            seed=int(config.raw["object_pilot"]["selection_seed"]),
        ))
        asset, digest = copy_hashed_asset(source.mesh_path, config.output_root / "assets/objects")
        asset_paths.append(asset.relative_to(config.output_root).as_posix())
        asset_hashes.append(digest)
    source_timestamps = tracks[0][0]
    source_translation = np.stack([track[1] for track in tracks], axis=1)
    source_quaternion = np.stack([track[2] for track in tracks], axis=1)
    object_translation = np.stack([track[3] for track in tracks], axis=1)
    object_quaternion = np.stack([track[4] for track in tracks], axis=1)
    source_contact_hash = source_contact_dependency_hash(
        reference, sources, str(metadata.get("text_annotation", "")),
        dict(config.raw["object_pilot"]),
    )
    baseline.quality["source_contact_dependency_hash"] = source_contact_hash
    baseline_copy.quality["source_contact_dependency_hash"] = source_contact_hash
    source_contact_cache = (
        config.output_root / "cache/source_contacts" /
        f"{reference.sequence_id}.{source_contact_hash}.npz"
    )
    if source_contact_cache.is_file():
        source_contacts = read_source_hand_contacts(source_contact_cache)
    else:
        source_contacts = recover_source_hand_contacts(
            native_motion, source_translation, source_quaternion, meshes,
            baseline.fps, str(metadata.get("text_annotation", "")),
            dict(config.raw["object_pilot"]),
        )
        write_source_hand_contacts(source_contact_cache, source_contacts)
    solver_version = str(config.raw["object_pilot"].get("solver_version", "legacy_weighted_v3"))
    binding_contacts = source_contacts
    if solver_version == "constrained_se3_v4" and pilot_windows:
        from .constrained_object_v4 import deterministic_gold_windows

        selected_frames = np.zeros(len(baseline.timestamps), dtype=bool)
        for start, end in deterministic_gold_windows(
            source_contacts.contact, len(baseline.timestamps),
            int(config.raw["object_pilot"]["ik_window_frames"]),
        ):
            selected_frames[start:end] = True
        binding_contacts = replace(
            source_contacts,
            contact=source_contacts.contact & selected_frames[:, None, None],
            contact_valid=source_contacts.contact_valid & selected_frames[:, None, None, None],
            annotation_assisted=(
                source_contacts.annotation_assisted & selected_frames[:, None, None]
            ),
        )
    target_reference_landmarks_local = _target_shape_hand_reference(
        config, baseline, binding_contacts, object_translation, object_quaternion,
    )
    source_reference = _assign_source_contacts_to_target(
        config, baseline, binding_contacts, object_translation, object_quaternion,
        target_reference_landmarks_local,
    )
    if solver_version == "constrained_se3_v4" and pilot_windows:
        before = source_reference
        source_contact_patch_quality = {
            "status": "deferred_in_pilot_window_mode",
            "source_contact_frames": int(source_contacts.contact.sum()),
        }
    else:
        before = _measure_geometry(
            config, baseline, grids, object_translation, object_quaternion,
            contact_reference=source_reference,
        )
        source_contact_patch_quality = _contact_patch_quality(
            config, baseline_copy, before, object_translation, object_quaternion,
        )
    constrained = None
    if solver_version == "constrained_se3_v4":
        from .constrained_object_v4 import solve_constrained_object_motion

        coarse_retarget = {
            "applied": False,
            "reason": "v4 solves upper-body reach and MANO jointly in direct C4 control space",
        }
        constrained = solve_constrained_object_motion(
            config, baseline, source_contacts, meshes, grids,
            object_translation, object_quaternion, source_reference,
            target_reference_landmarks_local, pilot_windows=pilot_windows,
        )
        source_reference = dict(source_reference)
        source_reference.update({
            "dynamic_contact_constraint_valid": constrained["contact_constraint_valid"],
            "dynamic_contact_normal_constraint_valid": constrained[
                "contact_normal_constraint_valid"
            ],
            "dynamic_contact_target_vertex": constrained["contact_target_vertex"],
            "left_patch_vertex": constrained["contact_target_vertex"][:, :, 0],
            "right_patch_vertex": constrained["contact_target_vertex"][:, :, 1],
            "left_patch_anchor": constrained["remapped_contact_anchor_local"][:, :, 0],
            "right_patch_anchor": constrained["remapped_contact_anchor_local"][:, :, 1],
            "left_patch_normal": constrained["remapped_contact_normal_local"][:, :, 0],
            "right_patch_normal": constrained["remapped_contact_normal_local"][:, :, 1],
            "left_patch_valid": constrained["contact_constraint_valid"][:, :, 0],
            "right_patch_valid": constrained["contact_constraint_valid"][:, :, 1],
        })
        ik = {
            "solver_version": constrained["solver_version"],
            "pilot_windows": constrained["pilot_windows"],
            "windows": constrained["windows"],
            "direct_c4_control_optimization": True,
            "priority_solver": "per-side hierarchical augmented Lagrangian",
            "object_pose_optimized": False,
        }
    else:
        coarse_retarget = _coarse_wrist_retarget(
            config, baseline, source_contacts, object_translation, object_quaternion,
        )
        # The legacy branch must rebuild its target-shape reference after the
        # coarse reach stage.  v4 intentionally does not consume that state.
        target_reference_landmarks_local = _target_shape_hand_reference(
            config, baseline, source_contacts, object_translation, object_quaternion,
        )
        source_reference = _assign_source_contacts_to_target(
            config, baseline, source_contacts, object_translation, object_quaternion,
            target_reference_landmarks_local,
        )
        before = _measure_geometry(
            config, baseline, grids, object_translation, object_quaternion,
            contact_reference=source_reference,
        )
        ik = _joint_retarget(config, baseline, grids, object_translation, object_quaternion, before)
    baseline.quality["native_hand_fk"] = native_hand_fk_quality(
        config, profile, native_motion, baseline.global_orient, baseline.body_pose,
        baseline.left_hand_pose, baseline.right_hand_pose, baseline.transl,
        baseline.source_to_canonical, float(baseline.quality["source_height_m"]),
    )
    after = _measure_geometry(
        config, baseline, grids, object_translation, object_quaternion,
        contact_reference=source_reference,
    )
    deviation = _joint_deviation(config, baseline_copy, baseline)
    contact_patch_quality = _contact_patch_quality(
        config, baseline, after, object_translation, object_quaternion,
    )
    contact_pose_deviation = _contact_pose_deviation(
        baseline_copy, baseline, after,
        np.asarray(config.raw["object_pilot"]["optimized_body_joint_indices"], dtype=np.int64),
    )
    contact_values = np.concatenate([
        after["left_distance"][after["left_contact"]], after["right_distance"][after["right_contact"]],
    ])
    noncontact = ~(after["left_contact"].any(axis=1) | after["right_contact"].any(axis=1))
    noncontact_deviation = deviation[noncontact] if noncontact.any() else deviation
    target_hand_landmarks_local = (
        constrained["target_hand_landmarks_local"] if constrained is not None else
        _measure_target_hand_landmarks(
            config, baseline, object_translation, object_quaternion,
        )
    )
    dynamic_hand_quality = _dynamic_hand_quality(
        config, baseline_copy, baseline, after, target_hand_landmarks_local,
        object_translation, object_quaternion,
        evaluation_frame_mask=(
            constrained["pilot_frame_mask"]
            if constrained is not None and constrained["pilot_windows"] else None
        ),
    )
    penetration_quality = _penetration_metrics(after)
    baseline.quality.update({
        "config_hash": run_config_hash(config, reference, "bvh"),
        "input_hash": input_hash, "model_dependency_hash": model_hash,
        "object_joint_retarget": ik,
        "object_coarse_wrist_retarget": coarse_retarget,
        "hand_pose_convention": baseline.hand_pose_convention,
        "smplx_model_signature": smplx_model_signature(
            config.smplx_model_dir, profile.gender, len(profile.betas),
        ),
        "source_contact_method": "raw_bvh_dynamic_hand_proxy_before_smplx_retarget",
        "object_policy": "fixed released SE(3); rigid source-to-canonical only; no human-height scaling",
        "object_acceptance_thresholds": dict(config.raw["object_pilot"]["acceptance"]),
        "object_sdf": {
            "pitch_m": float(config.raw["object_pilot"]["sdf_pitch_m"]),
            "sign_method": "actual OBJ triangle distance with trimesh voxel-fill sign calibration",
            "orientation_agreement": [grid.sign_agreement for grid in grids],
            "mesh_topology": [{
                "watertight": bool(mesh.is_watertight),
                "winding_consistent_after_repair": bool(mesh.is_winding_consistent),
                "connected_components": int(len(mesh.split(only_watertight=False))),
            } for mesh in meshes],
        },
        "source_object_contact_quality": source_contact_patch_quality,
        "dynamic_hand_object_quality": dynamic_hand_quality,
        "object_quality": {
            "object_coverage": 1.0,
            "contact_median_m": float(np.median(contact_values)) if len(contact_values) else float("inf"),
            "contact_p95_m": float(np.percentile(contact_values, 95)) if len(contact_values) else float("inf"),
            **penetration_quality,
            "noncontact_joint_deviation_median_m": float(np.median(noncontact_deviation)),
            "noncontact_joint_deviation_p95_m": float(np.percentile(noncontact_deviation, 95)),
            "left_contact_fraction": float(after["left_contact"].mean()),
            "right_contact_fraction": float(after["right_contact"].mean()),
            **contact_patch_quality,
            **contact_pose_deviation,
            **{
                key: value for key, value in dynamic_hand_quality.items()
                if key != "sides"
            },
        },
    })
    if constrained is not None:
        baseline.quality["constrained_object_solver"] = {
            "solver_version": constrained["solver_version"],
            "pilot_windows": constrained["pilot_windows"],
            "surface_remap_max_geodesic_m": float(
                np.max(constrained["remapped_contact_geodesic_m"], initial=0.0)
            ),
            "fallback_frame_fraction": float(np.mean(constrained["full_body_fallback"])),
            "solver_layer_residual_p95": constrained["solver_layer_residual_p95"].tolist(),
            "solver_layer_converged": constrained["solver_layer_converged"].tolist(),
            "contact_segment_residual_summary": constrained[
                "contact_segment_residual_summary"
            ],
            "spline_degree": int(constrained["spline_degree"]),
            "spline_control_count": int(len(constrained["spline_knots"]) - constrained["spline_degree"] - 1),
            "soft_prior_normalization": {
                key: config.raw["object_pilot"][key]
                for key in (
                    "pose_prior_tolerance_degrees",
                    "source_velocity_tolerance_degrees_s",
                    "source_acceleration_tolerance_degrees_s2",
                    "root_translation_velocity_tolerance_m_s",
                    "root_translation_acceleration_tolerance_m_s2",
                )
            },
        }
    baseline.objects = ObjectTrackRecord(
        ids=np.asarray([source.object_id for source in sources]),
        categories=np.asarray([source.category for source in sources]),
        transl=object_translation, quat_xyzw=object_quaternion,
        valid=np.ones(object_translation.shape[:2], dtype=bool),
        source_transl=source_translation, source_quat_xyzw=source_quaternion,
        source_timestamps=source_timestamps, canonical_timestamps=baseline.timestamps.copy(),
        mesh_paths=np.asarray(asset_paths), mesh_sha256=np.asarray(asset_hashes),
        mesh_scale_to_m=np.full(len(sources), float(config.raw["object_pilot"]["mesh_scale_to_m"])),
        left_contact=after["left_contact"], right_contact=after["right_contact"],
        body_contact=after["body_contact"], left_distance_m=after["left_distance"],
        right_distance_m=after["right_distance"], body_distance_m=after["body_distance"],
        penetration_depth_m=after["penetration"], left_anchor_local=after["left_anchor"],
        right_anchor_local=after["right_anchor"], left_vertex_index=after["left_vertex"],
        right_vertex_index=after["right_vertex"],
        source_hand_landmark_names=source_contacts.landmark_names,
        source_hand_region_names=source_contacts.region_names,
        source_hand_segment_names=source_contacts.segment_names,
        source_hand_landmarks_local=source_contacts.landmarks_local,
        source_hand_landmark_velocity_local=source_contacts.landmark_velocity_local,
        target_hand_landmarks_local=target_hand_landmarks_local,
        target_reference_hand_landmarks_local=target_reference_landmarks_local,
        source_hand_observed=source_contacts.observed,
        source_hand_contact=source_contacts.contact,
        annotation_assisted=source_contacts.annotation_assisted,
        source_hand_proxy_gap_m=source_contacts.proxy_gap_m,
        contact_source_local=source_contacts.contact_source_local,
        contact_anchor_local=source_contacts.contact_anchor_local,
        contact_face=source_contacts.contact_face,
        contact_barycentric=source_contacts.contact_barycentric,
        contact_normal_local=source_contacts.contact_normal_local,
        contact_region=source_contacts.contact_region,
        contact_segment=source_contacts.contact_segment,
        contact_proxy_index=source_contacts.contact_proxy_index,
        contact_valid=source_contacts.contact_valid,
        contact_constraint_valid=source_reference["dynamic_contact_constraint_valid"],
        contact_normal_constraint_valid=source_reference.get(
            "dynamic_contact_normal_constraint_valid",
            source_reference["dynamic_contact_constraint_valid"],
        ),
        contact_target_vertex=source_reference["dynamic_contact_target_vertex"],
        source_hand_proxy_radii_m=source_contacts.proxy_radii_m,
        solver_version=(constrained["solver_version"] if constrained is not None else None),
        solver_pilot_windows=(constrained["pilot_windows"] if constrained is not None else None),
        solver_pilot_frame_mask=(constrained["pilot_frame_mask"] if constrained is not None else None),
        source_palm_position_local=(constrained["source_palm_position_local"] if constrained is not None else None),
        source_palm_quat_xyzw_local=(constrained["source_palm_quat_xyzw_local"] if constrained is not None else None),
        target_palm_position_local=(constrained["target_palm_position_local"] if constrained is not None else None),
        target_palm_quat_xyzw_local=(constrained["target_palm_quat_xyzw_local"] if constrained is not None else None),
        remapped_contact_anchor_local=(constrained["remapped_contact_anchor_local"] if constrained is not None else None),
        remapped_contact_face=(constrained["remapped_contact_face"] if constrained is not None else None),
        remapped_contact_barycentric=(constrained["remapped_contact_barycentric"] if constrained is not None else None),
        remapped_contact_geodesic_m=(constrained["remapped_contact_geodesic_m"] if constrained is not None else None),
        palm_constraint_multiplier=(constrained["palm_constraint_multiplier"] if constrained is not None else None),
        contact_constraint_multiplier=(constrained["contact_constraint_multiplier"] if constrained is not None else None),
        solver_layer_residual_p95=(constrained["solver_layer_residual_p95"] if constrained is not None else None),
        solver_layer_penalty=(constrained["solver_layer_penalty"] if constrained is not None else None),
        solver_layer_multiplier_norm=(constrained["solver_layer_multiplier_norm"] if constrained is not None else None),
        solver_layer_converged=(constrained["solver_layer_converged"] if constrained is not None else None),
        palm_jacobian_rank=(constrained["palm_jacobian_rank"] if constrained is not None else None),
        palm_jacobian_condition=(constrained["palm_jacobian_condition"] if constrained is not None else None),
        full_body_fallback=(constrained["full_body_fallback"] if constrained is not None else None),
        support_foot=(constrained["support_foot"] if constrained is not None else None),
        support_foot_baseline_world=(constrained["support_foot_baseline_world"] if constrained is not None else None),
        support_foot_target_world=(constrained["support_foot_target_world"] if constrained is not None else None),
        root_fallback_translation=(constrained["root_fallback_translation"] if constrained is not None else None),
        root_fallback_rotvec=(constrained["root_fallback_rotvec"] if constrained is not None else None),
        solver_spline_knots=(constrained["spline_knots"] if constrained is not None else None),
        solver_spline_degree=(constrained["spline_degree"] if constrained is not None else None),
        solver_spline_body_delta_coeff=(constrained["spline_body_delta_coeff"] if constrained is not None else None),
        solver_spline_root_delta_coeff=(constrained["spline_root_delta_coeff"] if constrained is not None else None),
        solver_spline_translation_delta_coeff=(constrained["spline_translation_delta_coeff"] if constrained is not None else None),
        solver_spline_left_hand_delta_coeff=(constrained["spline_left_hand_delta_coeff"] if constrained is not None else None),
        solver_spline_right_hand_delta_coeff=(constrained["spline_right_hand_delta_coeff"] if constrained is not None else None),
        solver_spline_rotation_basis=(constrained["spline_rotation_basis"] if constrained is not None else None),
    )
    baseline.validate()
    return baseline


def extract_object_reference(
    config_path: str | Path, reference: SequenceRef, resume: bool = True, retries: int = 0,
    pilot_windows: bool = False,
) -> dict[str, Any]:
    config = load_config(config_path)
    ledger = Ledger(config.output_root / "ledger.sqlite3")
    profile = profiles_for_reference(config, reference, ["source_native"])[0]
    target = output_path(config, reference, profile)
    key = (reference.dataset, reference.sequence_id, profile.name)
    task_hash = run_config_hash(config, reference, "bvh")
    if pilot_windows:
        task_hash = hashlib.sha256((task_hash + ":v4_pilot_windows").encode()).hexdigest()
    result: dict[str, Any] = {}
    try:
        for _ in range(retries + 1):
            try:
                base_input_hash = input_dependency_hash(config, ledger, reference, "bvh")
                metadata = load_hiphi_metadata(reference)
                sources = resolve_object_sources(config.dataset_path("hiphi"), reference, metadata)
                digest = hashlib.sha256(base_input_hash.encode())
                for source in sources:
                    digest.update(str(source.mesh_path.resolve()).encode())
                    digest.update(ledger.sha256(source.mesh_path).encode())
                    digest.update(str(source.trajectory_path.resolve()).encode())
                    digest.update(ledger.sha256(source.trajectory_path).encode())
                input_hash = digest.hexdigest()
                model_hash = model_dependency_hash(config, ledger, reference, profile, "bvh")
                if resume and ledger.completed(*key, task_hash, input_hash, model_hash):
                    return {"status": "skipped", "sequence_id": reference.sequence_id, "output": str(target)}
                ledger.upsert(*key, "running", task_hash, increment_attempts=True,
                              input_hash=input_hash, model_hash=model_hash)
                motion = build_object_motion(
                    config, reference, profile, input_hash, model_hash,
                    pilot_windows=pilot_windows,
                )
                write_motion_npz(target, motion)
                metrics = dict(motion.quality)
                metrics.update({"frames": len(motion.timestamps), "fps": motion.fps, "output_bytes": target.stat().st_size})
                ledger.upsert(*key, "extracted", task_hash, str(target), metrics,
                              input_hash=input_hash, model_hash=model_hash)
                result = {"status": "extracted", "sequence_id": reference.sequence_id,
                          "output": str(target), "metrics": metrics}
                break
            except Exception as exc:
                error = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                diagnostic = traceback.format_exc()
                print(diagnostic, flush=True)
                ledger.upsert(*key, "failed", task_hash, str(target), error=diagnostic)
                result = {"status": "failed", "sequence_id": reference.sequence_id, "error": error}
        return result
    finally:
        ledger.close()
