from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .bvh import BVHMotion
from .objects import TriangleSurfaceQuery, contact_hysteresis


FINGERS = ("Index", "Middle", "Pinky", "Ring", "Thumb")
LANDMARK_NAMES = ("wrist", "palm", *(
    name for finger in FINGERS for name in (
        f"{finger.lower()}_1", f"{finger.lower()}_2",
        f"{finger.lower()}_3", f"{finger.lower()}_tip",
    )
))
REGION_NAMES = ("palm", "index", "middle", "pinky", "ring", "thumb")
SEGMENT_NAMES = (
    "wrist_palm", "palm_index", "palm_middle", "palm_pinky", "palm_ring", "palm_thumb",
    *(f"{finger.lower()}_{phalange}" for finger in FINGERS for phalange in range(1, 4)),
)


@dataclass
class SourceHandContacts:
    landmark_names: np.ndarray
    region_names: np.ndarray
    segment_names: np.ndarray
    landmarks_local: np.ndarray          # [T,N,2,L,3]
    landmark_velocity_local: np.ndarray  # [T,N,2,L,3]
    observed: np.ndarray                 # [T,N,2]
    contact: np.ndarray                  # [T,N,2]
    annotation_assisted: np.ndarray      # [T,N,2]
    proxy_gap_m: np.ndarray              # [T,N,2]
    contact_source_local: np.ndarray     # [T,N,2,K,3]
    contact_anchor_local: np.ndarray     # [T,N,2,K,3]
    contact_face: np.ndarray             # [T,N,2,K]
    contact_barycentric: np.ndarray      # [T,N,2,K,3]
    contact_normal_local: np.ndarray     # [T,N,2,K,3]
    contact_region: np.ndarray           # [T,N,2,K]
    contact_segment: np.ndarray          # [T,N,2,K]
    contact_proxy_index: np.ndarray      # [T,N,2,K]
    contact_valid: np.ndarray            # [T,N,2,K]
    proxy_radii_m: np.ndarray            # [2,6]


def write_source_hand_contacts(path: Path, value: SourceHandContacts) -> None:
    """Atomically cache source-only geometry; never target bindings or IK state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(stream, **{
                name: np.asarray(getattr(value, name))
                for name in SourceHandContacts.__dataclass_fields__
            })
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_source_hand_contacts(path: Path) -> SourceHandContacts:
    with np.load(path, allow_pickle=False) as payload:
        values = {
            name: payload[name]
            for name in SourceHandContacts.__dataclass_fields__
        }
    return SourceHandContacts(**values)


def _tip_position(motion: BVHMotion, joint_index: int, positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    joint = motion.joints[joint_index]
    if joint.end_sites:
        offset = np.asarray(joint.end_sites[0], dtype=np.float32) / 100.0
    else:
        parent = joint.parent
        offset = np.asarray(joint.offset if parent < 0 else joint.offset * 0.65) / 100.0
    return positions[:, joint_index] + np.einsum("tij,j->ti", rotations[:, joint_index], offset)


def source_hand_landmarks(motion: BVHMotion) -> np.ndarray:
    """Return wrist, palm, 15 joints and five fingertips in source metres."""
    positions, rotations = motion.forward_kinematics(include_nonroot_translation=False)
    positions = positions.astype(np.float64) / 100.0
    sides = []
    for side in ("Left", "Right"):
        wrist = positions[:, motion.joint_index(f"{side}Hand")]
        finger_landmarks = []
        bases = []
        for finger in FINGERS:
            ids = [motion.joint_index(f"{side}Hand{finger}{segment}") for segment in range(1, 4)]
            chain = [positions[:, index] for index in ids]
            chain.append(_tip_position(motion, ids[-1], positions, rotations))
            bases.append(chain[0])
            finger_landmarks.extend(chain)
        palm = 0.45 * wrist + 0.55 * np.mean(np.stack(bases, axis=1), axis=1)
        sides.append(np.stack([wrist, palm, *finger_landmarks], axis=1))
    return np.stack(sides, axis=1).astype(np.float32)


def _hand_proxy(
    landmarks: np.ndarray, samples_per_segment: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample palm/phalange capsule axes and actor-scaled surface radii."""
    frames = len(landmarks)
    bases = np.stack([landmarks[:, 2 + finger * 4] for finger in range(5)], axis=1)
    palm_width = np.median(np.linalg.norm(bases[:, 0] - bases[:, 2], axis=1))
    palm_radius = float(np.clip(0.18 * palm_width, 0.012, 0.025))
    points, radii, regions, segments = [], [], [], []

    def segment(a: np.ndarray, b: np.ndarray, radius: float, region: int, segment_id: int) -> None:
        alpha = np.linspace(0.0, 1.0, samples_per_segment, dtype=np.float32)
        points.append(a[:, None] * (1.0 - alpha[None, :, None]) + b[:, None] * alpha[None, :, None])
        radii.extend([radius] * samples_per_segment)
        regions.extend([region] * samples_per_segment)
        segments.extend([segment_id] * samples_per_segment)

    segment(landmarks[:, 0], landmarks[:, 1], palm_radius, 0, 0)
    for finger in range(5):
        base = 2 + finger * 4
        segment(landmarks[:, 1], landmarks[:, base], palm_radius, 0, finger + 1)
        chain = landmarks[:, base:base + 4]
        for phalange in range(3):
            length = np.median(np.linalg.norm(chain[:, phalange + 1] - chain[:, phalange], axis=1))
            radius = float(np.clip(0.20 * length, 0.004, 0.010))
            segment(
                chain[:, phalange], chain[:, phalange + 1], radius,
                finger + 1, 6 + finger * 3 + phalange,
            )
    return (
        np.concatenate(points, axis=1).reshape(frames, -1, 3).astype(np.float32),
        np.asarray(radii, dtype=np.float32), np.asarray(regions, dtype=np.int16),
        np.asarray(segments, dtype=np.int16),
    )


def _bridge_annotation_prior(
    contact: np.ndarray, eligible: np.ndarray, maximum_gap_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(contact, dtype=bool).copy()
    assisted = np.zeros_like(result)
    padded = np.r_[False, result, False]
    runs = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    for (_, left_end), (right_start, _) in zip(runs[:-1], runs[1:]):
        if right_start - left_end <= maximum_gap_frames and eligible[left_end:right_start].all():
            result[left_end:right_start] = True
            assisted[left_end:right_start] = True
    # Extend only from an observed contact boundary; text never creates an
    # isolated contact interval without geometric support.
    for _ in range(maximum_gap_frames):
        neighbor = np.r_[False, result[:-1]] | np.r_[result[1:], False]
        add = ~result & eligible & neighbor
        if not add.any():
            break
        result[add] = True
        assisted[add] = True
    return result, assisted


def contact_facing_normals(
    face_normals: np.ndarray, anchors: np.ndarray, hand_surface_points: np.ndarray,
) -> np.ndarray:
    """Orient an OBJ triangle-normal axis toward the observed source hand.

    HiPHI objects such as the cloth rack are non-watertight thin surfaces.
    Their published triangle winding supplies a geometric normal *axis*, but
    not a reliable inside/outside sign. Contact itself removes that ambiguity:
    the relevant object-side normal points from the triangle anchor toward the
    source hand *interior*. Callers must therefore pass the capsule axis, not a
    capsule surface sample. A penetrating surface sample can overshoot the OBJ
    triangle and invert the sign exactly when contact is deepest. This changes
    neither the triangle, face id, barycentric coordinate, mesh hash, nor pose.
    """
    normal = np.asarray(face_normals, dtype=np.float64)
    anchor = np.asarray(anchors, dtype=np.float64)
    hand = np.asarray(hand_surface_points, dtype=np.float64)
    if normal.shape != anchor.shape or normal.shape != hand.shape or normal.shape[-1] != 3:
        raise ValueError("Contact-facing normal inputs must have matching [...,3] shapes")
    direction = hand - anchor
    dot = np.sum(normal * direction, axis=-1, keepdims=True)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    return (normal * sign).astype(np.float32)


def align_normal_axis_to_reference(
    face_normals: np.ndarray, reference_normals: np.ndarray,
) -> np.ndarray:
    """Choose a remapped triangle-normal sign from a trusted source axis."""
    normal = np.asarray(face_normals, dtype=np.float64)
    reference = np.asarray(reference_normals, dtype=np.float64)
    if normal.shape != reference.shape or normal.shape[-1] != 3:
        raise ValueError("Normal-axis alignment inputs must have matching [...,3] shapes")
    sign = np.where(
        np.sum(normal * reference, axis=-1, keepdims=True) < 0.0, -1.0, 1.0,
    )
    return (normal * sign).astype(np.float32)


def recover_source_hand_contacts(
    motion: BVHMotion, object_translation: np.ndarray, object_quaternion: np.ndarray,
    meshes: list[Any], fps: float, text_annotation: str, settings: dict[str, Any],
) -> SourceHandContacts:
    """Recover released hand/object motion before any SMPL-X retargeting."""
    landmarks_world = source_hand_landmarks(motion)
    frames, objects = object_translation.shape[:2]
    if len(landmarks_world) != frames:
        raise ValueError("Source hand and object frame counts disagree")
    landmark_count = landmarks_world.shape[2]
    maximum_contacts = int(settings.get("source_contact_points_per_hand", 8))
    landmarks_local = np.zeros((frames, objects, 2, landmark_count, 3), dtype=np.float32)
    velocity_local = np.zeros_like(landmarks_local)
    observed = np.ones((frames, objects, 2), dtype=bool)
    contact = np.zeros_like(observed)
    assisted = np.zeros_like(observed)
    proxy_gap = np.full((frames, objects, 2), np.inf, dtype=np.float32)
    source_points = np.zeros((frames, objects, 2, maximum_contacts, 3), dtype=np.float32)
    anchors = np.zeros_like(source_points)
    normals = np.zeros_like(source_points)
    barycentric = np.zeros_like(source_points)
    faces = np.full((frames, objects, 2, maximum_contacts), -1, dtype=np.int32)
    regions = np.full_like(faces, -1, dtype=np.int16)
    segments = np.full_like(faces, -1, dtype=np.int16)
    proxy_indices = np.full_like(faces, -1, dtype=np.int16)
    valid = np.zeros_like(faces, dtype=bool)
    radii_by_side = np.zeros((2, len(REGION_NAMES)), dtype=np.float32)
    requires_both = "both hands" in text_annotation.lower()
    rotation = Rotation.from_quat(object_quaternion.reshape(-1, 4)).as_matrix().reshape(frames, objects, 3, 3)

    for side in range(2):
        proxy_world, proxy_radius, proxy_region, proxy_segment = _hand_proxy(landmarks_world[:, side])
        for region in range(len(REGION_NAMES)):
            values = proxy_radius[proxy_region == region]
            radii_by_side[side, region] = float(np.median(values)) if len(values) else 0.0
        for object_index, mesh in enumerate(meshes):
            local_landmarks = np.einsum(
                "tlj,tjk->tlk",
                landmarks_world[:, side] - object_translation[:, object_index, None],
                rotation[:, object_index],
            )
            landmarks_local[:, object_index, side] = local_landmarks
            local_proxy = np.einsum(
                "tpj,tjk->tpk",
                proxy_world - object_translation[:, object_index, None],
                rotation[:, object_index],
            )
            query = TriangleSurfaceQuery(mesh)
            closest, distance, face, bary = query.query(local_proxy.reshape(-1, 3))
            closest = closest.reshape(frames, -1, 3)
            distance = distance.reshape(frames, -1)
            face = face.reshape(frames, -1)
            bary = bary.reshape(frames, -1, 3)
            gap = distance - proxy_radius[None]
            direction = closest - local_proxy
            length = np.linalg.norm(direction, axis=2, keepdims=True)
            proxy_surface = local_proxy + direction / np.maximum(length, 1e-8) * proxy_radius[None, :, None]
            speed = np.zeros_like(gap)
            speed[1:] = np.linalg.norm(np.diff(proxy_surface, axis=0), axis=2) * float(fps)
            speed[0] = speed[1] if frames > 1 else 0.0
            best = np.argmin(gap, axis=1)
            rows = np.arange(frames)
            best_gap = gap[rows, best]
            best_speed = speed[rows, best]
            proxy_gap[:, object_index, side] = best_gap
            credible = np.where(
                best_speed <= float(settings.get("source_contact_relative_speed_m_s", 0.25)),
                best_gap, np.inf,
            )
            mask = contact_hysteresis(
                credible, float(settings.get("source_contact_enter_m", 0.010)),
                float(settings.get("source_contact_exit_m", 0.020)),
                int(settings.get("source_contact_minimum_frames", max(3, round(0.1 * fps)))),
            )
            side_prior = requires_both
            if side_prior:
                eligible = (
                    (best_gap <= float(settings.get("annotation_max_gap_m", 0.030))) &
                    (best_speed <= float(settings.get("annotation_max_speed_m_s", 0.35)))
                )
                mask, side_assisted = _bridge_annotation_prior(
                    mask, eligible, int(round(float(settings.get("annotation_max_bridge_s", 0.25)) * fps)),
                )
                assisted[:, object_index, side] = side_assisted
            contact[:, object_index, side] = mask
            previous: list[int] = []
            switch_tolerance = float(settings.get("source_contact_switch_hysteresis_m", 0.003))
            for frame_index in range(frames):
                if not mask[frame_index]:
                    previous = []
                    continue
                limit = float(settings.get("annotation_max_gap_m", 0.030)) if assisted[frame_index, object_index, side] else float(settings.get("source_contact_exit_m", 0.020))
                candidates = np.flatnonzero(gap[frame_index] <= limit)
                if not len(candidates):
                    previous = []
                    continue
                order = candidates[np.argsort(gap[frame_index, candidates])]
                chosen = []
                # Region diversity before additional points prevents a single
                # fingertip vertex from masquerading as an entire grasp.  A
                # small nearest-proxy hysteresis suppresses 90 Hz identity
                # chatter without preventing real rolling or finger switches.
                for region in np.unique(proxy_region[order]):
                    region_order = order[proxy_region[order] == region]
                    best_region_gap = gap[frame_index, region_order[0]]
                    retained = next((
                        proxy for proxy in previous
                        if proxy_region[proxy] == region and proxy in candidates and
                        gap[frame_index, proxy] <= best_region_gap + switch_tolerance
                    ), None)
                    chosen.append(int(region_order[0] if retained is None else retained))
                for proxy in previous:
                    if (
                        proxy in candidates and proxy not in chosen and
                        gap[frame_index, proxy] <= gap[frame_index, order[0]] + switch_tolerance
                    ):
                        chosen.append(int(proxy))
                for candidate in order:
                    if int(candidate) not in chosen:
                        chosen.append(int(candidate))
                    if len(chosen) >= maximum_contacts:
                        break
                chosen = sorted(
                    chosen[:maximum_contacts],
                    key=lambda proxy: (int(proxy_region[proxy]), int(proxy_segment[proxy]), int(proxy)),
                )
                previous = chosen.copy()
                count = len(chosen)
                valid[frame_index, object_index, side, :count] = True
                source_points[frame_index, object_index, side, :count] = proxy_surface[frame_index, chosen]
                anchors[frame_index, object_index, side, :count] = closest[frame_index, chosen]
                faces[frame_index, object_index, side, :count] = face[frame_index, chosen]
                barycentric[frame_index, object_index, side, :count] = bary[frame_index, chosen]
                raw_normal = np.asarray(mesh.face_normals)[face[frame_index, chosen]]
                normals[frame_index, object_index, side, :count] = contact_facing_normals(
                    raw_normal,
                    closest[frame_index, chosen],
                    local_proxy[frame_index, chosen],
                )
                regions[frame_index, object_index, side, :count] = proxy_region[chosen]
                segments[frame_index, object_index, side, :count] = proxy_segment[chosen]
                proxy_indices[frame_index, object_index, side, :count] = chosen
    if frames > 1:
        velocity_local[1:] = np.diff(landmarks_local, axis=0) * float(fps)
        velocity_local[0] = velocity_local[1]
    return SourceHandContacts(
        np.asarray(LANDMARK_NAMES), np.asarray(REGION_NAMES), np.asarray(SEGMENT_NAMES), landmarks_local,
        velocity_local, observed, contact, assisted, proxy_gap, source_points,
        anchors, faces, barycentric, normals, regions, segments, proxy_indices,
        valid, radii_by_side,
    )


def hand_vertex_regions(model, side: str) -> dict[int, np.ndarray]:
    """Map target SMPL-X vertices to palm/finger anatomical regions."""
    weights = model.lbs_weights.detach().cpu().numpy()
    if side == "left":
        wrist, starts = 20, (25, 28, 31, 34, 37)
    elif side == "right":
        wrist, starts = 21, (40, 43, 46, 49, 52)
    else:
        raise ValueError(side)
    result = {0: np.flatnonzero(weights[:, wrist] > 0.08)}
    for region, start in enumerate(starts, 1):
        score = weights[:, start:start + 3].sum(axis=1)
        result[region] = np.flatnonzero(score > 0.08)
    if any(len(values) < 3 for values in result.values()):
        raise ValueError(f"SMPL-X {side} hand has an empty anatomical vertex region")
    return result


def hand_vertex_segments(model, side: str) -> dict[int, np.ndarray]:
    """Return surface candidates for the 21 source proxy segments.

    Segment identities are deliberately finer than the six reporting regions.
    They prevent a contact on one phalange from being rebound to an unrelated
    vertex elsewhere on the same finger on every frame.
    """
    weights = model.lbs_weights.detach().cpu().numpy()
    if side == "left":
        wrist, starts = 20, (25, 28, 31, 34, 37)
    elif side == "right":
        wrist, starts = 21, (40, 43, 46, 49, 52)
    else:
        raise ValueError(side)

    def candidates(score: np.ndarray, minimum: float = 0.035) -> np.ndarray:
        result = np.flatnonzero(score > minimum)
        if len(result) < 3:
            result = np.argsort(score)[-12:]
        return np.asarray(result, dtype=np.int64)

    result: dict[int, np.ndarray] = {0: candidates(weights[:, wrist])}
    for finger, start in enumerate(starts):
        # A metacarpal/palm bridge must genuinely blend the rigid palm and the
        # first articulated finger joint. Summing the weights admitted vertices
        # with ~100% finger influence and ~0% wrist influence, binding a source
        # palm contact to a moving phalanx and creating a false conflict with
        # the dynamic MANO task. ``minimum`` is positive only when both sides
        # of the anatomical bridge influence the target surface vertex.
        result[finger + 1] = candidates(
            np.minimum(weights[:, wrist], weights[:, start])
        )
        for phalange in range(3):
            result[6 + finger * 3 + phalange] = candidates(weights[:, start + phalange])
    return result
