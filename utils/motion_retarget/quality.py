from __future__ import annotations


import numpy as np

from .bvh import BVHMotion
from .config import RetargetConfig
from .schema import RetargetProfile
from .gpu import assigned_gpu_index
from .trajectory import c3_smooth_rotvec
from .smplx_model import (
    HAND_TIP_VERTEX_IDS, create_smplx_model, smplx_hand_landmarks,
)
from .source_hands import source_hand_landmarks


def _wrist_basis_alignment(source_bases: np.ndarray, target_bases: np.ndarray) -> np.ndarray:
    """Return the proper row-vector rotation from BVH to MANO wrist coordinates.

    BVH publishers and SMPL-X do not guarantee identical wrist-local axes. The
    five MCP directions provide a stable, articulation-independent calibration;
    normalizing them prevents actor-specific palm lengths from biasing the
    coordinate-frame estimate.
    """
    source = np.asarray(source_bases, dtype=np.float64).copy()
    target = np.asarray(target_bases, dtype=np.float64).copy()
    if source.shape != (5, 3) or target.shape != (5, 3):
        raise ValueError("Wrist basis calibration requires five 3D MCP vectors")
    source /= np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    target /= np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    u, _, vt = np.linalg.svd(source.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation.astype(np.float32)


def _selected_torch_device(config: RetargetConfig):
    import torch
    requested = str(config.raw["fitting"].get("device", "auto"))
    use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda"}
    device_index = assigned_gpu_index(torch.cuda.device_count()) if use_cuda else 0
    return torch.device(f"cuda:{device_index}" if use_cuda else "cpu"), use_cuda, device_index


def retarget_native_hands(
    config: RetargetConfig, profile: RetargetProfile, motion: BVHMotion,
    left_pose: np.ndarray, right_pose: np.ndarray, source_height_m: float,
    global_orient: np.ndarray, body_pose: np.ndarray, source_to_canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """IK HiPHI finger rotations onto the fixed-profile SMPL-X hand bones."""
    from .fitting import _device_lock
    device, use_cuda, device_index = _selected_torch_device(config)
    with _device_lock(config, use_cuda, device_index):
        return _retarget_native_hands_impl(
            config, profile, motion, left_pose, right_pose, source_height_m,
            global_orient, body_pose, source_to_canonical, device,
        )


def _retarget_native_hands_impl(
    config: RetargetConfig, profile: RetargetProfile, motion: BVHMotion,
    left_pose: np.ndarray, right_pose: np.ndarray, source_height_m: float,
    global_orient: np.ndarray, body_pose: np.ndarray, source_to_canonical: np.ndarray,
    device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    import torch
    import torch.nn.functional as F
    from smplx.lbs import batch_rodrigues

    model = create_smplx_model(
        config.smplx_model_dir, profile.gender, len(profile.betas), device=device,
    ).eval()
    betas = torch.tensor(profile.betas, dtype=torch.float32, device=device)[None]
    with torch.no_grad():
        neutral = model(betas=betas)
    scale = profile.height_m / float(neutral.vertices[0, :, 1].max() - neutral.vertices[0, :, 1].min())
    joints = neutral.joints[0].detach().cpu().numpy() * scale
    source_positions, source_rotations = motion.forward_kinematics(include_nonroot_translation=False)
    ratio = profile.height_m / source_height_m / 100.0
    pose_body_t = torch.tensor(
        np.concatenate((global_orient[:, None], body_pose.reshape(len(body_pose), 21, 3)), axis=1),
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        local_body_r = batch_rodrigues(pose_body_t.reshape(-1, 3)).reshape(len(body_pose), 22, 3, 3)
        global_body_r = [local_body_r[:, 0]]
        parents = model.parents[:22].detach().cpu().tolist()
        for joint_index in range(1, 22):
            global_body_r.append(global_body_r[parents[joint_index]] @ local_body_r[:, joint_index])
        global_body_r = torch.stack(global_body_r, dim=1).cpu().numpy()
    canonical_rotation = source_to_canonical[:3, :3]
    finger_names = ("Index", "Middle", "Pinky", "Ring", "Thumb")
    released_landmarks = source_hand_landmarks(motion).astype(np.float64)
    outputs = []
    side_metrics = {}
    for side_index, (side, initial, wrist_joint, target_start) in enumerate((
        ("Left", left_pose, 20, 25), ("Right", right_pose, 21, 40),
    )):
        wrist_source_index = motion.joint_index(f"{side}Hand")
        relative_world = (
            released_landmarks[:, side_index, 2:] -
            released_landmarks[:, side_index, 0, None]
        ) * (profile.height_m / source_height_m)
        # Finger-chain fitting is intrinsically wrist local.  Projecting the
        # released BVH fingers through the *target* wrist orientation couples
        # MANO articulation to any body-retarget wrist error and can never be
        # validated consistently after object-aware arm IK.
        desired = np.einsum(
            "tfj,tjk->tfk", relative_world, source_rotations[:, wrist_source_index],
        )

        offsets = np.zeros((20, 3), dtype=np.float32)
        tip_ids = HAND_TIP_VERTEX_IDS[side.lower()]
        neutral_vertices = neutral.vertices[0].detach().cpu().numpy() * scale
        for finger in range(5):
            base = target_start + finger * 3
            offset = finger * 4
            offsets[offset] = joints[base] - joints[wrist_joint]
            offsets[offset + 1] = joints[base + 1] - joints[base]
            offsets[offset + 2] = joints[base + 2] - joints[base + 1]
            offsets[offset + 3] = neutral_vertices[tip_ids[finger]] - joints[base + 2]
        basis_alignment = _wrist_basis_alignment(
            np.median(desired[:, ::4], axis=0), offsets[::4],
        )
        desired = desired @ basis_alignment
        desired_t = torch.tensor(desired, dtype=torch.float32, device=device)
        offsets_t = torch.tensor(offsets, device=device)
        initial_t = torch.tensor(initial.reshape(len(initial), 15, 3), device=device)
        pose = torch.nn.Parameter(initial_t.clone())
        fitting = config.raw["fitting"]
        optimizer = torch.optim.Adam(
            [pose], lr=float(fitting["native_hand_learning_rate"]),
        )

        def forward_positions():
            local = batch_rodrigues(pose.reshape(-1, 3)).reshape(len(pose), 15, 3, 3)
            predicted = []
            for finger in range(5):
                joint_index = finger * 3
                offset_index = finger * 4
                p1 = offsets_t[offset_index].expand(len(pose), -1)
                r1 = local[:, joint_index]
                p2 = p1 + torch.einsum("tij,j->ti", r1, offsets_t[offset_index + 1])
                r2 = r1 @ local[:, joint_index + 1]
                p3 = p2 + torch.einsum("tij,j->ti", r2, offsets_t[offset_index + 2])
                r3 = r2 @ local[:, joint_index + 2]
                tip = p3 + torch.einsum("tij,j->ti", r3, offsets_t[offset_index + 3])
                predicted.extend((p1, p2, p3, tip))
            return torch.stack(predicted, dim=1), local

        def articulation_losses(predicted):
            """Compare the released dynamic finger shape, not optimizer delta.

            MCP positions are fixed by the target hand shape, so the native
            MANO fit can only control the three phalange directions.  Keeping
            these losses separate from absolute point distance prevents the
            immutable actor/SMPL-X palm-width mismatch from overwhelming the
            actual finger articulation (especially the right thumb).
            """
            predicted_chain = predicted.reshape(len(pose), 5, 4, 3)
            desired_chain = desired_t.reshape(len(pose), 5, 4, 3)
            predicted_bones = torch.diff(predicted_chain, dim=2)
            desired_bones = torch.diff(desired_chain, dim=2)
            predicted_unit = F.normalize(predicted_bones, dim=-1)
            desired_unit = F.normalize(desired_bones, dim=-1)
            direction = (1.0 - (predicted_unit * desired_unit).sum(-1)).mean()
            predicted_bend = (predicted_unit[:, :, :-1] * predicted_unit[:, :, 1:]).sum(-1)
            desired_bend = (desired_unit[:, :, :-1] * desired_unit[:, :, 1:]).sum(-1)
            bend = (predicted_bend - desired_bend).square().mean()
            # Thumb tip to index MCP/tip is a stable, shape-aware grasp
            # aperture.  It catches a visually open or mirrored thumb even
            # when the average 20-landmark distance appears acceptable.
            predicted_opposition = torch.stack((
                torch.linalg.vector_norm(predicted[:, 19] - predicted[:, 0], dim=-1),
                torch.linalg.vector_norm(predicted[:, 19] - predicted[:, 3], dim=-1),
            ), dim=-1)
            desired_opposition = torch.stack((
                torch.linalg.vector_norm(desired_t[:, 19] - desired_t[:, 0], dim=-1),
                torch.linalg.vector_norm(desired_t[:, 19] - desired_t[:, 3], dim=-1),
            ), dim=-1)
            opposition = (predicted_opposition - desired_opposition).square().mean()
            return direction, bend, opposition

        for _ in range(int(fitting["native_hand_iterations"])):
            optimizer.zero_grad(set_to_none=True)
            predicted, local_rotations = forward_positions()
            distance = torch.linalg.vector_norm(predicted - desired_t, dim=2)
            data_loss = F.huber_loss(
                distance, torch.zeros_like(distance),
                delta=float(fitting["native_hand_huber_delta_m"]),
            )
            direction, bend, opposition = articulation_losses(predicted)
            prior = ((pose - initial_t) ** 2).mean() * float(fitting["native_hand_pose_prior_weight"])
            temporal = (
                ((local_rotations[1:] - local_rotations[:-1]) ** 2).mean() *
                float(fitting["native_hand_velocity_weight"])
                if len(pose) > 1 else 0.0
            )
            acceleration = torch.zeros((), device=device)
            if len(pose) > 2:
                relative = local_rotations[:-1].transpose(-1, -2) @ local_rotations[1:]
                acceleration = (
                    (relative[1:] - relative[:-1]) ** 2
                ).mean() * float(fitting["native_hand_acceleration_weight"])
            loss = (
                float(fitting.get("native_hand_data_weight", 1.0)) * data_loss +
                float(fitting.get("native_hand_direction_weight", 0.0)) * direction +
                float(fitting.get("native_hand_bend_weight", 0.0)) * bend +
                float(fitting.get("native_hand_thumb_opposition_weight", 0.0)) * opposition +
                prior + temporal + acceleration
            )
            loss.backward()
            optimizer.step()
        temporal_config = config.raw["temporal_continuity"]
        smoothed, temporal_metadata = c3_smooth_rotvec(
            np.arange(len(pose), dtype=np.float64) / float(motion.fps),
            pose.detach().cpu().numpy().reshape(len(pose), 45),
            float(temporal_config["hand_knot_seconds"]),
            int(temporal_config["spline_degree"]),
            int(temporal_config["robust_iterations"]),
            float(temporal_config["rotation_huber_degrees"]),
        )
        with torch.no_grad():
            pose.copy_(torch.tensor(smoothed.reshape(len(pose), 15, 3), device=device))
            predicted, _ = forward_positions()
            error = torch.linalg.vector_norm(predicted - desired_t, dim=2).cpu().numpy()
        outputs.append(pose.detach().cpu().numpy().reshape(len(pose), 45).astype(np.float32))
        side_metrics[side.lower()] = {
            "median_error_m": float(np.median(error)), "p95_error_m": float(np.percentile(error, 95)),
            "source_to_mano_wrist_rotation": basis_alignment.tolist(),
            "c3_projection": temporal_metadata,
        }
    return outputs[0], outputs[1], {
        "method": "wrist_local_fixed_bone_temporal_ik",
        "rotation_continuity": "quintic_quaternion_spline_C4_satisfies_C3", **side_metrics,
    }


def native_hand_fk_quality(
    config: RetargetConfig, profile: RetargetProfile, motion: BVHMotion,
    global_orient: np.ndarray, body_pose: np.ndarray, left_hand_pose: np.ndarray,
    right_hand_pose: np.ndarray, translation: np.ndarray,
    source_to_canonical: np.ndarray, source_height_m: float,
) -> dict:
    from .fitting import _device_lock
    device, use_cuda, device_index = _selected_torch_device(config)
    with _device_lock(config, use_cuda, device_index):
        return _native_hand_fk_quality_impl(
            config, profile, motion, global_orient, body_pose, left_hand_pose,
            right_hand_pose, translation, source_to_canonical, source_height_m, device,
        )


def _native_hand_fk_quality_impl(
    config: RetargetConfig, profile: RetargetProfile, motion: BVHMotion,
    global_orient: np.ndarray, body_pose: np.ndarray, left_hand_pose: np.ndarray,
    right_hand_pose: np.ndarray, translation: np.ndarray,
    source_to_canonical: np.ndarray, source_height_m: float, device,
) -> dict:
    import torch
    model = create_smplx_model(
        config.smplx_model_dir, profile.gender, len(profile.betas), device=device,
    ).eval()
    betas = torch.tensor(profile.betas, dtype=torch.float32, device=device)[None]
    with torch.no_grad():
        neutral_output = model(betas=betas)
        neutral = neutral_output.vertices[0]
    scale = profile.height_m / float(neutral[:, 1].max() - neutral[:, 1].min())
    neutral_joints = neutral_output.joints[0].cpu().numpy() * scale
    target_chunks = []
    target_wrist_rotation_chunks = []
    with torch.no_grad():
        for start in range(0, motion.frame_count, 128):
            end = min(motion.frame_count, start + 128)
            count = end - start
            zeros3 = torch.zeros((count, 3), device=device)
            output = model(
                global_orient=torch.tensor(global_orient[start:end], device=device),
                body_pose=torch.tensor(body_pose[start:end], device=device),
                left_hand_pose=torch.tensor(left_hand_pose[start:end], device=device),
                right_hand_pose=torch.tensor(right_hand_pose[start:end], device=device),
                betas=betas.expand(count, -1), jaw_pose=zeros3, leye_pose=zeros3,
                reye_pose=zeros3, expression=torch.zeros((count, model.num_expression_coeffs), device=device),
            )
            landmarks = smplx_hand_landmarks(output.joints, output.vertices)
            fingers = (
                landmarks[:, :, 2:] - landmarks[:, :, 0:1]
            ).cpu().numpy() * scale
            target_chunks.append(fingers.reshape(count, 40, 3))
            local_body = torch.cat([
                torch.tensor(global_orient[start:end], device=device)[:, None],
                torch.tensor(body_pose[start:end], device=device).reshape(count, 21, 3),
            ], dim=1)
            from smplx.lbs import batch_rodrigues
            local_rotation = batch_rodrigues(local_body.reshape(-1, 3)).reshape(count, 22, 3, 3)
            global_rotation = [local_rotation[:, 0]]
            parents = model.parents[:22].detach().cpu().tolist()
            for joint_index in range(1, 22):
                global_rotation.append(global_rotation[parents[joint_index]] @ local_rotation[:, joint_index])
            target_wrist_rotation_chunks.append(
                torch.stack([global_rotation[20], global_rotation[21]], dim=1).cpu().numpy()
            )
    target = np.concatenate(target_chunks, axis=0)
    target_wrist_rotation = np.concatenate(target_wrist_rotation_chunks, axis=0)
    target[:, :20] = np.einsum("tfj,tjk->tfk", target[:, :20], target_wrist_rotation[:, 0])
    target[:, 20:] = np.einsum("tfj,tjk->tfk", target[:, 20:], target_wrist_rotation[:, 1])

    source_positions, source_rotations = motion.forward_kinematics(include_nonroot_translation=False)
    ratio = profile.height_m / source_height_m
    source_world = source_positions / 100.0 * ratio
    source_world = source_world @ source_to_canonical[:3, :3].T + source_to_canonical[:3, 3]
    released = source_hand_landmarks(motion).astype(np.float64) * ratio
    released = released @ source_to_canonical[:3, :3].T + source_to_canonical[:3, 3]
    source = (released[:, :, 2:] - released[:, :, 0:1]).reshape(len(released), 40, 3)
    # Compare both hands in their own wrist-local frames. Merely subtracting
    # wrist translation (the old implementation) incorrectly treated an
    # object-aware whole-arm rotation as a MANO articulation error.
    source_left_wrist_rotation = (
        source_to_canonical[:3, :3] @ source_rotations[:, motion.joint_index("LeftHand")]
    )
    source_right_wrist_rotation = (
        source_to_canonical[:3, :3] @ source_rotations[:, motion.joint_index("RightHand")]
    )
    source[:, :20] = np.einsum("tfj,tjk->tfk", source[:, :20], source_left_wrist_rotation)
    source[:, 20:] = np.einsum("tfj,tjk->tfk", source[:, 20:], source_right_wrist_rotation)
    for side_index, (wrist_joint, target_start) in enumerate(((20, 25), (21, 40))):
        side_slice = slice(side_index * 20, (side_index + 1) * 20)
        target_bases = np.stack([
            neutral_joints[target_start + finger * 3] - neutral_joints[wrist_joint]
            for finger in range(5)
        ])
        alignment = _wrist_basis_alignment(
            np.median(source[:, side_slice][:, ::4], axis=0), target_bases,
        )
        source[:, side_slice] = source[:, side_slice] @ alignment
    errors = np.linalg.norm(source - target, axis=2)
    with_channels = motion.local_positions(include_nonroot_translation=True)
    offsets_only = motion.local_positions(include_nonroot_translation=False)
    auxiliary_translation = np.linalg.norm(with_channels[:, 1:] - offsets_only[:, 1:], axis=2) / 100.0
    left, right = errors[:, :20], errors[:, 20:]
    threshold = float(config.raw["quality"]["native_hand_median_error_m"])
    median = float(np.median(errors))
    return {
        "median_error_m": median, "p95_error_m": float(np.percentile(errors, 95)),
        "left_median_error_m": float(np.median(left)), "right_median_error_m": float(np.median(right)),
        "nonroot_translation_channel_median_m": float(np.median(auxiliary_translation)),
        "nonroot_translation_channel_p95_m": float(np.percentile(auxiliary_translation, 95)),
        "passed": median <= threshold, "threshold_m": threshold,
        "scope": "Wrist-aligned native BVH finger-joint and End-Site FK vs fixed-profile SMPL-X/MANO joints and true fingertip vertices; non-root BVH translations used only as diagnostics.",
    }
