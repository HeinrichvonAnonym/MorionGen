from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .config import RetargetConfig
from .schema import RetargetProfile
from .trajectory import c3_smooth_rotvec, c3_smooth_translation
from .gpu import assigned_gpu_index
from .smplx_model import create_smplx_model


def summarize_fit_errors(
    errors: np.ndarray, valid: np.ndarray, target_kind: str, settings: dict,
) -> dict:
    """Summarize a robust marker gate without hiding the raw residuals."""
    raw_finite = errors[np.isfinite(errors)]
    per_target_median = np.nanmedian(errors, axis=0)
    per_target_p95 = np.nanpercentile(errors, 95, axis=0)
    threshold_key = "marker_track_outlier_m" if target_kind == "vertices" else "joint_track_outlier_m"
    minimum_key = "min_marker_tracks" if target_kind == "vertices" else "min_joint_tracks"
    track_threshold = float(settings[threshold_key])
    reliable = np.isfinite(per_target_median) & (per_target_median <= track_threshold)
    robust_errors = errors[:, reliable]
    finite = robust_errors[np.isfinite(robust_errors)]
    minimum = int(settings.get(minimum_key, 1))
    enough_targets = int(reliable.sum()) >= minimum
    return {
        "raw_observation_count": int(valid.sum()),
        "observation_count": int(valid[:, reliable].sum()),
        "median_error_m": float(np.median(finite)) if len(finite) and enough_targets else float("inf"),
        "p95_error_m": float(np.percentile(finite, 95)) if len(finite) and enough_targets else float("inf"),
        "raw_median_error_m": float(np.median(raw_finite)) if len(raw_finite) else float("inf"),
        "raw_p95_error_m": float(np.percentile(raw_finite, 95)) if len(raw_finite) else float("inf"),
        "reliable_target_count": int(reliable.sum()),
        "excluded_target_indices": np.flatnonzero(~reliable).astype(int).tolist(),
        "track_outlier_threshold_m": track_threshold,
        "per_target_median_error_m": per_target_median.astype(float).tolist(),
        "per_target_p95_error_m": per_target_p95.astype(float).tolist(),
    }


@contextmanager
def _device_lock(config: RetargetConfig, use_cuda: bool, device_index: int = 0):
    if not use_cuda:
        yield
        return
    path = Path(__file__).resolve().parents[2] / f"develop_logs/test_runs/.motion_retarget_gpu_{device_index}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def fit_smplx_observations(
    config: RetargetConfig, profile: RetargetProfile,
    target: np.ndarray, valid: np.ndarray, target_indices: np.ndarray,
    initial_root: np.ndarray, initial_body: np.ndarray, initial_translation: np.ndarray,
    target_kind: str, fps: float, fallback_body: np.ndarray | None = None,
    fallback_body_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Windowed robust IK against SMPL-X joints or marker vertices."""
    import torch
    import torch.nn.functional as F
    from smplx.lbs import batch_rodrigues

    settings = config.raw["fitting"]
    requested = str(settings.get("device", "auto"))
    use_cuda = torch.cuda.is_available() and requested in {"auto", "cuda"}
    device_index = assigned_gpu_index(torch.cuda.device_count()) if use_cuda else 0
    device = torch.device(f"cuda:{device_index}" if use_cuda else "cpu")
    model = create_smplx_model(
        config.smplx_model_dir, profile.gender, len(profile.betas), device=device,
    ).eval()
    betas = torch.tensor(profile.betas, dtype=torch.float32, device=device)[None]
    base_output = model(betas=betas)
    base_vertices = base_output.vertices[0]
    scale = profile.height_m / float(base_vertices[:, 1].max() - base_vertices[:, 1].min())
    root_pivot = base_output.joints[0, 0].detach().cpu().numpy() * scale
    t = len(target)
    roots = np.asarray(initial_root, dtype=np.float32).copy()
    bodies = np.asarray(initial_body, dtype=np.float32).reshape(t, 63).copy()
    # Callers provide the observed pelvis/world-root trajectory. SMPL-X
    # ``transl`` is the model-origin translation; its neutral pelvis has a
    # non-zero offset, so subtract it before optimization. Starting 36 cm away
    # from every JCP made the robust Huber loss saturate and encouraged local
    # minima, especially for short/static COMFI clips.
    translations = np.asarray(initial_translation, dtype=np.float32).copy() - root_pivot.astype(np.float32)
    window = int(settings["window_frames"])
    iterations = int(settings["iterations"])
    delta = float(settings["huber_delta_m"])
    all_errors: list[np.ndarray] = []
    previous = None
    elbow_prior_active = np.zeros(2, dtype=bool)
    if t > 1:
        initial_body_matrices = Rotation.from_rotvec(
            bodies.reshape(-1, 21, 3).reshape(-1, 3)
        ).as_matrix().reshape(t, 21, 3, 3)
        initial_relative = (
            initial_body_matrices[:-1].transpose(0, 1, 3, 2) @ initial_body_matrices[1:]
        )
        initial_steps = np.degrees(
            Rotation.from_matrix(initial_relative.reshape(-1, 3, 3)).magnitude()
        ).reshape(t - 1, 21)
        elbow_prior_active = np.percentile(initial_steps[:, [17, 18]], 95, axis=0) >= float(
            settings["elbow_prior_activation_degrees"]
        )
    def forward_batch(root_value, body_value, beta_value):
        batch = len(root_value)
        zeros3 = torch.zeros((batch, 3), dtype=root_value.dtype, device=device)
        zeros45 = torch.zeros((batch, 45), dtype=root_value.dtype, device=device)
        expression = torch.zeros((batch, model.num_expression_coeffs), dtype=root_value.dtype, device=device)
        return model(
            global_orient=root_value, body_pose=body_value, betas=beta_value,
            left_hand_pose=zeros45, right_hand_pose=zeros45, jaw_pose=zeros3,
            leye_pose=zeros3, reye_pose=zeros3, expression=expression,
        )
    contact_mask = np.zeros((t, 2), dtype=bool)
    with _device_lock(config, use_cuda, device_index):
        # Infer conservative stance frames from the native initialization.  We
        # only constrain low-speed feet close to their own lower envelope, so
        # jumps and swing phases are not flattened by the contact regularizer.
        initial_feet_chunks = []
        with torch.no_grad():
            for contact_start in range(0, t, window):
                contact_end = min(t, contact_start + window)
                root_value = torch.tensor(roots[contact_start:contact_end], device=device)
                body_value = torch.tensor(bodies[contact_start:contact_end], device=device)
                trans_value = torch.tensor(translations[contact_start:contact_end], device=device)
                initialized = forward_batch(
                    root_value, body_value, betas.expand(contact_end - contact_start, -1),
                )
                initial_feet_chunks.append(
                    (initialized.joints[:, [10, 11]] * scale + trans_value[:, None]).cpu().numpy()
                )
        initial_feet = np.concatenate(initial_feet_chunks, axis=0)
        if t > 1:
            speed = np.zeros((t, 2), dtype=np.float32)
            speed[1:] = np.linalg.norm(np.diff(initial_feet, axis=0), axis=2) * float(fps)
            speed[0] = speed[1]
            floor = np.percentile(initial_feet[:, :, 2], 5, axis=0)
            contact_mask = (
                (speed <= float(settings["foot_contact_speed_m_s"])) &
                (initial_feet[:, :, 2] <= floor[None] + float(settings["foot_contact_height_m"]))
            )
        for start in range(0, t, window):
            end = min(t, start + window)
            root0 = torch.tensor(roots[start:end], device=device)
            body0 = torch.tensor(bodies[start:end], device=device)
            trans0 = torch.tensor(translations[start:end], device=device)
            root = torch.nn.Parameter(root0.clone())
            body = torch.nn.Parameter(body0.clone())
            trans = torch.nn.Parameter(trans0.clone())
            optimizer = torch.optim.Adam([
                {"params": (root, body), "lr": float(settings["learning_rate_pose"])},
                {"params": (trans,), "lr": float(settings["learning_rate_translation"])},
            ])
            observed = torch.tensor(target[start:end], device=device)
            mask = torch.tensor(valid[start:end], device=device)
            indices = torch.tensor(target_indices, dtype=torch.long, device=device)
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                batch_betas = betas.expand(end - start, -1)
                output = forward_batch(root, body, batch_betas)
                predicted = output.vertices[:, indices] if target_kind == "vertices" else output.joints[:, indices]
                predicted = predicted * scale + trans[:, None]
                distance = torch.linalg.vector_norm(predicted - observed, dim=-1)
                data_loss = F.huber_loss(distance[mask], torch.zeros_like(distance[mask]), delta=delta)
                local_rotation = batch_rodrigues(
                    torch.cat([root[:, None], body.reshape(len(body), 21, 3)], dim=1).reshape(-1, 3)
                ).reshape(len(root), 22, 3, 3)
                initial_local_rotation = batch_rodrigues(
                    torch.cat([root0[:, None], body0.reshape(len(body0), 21, 3)], dim=1).reshape(-1, 3)
                ).reshape(len(root0), 22, 3, 3)
                # Position observations cannot identify axial twist (notably
                # elbow pronation). A chordal SO(3) prior prevents the IK from
                # inventing rotations that leave the observed JCPs unchanged.
                prior_per_joint = (
                    (local_rotation[:, 1:] - initial_local_rotation[:, 1:]) ** 2
                ).mean(dim=(-1, -2))
                prior_weights = torch.ones((1, 21), device=device)
                # Elbow JCPs observe flexion through the wrist position but do
                # not observe axial twist. Preserve the native elbow rotation
                # more strongly to remove the otherwise unconstrained flips.
                for elbow_offset, body_index in enumerate((17, 18)):
                    if elbow_prior_active[elbow_offset]:
                        prior_weights[:, body_index] = float(settings["elbow_prior_multiplier"])
                prior = (prior_per_joint * prior_weights).mean() * float(settings["pose_prior_weight"])
                velocity = torch.zeros((), device=device)
                acceleration = torch.zeros((), device=device)
                if len(root) > 1:
                    velocity = (((local_rotation[1:] - local_rotation[:-1]) ** 2).mean() +
                                0.1 * ((trans[1:] - trans[:-1]) ** 2).mean())
                    velocity *= float(settings["velocity_weight"])
                    if len(root) > 2:
                        relative = local_rotation[:-1].transpose(-1, -2) @ local_rotation[1:]
                        acceleration = (
                            ((relative[1:] - relative[:-1]) ** 2).mean() +
                            0.1 * ((trans[2:] - 2.0 * trans[1:-1] + trans[:-2]) ** 2).mean()
                        ) * float(settings["acceleration_weight"])
                contact = torch.zeros((), device=device)
                if len(root) > 1 and float(settings["foot_contact_weight"]) > 0:
                    feet = output.joints[:, [10, 11]] * scale + trans[:, None]
                    stance = torch.tensor(
                        contact_mask[start + 1:end] & contact_mask[start:end - 1],
                        dtype=torch.bool, device=device,
                    )
                    if stance.any():
                        foot_speed = torch.linalg.vector_norm(
                            (feet[1:] - feet[:-1]) * float(fps), dim=2,
                        )
                        contact = F.huber_loss(
                            foot_speed[stance], torch.zeros_like(foot_speed[stance]), delta=0.05,
                        ) * float(settings["foot_contact_weight"])
                boundary = torch.zeros((), device=device)
                if previous is not None:
                    current_boundary = batch_rodrigues(
                        torch.cat([root[0].reshape(1, 1, 3), body[[0]].reshape(1, 21, 3)], dim=1).reshape(-1, 3)
                    )
                    previous_boundary = batch_rodrigues(
                        torch.cat([previous[0].reshape(1, 1, 3), previous[1].reshape(1, 21, 3)], dim=1).reshape(-1, 3)
                    )
                    boundary = (((current_boundary - previous_boundary) ** 2).mean() +
                                ((trans[0] - previous[2]) ** 2).mean()) * float(settings["boundary_weight"])
                (data_loss + prior + velocity + acceleration + contact + boundary).backward()
                optimizer.step()
            with torch.no_grad():
                output = forward_batch(root, body, betas.expand(end - start, -1))
                predicted = output.vertices[:, indices] if target_kind == "vertices" else output.joints[:, indices]
                predicted = predicted * scale + trans[:, None]
                errors = torch.linalg.vector_norm(predicted - observed, dim=-1).cpu().numpy()
                errors[~valid[start:end]] = np.nan
                all_errors.append(errors)
                roots[start:end] = root.detach().cpu().numpy()
                bodies[start:end] = body.detach().cpu().numpy()
                translations[start:end] = trans.detach().cpu().numpy()
                previous = (root[-1].detach(), body[-1].detach(), trans[-1].detach())
    pre_c3_errors = np.concatenate(all_errors, axis=0)
    pre_c3_metrics = summarize_fit_errors(pre_c3_errors, valid, target_kind, settings)
    fallback_count = 0
    if fallback_body is not None or fallback_body_mask is not None:
        if fallback_body is None or fallback_body_mask is None:
            raise ValueError("fallback_body and fallback_body_mask must be provided together")
        fallback = np.asarray(fallback_body, dtype=np.float32).reshape(t, 21, 3)
        fallback_mask = np.asarray(fallback_body_mask, dtype=bool)
        if fallback_mask.shape != (t, 21):
            raise ValueError(f"fallback_body_mask shape {fallback_mask.shape}, expected {(t, 21)}")
        body_view = bodies.reshape(t, 21, 3)
        body_view[fallback_mask] = fallback[fallback_mask]
        bodies = body_view.reshape(t, 63)
        fallback_count = int(fallback_mask.sum())
    temporal = config.raw["temporal_continuity"]
    times = np.arange(t, dtype=np.float64) / float(fps)
    combined, rotation_temporal = c3_smooth_rotvec(
        times, np.concatenate([roots, bodies], axis=1),
        float(temporal["body_knot_seconds"]), int(temporal["spline_degree"]),
        int(temporal["robust_iterations"]), float(temporal["rotation_huber_degrees"]),
    )
    roots, bodies = combined[:, :3], combined[:, 3:]
    translations, translation_temporal = c3_smooth_translation(
        times, translations, float(temporal["translation_knot_seconds"]),
        int(temporal["spline_degree"]), int(temporal["robust_iterations"]),
        float(temporal["translation_huber_m"]),
    )
    post_c3_errors: list[np.ndarray] = []
    with _device_lock(config, use_cuda, device_index), torch.no_grad():
        indices = torch.tensor(target_indices, dtype=torch.long, device=device)
        for start in range(0, t, window):
            end = min(t, start + window)
            root_value = torch.tensor(roots[start:end], device=device)
            body_value = torch.tensor(bodies[start:end], device=device)
            trans_value = torch.tensor(translations[start:end], device=device)
            output = forward_batch(root_value, body_value, betas.expand(end - start, -1))
            predicted = output.vertices[:, indices] if target_kind == "vertices" else output.joints[:, indices]
            observed = torch.tensor(target[start:end], device=device)
            error = torch.linalg.vector_norm(
                predicted * scale + trans_value[:, None] - observed, dim=-1,
            ).cpu().numpy()
            error[~valid[start:end]] = np.nan
            post_c3_errors.append(error)
    metrics = summarize_fit_errors(np.concatenate(post_c3_errors, axis=0), valid, target_kind, settings)
    metrics.update({
        "observation_kind": target_kind, "device": str(device),
        "windows": int(np.ceil(t / window)),
        "foot_contact_frames_left": int(contact_mask[:, 0].sum()),
        "foot_contact_frames_right": int(contact_mask[:, 1].sum()),
        "foot_contact_method": "native_initialization_low_speed_near_floor",
        "rotation_continuity": "SO3_chordal_matrix_difference",
        "pre_c3_median_error_m": pre_c3_metrics["median_error_m"],
        "pre_c3_p95_error_m": pre_c3_metrics["p95_error_m"],
        "c3_continuity_order": int(temporal["spline_degree"]) - 1,
        "c3_rotation_projection": rotation_temporal,
        "c3_translation_projection": translation_temporal,
        "native_pose_fallback_joint_frames": fallback_count,
        "elbow_prior_active_left": bool(elbow_prior_active[0]),
        "elbow_prior_active_right": bool(elbow_prior_active[1]),
        "root_pivot_m": root_pivot.astype(float).tolist(),
    })
    return roots, bodies, translations, metrics
