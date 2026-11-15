#!/usr/bin/env python3
"""Render retargeted SMPL-X surface vertices, not MuJoCo collision geometry."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Select EGL before importing pyrender/PyOpenGL so batch rendering does not
# depend on the desktop player or an active X11 window.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.motion_retarget.config import load_config
from utils.motion_retarget.smplx_model import HAND_POSE_CONVENTION, create_smplx_model


def nearest_frame_indices(timestamps: np.ndarray, source_fps: float, target_fps: float) -> np.ndarray:
    """Resample a strictly increasing source clock without changing duration."""
    count = len(timestamps)
    if count <= 1:
        return np.zeros(1, dtype=np.int64)
    source = np.asarray(timestamps, dtype=np.float64) - float(timestamps[0])
    duration = max(float(source[-1]) + 1.0 / source_fps, 1.0 / target_fps)
    target = np.arange(max(1, int(np.ceil(duration * target_fps))), dtype=np.float64) / target_fps
    right = np.clip(np.searchsorted(source, target, side="left"), 0, count - 1)
    left = np.maximum(0, right - 1)
    choose_right = np.abs(source[right] - target) < np.abs(source[left] - target)
    return np.where(choose_right, right, left).astype(np.int64)


def camera_pose(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return an OpenGL camera-to-world pose looking from eye to target."""
    forward = target - eye
    forward /= max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(right, forward)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.column_stack([right, up, -forward])
    pose[:3, 3] = eye
    return pose


def project_world_points(
    points: np.ndarray, pose: np.ndarray, width: int, height: int, yfov: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points into the active OpenGL camera for diagnostic ink."""
    values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    homogeneous = np.c_[values, np.ones(len(values))]
    camera = homogeneous @ np.linalg.inv(pose).T
    depth = -camera[:, 2]
    aspect = float(width) / float(height)
    tangent = np.tan(0.5 * float(yfov))
    x = camera[:, 0] / np.maximum(depth * tangent * aspect, 1e-8)
    y = camera[:, 1] / np.maximum(depth * tangent, 1e-8)
    pixel = np.stack([
        0.5 * (x + 1.0) * width,
        0.5 * (1.0 - y) * height,
    ], axis=1)
    visible = (
        (depth > 0.0) & (pixel[:, 0] >= 0) & (pixel[:, 0] < width) &
        (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
    )
    return np.round(pixel).astype(np.int32), visible


def draw_source_hand_proxy(
    image: np.ndarray, landmarks_world: np.ndarray, pose: np.ndarray, yfov: float,
) -> None:
    """Overlay raw-BVH dynamic hand proxies, clearly distinct from SMPL-X."""
    connections = [(0, 1)]
    for base in (2, 6, 10, 14, 18):
        connections.append((1, base))
        connections.extend((base + offset, base + offset + 1) for offset in range(3))
    colors = ((235, 80, 230), (80, 235, 235))
    for side in range(2):
        points, visible = project_world_points(
            landmarks_world[side], pose, image.shape[1], image.shape[0], yfov,
        )
        for first, second in connections:
            if visible[first] and visible[second]:
                cv2.line(image, tuple(points[first]), tuple(points[second]), colors[side], 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            if visible[index]:
                cv2.circle(image, tuple(point), 3 if index > 1 else 5, colors[side], -1, cv2.LINE_AA)


def draw_palm_frame(
    image: np.ndarray, origin: np.ndarray, matrix: np.ndarray, pose: np.ndarray,
    yfov: float, axis_length: float = 0.08, thickness: int = 2,
) -> None:
    endpoints = origin[None] + np.asarray(matrix).T * float(axis_length)
    points, visible = project_world_points(
        np.vstack([origin, endpoints]), pose, image.shape[1], image.shape[0], yfov,
    )
    for axis, color in enumerate(((240, 70, 70), (70, 240, 70), (70, 120, 250))):
        if visible[0] and visible[axis + 1]:
            cv2.arrowedLine(
                image, tuple(points[0]), tuple(points[axis + 1]), color,
                thickness, cv2.LINE_AA, tipLength=0.18,
            )


def ground_mesh() -> tuple[pyrender.Mesh, pyrender.Mesh]:
    ground = trimesh.creation.box(extents=(12.0, 12.0, 0.02))
    ground.apply_translation((0.0, 0.0, -0.011))
    ground_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.17, 0.19, 0.22, 1.0), metallicFactor=0.0, roughnessFactor=0.95,
    )
    strips = []
    for coordinate in np.arange(-6.0, 6.01, 0.5):
        horizontal = trimesh.creation.box(extents=(12.0, 0.008, 0.004))
        horizontal.apply_translation((0.0, coordinate, 0.002))
        strips.append(horizontal)
        vertical = trimesh.creation.box(extents=(0.008, 12.0, 0.004))
        vertical.apply_translation((coordinate, 0.0, 0.002))
        strips.append(vertical)
    grid_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.46, 0.50, 0.55, 1.0), metallicFactor=0.0, roughnessFactor=0.9,
    )
    return (
        pyrender.Mesh.from_trimesh(ground, material=ground_material, smooth=False),
        pyrender.Mesh.from_trimesh(trimesh.util.concatenate(strips), material=grid_material, smooth=False),
    )


def add_overlay(
    image: np.ndarray, frame_index: int, gender: str, height_m: float,
    left_valid: bool, right_valid: bool, object_status: str | None = None,
    contact_status: str | None = None,
) -> None:
    cv2.putText(
        image, f"SMPL-X mesh | {gender} {height_m:.2f}m | source frame {frame_index}",
        (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA,
    )
    if object_status is not None:
        cv2.putText(
            image, object_status, (14, 73), cv2.FONT_HERSHEY_SIMPLEX, 0.49,
            (80, 220, 245), 1, cv2.LINE_AA,
        )
    if contact_status is not None:
        cv2.putText(
            image, contact_status, (14, 97), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
            (120, 245, 170), 1, cv2.LINE_AA,
        )
    left_label = "L HAND: OBS" if left_valid else "L HAND: PRIOR"
    right_label = "R HAND: OBS" if right_valid else "R HAND: PRIOR"
    cv2.putText(
        image, left_label, (14, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (80, 230, 110) if left_valid else (245, 85, 85), 1, cv2.LINE_AA,
    )
    cv2.putText(
        image, right_label, (170, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
        (80, 230, 110) if right_valid else (245, 85, 85), 1, cv2.LINE_AA,
    )


def resolve_asset(input_path: Path, relative: str) -> Path:
    candidate = Path(relative)
    for parent in input_path.parents:
        resolved = parent / candidate
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(f"Cannot resolve object asset {relative} from {input_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an actual SMPL-X surface mesh video")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--camera-mode", choices=("full", "contact"), default="full")
    parser.add_argument("--hide-trajectory", action="store_true")
    parser.add_argument("--contact-sheet", type=Path)
    args = parser.parse_args()
    if args.width % 2 or args.height % 2:
        raise ValueError("H.264 output dimensions must be even")

    config = load_config(args.config)
    with np.load(args.input, allow_pickle=False) as archive:
        convention = str(archive.get("hand_pose_convention", np.asarray("")).item())
        if convention != HAND_POSE_CONVENTION:
            raise ValueError(
                f"Unsupported or missing hand pose convention: {convention!r}"
            )
        global_orient = np.asarray(archive["global_orient"], dtype=np.float32)
        body_pose = np.asarray(archive["body_pose"], dtype=np.float32)
        left_hand_pose = np.asarray(archive["left_hand_pose"], dtype=np.float32)
        right_hand_pose = np.asarray(archive["right_hand_pose"], dtype=np.float32)
        translation = np.asarray(archive["transl"], dtype=np.float32)
        timestamps = np.asarray(archive["timestamps"], dtype=np.float64)
        left_valid = np.asarray(archive["left_hand_valid"], dtype=bool)
        right_valid = np.asarray(archive["right_hand_valid"], dtype=bool)
        betas = np.asarray(archive["betas"], dtype=np.float32)
        scale = float(archive["scale"])
        source_fps = float(archive["fps"])
        gender = str(archive["gender"].item())
        height_m = float(archive["height_m"])
        schema_version = int(np.asarray(archive.get("schema_version", 0)))
        has_objects = schema_version >= 3 and "object_transl" in archive
        constrained_v5 = schema_version >= 5
        if has_objects:
            object_translation = np.asarray(archive["object_transl"], dtype=np.float32)
            object_quaternion = np.asarray(archive["object_quat_xyzw"], dtype=np.float32)
            object_valid = np.asarray(archive["object_valid"], dtype=bool)
            object_ids = [str(value) for value in archive["object_ids"]]
            object_mesh_paths = [str(value) for value in archive["object_mesh_paths"]]
            object_mesh_scale = np.asarray(archive["object_mesh_scale_to_m"], dtype=np.float32)
            left_object_contact = np.asarray(archive["left_hand_object_contact"], dtype=bool)
            right_object_contact = np.asarray(archive["right_hand_object_contact"], dtype=bool)
            penetration = np.asarray(archive["body_object_penetration_depth_m"], dtype=np.float32)
            left_object_distance = np.asarray(archive["left_hand_object_distance_m"], dtype=np.float32)
            right_object_distance = np.asarray(archive["right_hand_object_distance_m"], dtype=np.float32)
            source_hand_contact = np.asarray(
                archive.get("source_hand_contact", np.zeros((len(timestamps), len(object_ids), 2))),
                dtype=bool,
            )
            annotation_assisted = np.asarray(
                archive.get("annotation_assisted", np.zeros_like(source_hand_contact)), dtype=bool,
            )
            target_hand_landmarks_local = np.asarray(
                archive.get("target_hand_landmarks_local", np.empty((0,))), dtype=np.float32,
            )
            source_hand_landmarks_local = np.asarray(
                archive.get("source_hand_landmarks_local", np.empty((0,))), dtype=np.float32,
            )
            source_palm_position_local = np.asarray(
                archive.get("source_palm_position_local", np.empty((0,))), dtype=np.float32,
            )
            source_palm_quaternion_local = np.asarray(
                archive.get("source_palm_quat_xyzw_local", np.empty((0,))), dtype=np.float32,
            )
            target_palm_position_local = np.asarray(
                archive.get("target_palm_position_local", np.empty((0,))), dtype=np.float32,
            )
            target_palm_quaternion_local = np.asarray(
                archive.get("target_palm_quat_xyzw_local", np.empty((0,))), dtype=np.float32,
            )
            remapped_geodesic = np.asarray(
                archive.get("remapped_contact_geodesic_m", np.empty((0,))), dtype=np.float32,
            )
            remapped_valid = np.asarray(
                archive.get("dynamic_contact_constraint_valid", np.empty((0,))), dtype=bool,
            )

    selected = nearest_frame_indices(timestamps, source_fps, args.fps)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # The configured path already is the terminal ``models/smplx`` directory;
    # smplx.create() would append another ``smplx`` component.
    model = create_smplx_model(
        config.smplx_model_dir, gender, len(betas),
        batch_size=args.batch_size, device=device,
    )
    model.eval()
    faces = np.asarray(model.faces, dtype=np.int32)
    beta_tensor = torch.from_numpy(betas).to(device)[None]

    scene = pyrender.Scene(
        bg_color=(0.035, 0.045, 0.06, 1.0), ambient_light=(0.22, 0.22, 0.24),
    )
    ground, grid = ground_mesh()
    scene.add(ground)
    scene.add(grid)
    pane_count = 3 if has_objects and constrained_v5 else (2 if has_objects else 1)
    pane_width = args.width // pane_count
    camera_yfov = np.deg2rad(42.0)
    camera = pyrender.PerspectiveCamera(yfov=camera_yfov, aspectRatio=pane_width / args.height)
    camera_node = scene.add(camera, pose=np.eye(4))
    for intensity, offset in (
        (3.5, (3.0, -4.0, 6.0)), (2.0, (-4.0, -1.0, 3.0)), (1.5, (1.0, 4.0, 2.0)),
    ):
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=intensity)
        light_eye = np.asarray(offset, dtype=np.float64)
        scene.add(light, pose=camera_pose(light_eye, np.zeros(3)))
    body_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.38, 0.67, 0.88, 1.0), metallicFactor=0.05, roughnessFactor=0.62,
    )
    renderer = pyrender.OffscreenRenderer(pane_width, args.height)
    object_nodes = []
    if has_objects:
        object_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.92, 0.55, 0.18, 1.0), metallicFactor=0.08, roughnessFactor=0.58,
        )
        trajectory_material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.95, 0.23, 0.22, 0.85), metallicFactor=0.0, roughnessFactor=0.7,
        )
        for object_index, relative in enumerate(object_mesh_paths):
            object_mesh = trimesh.load(resolve_asset(args.input, relative), force="mesh", process=False)
            object_mesh.apply_scale(float(object_mesh_scale[object_index]))
            object_nodes.append(scene.add(
                pyrender.Mesh.from_trimesh(object_mesh, material=object_material, smooth=True),
                pose=np.eye(4),
            ))
            points = object_translation[:, object_index]
            sample = np.unique(np.linspace(0, len(points) - 1, min(120, len(points))).astype(int))
            segments = []
            for left_index, right_index in zip(sample[:-1], sample[1:]):
                if np.linalg.norm(points[right_index] - points[left_index]) > 1e-5:
                    segments.append(trimesh.creation.cylinder(
                        radius=0.006, segment=np.stack([points[left_index], points[right_index]]), sections=8,
                    ))
            if segments and not args.hide_trajectory:
                scene.add(pyrender.Mesh.from_trimesh(
                    trimesh.util.concatenate(segments), material=trajectory_material, smooth=False,
                ))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=args.output.stem + ".", suffix=".partial.mp4", dir=args.output.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    writer = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s", f"{args.width}x{args.height}", "-r", str(args.fps),
            "-i", "-", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
        ],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    mesh_node = None
    smoothed_full_target = None
    smoothed_contact_target = None
    sheet_path = args.contact_sheet or args.output.with_suffix(".contact_sheet.jpg")
    sheet_source_frames: set[int] = set(np.linspace(
        0, len(global_orient) - 1, min(8, len(global_orient)), dtype=np.int64,
    ).tolist())
    if has_objects:
        source_any = source_hand_contact.any(axis=(1, 2))
        transitions = np.flatnonzero(np.diff(source_any.astype(np.int8), prepend=0) != 0)
        sheet_source_frames.update(int(value) for value in transitions[:6])
        frame_penetration = penetration.max(axis=1) if penetration.ndim > 1 else penetration
        deepest = np.argsort(frame_penetration)[-2:]
        sheet_source_frames.update(int(value) for value in deepest)
    # The video clock may not contain the exact 90 Hz transition frame.  Snap
    # requested evidence frames onto frames that will actually be rendered.
    sheet_source_frames = {
        int(selected[np.argmin(np.abs(selected - requested))])
        for requested in sheet_source_frames
    }
    sheet_frames: dict[int, np.ndarray] = {}
    rendered = 0
    try:
        with torch.inference_mode():
            for chunk_start in range(0, len(selected), args.batch_size):
                source_indices = selected[chunk_start:chunk_start + args.batch_size]
                count = len(source_indices)
                if count < args.batch_size:
                    padded_indices = np.concatenate([
                        source_indices,
                        np.full(args.batch_size - count, source_indices[-1], dtype=np.int64),
                    ])
                else:
                    padded_indices = source_indices

                def tensor(values: np.ndarray) -> torch.Tensor:
                    return torch.from_numpy(np.ascontiguousarray(values[padded_indices])).to(device)

                zeros3 = torch.zeros((args.batch_size, 3), dtype=torch.float32, device=device)
                output = model(
                    betas=beta_tensor.expand(args.batch_size, -1),
                    global_orient=tensor(global_orient), body_pose=tensor(body_pose),
                    left_hand_pose=tensor(left_hand_pose), right_hand_pose=tensor(right_hand_pose),
                    jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                    expression=torch.zeros((args.batch_size, model.num_expression_coeffs), device=device),
                    return_verts=True,
                )
                vertices = output.vertices[:count].detach().cpu().numpy() * scale
                vertices += translation[source_indices, None]
                for source_index, frame_vertices in zip(source_indices, vertices):
                    if mesh_node is not None:
                        scene.remove_node(mesh_node)
                    surface = trimesh.Trimesh(vertices=frame_vertices, faces=faces, process=False)
                    mesh_node = scene.add(
                        pyrender.Mesh.from_trimesh(surface, material=body_material, smooth=True)
                    )
                    if has_objects:
                        for object_index, node in enumerate(object_nodes):
                            pose = np.eye(4)
                            pose[:3, :3] = Rotation.from_quat(
                                object_quaternion[source_index, object_index]
                            ).as_matrix()
                            pose[:3, 3] = object_translation[source_index, object_index]
                            scene.set_pose(node, pose)
                    bounds = np.asarray([frame_vertices.min(axis=0), frame_vertices.max(axis=0)])
                    target = np.asarray([
                        translation[source_index, 0], translation[source_index, 1],
                        0.52 * bounds[0, 2] + 0.48 * bounds[1, 2],
                    ])
                    if has_objects:
                        valid_centers = object_translation[source_index, object_valid[source_index]]
                        if len(valid_centers):
                            if args.camera_mode == "contact":
                                target = valid_centers.mean(axis=0)
                            else:
                                target = 0.65 * target + 0.35 * valid_centers.mean(axis=0)
                    smoothed_full_target = (
                        target if smoothed_full_target is None
                        else 0.88 * smoothed_full_target + 0.12 * target
                    )
                    full_offset = np.asarray([2.65, -2.65, 1.45]) * max(height_m / 1.72, 0.75)
                    full_camera_pose = camera_pose(
                        smoothed_full_target + full_offset, smoothed_full_target,
                    )
                    scene.set_pose(camera_node, full_camera_pose)
                    full_image, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
                    full_image = np.array(full_image, copy=True)

                    if has_objects:
                        contact_target = object_translation[source_index, object_valid[source_index]].mean(axis=0)
                        if target_hand_landmarks_local.ndim == 5:
                            active_sides = source_hand_contact[source_index].any(axis=0)
                            local_palms = target_hand_landmarks_local[source_index, 0, active_sides, :2]
                            if local_palms.size:
                                rotation = Rotation.from_quat(object_quaternion[source_index, 0]).as_matrix()
                                contact_target = (
                                    local_palms.reshape(-1, 3) @ rotation.T
                                    + object_translation[source_index, 0]
                                ).mean(axis=0)
                        smoothed_contact_target = (
                            contact_target if smoothed_contact_target is None
                            else 0.82 * smoothed_contact_target + 0.18 * contact_target
                        )
                        close_offset = np.asarray([0.72, -0.72, 0.42]) * max(height_m / 1.72, 0.75)
                        close_camera_pose = camera_pose(
                            smoothed_contact_target + close_offset, smoothed_contact_target,
                        )
                        scene.set_pose(camera_node, close_camera_pose)
                        close_image, _ = renderer.render(scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
                        close_image = np.array(close_image, copy=True)
                        source_proxy_world = None
                        object_rotation = Rotation.from_quat(
                            object_quaternion[source_index, 0]
                        ).as_matrix()
                        if constrained_v5 and source_hand_landmarks_local.ndim == 5:
                            source_proxy_world = (
                                source_hand_landmarks_local[source_index, 0] @ object_rotation.T +
                                object_translation[source_index, 0]
                            )
                            draw_source_hand_proxy(
                                close_image, source_proxy_world, close_camera_pose, camera_yfov,
                            )
                            for side in range(2):
                                for positions, quaternions, thickness in (
                                    (source_palm_position_local, source_palm_quaternion_local, 1),
                                    (target_palm_position_local, target_palm_quaternion_local, 3),
                                ):
                                    if positions.ndim != 4 or quaternions.ndim != 4:
                                        continue
                                    origin = (
                                        positions[source_index, 0, side] @ object_rotation.T +
                                        object_translation[source_index, 0]
                                    )
                                    local_matrix = Rotation.from_quat(
                                        quaternions[source_index, 0, side]
                                    ).as_matrix()
                                    draw_palm_frame(
                                        close_image, origin, object_rotation @ local_matrix,
                                        close_camera_pose, camera_yfov, thickness=thickness,
                                    )
                        if constrained_v5:
                            right_target = smoothed_contact_target
                            if target_hand_landmarks_local.ndim == 5:
                                right_target = (
                                    target_hand_landmarks_local[source_index, 0, 1, 1] @
                                    object_rotation.T + object_translation[source_index, 0]
                                )
                            right_offset = np.asarray([0.34, -0.34, 0.20]) * max(height_m / 1.72, 0.75)
                            right_camera_pose = camera_pose(right_target + right_offset, right_target)
                            scene.set_pose(camera_node, right_camera_pose)
                            right_image, _ = renderer.render(
                                scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL,
                            )
                            right_image = np.array(right_image, copy=True)
                            if source_proxy_world is not None:
                                draw_source_hand_proxy(
                                    right_image, source_proxy_world, right_camera_pose, camera_yfov,
                                )
                                for positions, quaternions, thickness in (
                                    (source_palm_position_local, source_palm_quaternion_local, 1),
                                    (target_palm_position_local, target_palm_quaternion_local, 3),
                                ):
                                    if positions.ndim == 4 and quaternions.ndim == 4:
                                        origin = (
                                            positions[source_index, 0, 1] @ object_rotation.T +
                                            object_translation[source_index, 0]
                                        )
                                        local_matrix = Rotation.from_quat(
                                            quaternions[source_index, 0, 1]
                                        ).as_matrix()
                                        draw_palm_frame(
                                            right_image, origin, object_rotation @ local_matrix,
                                            right_camera_pose, camera_yfov, thickness=thickness,
                                        )
                            remaining = args.width - 2 * pane_width
                            if right_image.shape[1] != remaining:
                                right_image = cv2.resize(
                                    right_image, (remaining, args.height), interpolation=cv2.INTER_AREA,
                                )
                            image = np.concatenate([full_image, close_image, right_image], axis=1)
                        else:
                            image = np.concatenate([full_image, close_image], axis=1)
                        cv2.putText(image, "FULL BODY", (14, args.height - 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
                        cv2.putText(image, "HANDS / OBJECT + SOURCE PROXY", (pane_width + 14, args.height - 18),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
                        if constrained_v5:
                            cv2.putText(image, "RIGHT HAND CLOSE-UP", (2 * pane_width + 14, args.height - 18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
                    else:
                        image = full_image
                    object_status = None
                    contact_status = None
                    if has_objects:
                        contacts = []
                        if left_object_contact[source_index].any(): contacts.append("L-contact")
                        if right_object_contact[source_index].any(): contacts.append("R-contact")
                        maximum_penetration = float(penetration[source_index].max(initial=0.0))
                        warning = f" | PEN {maximum_penetration * 1000:.1f}mm" if maximum_penetration > 0.003 else ""
                        object_status = f"OBJ {','.join(object_ids)} | {'+'.join(contacts) or 'no target contact'}{warning}"
                        source_sides = source_hand_contact[source_index].any(axis=0)
                        assisted_sides = annotation_assisted[source_index].any(axis=0)
                        left_gap = float(np.min(left_object_distance[source_index])) * 1000.0
                        right_gap = float(np.min(right_object_distance[source_index])) * 1000.0
                        contact_status = (
                            f"SRC L:{int(source_sides[0])} R:{int(source_sides[1])} | "
                            f"AUX L:{int(assisted_sides[0])} R:{int(assisted_sides[1])} | "
                            f"TARGET GAP L:{left_gap:.1f} R:{right_gap:.1f} mm"
                        )
                        if constrained_v5 and remapped_geodesic.size and remapped_valid.size:
                            active_remap = remapped_valid[source_index]
                            geodesic = (
                                float(remapped_geodesic[source_index][active_remap].max(initial=0.0))
                                if active_remap.any() else 0.0
                            )
                            contact_status += f" | REMAP MAX:{geodesic * 1000.0:.2f}mm"
                    add_overlay(
                        image, int(source_index), gender, height_m,
                        bool(left_valid[source_index]), bool(right_valid[source_index]),
                        object_status, contact_status,
                    )
                    if int(source_index) in sheet_source_frames and int(source_index) not in sheet_frames:
                        sheet_frames[int(source_index)] = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    assert writer.stdin is not None
                    writer.stdin.write(image.tobytes())
                    rendered += 1
        assert writer.stdin is not None
        writer.stdin.close()
        stderr = writer.stderr.read().decode(errors="replace") if writer.stderr is not None else ""
        code = writer.wait()
        if code:
            raise RuntimeError(f"ffmpeg failed ({code}): {stderr[-4000:]}")
        os.replace(temporary, args.output)
        if sheet_frames:
            ordered = [sheet_frames[index] for index in sorted(sheet_frames)[:16]]
            tile_width, tile_height = 480, 270
            tiles = [cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                     for frame in ordered]
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.concatenate([
                np.concatenate(tiles[start:start + 4], axis=1)
                for start in range(0, len(tiles), 4)
            ], axis=0)
            sheet_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"Cannot write contact sheet {sheet_path}")
    except Exception:
        if writer.stdin is not None and not writer.stdin.closed:
            writer.stdin.close()
        writer.kill()
        writer.wait()
        temporary.unlink(missing_ok=True)
        raise
    finally:
        renderer.delete()

    print(json.dumps({
        "renderer": "smplx_surface_mesh", "input": str(args.input), "output": str(args.output),
        "frames": rendered, "fps": args.fps, "width": args.width, "height": args.height,
        "source_frames": len(global_orient), "device": str(device),
        "camera_mode": args.camera_mode,
        "trajectory_visible": not args.hide_trajectory,
        "layout": (
            "full_body_plus_hands_object_plus_right_hand" if has_objects and constrained_v5
            else ("full_body_plus_hands_object_split" if has_objects else "full_body")
        ),
        "contact_sheet": str(sheet_path) if sheet_frames else None,
        "hand_overlay": "OBS uses valid mask; PRIOR is smooth imputation and not an observation",
        "diagnostic_overlays": (
            ["raw_bvh_hand_proxy", "source_and_target_palm_frames", "remap_distance", "right_hand_closeup"]
            if has_objects and constrained_v5 else []
        ),
        "objects": object_ids if has_objects else [],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
