#!/usr/bin/env python3
"""SMPLSim/MuJoCo kinematic replay; deliberately makes no controller claim."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
for _name, _value in {"bool": bool, "int": int, "float": float, "complex": complex,
                      "object": object, "unicode": str, "str": str}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

import mujoco
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.motion_retarget.config import load_config
from utils.motion_retarget.smplx_model import HAND_POSE_CONVENTION


def _scale_values(value: str, scale: float) -> str:
    return " ".join(f"{float(item) * scale:.10g}" for item in value.split())


def scale_humanoid_xml(payload: bytes, scale: float) -> bytes:
    root = ET.fromstring(payload)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("SMPLSim XML has no worldbody")
    humanoid = next((node for node in worldbody.findall("body") if node.get("name") == "Pelvis"), None)
    if humanoid is None:
        humanoid = worldbody.find("body")
    if humanoid is None:
        raise ValueError("SMPLSim XML has no humanoid body")
    for node in humanoid.iter():
        for attribute in ("pos", "size", "fromto"):
            if attribute in node.attrib:
                node.set(attribute, _scale_values(node.get(attribute, ""), scale))
    return ET.tostring(root, encoding="utf-8")


def resolve_asset(input_path: Path, relative: str) -> Path:
    for parent in input_path.parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve object asset {relative} from {input_path}")


def add_object_bodies(
    payload: bytes, input_path: Path, object_ids: list[str], mesh_paths: list[str],
    mesh_scales: np.ndarray,
) -> bytes:
    root = ET.fromstring(payload)
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("SMPLSim XML has no worldbody")
    for index, (object_id, relative, mesh_scale) in enumerate(zip(object_ids, mesh_paths, mesh_scales)):
        mesh_name = f"pilot_object_mesh_{index}"
        body_name = f"pilot_object_body_{index}"
        resolved = resolve_asset(input_path, relative)
        ET.SubElement(asset, "mesh", {
            "name": mesh_name, "file": str(resolved),
            "scale": f"{float(mesh_scale):.10g} {float(mesh_scale):.10g} {float(mesh_scale):.10g}",
        })
        body = ET.SubElement(worldbody, "body", {"name": body_name, "pos": "0 0 0"})
        ET.SubElement(body, "freejoint", {"name": f"pilot_object_joint_{index}"})
        ET.SubElement(body, "geom", {
            "name": f"pilot_object_geom_{index}", "type": "mesh", "mesh": mesh_name,
            "rgba": "0.92 0.55 0.18 1", "mass": "1", "contype": "1", "conaffinity": "1",
        })
    return ET.tostring(root, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--render", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    repo = Path(config.raw["paths"]["smplsim_repo"])
    sys.path.insert(0, str(repo))
    from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
    from smpl_sim.smpllib.smpl_mujoco_new import smpl_to_qpose

    with np.load(args.input, allow_pickle=False) as archive:
        convention = str(archive.get("hand_pose_convention", np.asarray("")).item())
        if convention != HAND_POSE_CONVENTION:
            raise ValueError(
                f"Unsupported or missing hand pose convention: {convention!r}"
            )
        pose = np.asarray(archive["pose_aa_smplsim"], dtype=np.float32)
        translation = np.asarray(archive["transl"], dtype=np.float32)
        betas = np.asarray(archive["betas"], dtype=np.float32)
        scale = float(archive["scale"])
        gender = str(archive["gender"].item())
        fps = float(archive["fps"])
        has_objects = int(np.asarray(archive.get("schema_version", 0))) >= 3 and "object_transl" in archive
        if has_objects:
            object_ids = [str(value) for value in archive["object_ids"]]
            object_translation = np.asarray(archive["object_transl"], dtype=np.float64)
            object_quaternion = np.asarray(archive["object_quat_xyzw"], dtype=np.float64)
            object_mesh_paths = [str(value) for value in archive["object_mesh_paths"]]
            object_mesh_scales = np.asarray(archive["object_mesh_scale_to_m"], dtype=np.float64)

    torch.set_num_threads(2)
    robot_cfg = {
        "model": "smplx", "mesh": False, "replace_feet": True, "freeze_hand": False,
        "big_ankle": True, "box_body": True, "real_weight": True,
        "real_weight_porpotion_capsules": True, "real_weight_porpotion_boxes": True,
        "rel_joint_lm": True, "upright_start": False, "create_vel_sensors": False,
    }
    gender_code = {"neutral": 0, "male": 1, "female": 2}[gender]
    robot = SMPL_Robot(robot_cfg, data_dir=str(config.smplx_model_dir))
    robot.load_from_skeleton(betas=torch.from_numpy(betas[None]), gender=[gender_code])
    xml = scale_humanoid_xml(robot.export_xml_string(), scale)
    human_model = mujoco.MjModel.from_xml_string(xml.decode())
    # The generated SMPL-X XML declares three hinge joints in x,y,z order.
    # SMPLSim's helper defaults to its older ZYX XML convention, so pass the
    # actual order explicitly or the same pose yields a different FK chain.
    qposes = smpl_to_qpose(
        pose, human_model, trans=translation, count_offset=True, model="smplx", euler_order="XYZ",
    )
    if qposes.shape != (len(pose), human_model.nq):
        raise ValueError(f"qpos mapping shape {qposes.shape}, expected {(len(pose), human_model.nq)}")
    if has_objects:
        xml = add_object_bodies(xml, args.input, object_ids, object_mesh_paths, object_mesh_scales)
    model = mujoco.MjModel.from_xml_string(xml.decode())
    data = mujoco.MjData(model)
    full_qposes = np.zeros((len(pose), model.nq), dtype=qposes.dtype)
    full_qposes[:, :human_model.nq] = qposes
    object_body_ids = []
    object_geom_ids = set()
    if has_objects:
        for object_index in range(len(object_ids)):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"pilot_object_body_{object_index}")
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"pilot_object_joint_{object_index}")
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"pilot_object_geom_{object_index}")
            address = model.jnt_qposadr[joint_id]
            full_qposes[:, address:address + 3] = object_translation[:, object_index]
            full_qposes[:, address + 3:address + 7] = object_quaternion[:, object_index][:, [3, 0, 1, 2]]
            object_body_ids.append(body_id)
            object_geom_ids.add(geom_id)

    body_lookup = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index): index
                   for index in range(model.nbody)}
    joint_names = list(robot.smpl_parser.joint_names)
    mapped_names = [name for name in joint_names if name in body_lookup]
    mapped_model_bodies = np.asarray([body_lookup[name] for name in mapped_names], dtype=np.int64)
    mapped_smpl_joints = np.asarray([robot.smpl_parser.joint_to_use[joint_names.index(name)]
                                     for name in mapped_names], dtype=np.int64)
    reference_chunks = []
    mesh_minimum_z_chunks = []
    with torch.no_grad():
        beta_tensor = torch.from_numpy(betas[None])
        if beta_tensor.shape[1] < 20:
            beta_tensor = torch.cat([beta_tensor, torch.zeros((1, 20 - beta_tensor.shape[1]))], dim=1)
        for start in range(0, len(pose), 128):
            end = min(len(pose), start + 128)
            pose_chunk = torch.from_numpy(pose[start:end])
            count = end - start
            zeros3 = torch.zeros((count, 3))
            output = robot.smpl_parser.forward(
                global_orient=pose_chunk[:, :3], body_pose=pose_chunk[:, 3:66],
                left_hand_pose=pose_chunk[:, 66:111], right_hand_pose=pose_chunk[:, 111:156],
                betas=beta_tensor.expand(count, -1), jaw_pose=zeros3, leye_pose=zeros3,
                reye_pose=zeros3, expression=torch.zeros((count, robot.smpl_parser.num_expression_coeffs)),
            )
            selected = output.joints[:, mapped_smpl_joints].cpu().numpy() * scale + translation[start:end, None]
            reference_chunks.append(selected)
            world_vertices = output.vertices.cpu().numpy() * scale + translation[start:end, None]
            mesh_minimum_z_chunks.append(world_vertices[:, :, 2].min(axis=1))
    reference_joints = np.concatenate(reference_chunks, axis=0)
    mesh_minimum_z = np.concatenate(mesh_minimum_z_chunks)
    mesh_penetration = np.maximum(0.0, -mesh_minimum_z)

    finite = True
    warning_frames = 0
    limited_values = 0
    limit_violations = 0
    penetration_frames = 0
    contact_frames = 0
    max_penetration = 0.0
    ground_contact_frames = 0
    ground_penetration_frames = 0
    max_ground_penetration = 0.0
    self_contact_frames = 0
    self_penetration_frames = 0
    max_self_penetration = 0.0
    human_object_contact_frames = 0
    human_object_penetration_frames = 0
    human_object_max_penetration = 0.0
    object_translation_errors = []
    object_rotation_errors = []
    ground_geoms = {
        geom_id for geom_id in range(model.ngeom)
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE or
        (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower() in {"floor", "ground"}
    }
    min_root_height = float("inf")
    fk_errors = []
    renderer = None
    video_writer = None
    render_camera = None
    render_temporary = None
    if args.render is not None:
        renderer = mujoco.Renderer(model, height=480, width=640)
        render_camera = mujoco.MjvCamera()
        render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        render_camera.distance = 3.2
        render_camera.azimuth = 135.0
        render_camera.elevation = -12.0
        try:
            args.render.parent.mkdir(parents=True, exist_ok=True)
            handle, temporary_name = tempfile.mkstemp(
                prefix=args.render.stem + ".", suffix=".partial.mp4", dir=args.render.parent,
            )
            os.close(handle)
            render_temporary = Path(temporary_name)
            video_writer = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", "640x480", "-r", "30", "-i", "-", "-an",
                 "-vcodec", "libx264", "-pix_fmt", "yuv420p", str(render_temporary)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            renderer.close()
            if render_temporary is not None:
                render_temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Cannot open render output: {exc}") from exc
    render_stride = max(1, round(fps / 30.0))

    for frame, qpos in enumerate(full_qposes):
        data.qpos[:] = qpos
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        finite = finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.xpos).all())
        warning_frames += any(item.number for item in data.warning)
        min_root_height = min(min_root_height, float(data.xpos[1, 2]))
        fk_errors.append(np.linalg.norm(data.xpos[mapped_model_bodies] - reference_joints[frame], axis=1))
        if has_objects:
            for object_index, body_id in enumerate(object_body_ids):
                object_translation_errors.append(float(np.linalg.norm(
                    data.xpos[body_id] - object_translation[frame, object_index]
                )))
                expected = Rotation.from_quat(object_quaternion[frame, object_index]).as_matrix()
                actual = Rotation.from_quat(data.xquat[body_id][[1, 2, 3, 0]]).as_matrix()
                object_rotation_errors.append(float(np.degrees(Rotation.from_matrix(expected.T @ actual).magnitude())))
        if data.ncon:
            contact_frames += 1
            contacts = [data.contact[index] for index in range(data.ncon)]
            distances = [float(contact.dist) for contact in contacts]
            depth = max(0.0, -min(distances))
            if depth > 0:
                penetration_frames += 1
                max_penetration = max(max_penetration, depth)
            ground_distances = [
                float(contact.dist) for contact in contacts
                if int(contact.geom1) in ground_geoms or int(contact.geom2) in ground_geoms
            ]
            self_distances = [
                float(contact.dist) for contact in contacts
                if int(contact.geom1) not in ground_geoms and int(contact.geom2) not in ground_geoms
                and int(contact.geom1) not in object_geom_ids and int(contact.geom2) not in object_geom_ids
            ]
            if ground_distances:
                ground_contact_frames += 1
                ground_depth = max(0.0, -min(ground_distances))
                if ground_depth > 0:
                    ground_penetration_frames += 1
                    max_ground_penetration = max(max_ground_penetration, ground_depth)
            if self_distances:
                self_contact_frames += 1
                self_depth = max(0.0, -min(self_distances))
                if self_depth > 0:
                    self_penetration_frames += 1
                    max_self_penetration = max(max_self_penetration, self_depth)
            object_distances = [
                float(contact.dist) for contact in contacts
                if int(contact.geom1) in object_geom_ids or int(contact.geom2) in object_geom_ids
            ]
            if object_distances:
                human_object_contact_frames += 1
                object_depth = max(0.0, -min(object_distances))
                if object_depth > 0:
                    human_object_penetration_frames += 1
                    human_object_max_penetration = max(human_object_max_penetration, object_depth)
        for joint_id in range(model.njnt):
            if not model.jnt_limited[joint_id] or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            value = data.qpos[model.jnt_qposadr[joint_id]]
            low, high = model.jnt_range[joint_id]
            limited_values += 1
            limit_violations += bool(value < low - 1e-5 or value > high + 1e-5)
        if renderer is not None and frame % render_stride == 0:
            render_camera.lookat[:] = data.xpos[1] + np.asarray([0.0, 0.0, 0.65])
            renderer.update_scene(data, camera=render_camera)
            video_writer.stdin.write(renderer.render().tobytes())

    if renderer is not None:
        renderer.close()
        video_writer.stdin.close()
        if video_writer.wait() != 0:
            render_temporary.unlink(missing_ok=True)
            raise RuntimeError("ffmpeg failed while writing MuJoCo preview")
        os.replace(render_temporary, args.render)

    ordered_limits = bool(np.all(model.jnt_range[model.jnt_limited.astype(bool), 0] <=
                                 model.jnt_range[model.jnt_limited.astype(bool), 1]))
    fk_errors = np.concatenate(fk_errors)
    fk_median = float(np.median(fk_errors))
    fk_p95 = float(np.percentile(fk_errors, 95))
    fk_threshold = float(config.raw["quality"]["mujoco_fk_error_m"])
    checks = {
        "finite_forward_kinematics": finite,
        "ordered_joint_limits": ordered_limits,
        "no_mujoco_warnings": warning_frames == 0,
        "qpos_mapping_complete": qposes.shape[1] == human_model.nq,
        "mapped_joint_fk": fk_p95 <= fk_threshold,
    }
    if has_objects:
        acceptance = config.raw["object_pilot"]["acceptance"]
        checks["fixed_object_pose_replay"] = bool(
            max(object_translation_errors, default=0.0) <= float(acceptance["mujoco_object_translation_m"]) and
            max(object_rotation_errors, default=0.0) <= float(acceptance["mujoco_object_rotation_deg"])
        )
    report = {
        "passed": all(checks.values()), "checks": checks, "frames": len(pose), "fps": fps,
        "nq": model.nq, "nv": model.nv, "body_count": model.nbody,
        "joint_limit_violation_fraction": limit_violations / max(1, limited_values),
        "contact_frame_fraction": contact_frames / len(pose),
        "penetration_frame_fraction": penetration_frames / len(pose),
        "max_contact_penetration_m": max_penetration, "minimum_root_height_m": min_root_height,
        "ground_contact_frame_fraction": ground_contact_frames / len(pose),
        "ground_penetration_frame_fraction": ground_penetration_frames / len(pose),
        "max_ground_penetration_m": max_ground_penetration,
        "self_contact_frame_fraction": self_contact_frames / len(pose),
        "self_penetration_frame_fraction": self_penetration_frames / len(pose),
        "max_self_penetration_m": max_self_penetration,
        "smplx_mesh_ground_penetration_frame_fraction": float(np.mean(mesh_penetration > 0.001)),
        "max_smplx_mesh_ground_penetration_m": float(mesh_penetration.max(initial=0.0)),
        "mapped_joint_count": len(mapped_names), "mapped_joint_fk_median_m": fk_median,
        "mapped_joint_fk_p95_m": fk_p95,
        "warnings_frames": int(warning_frames),
        "object_count": len(object_body_ids),
        "object_translation_max_error_m": max(object_translation_errors, default=0.0),
        "object_rotation_max_error_deg": max(object_rotation_errors, default=0.0),
        "human_object_contact_frame_fraction": human_object_contact_frames / len(pose),
        "human_object_penetration_frame_fraction": human_object_penetration_frames / len(pose),
        "human_object_max_mujoco_penetration_m": human_object_max_penetration,
        "scope": "Per-frame mj_forward, joint-limit and contact sanity only. No dynamics or controller tracking claim.",
        "fk_error_status": "independent_smplx_joints_vs_mujoco_body_xpos",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=args.report.name, suffix=".partial", dir=args.report.parent)
    with os.fdopen(handle, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
