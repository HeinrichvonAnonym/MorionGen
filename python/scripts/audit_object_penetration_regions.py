#!/usr/bin/env python3
"""Audit rigid SMPL-X/object overlap by anatomical and contact region.

This is deliberately separate from MuJoCo collision geoms.  It evaluates the
canonical SMPL-X surface against the same real-OBJ triangle SDF used by the v3
retargeter, then distinguishes contact-hand compression from unexplained body
or non-contact-hand penetration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.motion_retarget.config import load_config
from utils.motion_retarget.objects import build_signed_distance_grid, load_repaired_mesh
from utils.motion_retarget.smplx_model import create_smplx_model


def resolve_asset(input_path: Path, relative: str) -> Path:
    for parent in input_path.parents:
        candidate = parent / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve object asset {relative} from {input_path}")


def histogram_percentile(counts: np.ndarray, edges: np.ndarray, percentile: float) -> float:
    total = int(counts.sum())
    if not total:
        return 0.0
    target = percentile / 100.0 * total
    cumulative = np.cumsum(counts)
    index = min(int(np.searchsorted(cumulative, target, side="left")), len(counts) - 1)
    previous = int(cumulative[index - 1]) if index else 0
    fraction = (target - previous) / max(1, int(counts[index]))
    return float(edges[index] + fraction * (edges[index + 1] - edges[index]))


def classify_penetration(
    metrics: dict[str, dict[str, float]], strict_p95: float, maximum: float,
    soft_p95: float = 0.005,
) -> tuple[bool, bool, float]:
    noncontact_positive_fraction = (
        metrics["noncontact_hand"]["positive_sample_count"] /
        max(1, metrics["all"]["positive_sample_count"])
    )
    strict = bool(
        metrics["all"]["penetration_p95_m"] <= strict_p95 and
        metrics["all"]["penetration_max_m"] <= maximum
    )
    soft = bool(
        metrics["all"]["penetration_p95_m"] <= soft_p95 and
        metrics["all"]["penetration_max_m"] <= maximum and
        metrics["nonhand"]["penetration_max_m"] <= strict_p95 and
        metrics["noncontact_hand"]["penetration_p95_m"] <= soft_p95 and
        metrics["noncontact_hand"]["penetration_max_m"] <= maximum and
        noncontact_positive_fraction <= 0.01
    )
    return strict, soft, noncontact_positive_fraction


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    config = load_config(args.config)
    with np.load(args.input, allow_pickle=False) as archive:
        global_orient = np.asarray(archive["global_orient"], dtype=np.float32)
        body_pose = np.asarray(archive["body_pose"], dtype=np.float32)
        left_hand_pose = np.asarray(archive["left_hand_pose"], dtype=np.float32)
        right_hand_pose = np.asarray(archive["right_hand_pose"], dtype=np.float32)
        translation = np.asarray(archive["transl"], dtype=np.float32)
        betas = np.asarray(archive["betas"], dtype=np.float32)
        scale = float(archive["scale"])
        gender = str(archive["gender"].item())
        object_translation = np.asarray(archive["object_transl"], dtype=np.float64)
        object_quaternion = np.asarray(archive["object_quat_xyzw"], dtype=np.float64)
        mesh_paths = [str(value) for value in archive["object_mesh_paths"]]
        mesh_scales = np.asarray(archive["object_mesh_scale_to_m"], dtype=np.float64)
        source_contact = np.asarray(archive["source_hand_contact"], dtype=bool)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_smplx_model(
        config.smplx_model_dir, gender, len(betas), batch_size=args.batch_size, device=device,
    ).eval()
    weights = model.lbs_weights.detach().cpu().numpy()
    left_score = weights[:, [20, *range(25, 40)]].sum(axis=1)
    right_score = weights[:, [21, *range(40, 55)]].sum(axis=1)
    hand_mask = (left_score > 0.01) | (right_score > 0.01)
    nonhand_mask = ~hand_mask
    grids = []
    settings = config.raw["object_pilot"]
    for relative, mesh_scale in zip(mesh_paths, mesh_scales):
        mesh = load_repaired_mesh(resolve_asset(args.input, relative), float(mesh_scale))
        grids.append(build_signed_distance_grid(
            mesh, pitch=float(settings["sdf_pitch_m"]),
            padding=float(settings["sdf_padding_m"]),
            sample_count=int(settings["sdf_surface_samples"]),
            seed=int(settings["selection_seed"]),
        ))

    edges = np.linspace(0.0, 0.05, 501, dtype=np.float64)
    names = ("all", "contact_hand", "noncontact_hand", "nonhand")
    counts = {name: np.zeros(len(edges) - 1, dtype=np.int64) for name in names}
    maxima = {name: 0.0 for name in names}
    frames = len(global_orient)
    beta_tensor = torch.from_numpy(betas).to(device)[None]
    with torch.inference_mode():
        for start in range(0, frames, args.batch_size):
            end = min(frames, start + args.batch_size)
            count = end - start
            indices = np.arange(start, end)
            if count < args.batch_size:
                indices = np.r_[indices, np.full(args.batch_size - count, end - 1)]

            def tensor(value: np.ndarray) -> torch.Tensor:
                return torch.from_numpy(np.ascontiguousarray(value[indices])).to(device)

            zeros3 = torch.zeros((args.batch_size, 3), dtype=torch.float32, device=device)
            output = model(
                betas=beta_tensor.expand(args.batch_size, -1),
                global_orient=tensor(global_orient), body_pose=tensor(body_pose),
                left_hand_pose=tensor(left_hand_pose), right_hand_pose=tensor(right_hand_pose),
                jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                expression=torch.zeros((args.batch_size, model.num_expression_coeffs), device=device),
                return_verts=True,
            )
            vertices = output.vertices[:count].cpu().numpy() * scale + translation[start:end, None]
            for object_index, grid in enumerate(grids):
                rotation = Rotation.from_quat(object_quaternion[start:end, object_index]).as_matrix()
                local = np.einsum(
                    "tvj,tjk->tvk",
                    vertices - object_translation[start:end, object_index, None], rotation,
                )
                depth = np.maximum(0.0, -grid.sample(local.reshape(-1, 3))).reshape(count, -1)
                contact_frames = source_contact[start:end, object_index].any(axis=1)
                regions = {
                    "all": depth,
                    "contact_hand": depth[:, hand_mask][contact_frames],
                    "noncontact_hand": depth[:, hand_mask][~contact_frames],
                    "nonhand": depth[:, nonhand_mask],
                }
                for name, values in regions.items():
                    positive = values[values > 0.0]
                    if len(positive):
                        counts[name] += np.histogram(positive, bins=edges)[0]
                        maxima[name] = max(maxima[name], float(positive.max()))

    metrics = {
        name: {
            "positive_sample_count": int(counts[name].sum()),
            "penetration_p95_m": histogram_percentile(counts[name], edges, 95.0),
            "penetration_max_m": maxima[name],
        }
        for name in names
    }
    strict_p95 = float(settings["acceptance"]["penetration_p95_m"])
    maximum = float(settings["acceptance"]["penetration_max_m"])
    soft_p95 = 0.005
    strict, soft, noncontact_positive_fraction = classify_penetration(
        metrics, strict_p95, maximum, soft_p95,
    )
    report = {
        "input": str(args.input), "frames": frames,
        "surface": "canonical SMPL-X vertices vs actual-OBJ triangle-distance SDF",
        "strict_thresholds": {"p95_m": strict_p95, "max_m": maximum},
        "soft_tissue_thresholds": {
            "contact_hand_p95_m": soft_p95, "max_m": maximum,
            "nonhand_max_m": strict_p95,
            "noncontact_hand_p95_m": soft_p95,
            "noncontact_hand_positive_fraction_of_all": 0.01,
        },
        "noncontact_hand_positive_fraction_of_all": noncontact_positive_fraction,
        "metrics": metrics,
        "strict_geometric_pass": bool(strict),
        "soft_tissue_tolerance_pass": soft,
        "acceptance_mode": "strict_geometric" if strict else ("soft_tissue_tolerance" if soft else "failed"),
        "passed": bool(strict or soft),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=args.report.name, suffix=".partial", dir=args.report.parent)
    with os.fdopen(handle, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, args.report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
