#!/usr/bin/env python3
"""Render fixed-scale, root-centred 3D snapshots for the A/B/E/H report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


CHAINS = [
    ["pelvis", "spine1", "spine2", "spine3", "neck", "head"],
    ["pelvis", "left_hip", "left_knee", "left_ankle", "left_foot"],
    ["pelvis", "right_hip", "right_knee", "right_ankle", "right_foot"],
    ["neck", "left_shoulder", "left_elbow", "left_wrist"],
    ["neck", "right_shoulder", "right_elbow", "right_wrist"],
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_frame(motion: dict[str, np.ndarray], scene: dict) -> tuple[int, str]:
    fps = float(motion["fps"])
    n_frames = len(motion["joints"])
    constraints = scene.get("constraints", [])
    if constraints:
        index = min(round(float(constraints[0]["time_seconds"]) * fps), n_frames - 1)
        return index, "constraint_time"
    if "chunk_boundaries" in motion and len(motion["chunk_boundaries"]):
        boundaries = motion["chunk_boundaries"]
        return int(boundaries[len(boundaries) // 2]), "middle_chunk_boundary"
    return n_frames // 2, "sequence_midpoint"


def _render_case(repo: Path, case: dict, view: dict, output_dir: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = repo / case["run_dir"]
    motion_path = run_dir / "motion.npz"
    scene_path = run_dir / "scenario.json"
    metrics_path = run_dir / "metrics.json"
    with np.load(motion_path, allow_pickle=False) as loaded:
        motion = {key: loaded[key] for key in loaded.files}
    scene = json.loads(scene_path.read_text())
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    frame_index, frame_reason = _select_frame(motion, scene)
    fps = float(motion["fps"])
    names = [str(name) for name in motion["joint_names"]]
    index = {name: i for i, name in enumerate(names)}
    xyz = np.asarray(motion["joints"], dtype=float)
    frame = xyz[frame_index].copy()
    pelvis_xy = frame[index.get("pelvis", 0), :2].copy()
    frame[:, :2] -= pelvis_xy

    fig = plt.figure(figsize=(5.2, 5.2), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*frame.T, s=9, color="#2f6faa", alpha=0.72)
    colors = ["#3157d5", "#4d6d42", "#c85218", "#8661c1", "#d44a42"]
    for chain, color in zip(CHAINS, colors, strict=True):
        ids = [index[name] for name in chain if name in index]
        if len(ids) > 1:
            ax.plot(*frame[ids].T, color=color, linewidth=2.4, marker="o", markersize=2.8)

    if "object_positions" in motion:
        obj = np.asarray(motion["object_positions"][frame_index], dtype=float).copy()
        obj[:2] -= pelvis_xy
        ax.scatter(*obj, color="#111827", marker="s", s=54, label="object")

    for constraint in scene.get("constraints", []):
        if "position_m" not in constraint:
            continue
        target = np.asarray(constraint["position_m"], dtype=float).copy()
        target[:2] -= pelvis_xy
        ax.scatter(*target, color="#c7392f", marker="x", s=90, linewidths=2.2)
        ax.text(*target, f" {constraint['joint']} target", fontsize=7, color="#8f2d27")
        if "rotation_matrix" in constraint:
            rotation = np.asarray(constraint["rotation_matrix"], dtype=float)
            for axis, color in enumerate(("#c7392f", "#41805a", "#3157d5")):
                tip = target + 0.16 * rotation[:, axis]
                ax.plot(*np.stack([target, tip]).T, color=color, linestyle="--", linewidth=1.4)

    ax.set(
        xlim=tuple(view["xlim_m"]), ylim=tuple(view["ylim_m"]), zlim=tuple(view["zlim_m"]),
        xlabel="X from pelvis (m)", ylabel="Y from pelvis (m)", zlabel="Z (m)",
    )
    ax.set_box_aspect((
        view["xlim_m"][1] - view["xlim_m"][0],
        view["ylim_m"][1] - view["ylim_m"][0],
        view["zlim_m"][1] - view["zlim_m"][0],
    ))
    ax.view_init(elev=view["elevation_deg"], azim=view["azimuth_deg"])
    ax.set_title(
        f"{case['baseline'].upper()} · {case['scenario']} · seed {case['seed']} · t={frame_index / fps:.2f}s\n"
        "Fixed axes: X/Y ±1.4 m, Z 0–2.2 m; pelvis-centred in X/Y",
        fontsize=9,
    )
    ax.tick_params(labelsize=7)
    fig.patch.set_facecolor("white")
    output_path = output_dir / f"{case['id']}.png"
    fig.savefig(output_path, dpi=145, facecolor="white")
    plt.close(fig)

    return {
        **case,
        "motion": str(motion_path.relative_to(repo)),
        "motion_sha256": _sha256(motion_path),
        "original_video": str((run_dir / "motion.mp4").relative_to(repo)),
        "fixed_snapshot": str(output_path.relative_to(repo)),
        "frame_index": frame_index,
        "frame_seconds": frame_index / fps,
        "frame_reason": frame_reason,
        "n_frames": len(xyz),
        "fps": fps,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/abeh_report_cases.json"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("develop_logs/test_runs/20261007_abeh_fixed_scale_gallery"),
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    config_path = args.config if args.config.is_absolute() else repo / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    rendered = [_render_case(repo, case, config["fixed_view"], output_dir) for case in config["cases"]]
    manifest = {
        "schema_version": 1,
        "source_config": str(config_path.relative_to(repo)),
        "fixed_view": config["fixed_view"],
        "case_count": len(rendered),
        "cases": rendered,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": "success", "output_dir": str(output_dir), "case_count": len(rendered)}))


if __name__ == "__main__":
    main()
