#!/usr/bin/env python3
"""Run pinned WiLoR on one synchronized multi-view sequence and emit MANO tracks."""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation


def read_times(path: Path | None, count: int, fps: float) -> np.ndarray:
    if path is None:
        return np.arange(count, dtype=np.float64) / fps
    values = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            values.append(np.datetime64(row["timestamp"], "us").astype(np.int64) / 1e6)
    result = np.asarray(values[:count], dtype=np.float64)
    return result


def rotation_mean(rotations: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(rotations) == 1:
        return rotations[0]
    matrices = np.stack(rotations)
    return Rotation.from_matrix(matrices).mean(weights=np.asarray(weights)).as_matrix()


def reflect_left_mano_matrices(matrices: np.ndarray) -> np.ndarray:
    """Convert WiLoR's mirrored right-hand canonical rotations to left MANO."""
    reflection = np.diag([-1.0, 1.0, 1.0])
    values = np.asarray(matrices)
    return reflection @ values @ reflection


def normalize_keypoints_2d(keypoints: np.ndarray) -> np.ndarray:
    """Normalize WiLoR/OpenCV keypoints to an unambiguous ``[J, 2]`` view."""
    values = np.asarray(keypoints, dtype=np.float64)
    values = np.squeeze(values)
    if values.ndim != 2 or values.shape[-1] != 2:
        raise ValueError(f"Expected 2D keypoints shaped [J,2], got {values.shape}")
    return values


def load_calibration(intrinsic_path: Path, extrinsic_path: Path) -> dict:
    storage = cv2.FileStorage(str(intrinsic_path), cv2.FILE_STORAGE_READ)
    K = storage.getNode("K").mat()
    D = storage.getNode("D").mat()
    storage.release()
    payload = yaml.safe_load(extrinsic_path.read_text())["camera_extrinsics"]
    rotation_cw = np.asarray(payload["rotation_matrix"], dtype=np.float64)
    translation_cw = np.asarray(payload["translation_vector"], dtype=np.float64)
    rotation_wc = rotation_cw.T
    translation_wc = -rotation_wc @ translation_cw
    projection = K @ np.concatenate([rotation_wc, translation_wc[:, None]], axis=1)
    return {"K": K, "D": D, "P": projection}


def triangulate_and_score(
    rows: list[dict], calibrations: list[dict], fit_indices: list[int] | None = None,
) -> tuple[np.ndarray, list[float]]:
    joint_count = rows[0]["keypoints"].shape[0]
    points = np.zeros((joint_count, 3), dtype=np.float64)
    undistorted = []
    for row in rows:
        calibration = calibrations[row["view"]]
        image_points = normalize_keypoints_2d(row["keypoints"]).reshape(-1, 1, 2)
        value = cv2.undistortPoints(
            image_points, calibration["K"], calibration["D"], P=calibration["K"],
        ).reshape(-1, 2)
        undistorted.append(value)
    for joint in range(joint_count):
        equations = []
        indices = fit_indices if fit_indices is not None else list(range(len(rows)))
        for index in indices:
            row, point = rows[index], undistorted[index]
            projection = calibrations[row["view"]]["P"]
            equations.extend([point[joint, 0] * projection[2] - projection[0],
                              point[joint, 1] * projection[2] - projection[1]])
        _, _, vh = np.linalg.svd(np.asarray(equations))
        homogeneous = vh[-1]
        points[joint] = homogeneous[:3] / homogeneous[3]
    errors = []
    homogeneous = np.concatenate([points, np.ones((joint_count, 1))], axis=1)
    for row, observed in zip(rows, undistorted):
        projected = (calibrations[row["view"]]["P"] @ homogeneous.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        diagonal = float(np.linalg.norm(row["img_size"]))
        errors.append(float(np.mean(np.linalg.norm(projected - observed, axis=1)) / diagonal))
    return points, errors


def robust_triangulate_and_score(
    rows: list[dict], calibrations: list[dict], max_normalized_error: float,
) -> tuple[np.ndarray, list[float], list[int]]:
    """Fit every camera pair, retain the hypothesis supported by most views."""
    if len(rows) < 2:
        raise ValueError("Robust triangulation needs at least two camera views")
    candidates = []
    for pair in itertools.combinations(range(len(rows)), 2):
        points, errors = triangulate_and_score(rows, calibrations, list(pair))
        inliers = [index for index, error in enumerate(errors) if error <= max_normalized_error]
        inlier_error = float(np.median([errors[index] for index in inliers])) if inliers else np.inf
        candidates.append((-len(inliers), inlier_error, float(np.mean(errors)), pair, points, errors, inliers))
    _, _, _, pair, points, errors, inliers = min(candidates, key=lambda item: item[:3])
    if len(inliers) < 2:
        inliers = list(pair)
    if set(inliers) != set(pair):
        points, errors = triangulate_and_score(rows, calibrations, inliers)
        refined = [index for index, error in enumerate(errors) if error <= max_normalized_error]
        if len(refined) >= 2:
            inliers = refined
    return points, errors, inliers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--video", type=Path, action="append", required=True)
    parser.add_argument("--timestamps", type=Path, action="append", default=[])
    parser.add_argument("--intrinsic", type=Path, action="append", default=[])
    parser.add_argument("--extrinsic", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--detector-confidence", type=float, default=0.3)
    parser.add_argument("--max-normalized-reprojection-error", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if len(args.intrinsic) != len(args.extrinsic):
        raise ValueError("Each intrinsic file needs one extrinsic file")
    calibrations = [load_calibration(intrinsic, extrinsic)
                    for intrinsic, extrinsic in zip(args.intrinsic, args.extrinsic)]
    if calibrations and len(calibrations) != len(args.video):
        raise ValueError("Calibration count must equal video count")
    os.chdir(args.repo)
    sys.path.insert(0, str(args.repo))
    from ultralytics import YOLO
    from wilor.models import load_wilor
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to

    checkpoint = args.repo / "pretrained_models/wilor_final.ckpt"
    detector_path = args.repo / "pretrained_models/detector.pt"
    model, model_cfg = load_wilor(checkpoint_path=str(checkpoint), cfg_path=str(args.repo / "pretrained_models/model_config.yaml"))
    detector = YOLO(str(detector_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, detector = model.to(device).eval(), detector.to(device)

    observations: list[dict] = []
    all_absolute_times: list[np.ndarray] = []
    for view, video_path in enumerate(args.video):
        capture = cv2.VideoCapture(str(video_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        timestamp_path = args.timestamps[view] if view < len(args.timestamps) else None
        times = read_times(timestamp_path, count, fps)
        all_absolute_times.append(times)
        detections = detector.predict(
            source=str(video_path), conf=args.detector_confidence, verbose=False, stream=True,
        )
        pending_samples = []
        pending_metadata = []

        def run_batch(samples: list[dict], metadata_rows: list[tuple[int, int, float]]) -> None:
            if not samples:
                return
            batch = recursive_to(torch.utils.data.default_collate(samples), device)
            try:
                with torch.no_grad():
                    output = model(batch)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or len(samples) == 1:
                    raise
                del batch
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                middle = len(samples) // 2
                run_batch(samples[:middle], metadata_rows[:middle])
                run_batch(samples[middle:], metadata_rows[middle:])
                return
            for sample, metadata in enumerate(metadata_rows):
                frame, side, confidence = metadata
                matrices = output["pred_mano_params"]["hand_pose"][sample].detach().float().cpu().numpy()
                keypoints = normalize_keypoints_2d(
                    output["pred_keypoints_2d"][sample].detach().float().cpu().numpy(),
                )
                if side == 0:
                    keypoints[:, 0] *= -1
                keypoints = keypoints * float(batch["box_size"][sample].detach().cpu())
                keypoints += batch["box_center"][sample].detach().float().cpu().numpy()
                observations.append({
                    "view": view, "frame": frame, "time": float(times[frame]),
                    "side": side, "confidence": confidence, "pose": matrices,
                    "keypoints": keypoints,
                    "img_size": batch["img_size"][sample].detach().cpu().numpy(),
                })

        def flush_pending() -> None:
            if not pending_samples:
                return
            run_batch(pending_samples, pending_metadata)
            pending_samples.clear()
            pending_metadata.clear()

        for frame, detection in enumerate(detections):
            if frame >= len(times):
                break
            image = detection.orig_img
            candidates = []
            for box in detection.boxes:
                data = box.data.detach().cpu().numpy().reshape(-1)
                candidates.append((data[:4], int(data[5]), float(data[4])))
            selected = []
            for side in (0, 1):
                matches = [item for item in candidates if item[1] == side]
                if not matches:
                    continue
                bbox, _, confidence = max(matches, key=lambda item: item[2])
                selected.append((side, bbox, confidence))
            if not selected:
                continue
            dataset = ViTDetDataset(
                model_cfg, image, np.stack([item[1] for item in selected]),
                np.asarray([item[0] for item in selected]), rescale_factor=2.0, fp16=False,
            )
            for sample, (side, _, confidence) in enumerate(selected):
                pending_samples.append(dataset[sample])
                pending_metadata.append((frame, side, confidence))
                if len(pending_samples) >= args.batch_size:
                    flush_pending()
        flush_pending()

    if not all_absolute_times:
        raise ValueError("No videos")
    origin = min(times[0] for times in all_absolute_times if len(times))
    for observation in observations:
        observation["time"] -= origin
    relative_times = [times - origin for times in all_absolute_times]
    if args.timestamps:
        target_times = relative_times[0]
    else:
        fps = float(cv2.VideoCapture(str(args.video[0])).get(cv2.CAP_PROP_FPS))
        target_times = np.arange(len(relative_times[0]), dtype=np.float64) / fps

    poses = {0: np.zeros((len(target_times), 45), dtype=np.float32),
             1: np.zeros((len(target_times), 45), dtype=np.float32)}
    valid = {0: np.zeros(len(target_times), dtype=bool), 1: np.zeros(len(target_times), dtype=bool)}
    confidence = {0: np.zeros(len(target_times), dtype=np.float32), 1: np.zeros(len(target_times), dtype=np.float32)}
    reprojection = {0: np.full(len(target_times), np.inf, dtype=np.float32),
                    1: np.full(len(target_times), np.inf, dtype=np.float32)}
    inlier_view_counts: list[int] = []
    for side in (0, 1):
        side_rows = [row for row in observations if row["side"] == side]
        by_view = {}
        for view in range(len(args.video)):
            rows = sorted((row for row in side_rows if row["view"] == view), key=lambda row: row["time"])
            by_view[view] = (np.asarray([row["time"] for row in rows]), rows)
        for frame, time_value in enumerate(target_times):
            per_view = []
            for view in range(len(args.video)):
                times, rows = by_view[view]
                if not rows:
                    continue
                insertion = int(np.searchsorted(times, time_value))
                choices = [max(0, min(len(rows) - 1, insertion + offset)) for offset in (-1, 0)]
                nearest = min((rows[index] for index in choices), key=lambda row: abs(row["time"] - time_value))
                tolerance = 0.75 * np.median(np.diff(relative_times[view])) if len(relative_times[view]) > 1 else 0.04
                if abs(nearest["time"] - time_value) <= tolerance:
                    per_view.append(nearest)
            if not per_view:
                continue
            view_errors = [0.0] * len(per_view)
            if calibrations and len(per_view) >= 2:
                _, view_errors, inlier_indices = robust_triangulate_and_score(
                    per_view, calibrations, args.max_normalized_reprojection_error,
                )
                per_view = [per_view[index] for index in inlier_indices]
                view_errors = [view_errors[index] for index in inlier_indices]
                inlier_view_counts.append(len(inlier_indices))
            fused = np.zeros((15, 3), dtype=np.float32)
            weights = [row["confidence"] / max(0.005, error) for row, error in zip(per_view, view_errors)]
            for joint in range(15):
                matrix = rotation_mean([row["pose"][joint] for row in per_view], weights)
                if side == 0:
                    matrix = reflect_left_mano_matrices(matrix)
                fused[joint] = Rotation.from_matrix(matrix).as_rotvec()
            poses[side][frame] = fused.reshape(45)
            confidence[side][frame] = max(row["confidence"] for row in per_view)
            if calibrations and len(per_view) >= 2:
                reprojection[side][frame] = float(np.mean(view_errors))
            valid[side][frame] = True

    metadata = {
        "model": "WiLoR", "views": len(args.video), "inference_batch_size": args.batch_size,
        "fusion": "confidence_weighted_SO3_mean",
        "left_hand_conversion": "M_x R M_x from WiLoR mirrored right-hand canonical frame",
        "reprojection_available": bool(calibrations),
        "triangulation": "camera-pair hypotheses ranked by inlier-view support, followed by inlier DLT refinement",
        "inlier_view_count_median": (float(np.median(inlier_view_counts)) if inlier_view_counts else None),
        "inlier_view_count_min": (int(np.min(inlier_view_counts)) if inlier_view_counts else None),
        "note": "Calibrated multi-view frames use robust DLT hand-joint triangulation, normalized reprojection gating, and confidence/reprojection-weighted SO(3) pose fusion.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=args.output.name, suffix=".partial", dir=args.output.parent)
    os.close(handle)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(
                stream, timestamps=target_times,
                left_hand_pose=poses[0], right_hand_pose=poses[1],
                left_valid=valid[0], right_valid=valid[1],
                left_confidence=confidence[0], right_confidence=confidence[1],
                left_reprojection_error=reprojection[0],
                right_reprojection_error=reprojection[1],
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary_path, args.output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output), "frames": len(target_times), **metadata}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
