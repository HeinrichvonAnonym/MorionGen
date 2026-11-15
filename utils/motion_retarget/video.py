from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

from .config import RetargetConfig
from .schema import SequenceRef
from .gpu import assigned_gpu_index, physical_gpu_count, physical_gpu_token


def _extract_members(archive: Path, members: list[str], target: Path) -> list[Path]:
    outputs: list[Path] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in members:
            output = target / Path(member).name
            with bundle.open(member) as source, output.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=8 << 20)
            outputs.append(output)
    return outputs


@lru_cache(maxsize=512)
def _find_comfi_video_archive(root_value: str, subject: str, task: str) -> Path:
    root = Path(root_value)
    for index in range(1, 7):
        candidate = root / f"data/raw/videos{index}.zip"
        prefix = f"videos{index}/{subject}/{task}/"
        wanted = [prefix + f"camera_{camera}.mp4" for camera in (0, 2, 4, 6)]
        with zipfile.ZipFile(candidate) as bundle:
            members = set(bundle.namelist())
        if all(item in members for item in wanted):
            return candidate
    raise FileNotFoundError(f"Four calibrated COMFI views not found for {subject}/{task}")


def find_comfi_video_archive(config: RetargetConfig, reference: SequenceRef) -> Path:
    return _find_comfi_video_archive(
        str(config.dataset_path("comfi")),
        str(reference.metadata["subject"]), str(reference.metadata["task"]),
    )


def stage_videos(config: RetargetConfig, reference: SequenceRef, target: Path) -> tuple[list[Path], list[Path], list[tuple[Path, Path]]]:
    target.mkdir(parents=True, exist_ok=True)
    videos: list[Path] = []
    timestamps: list[Path] = []
    calibrations: list[tuple[Path, Path]] = []
    if reference.dataset == "andy":
        participant = f"Participant_{reference.metadata['participant']}"
        for camera in ("camera1", "camera2"):
            archive = config.dataset_path("andy") / f"data/raw/videos_{camera}_mp4.zip"
            member = f"videos_{camera}_mp4/{participant}/{reference.sequence_id}.{camera}.mp4"
            videos.extend(_extract_members(archive, [member], target))
    elif reference.dataset == "comfi":
        subject, task = reference.metadata["subject"], reference.metadata["task"]
        archive = find_comfi_video_archive(config, reference)
        index = int(archive.stem.removeprefix("videos"))
        prefix = f"videos{index}/{subject}/{task}/"
        video_members = [prefix + f"camera_{camera}.mp4" for camera in (0, 2, 4, 6)]
        timestamp_members = [prefix + f"camera_{camera}_timestamps.csv" for camera in (0, 2, 4, 6)]
        videos.extend(_extract_members(archive, video_members, target))
        timestamps.extend(_extract_members(archive, timestamp_members, target))
        calibration_archive = config.dataset_path("comfi") / "data/raw/cam_params.zip"
        intrinsic_members = [f"cam_params/{subject}/intrinsics/camera_{camera}_intrinsics.yaml" for camera in (0, 2, 4, 6)]
        extrinsic_members = [f"cam_params/{subject}/extrinsics/cam_to_world/camera_{camera}/camera_{camera}_extrinsics.yaml"
                             for camera in (0, 2, 4, 6)]
        intrinsic_paths = _extract_members(calibration_archive, intrinsic_members, target)
        extrinsic_paths = _extract_members(calibration_archive, extrinsic_members, target)
        calibrations = list(zip(intrinsic_paths, extrinsic_paths))
    else:
        raise ValueError(f"No video hand path for {reference.dataset}")
    return videos, timestamps, calibrations


def ensure_wilor_cache(config: RetargetConfig, reference: SequenceRef, output: Path) -> dict:
    if output.is_file():
        return {"status": "cached", "path": str(output)}
    python = Path(config.raw["environments"]["wilor_python"])
    repo = Path(config.raw["paths"]["wilor_repo"])
    if not python.is_file():
        return {"status": "environment_missing", "path": str(python)}
    required = [repo / "pretrained_models/wilor_final.ckpt", repo / "pretrained_models/detector.pt",
                repo / "mano_data/MANO_RIGHT.pkl"]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        return {"status": "assets_missing", "missing": missing}
    scratch_root = config.data_root / config.raw["batch"]["scratch_subdir"] / "motion_retarget"
    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"wilor_{reference.dataset}_{reference.sequence_id}_", dir=scratch_root))
    try:
        videos, timestamps, calibrations = stage_videos(config, reference, temporary)
        worker = Path(__file__).resolve().parents[2] / "python/scripts/extract_wilor_mano.py"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [str(python), str(worker), "--repo", str(repo), "--output", str(output)]
        command += ["--batch-size", str(int(config.raw["hands"]["inference_batch_size"]))]
        command += ["--detector-confidence", str(
            float(config.raw["hands"]["min_detection_confidence"]),
        )]
        command += ["--max-normalized-reprojection-error", str(
            float(config.raw["hands"]["max_normalized_reprojection_error"]),
        )]
        for video in videos:
            command += ["--video", str(video)]
        for timestamp in timestamps:
            command += ["--timestamps", str(timestamp)]
        for intrinsic, extrinsic in calibrations:
            command += ["--intrinsic", str(intrinsic), "--extrinsic", str(extrinsic)]
        environment = dict(os.environ)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        from .fitting import _device_lock
        gpu_count = physical_gpu_count()
        gpu_index = assigned_gpu_index(gpu_count)
        if gpu_count:
            environment["CUDA_VISIBLE_DEVICES"] = physical_gpu_token(gpu_index)
        with _device_lock(config, gpu_count > 0, gpu_index):
            result = subprocess.run(
                command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
            )
        if result.returncode:
            return {"status": "worker_failed", "returncode": result.returncode, "log_tail": result.stdout[-8000:]}
        return {"status": "created", "path": str(output), "views": len(videos)}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
