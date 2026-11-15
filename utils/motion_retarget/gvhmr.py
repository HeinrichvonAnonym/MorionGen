from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .config import RetargetConfig
from .gpu import assigned_gpu_index, physical_gpu_count, physical_gpu_token


def gvhmr_runtime_assets(repo: Path) -> list[Path]:
    return [
        repo / "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt",
        repo / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt",
        repo / "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth",
        repo / "inputs/checkpoints/yolo/yolov8x.pt",
    ]


def run_gvhmr(config: RetargetConfig, video: Path, output_dir: Path, static_camera: bool = True) -> Path:
    """Invoke pinned official preprocessing/inference in its isolated environment."""
    python = Path(config.raw["environments"]["gvhmr_python"])
    repo = Path(config.raw["paths"]["gvhmr_repo"])
    worker = Path(__file__).resolve().parents[2] / "python/scripts/extract_gvhmr_body.py"
    missing = [path for path in (
        python, *gvhmr_runtime_assets(repo), repo / "tools/demo/demo.py", worker,
    ) if not path.exists()]
    if missing:
        raise FileNotFoundError("GVHMR runtime is incomplete: " + ", ".join(map(str, missing)))
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), str(worker), "--repo", str(repo), "--video", str(video),
        "--output-root", str(output_dir),
    ]
    if static_camera:
        command.append("--static-cam")
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
            command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=environment,
        )
    if result.returncode:
        raise RuntimeError(f"GVHMR failed ({result.returncode}):\n{result.stdout[-8000:]}")
    candidates = list(output_dir.rglob("hmr4d_results.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one GVHMR result, found {len(candidates)} in {output_dir}")
    return candidates[0]


def ensure_gvhmr_cache(config: RetargetConfig, reference, output: Path) -> dict:
    """Run official GVHMR in its isolated environment and atomically cache its result."""
    if output.is_file():
        return {"status": "cached", "path": str(output)}
    repo = Path(config.raw["paths"]["gvhmr_repo"])
    required = [Path(config.raw["environments"]["gvhmr_python"]),
                *gvhmr_runtime_assets(repo), repo / "tools/demo/demo.py"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return {"status": "assets_missing", "missing": missing}
    scratch_root = config.data_root / config.raw["batch"]["scratch_subdir"] / "motion_retarget"
    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f"gvhmr_{reference.dataset}_{reference.sequence_id}_", dir=scratch_root))
    partial = output.with_suffix(output.suffix + ".partial")
    try:
        from .video import stage_videos
        videos, _, _ = stage_videos(config, reference, temporary / "input")
        result_path = run_gvhmr(config, videos[0], temporary / "output", static_camera=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result_path, partial)
        with partial.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(partial, output)
        return {"status": "created", "path": str(output), "video": str(videos[0]), "static_camera": True}
    except Exception as exc:
        partial.unlink(missing_ok=True)
        return {"status": "worker_failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def load_gvhmr_result(path: Path) -> dict[str, np.ndarray]:
    import torch
    data = torch.load(path, map_location="cpu", weights_only=False)
    world = data.get("smpl_params_global") or data.get("smpl_params_incam")
    if world is None:
        raise KeyError(f"No SMPL parameters in {path}")
    result = {key: value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
              for key, value in world.items()}
    result["provenance_json"] = np.asarray(json.dumps({"source": str(path), "model": "GVHMR"}))
    return result
