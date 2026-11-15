from __future__ import annotations

import multiprocessing
import os
import subprocess
from functools import lru_cache
from collections.abc import Iterable


GPU_ENV = "MOTION_RETARGET_GPU_INDEX"


def parse_gpu_ids(value: str | Iterable[int] | None) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
        return [int(item) for item in values]
    return [int(item) for item in value]


def assigned_gpu_index(gpu_count: int) -> int:
    """Return the physical GPU assigned by the parent worker initializer."""
    configured = os.environ.get(GPU_ENV)
    if configured is not None:
        index = int(configured)
        if index < 0 or (gpu_count and index >= gpu_count):
            raise ValueError(f"{GPU_ENV}={index} is outside the {gpu_count} visible GPUs")
        return index
    return 0


@lru_cache(maxsize=8)
def physical_gpu_token(index: int) -> str:
    """Resolve an nvidia-smi physical index to a CUDA UUID visibility token."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        mapping = {}
        for line in output.splitlines():
            raw_index, uuid = (item.strip() for item in line.split(",", 1))
            mapping[int(raw_index)] = uuid
        if index in mapping:
            return mapping[index]
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return str(index)


@lru_cache(maxsize=1)
def physical_gpu_count() -> int:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        return sum(bool(line.strip()) for line in output.splitlines())
    except (OSError, subprocess.SubprocessError):
        return 0


def initialize_gpu_worker(gpu_ids: Iterable[int]) -> None:
    """Give each ProcessPool worker one stable physical GPU."""
    selected = parse_gpu_ids(gpu_ids)
    if not selected:
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    identity = multiprocessing.current_process()._identity
    worker_number = int(identity[-1]) - 1 if identity else 0
    os.environ[GPU_ENV] = str(selected[worker_number % len(selected)])
