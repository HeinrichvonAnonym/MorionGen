from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .config import RetargetConfig
from .schema import SequenceRef


STAGE_SENTINEL = ".motion_retarget_stage.json"


def cleanup_stale_hiphi_stages(config: RetargetConfig, minimum_age_seconds: float = 21600.0) -> list[Path]:
    """Remove only old, sentinel-owned stages whose creating process is gone."""
    removed: list[Path] = []
    now = time.time()
    base = config.data_root / config.raw["batch"]["scratch_subdir"] / "motion_retarget"
    roots = [base / "hiphi", base / "hiphi_object"]
    for candidate in (item for root in roots if root.is_dir() for item in root.iterdir()):
        sentinel = candidate / STAGE_SENTINEL
        if not candidate.is_dir() or not sentinel.is_file():
            continue
        try:
            payload = json.loads(sentinel.read_text())
            pid = int(payload["pid"])
            created_at = float(payload["created_at"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            continue
        if now - created_at < minimum_age_seconds:
            continue
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        if alive:
            continue
        shutil.rmtree(candidate)
        removed.append(candidate)
    return removed


@contextmanager
def stage_hiphi_archive(
    config: RetargetConfig, references: list[SequenceRef],
) -> Iterator[list[SequenceRef]]:
    """Decompress selected members once, never the complete HiPHI archive."""
    if not references:
        yield []
        return
    archives = {reference.source for reference in references}
    if len(archives) != 1:
        raise ValueError("stage_hiphi_archive accepts one archive at a time")
    archive = next(iter(archives))
    scratch_root = config.data_root / config.raw["batch"]["scratch_subdir"] / "motion_retarget/hiphi"
    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=archive.stem + "_", dir=scratch_root))
    (temporary / STAGE_SENTINEL).write_text(json.dumps({
        "pid": os.getpid(), "created_at": time.time(), "archive": str(archive.resolve()),
    }, sort_keys=True))
    try:
        members: list[str] = []
        for reference in references:
            if reference.member is None:
                raise ValueError("Expected archive member")
            members.append(reference.member)
            members.append(str(reference.member).rsplit("/", 1)[0] + "/metadata.json")
        command = ["tar", "--zstd", "-xf", str(archive), "-C", str(temporary), "--", *members]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(f"Selective HiPHI staging failed for {archive}: {result.stderr[-4000:]}")
        staged = []
        for reference in references:
            bvh = temporary / str(reference.member)
            metadata_path = bvh.parent / "metadata.json"
            if not bvh.is_file() or not metadata_path.is_file():
                raise FileNotFoundError(f"Staged HiPHI pair is incomplete: {bvh}")
            metadata = dict(reference.metadata)
            metadata.update({
                "origin_archive": str(archive), "origin_member": str(reference.member),
                "metadata_path": str(metadata_path),
            })
            staged.append(SequenceRef(reference.dataset, reference.sequence_id, bvh, None, metadata))
        yield staged
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@contextmanager
def stage_hiphi_object_archive(
    config: RetargetConfig, references: list[SequenceRef],
) -> Iterator[list[SequenceRef]]:
    """Stage complete selected motion directories (BVH, metadata, object CSVs only)."""
    if not references:
        yield []
        return
    archives = {reference.source for reference in references}
    if len(archives) != 1:
        raise ValueError("stage_hiphi_object_archive accepts one archive at a time")
    archive = next(iter(archives))
    scratch_root = config.data_root / config.raw["batch"]["scratch_subdir"] / "motion_retarget/hiphi_object"
    scratch_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=archive.stem + "_", dir=scratch_root))
    (temporary / STAGE_SENTINEL).write_text(json.dumps({
        "pid": os.getpid(), "created_at": time.time(), "archive": str(archive.resolve()),
        "purpose": "hiphi_object_pilot",
    }, sort_keys=True))
    try:
        patterns = []
        for reference in references:
            if reference.member is None:
                raise ValueError("Expected archive member")
            directory = str(reference.member).rsplit("/", 1)[0]
            patterns.extend([
                directory + "/metadata.json", directory + "/motion_actor.bvh",
                directory + "/object_tracks/*.csv",
            ])
        command = [
            "tar", "--zstd", "--wildcards", "--no-wildcards-match-slash",
            "-xf", str(archive), "-C", str(temporary), "--", *patterns,
        ]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError(f"Selective HiPHI object staging failed for {archive}: {result.stderr[-4000:]}")
        staged = []
        for reference in references:
            bvh = temporary / str(reference.member)
            metadata_path = bvh.parent / "metadata.json"
            tracks = bvh.parent / "object_tracks"
            if not bvh.is_file() or not metadata_path.is_file() or not tracks.is_dir():
                raise FileNotFoundError(f"Staged HiPHI object sequence is incomplete: {bvh}")
            metadata = dict(reference.metadata)
            metadata.update({
                "origin_archive": str(archive), "origin_member": str(reference.member),
                "metadata_path": str(metadata_path), "object_stage_dir": str(bvh.parent),
            })
            staged.append(SequenceRef(reference.dataset, reference.sequence_id, bvh, None, metadata))
        yield staged
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def group_hiphi_by_archive(references: list[SequenceRef]) -> list[list[SequenceRef]]:
    groups: dict[Path, list[SequenceRef]] = defaultdict(list)
    for reference in references:
        groups[reference.source].append(reference)
    return [groups[path] for path in sorted(groups)]
