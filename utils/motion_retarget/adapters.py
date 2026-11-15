from __future__ import annotations

import csv
import io
import json
import re
import subprocess
import zipfile
from functools import lru_cache
import concurrent.futures
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from .bvh import BVHMotion, parse_bvh
from .config import RetargetConfig
from .joints import bvh_to_smplx_pose
from .schema import SequenceRef


_ANDY_BVH_RE = re.compile(r"xsens_bvh/(?P<participant>Participant_\d+)/(?P<name>.+)\.xsens\.bvh$")
_COMFI_ANGLE_RE = re.compile(r"mocap/aligned/(?P<subject>\d+)/(?P<task>[^/]+)/joint_angles\.csv$")
_HIPHI_BVH_RE = re.compile(r"HiPHI/data/(?P<frame>[^/]+)/(?P<lu>[^/]+)/(?P<id>[^/]+)/motion_actor\.bvh$")


def _published_smplx_metadata(*rows: dict[str, str]) -> dict | None:
    """Read optional future release columns without inferring body shape."""
    merged = {key: value for row in rows for key, value in row.items() if value not in {None, ""}}
    raw_betas = merged.get("smplx_betas")
    if raw_betas is None:
        return None
    try:
        betas = json.loads(raw_betas)
        gender = merged["smplx_gender"]
        height_m = float(merged["smplx_height_m"])
        betas = [float(value) for value in betas]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if len(betas) not in {10, 16}:
        return None
    return {"gender": gender, "height_m": height_m, "betas": betas}


def inventory_andy(config: RetargetConfig) -> list[SequenceRef]:
    archive = config.dataset_path("andy") / "data/raw/xsens_bvh.zip"
    specifications = config.dataset_path("andy") / "data/raw/participant_specifications.csv"
    with specifications.open(newline="") as stream:
        participants = {row["ID"]: row for row in csv.DictReader(stream)}
    rows: list[SequenceRef] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            match = _ANDY_BVH_RE.fullmatch(member)
            if not match:
                continue
            participant = match.group("participant").split("_")[-1]
            details = participants.get(participant, {})
            metadata = {
                "participant": participant, "actor_id": participant,
                "gender": details.get("gender"),
                "height_m": float(details["body_height"]) / 100.0 if details.get("body_height") else None,
                "setup": details.get("setup"),
                "profile_metadata_source": "participant_specifications.csv",
            }
            published = _published_smplx_metadata(details)
            if published is not None:
                metadata["smplx"] = published
            rows.append(SequenceRef("andy", match.group("name"), archive, member, metadata))
    return sorted(rows, key=lambda row: row.sequence_id)


def inventory_comfi(config: RetargetConfig) -> list[SequenceRef]:
    archive = config.dataset_path("comfi") / "data/raw/mocap.zip"
    rows: list[SequenceRef] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            match = _COMFI_ANGLE_RE.fullmatch(member)
            if not match:
                continue
            sequence_id = f"{match.group('subject')}_{match.group('task')}"
            rows.append(SequenceRef("comfi", sequence_id, archive, member,
                                    {"subject": match.group("subject"), "task": match.group("task")}))
    return sorted(rows, key=lambda row: row.sequence_id)


def _list_tar_members(archive: Path) -> Iterable[str]:
    process = subprocess.Popen(
        ["tar", "--zstd", "-tf", str(archive)], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        yield line.rstrip("\n")
    stderr = process.stderr.read() if process.stderr else ""
    code = process.wait()
    if code:
        raise RuntimeError(f"Cannot list {archive}: {stderr.strip()}")


def inventory_hiphi(
    config: RetargetConfig, allow_incomplete: bool = False,
    sequence_ids: set[str] | None = None,
) -> list[SequenceRef]:
    root = config.dataset_path("hiphi")
    if config.raw["datasets"]["hiphi"].get("require_download_verified", True):
        if not (root / ".download_verified").is_file() and not allow_incomplete:
            raise RuntimeError(f"HiPHI download is not verified: {root / '.download_verified'}")
    rows: list[SequenceRef] = []
    archives = [archive for archive in sorted((root / "data").glob("HiPHI_data_part_*.tar.zst"))
                if not archive.with_name(archive.name + ".aria2").exists()]

    # Current HiPHI releases ship an authoritative motion-to-shard table and
    # a metadata table containing the logical frame/LU path.  Reading those
    # indexes avoids decompressing all 32 multi-gigabyte tar directory streams
    # merely to discover member names.  Retain the tar scan below for older or
    # partially mirrored releases that do not include the indexes.
    part_index = root / "data/motion_to_part.csv"
    metadata_index = root / "metadata/hiphi_metadata.csv"
    if part_index.is_file() and metadata_index.is_file():
        archive_by_name = {archive.name: archive for archive in archives}
        metadata: dict[str, dict[str, str]] = {}
        actor_index = root / "metadata/hiphi_actor_metadata.csv"
        actors: dict[str, dict[str, str]] = {}
        if actor_index.is_file():
            with actor_index.open(newline="") as stream:
                actors = {row["actor_id"]: row for row in csv.DictReader(stream)}
        with metadata_index.open(newline="") as stream:
            for row in csv.DictReader(stream):
                motion_id = row.get("motion_id", "")
                if motion_id and row.get("mirrored", "false").lower() != "true" and not motion_id.endswith("__mirror"):
                    metadata[motion_id] = row
        with part_index.open(newline="") as stream:
            for row in csv.DictReader(stream):
                motion_id = row.get("motion_id", "")
                if not motion_id or (sequence_ids is not None and motion_id not in sequence_ids):
                    continue
                archive = archive_by_name.get(row.get("archive_name", ""))
                details = metadata.get(motion_id)
                if archive is None or details is None:
                    continue
                frame, lu = details.get("frame", ""), details.get("lu", "")
                if not frame or not lu:
                    continue
                actor_id = details.get("actor_id", "")
                actor = actors.get(actor_id, {})
                member = f"HiPHI/data/{frame}/{lu}/{motion_id}/motion_actor.bvh"
                reference_metadata = {
                    "frame": frame, "lu": lu, "actor_id": actor_id,
                    "gender": actor.get("gender"),
                    "height_m": float(actor["height_cm"]) / 100.0 if actor.get("height_cm") else None,
                    "frame_count": int(details["frame_count"]) if details.get("frame_count") else None,
                    "duration_sec": float(details["duration_sec"]) if details.get("duration_sec") else None,
                    "is_hoi": details.get("is_hoi", "").lower() == "true",
                    "object_categories": details.get("object_categories", ""),
                    "profile_metadata_source": "hiphi_actor_metadata.csv",
                }
                published = _published_smplx_metadata(actor, details)
                if published is not None:
                    reference_metadata["smplx"] = published
                rows.append(SequenceRef(
                    "hiphi", motion_id, archive, member,
                    reference_metadata,
                ))
        if sequence_ids is not None:
            missing = sequence_ids - {row.sequence_id for row in rows}
            if missing:
                raise RuntimeError(f"HiPHI metadata index is missing requested motions: {sorted(missing)}")
        return sorted(rows, key=lambda row: row.sequence_id)

    def index_archive(archive: Path) -> list[SequenceRef]:
        cache_root = config.data_root / config.raw["batch"]["cache_subdir"] / "motion_retarget/inventory"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache = cache_root / f"{archive.name}.json"
        stat = archive.stat()
        entries = None
        if cache.is_file():
            try:
                payload = json.loads(cache.read_text())
                if payload.get("size") == stat.st_size and payload.get("mtime_ns") == stat.st_mtime_ns:
                    entries = payload["entries"]
            except (OSError, ValueError, KeyError):
                entries = None
        if entries is None:
            entries = []
            for member in _list_tar_members(archive):
                match = _HIPHI_BVH_RE.fullmatch(member)
                if not match or match.group("id").endswith("__mirror"):
                    continue
                entries.append({"member": member, "id": match.group("id"),
                                "frame": match.group("frame"), "lu": match.group("lu")})
            temporary = cache.with_suffix(cache.suffix + ".partial")
            temporary.write_text(json.dumps({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                             "entries": entries}, sort_keys=True))
            temporary.replace(cache)
        return [SequenceRef("hiphi", entry["id"], archive, entry["member"],
                            {"frame": entry["frame"], "lu": entry["lu"]}) for entry in entries]

    if sequence_ids:
        # Exact-sequence pilots should not decompress every multi-gigabyte
        # shard.  Consume valid index caches first (archives are stable and
        # sorted), and stop as soon as every requested id has been found.
        remaining = set(sequence_ids)
        for archive in archives:
            archive_rows = index_archive(archive)
            matches = [row for row in archive_rows if row.sequence_id in remaining]
            rows.extend(matches)
            remaining -= {row.sequence_id for row in matches}
            if not remaining:
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(archives) or 1)) as executor:
            indexed = list(executor.map(index_archive, archives))
        for archive_rows in indexed:
            rows.extend(archive_rows)
    return sorted(rows, key=lambda row: row.sequence_id)


def inventory(
    config: RetargetConfig, dataset: str, allow_incomplete: bool = False,
    sequence_ids: set[str] | None = None,
) -> list[SequenceRef]:
    if dataset == "andy":
        rows = inventory_andy(config)
        return rows if sequence_ids is None else [row for row in rows if row.sequence_id in sequence_ids]
    if dataset == "comfi":
        rows = inventory_comfi(config)
        return rows if sequence_ids is None else [row for row in rows if row.sequence_id in sequence_ids]
    if dataset == "hiphi":
        return inventory_hiphi(
            config, allow_incomplete=allow_incomplete, sequence_ids=sequence_ids,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


@lru_cache(maxsize=2)
def _load_bvh_cached(source: str, member: str) -> BVHMotion:
    source_path = Path(source)
    if source_path.suffix == ".zip":
        with zipfile.ZipFile(source_path) as bundle:
            payload = bundle.read(member)
    else:
        result = subprocess.run(
            ["tar", "--zstd", "-xOf", str(source_path), member],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        payload = result.stdout
    return parse_bvh(payload)


def load_bvh_reference(reference: SequenceRef) -> BVHMotion:
    if reference.member is None:
        return parse_bvh(reference.source.read_bytes())
    return _load_bvh_cached(str(reference.source), str(reference.member))


@lru_cache(maxsize=32)
def _load_tar_json(source: str, member: str) -> dict:
    result = subprocess.run(
        ["tar", "--zstd", "-xOf", source, member],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return json.loads(result.stdout)


def load_hiphi_metadata(reference: SequenceRef) -> dict:
    if reference.member is None and reference.metadata.get("metadata_path"):
        return json.loads(Path(str(reference.metadata["metadata_path"])).read_text())
    member = str(reference.member).rsplit("/", 1)[0] + "/metadata.json"
    return _load_tar_json(str(reference.source), member)


def load_andy_heights(config: RetargetConfig) -> dict[str, float]:
    path = config.dataset_path("andy") / "data/raw/participant_specifications.csv"
    with path.open(newline="") as stream:
        return {row["ID"]: float(row["body_height"]) / 100.0 for row in csv.DictReader(stream)}


_COMFI_TARGETS = {
    "Left_Hip": 0, "Right_Hip": 1,
    "Left_Knee": 3, "Right_Knee": 4,
    "Left_Ankle": 6, "Right_Ankle": 7,
    "Lumbar": 2, "Thoracic": 5,
    "Left_Clavicle": 12, "Right_Clavicle": 13,
    "Left_Shoulder": 15, "Right_Shoulder": 16,
    "Left_Elbow": 17, "Right_Elbow": 18,
    "Left_Wrist": 19, "Right_Wrist": 20,
    "Cervical": 11,
}


def _axis_rotation(angles: np.ndarray, axis: tuple[float, float, float]) -> np.ndarray:
    rotvec = np.asarray(angles, dtype=np.float64)[:, None] * np.asarray(axis, dtype=np.float64)[None]
    return Rotation.from_rotvec(rotvec).as_matrix()


def _serial_axis_rotation(
    angle_axes: tuple[tuple[np.ndarray, tuple[float, float, float]], ...],
) -> np.ndarray:
    """Compose the single-DoF URDF joints in their actual parent-to-child order."""
    count = len(angle_axes[0][0])
    result = np.broadcast_to(np.eye(3), (count, 3, 3)).copy()
    for angles, axis in angle_axes:
        result = result @ _axis_rotation(angles, axis)
    return result


# COMFI's human URDF uses +X forward, +Y up and +Z to the subject's right.
# SMPL-X uses +X to the subject's left, +Y up and +Z forward.  This proper
# rotation maps SMPL-X rest-frame vectors into the URDF link frame.  Applying
# URDF joint axes directly as SMPL-X rotvec components leaves every segment
# rotated by a fixed quarter turn; JCP-only IK can repair joint positions but
# cannot repair the resulting axial limb twist.
_COMFI_URDF_FROM_SMPLX = np.asarray([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [-1.0, 0.0, 0.0],
], dtype=np.float64)


def _comfi_local_rotation_to_smplx(matrix: np.ndarray) -> np.ndarray:
    basis = _COMFI_URDF_FROM_SMPLX
    return basis.T @ matrix @ basis


def _comfi_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def comfi_biomechanical_body_pose(rows: list[dict[str, str]]) -> np.ndarray:
    """Convert COMFI q values using the released subject-URDF joint axes.

    COMFI's CSV labels are biomechanical names, while the q vector follows a
    serial URDF chain. They are not XYZ Euler components. All released scaled
    human URDFs use the same axes/signs; only link lengths differ by subject.
    """
    count = len(rows)
    body = np.zeros((count, 21, 3), dtype=np.float32)

    def values(name: str) -> np.ndarray:
        return _comfi_column(rows, name + "[rad]")

    def assign(target: int, angle_axes):
        urdf_rotation = _serial_axis_rotation(tuple(angle_axes))
        body[:, target] = Rotation.from_matrix(
            _comfi_local_rotation_to_smplx(urdf_rotation)
        ).as_rotvec().astype(np.float32)

    for side, sign_x, sign_y, hip_target, knee_target, ankle_target in (
        ("Left", 1.0, -1.0, 0, 3, 6),
        ("Right", -1.0, 1.0, 1, 4, 7),
    ):
        assign(hip_target, (
            (values(f"{side}_Hip_Flexion_Extension"), (0.0, 0.0, 1.0)),
            (values(f"{side}_Hip_Abduction_Adduction"), (sign_x, 0.0, 0.0)),
            (values(f"{side}_Hip_Internal_External_Rotation"), (0.0, sign_y, 0.0)),
        ))
        assign(knee_target, (
            (values(f"{side}_Knee_Flexion_Extension"), (0.0, 0.0, -1.0)),
        ))
        assign(ankle_target, (
            (values(f"{side}_Ankle_Plantarflexion_Dorsiflexion"), (0.0, 0.0, 1.0)),
            (values(f"{side}_Ankle_Inversion_Eversion"), (-sign_x, 0.0, 0.0)),
        ))

    assign(2, (
        (values("Lumbar_Flexion_Extension"), (0.0, 0.0, -1.0)),
        (values("Lumbar_Lateral_Bending"), (1.0, 0.0, 0.0)),
    ))
    thoracic_urdf = _serial_axis_rotation((
        (values("Thoracic_Flexion_Extension"), (0.0, 0.0, -1.0)),
        (values("Thoracic_Lateral_Bending"), (1.0, 0.0, 0.0)),
        (values("Thoracic_Internal_External_Rotation"), (0.0, 1.0, 0.0)),
    ))
    # COMFI has one thoracic joint where SMPL-X has spine2 and spine3. Split
    # the logarithm equally so their product preserves the released motion
    # without concentrating all torso curvature into one vertebral joint.
    thoracic_half = Rotation.from_matrix(
        _comfi_local_rotation_to_smplx(thoracic_urdf)
    ).as_rotvec() * 0.5
    body[:, 5] = thoracic_half.astype(np.float32)
    body[:, 8] = thoracic_half.astype(np.float32)
    assign(11, (
        (values("Cervical_Flexion_Extension"), (0.0, 0.0, -1.0)),
        (values("Cervical_Lateral_Bending"), (1.0, 0.0, 0.0)),
        (values("Cervical_Internal_External_Rotation"), (0.0, 1.0, 0.0)),
    ))
    for side, sign_x, sign_y, collar_target, shoulder_target, elbow_target, wrist_target in (
        ("Left", 1.0, -1.0, 12, 15, 17, 19),
        ("Right", -1.0, 1.0, 13, 16, 18, 20),
    ):
        assign(collar_target, (
            (values(f"{side}_Clavicle_Elevation_Depression"), (sign_x, 0.0, 0.0)),
        ))
        assign(shoulder_target, (
            (values(f"{side}_Shoulder_Flexion_Extension"), (0.0, 0.0, 1.0)),
            (values(f"{side}_Shoulder_Abduction_Adduction"), (sign_x, 0.0, 0.0)),
            (values(f"{side}_Shoulder_Internal_External_Rotation"), (0.0, sign_y, 0.0)),
        ))
        assign(elbow_target, (
            (values(f"{side}_Elbow_Flexion_Extension"), (0.0, 0.0, 1.0)),
            (values(f"{side}_Elbow_Pronation_Supination"), (0.0, sign_y, 0.0)),
        ))
        assign(wrist_target, (
            (values(f"{side}_Wrist_Flexion_Extension"), (0.0, 0.0, 1.0)),
            (values(f"{side}_Wrist_Radial_Ulnar_Deviation"), (sign_x, 0.0, 0.0)),
        ))
    return body


def load_comfi_pose(reference: SequenceRef, mode: str = "mocap_aligned") -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    member = str(reference.member)
    if mode == "mocap_raw":
        member = member.replace("mocap/aligned/", "mocap/raw/")
    with zipfile.ZipFile(reference.source) as bundle:
        text = bundle.read(member).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise ValueError(f"Empty COMFI motion: {reference.sequence_id}")
    fps = 100.0 if mode == "mocap_raw" else 40.0
    t = len(rows)
    transl = np.asarray([[float(row[f"Freeflyer_{axis}[m]"]) for axis in "XYZ"] for row in rows], dtype=np.float32)
    quaternion = np.asarray(
        [[float(row[f"Freeflyer_quaternion_{axis}"]) for axis in "XYZW"] for row in rows], dtype=np.float64
    )
    # Freeflyer rotates URDF pelvis-frame vectors into the mocap world.  Append
    # the fixed SMPL-X-rest -> URDF-pelvis basis instead of asking position-only
    # IK to absorb that quarter turn into the hips, spine and shoulders.
    root = Rotation.from_matrix(
        Rotation.from_quat(quaternion).as_matrix() @ _COMFI_URDF_FROM_SMPLX
    ).as_rotvec().astype(np.float32)
    body = comfi_biomechanical_body_pose(rows)
    zeros = np.zeros((t, 45), dtype=np.float32)
    return fps, root, body.reshape(t, 63), transl, zeros


def bvh_pose(reference: SequenceRef) -> tuple[BVHMotion, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    motion = load_bvh_reference(reference)
    return (motion, *bvh_to_smplx_pose(motion))

