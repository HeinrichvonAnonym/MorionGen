from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from .config import RetargetConfig
from .schema import SequenceRef


def load_andy_markers(config: RetargetConfig, reference: SequenceRef) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    archive = config.dataset_path("andy") / "data/raw/qualisys_csv.zip"
    participant = f"Participant_{reference.metadata['participant']}"
    member = f"qualisys_csv/{participant}/{reference.sequence_id}.qualisys.csv"
    with zipfile.ZipFile(archive) as bundle:
        text = bundle.read(member).decode("utf-8")
    lines = text.splitlines()
    header_index = next(index for index, line in enumerate(lines) if line.startswith("timestamp,"))
    header = next(csv.reader([lines[header_index]]))
    names = [header[index][:-2] for index in range(1, len(header), 3)]
    values = np.genfromtxt(io.StringIO("\n".join(lines[header_index + 2:])), delimiter=",", dtype=np.float64)
    values = np.atleast_2d(values)
    timestamps = values[:, 0] - values[0, 0]
    markers = values[:, 1:].reshape(len(values), len(names), 3) / 1000.0
    valid = np.isfinite(markers).all(axis=2)
    markers[~valid] = 0.0
    return timestamps, names, markers.astype(np.float32), valid


def load_ssm2_vertices(config: RetargetConfig) -> dict[str, int]:
    path = Path(config.raw["paths"]["ssm2_marker_map"]).expanduser().resolve()
    data = json.loads(path.read_text())
    return {str(key): int(value) for key, value in data["markersets"][0]["indices"].items()}


def load_andy_glove(config: RetargetConfig, reference: SequenceRef) -> tuple[np.ndarray, np.ndarray]:
    archive = config.dataset_path("andy") / "data/raw/glove_csv.zip"
    participant = f"Participant_{reference.metadata['participant']}"
    member = f"glove_csv/{participant}/{reference.sequence_id}.glove.csv"
    with zipfile.ZipFile(archive) as bundle:
        text = bundle.read(member).decode("utf-8")
    lines = text.splitlines()
    reader = csv.DictReader(io.StringIO("\n".join([lines[0], *lines[2:]])))
    rows = list(reader)
    times = np.asarray([float(row["timestamp"]) for row in rows], dtype=np.float64)
    angles = np.asarray([[float(row[f"{finger}_angle"]) for finger in ("index", "middle", "thumb")]
                         for row in rows], dtype=np.float32)
    order = np.argsort(times)
    times, angles = times[order], angles[order]
    unique = np.concatenate([[True], np.diff(times) > 0])
    return times[unique] - times[unique][0], angles[unique]


def load_comfi_joint_centers(reference: SequenceRef, mode: str = "mocap_aligned") -> tuple[float, list[str], np.ndarray, np.ndarray]:
    folder = "raw" if mode == "mocap_raw" else "aligned"
    fps = 100.0 if folder == "raw" else 40.0
    member = f"mocap/{folder}/{reference.metadata['subject']}/{reference.metadata['task']}/joint_center_positions.csv"
    with zipfile.ZipFile(reference.source) as bundle:
        text = bundle.read(member).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    names: list[str] = []
    for key in reader.fieldnames or []:
        if key.endswith("_X[mm]"):
            names.append(key[:-6])
    points = np.empty((len(rows), len(names), 3), dtype=np.float32)
    for joint, name in enumerate(names):
        for axis, label in enumerate("XYZ"):
            points[:, joint, axis] = [float(row[f"{name}_{label}[mm]"]) / 1000.0 for row in rows]
    valid = np.isfinite(points).all(axis=2)
    points[~valid] = 0.0
    return fps, names, points, valid


COMFI_TO_SMPLX = {
    "Left_Shoulder": 16, "Right_Shoulder": 17, "Neck": 12,
    "Left_Elbow": 18, "Right_Elbow": 19, "Left_Wrist": 20, "Right_Wrist": 21,
    "Left_Hip": 1, "Right_Hip": 2, "Mid_Hip": 0,
    "Left_Knee": 4, "Right_Knee": 5, "Left_Ankle": 7, "Right_Ankle": 8,
}

# COMFI's released JCP construction applies the same signed torso-frame
# acromial offset to both shoulders. The resulting Left_Shoulder trajectory is
# systematically incompatible with the SMPL-X glenohumeral center and can make
# an otherwise position-equivalent collar/shoulder chain flip. Keep the raw JCP
# in provenance, but do not use this known-bad track as an IK objective.
COMFI_EXCLUDED_JCP = {
    "Left_Shoulder": "released COMFI shoulder-center construction uses the same signed acromial offset for both sides",
}


def select_comfi_joints(names: list[str], points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source, targets = [], []
    for name, target in COMFI_TO_SMPLX.items():
        if name in names and name not in COMFI_EXCLUDED_JCP:
            source.append(names.index(name)); targets.append(target)
    return np.asarray(targets, dtype=np.int64), points[:, source], valid[:, source]
