from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .schema import SequenceRef


@dataclass(frozen=True)
class ObjectSource:
    object_id: str
    category: str
    mesh_path: Path
    trajectory_path: Path


@dataclass
class SignedDistanceGrid:
    origin: np.ndarray
    pitch: float
    values: np.ndarray
    surface_points: np.ndarray
    surface_normals: np.ndarray
    sign_agreement: float
    triangle_query: object | None = None

    @property
    def maximum(self) -> np.ndarray:
        return self.origin + self.pitch * (np.asarray(self.values.shape) - 1)

    def sample(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        coordinates = (points - self.origin) / self.pitch
        lower = np.floor(coordinates).astype(np.int64)
        fraction = coordinates - lower
        shape = np.asarray(self.values.shape)
        outside = np.any((lower < 0) | (lower + 1 >= shape), axis=1)
        lower = np.clip(lower, 0, shape - 2)
        result = np.zeros(len(points), dtype=np.float64)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    weight = (
                        (fraction[:, 0] if dx else 1.0 - fraction[:, 0]) *
                        (fraction[:, 1] if dy else 1.0 - fraction[:, 1]) *
                        (fraction[:, 2] if dz else 1.0 - fraction[:, 2])
                    )
                    index = lower + np.asarray([dx, dy, dz])
                    result += weight * self.values[index[:, 0], index[:, 1], index[:, 2]]
        if outside.any():
            clipped = np.minimum(np.maximum(points[outside], self.origin), self.maximum)
            result[outside] = np.linalg.norm(points[outside] - clipped, axis=1)
        return result

    def query_surface(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return distance and normal on the actual OBJ triangle surface.

        ``surface_points`` remains in the archive object for compatibility,
        but v3 contact measurement must not turn a random point cloud into the
        geometry being validated.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if self.triangle_query is not None:
            _, distance, face, _ = self.triangle_query.query(points)
            return distance, self.triangle_query.face_normals[face]
        distance, nearest = cKDTree(self.surface_points).query(points, workers=-1)
        return distance, self.surface_normals[nearest]


class TriangleSurfaceQuery:
    """Vectorized closest-point queries without the optional ``rtree`` wheel.

    HiPHI scan meshes are dense, so nearest face centroids plus faces incident
    to the nearest vertices form a reliable candidate set.  Distances and
    barycentric coordinates are then evaluated on the actual triangles rather
    than on a random surface cloud.
    """

    def __init__(self, mesh, centroid_candidates: int = 32, vertex_candidates: int = 4):
        import trimesh

        self.mesh = mesh
        self.triangles = np.asarray(mesh.triangles, dtype=np.float64)
        self.face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        self.centroid_tree = cKDTree(self.triangles.mean(axis=1))
        self.vertex_tree = cKDTree(np.asarray(mesh.vertices, dtype=np.float64))
        self.centroid_candidates = min(int(centroid_candidates), len(self.triangles))
        self.vertex_candidates = min(int(vertex_candidates), len(mesh.vertices))
        incident = np.full(len(mesh.vertices), -1, dtype=np.int64)
        for face_index, face in enumerate(np.asarray(mesh.faces, dtype=np.int64)):
            for vertex in face:
                if incident[vertex] < 0:
                    incident[vertex] = face_index
        if (incident < 0).any():
            raise ValueError("Object mesh contains vertices without incident faces")
        self.incident_face = incident
        self._trimesh = trimesh

    def query(self, points: np.ndarray, chunk: int = 50000) -> tuple[np.ndarray, ...]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        closest_chunks, distance_chunks, face_chunks, barycentric_chunks = [], [], [], []
        for start in range(0, len(points), chunk):
            query = points[start:start + chunk]
            _, centroid_faces = self.centroid_tree.query(
                query, k=self.centroid_candidates, workers=-1,
            )
            _, vertices = self.vertex_tree.query(
                query, k=self.vertex_candidates, workers=-1,
            )
            centroid_faces = np.asarray(centroid_faces).reshape(len(query), -1)
            vertices = np.asarray(vertices).reshape(len(query), -1)
            candidate_faces = np.concatenate([
                centroid_faces, self.incident_face[vertices],
            ], axis=1).astype(np.int64)
            candidate_triangles = self.triangles[candidate_faces]
            repeated = np.broadcast_to(
                query[:, None, :], candidate_triangles.shape[:2] + (3,),
            )
            closest = self._trimesh.triangles.closest_point(
                candidate_triangles.reshape(-1, 3, 3), repeated.reshape(-1, 3),
            ).reshape(candidate_triangles.shape[:2] + (3,))
            distance = np.linalg.norm(closest - query[:, None], axis=2)
            choice = np.argmin(distance, axis=1)
            rows = np.arange(len(query))
            chosen_face = candidate_faces[rows, choice]
            chosen_point = closest[rows, choice]
            chosen_triangle = self.triangles[chosen_face]
            barycentric = self._trimesh.triangles.points_to_barycentric(
                chosen_triangle, chosen_point,
            )
            closest_chunks.append(chosen_point)
            distance_chunks.append(distance[rows, choice])
            face_chunks.append(chosen_face)
            barycentric_chunks.append(barycentric)
        return (
            np.concatenate(closest_chunks), np.concatenate(distance_chunks),
            np.concatenate(face_chunks), np.concatenate(barycentric_chunks),
        )


def load_selection_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported object selection manifest: {path}")
    selected = payload.get("selected")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Object selection manifest has no selected sequences")
    ids = [str(item["sequence_id"]) for item in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Object selection manifest contains duplicate sequence ids")
    return payload


def select_references(references: Iterable[SequenceRef], manifest: dict) -> list[SequenceRef]:
    lookup = {reference.sequence_id: reference for reference in references}
    result = []
    for item in manifest["selected"]:
        sequence_id = str(item["sequence_id"])
        if sequence_id not in lookup:
            raise ValueError(f"Selected HiPHI sequence is unavailable: {sequence_id}")
        reference = lookup[sequence_id]
        if not bool(reference.metadata.get("is_hoi")):
            raise ValueError(f"Selected sequence is not marked HOI: {sequence_id}")
        expected = str(item["object_category"])
        categories = {value.strip() for value in str(reference.metadata.get("object_categories", "")).split("|")}
        if expected not in categories:
            raise ValueError(f"{sequence_id} does not contain expected category {expected}")
        result.append(reference)
    return result


def parse_object_track_csv(payload: bytes | str, expected_frames: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = payload.decode("utf-8-sig") if isinstance(payload, bytes) else payload
    rows = list(csv.DictReader(io.StringIO(text)))
    expected_columns = {"frame", "time_sec", "px", "py", "pz", "qx", "qy", "qz", "qw"}
    if not rows or set(rows[0]) != expected_columns:
        raise ValueError("HiPHI object CSV has an unexpected header")
    frames = np.asarray([int(row["frame"]) for row in rows], dtype=np.int64)
    timestamps = np.asarray([float(row["time_sec"]) for row in rows], dtype=np.float64)
    translation = np.asarray([[float(row[key]) for key in ("px", "py", "pz")] for row in rows])
    quaternion = np.asarray([[float(row[key]) for key in ("qx", "qy", "qz", "qw")] for row in rows])
    if expected_frames is not None and len(rows) != expected_frames:
        raise ValueError(f"Object track has {len(rows)} frames, expected {expected_frames}")
    if not np.array_equal(frames, np.arange(len(frames))):
        raise ValueError("Object track frame numbers are not contiguous from zero")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("Object track timestamps are not strictly increasing")
    if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
        raise ValueError("Object track contains non-finite values")
    norms = np.linalg.norm(quaternion, axis=1)
    if np.max(np.abs(norms - 1.0)) > 2e-4:
        raise ValueError("Object track quaternion is not unit length")
    quaternion /= norms[:, None]
    for frame in range(1, len(quaternion)):
        if np.dot(quaternion[frame - 1], quaternion[frame]) < 0:
            quaternion[frame] *= -1.0
    return timestamps, translation, quaternion


def canonicalize_object_track(
    translation_m: np.ndarray, quaternion_xyzw: np.ndarray, source_to_canonical: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply only the released source-world to canonical rigid transform."""
    transform = np.asarray(source_to_canonical, dtype=np.float64)
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError("source_to_canonical is not a proper rigid transform")
    canonical_translation = np.asarray(translation_m, dtype=np.float64) @ rotation.T + transform[:3, 3]
    source_rotation = Rotation.from_quat(np.asarray(quaternion_xyzw)).as_matrix()
    canonical_quaternion = Rotation.from_matrix(rotation @ source_rotation).as_quat()
    for frame in range(1, len(canonical_quaternion)):
        if np.dot(canonical_quaternion[frame - 1], canonical_quaternion[frame]) < 0:
            canonical_quaternion[frame] *= -1.0
    return canonical_translation.astype(np.float32), canonical_quaternion.astype(np.float32)


def contact_hysteresis(distance_m: np.ndarray, enter_m: float, exit_m: float, minimum_frames: int = 3) -> np.ndarray:
    if not 0 <= enter_m <= exit_m:
        raise ValueError("Contact hysteresis requires 0 <= enter <= exit")
    distance = np.asarray(distance_m, dtype=np.float64)
    output = np.zeros(distance.shape, dtype=bool)
    active = False
    for index, value in enumerate(distance):
        active = value <= (exit_m if active else enter_m)
        output[index] = active
    if minimum_frames > 1:
        padded = np.r_[False, output, False]
        edges = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in edges.reshape(-1, 2):
            if end - start < minimum_frames:
                output[start:end] = False
    return output


def load_repaired_mesh(path: Path, scale_to_m: float = 0.01):
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) < 4 or len(mesh.faces) < 4:
        raise ValueError(f"Invalid HiPHI object mesh: {path}")
    mesh = mesh.copy()
    mesh.apply_scale(scale_to_m)
    trimesh.repair.fix_normals(mesh, multibody=True)
    if not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.face_normals).all():
        raise ValueError(f"Non-finite object mesh: {path}")
    return mesh


def build_signed_distance_grid(
    mesh, pitch: float = 0.01, padding: float = 0.04, sample_count: int = 50000,
    seed: int = 20261101,
) -> SignedDistanceGrid:
    """Actual-triangle distance grid with voxel-calibrated interior sign.

    The old implementation estimated magnitude from random surface samples.
    At a 3 mm acceptance gate that approximation can move the measured
    surface by millimetres.  v3 evaluates closest points on OBJ triangles for
    every grid node; the voxel fill is used only for the inside/outside sign
    because released HiPHI scans are not watertight.
    """
    import trimesh

    if pitch <= 0 or padding < pitch:
        raise ValueError("Invalid SDF pitch or padding")
    del sample_count, seed
    triangle_query = TriangleSurfaceQuery(mesh)
    points = np.asarray(mesh.vertices, dtype=np.float64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)
    lengths = np.linalg.norm(normals, axis=1)
    keep = np.isfinite(points).all(axis=1) & np.isfinite(normals).all(axis=1) & (lengths > 1e-8)
    points = points[keep]
    normals = normals[keep] / lengths[keep, None]
    if len(points) < 4:
        raise ValueError("Object mesh has too few oriented vertices")
    low = np.floor((mesh.bounds[0] - padding) / pitch) * pitch
    high = np.ceil((mesh.bounds[1] + padding) / pitch) * pitch
    shape = np.rint((high - low) / pitch).astype(int) + 1
    if int(np.prod(shape)) > 30_000_000:
        raise ValueError(f"Object SDF would be too large: {tuple(shape)}")
    values = np.empty(tuple(shape), dtype=np.float32)
    voxel = mesh.voxelized(pitch).fill()
    agreement_count = 0
    compared_count = 0
    yz = int(shape[1] * shape[2])
    for flat_start in range(0, int(np.prod(shape)), 250000):
        flat = np.arange(flat_start, min(int(np.prod(shape)), flat_start + 250000))
        ix = flat // yz
        remainder = flat % yz
        iy = remainder // shape[2]
        iz = remainder % shape[2]
        query = low + np.column_stack((ix, iy, iz)) * pitch
        closest, distance, face, _ = triangle_query.query(query)
        closest_normal = triangle_query.face_normals[face]
        normal_sign = np.einsum("ij,ij->i", query - closest, closest_normal) >= 0
        occupied = np.asarray(voxel.is_filled(query), dtype=bool)
        near = distance <= 2.5 * pitch
        agreement_count += int(np.sum(normal_sign[near] == ~occupied[near]))
        compared_count += int(near.sum())
        signed = np.where(occupied, -distance, distance)
        values[ix, iy, iz] = signed.astype(np.float32)
    agreement = agreement_count / max(1, compared_count)
    if agreement < 0.55:
        raise ValueError(f"Object surface orientation is unreliable (agreement={agreement:.3f})")
    return SignedDistanceGrid(
        low, pitch, values, points, normals, agreement,
        triangle_query=triangle_query,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_hashed_asset(source: Path, asset_root: Path) -> tuple[Path, str]:
    digest = sha256_file(source)
    target = asset_root / f"{digest}{source.suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        handle, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".partial", dir=target.parent)
        os.close(handle)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != digest:
                raise IOError(f"Object asset copy hash mismatch: {source}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target, digest


def resolve_object_sources(dataset_root: Path, staged_reference: SequenceRef, metadata: dict) -> list[ObjectSource]:
    sequence_dir = staged_reference.source.parent
    result = []
    for item in metadata.get("objects", []):
        trajectory = sequence_dir / str(item["trajectory_path"])
        mesh = dataset_root / str(item["mesh_path"])
        if not trajectory.is_file():
            raise FileNotFoundError(trajectory)
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        result.append(ObjectSource(
            str(item["object_id"]), str(item["object_category"]), mesh, trajectory,
        ))
    if not result:
        raise ValueError(f"HiPHI sequence has no object entries: {staged_reference.sequence_id}")
    return result
