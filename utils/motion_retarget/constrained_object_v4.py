from __future__ import annotations

import heapq
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation


PALM_LANDMARKS = {
    "wrist": 0,
    "palm": 1,
    "index_mcp": 2,
    "middle_mcp": 6,
    "pinky_mcp": 10,
}

# Landmark order in the canonical 22-point hand representation. Keep every
# finger as a separate constrained task: pooling all 20 points into one P95
# lets one failed joint landmark (exactly 5% of the samples) disappear at the
# acceptance quantile.
FINGER_NAMES = ("index", "middle", "pinky", "ring", "thumb")
FINGER_LANDMARK_SLICES = {
    name: slice(2 + 4 * index, 2 + 4 * (index + 1))
    for index, name in enumerate(FINGER_NAMES)
}


def finger_constraint_names(label: str) -> tuple[str, ...]:
    """Return independently multiplied dynamic constraints for one hand."""
    if label not in {"left", "right"}:
        raise ValueError(f"Unsupported hand label: {label}")
    return tuple(
        f"finger_{quantity}_{label}_{finger}"
        for quantity in ("position", "direction", "bend")
        for finger in FINGER_NAMES
    ) + (f"finger_opposition_{label}",)


def contact_segment_constraint_name(
    quantity: str, object_index: int, label: str, segment: int,
) -> str:
    """Name one independently multiplied object/hand/anatomical contact task.

    Contact points from different phalanges must never share an AL state: a
    large, easy segment would otherwise dilute the P95 tail and penalty update
    of a short, failed segment. Object identity is included as well so the
    multi-object path cannot overwrite an earlier object's constraint.
    """
    if quantity not in {"contact", "normal"}:
        raise ValueError(f"Unsupported contact constraint quantity: {quantity}")
    if label not in {"left", "right"}:
        raise ValueError(f"Unsupported hand label: {label}")
    if int(object_index) < 0 or not 0 <= int(segment) < 21:
        raise ValueError("Object index and hand segment must be non-negative/valid")
    return f"{quantity}_{label}_object_{int(object_index)}_segment_{int(segment)}"


def contact_segment_constraint_names(
    quantity: str, segments: np.ndarray, valid: np.ndarray, label: str,
) -> tuple[str, ...]:
    """Return deterministic AL task names for every active source segment."""
    segment_ids = np.asarray(segments, dtype=np.int64)
    active = np.asarray(valid, dtype=bool)
    if segment_ids.shape != active.shape or segment_ids.ndim != 4 or segment_ids.shape[2] != 2:
        raise ValueError("Contact segments/valid must have shape [T,N,2,K]")
    side = 0 if label == "left" else 1 if label == "right" else -1
    if side < 0:
        raise ValueError(f"Unsupported hand label: {label}")
    result: list[str] = []
    for object_index in range(segment_ids.shape[1]):
        values = segment_ids[:, object_index, side][active[:, object_index, side]]
        for segment in np.unique(values):
            result.append(contact_segment_constraint_name(
                quantity, object_index, label, int(segment),
            ))
    return tuple(result)


def changed_contact_constraint_names(
    current: dict[str, np.ndarray], updated: dict[str, np.ndarray],
    segments: np.ndarray, anchor_tolerance_m: float,
) -> tuple[str, ...]:
    """Return only AL tasks whose discrete surface target actually changed."""
    changed = correspondence_change_mask(
        current, updated, anchor_tolerance_m,
    )
    segment_ids = np.asarray(segments, dtype=np.int64)
    if changed.shape != segment_ids.shape:
        raise ValueError("Correspondence change masks and segments disagree")
    names: set[str] = set()
    for frame_index, object_index, side, slot in np.argwhere(changed):
        label = "left" if int(side) == 0 else "right"
        segment = int(segment_ids[frame_index, object_index, side, slot])
        for quantity in ("contact", "normal"):
            names.add(contact_segment_constraint_name(
                quantity, int(object_index), label, segment,
            ))
    return tuple(sorted(names))


def correspondence_change_mask(
    current: dict[str, np.ndarray], updated: dict[str, np.ndarray],
    anchor_tolerance_m: float,
) -> np.ndarray:
    """Identify changed discrete contact rows using the solver's same rule."""
    old_valid = np.asarray(current["valid"], dtype=bool)
    new_valid = np.asarray(updated["valid"], dtype=bool)
    if old_valid.shape != new_valid.shape:
        raise ValueError("Correspondence validity masks disagree")
    old_vertex = np.asarray(current["target_vertex"], dtype=np.int64)
    new_vertex = np.asarray(updated["target_vertex"], dtype=np.int64)
    old_face = np.asarray(current["face"], dtype=np.int64)
    new_face = np.asarray(updated["face"], dtype=np.int64)
    old_anchor = np.asarray(current["anchor"], dtype=np.float64)
    new_anchor = np.asarray(updated["anchor"], dtype=np.float64)
    common = old_valid & new_valid
    changed = old_valid != new_valid
    changed |= common & (old_vertex != new_vertex)
    changed |= (
        common & (old_face != new_face) &
        (np.linalg.norm(old_anchor - new_anchor, axis=-1) > float(anchor_tolerance_m))
    )
    return changed


def changed_contact_constraint_entries(
    current: dict[str, np.ndarray], updated: dict[str, np.ndarray],
    segments: np.ndarray, normal_valid: np.ndarray,
    anchor_tolerance_m: float,
) -> dict[str, np.ndarray]:
    """Map changed discrete rows to each residual tensor's flattened order."""
    changed = correspondence_change_mask(current, updated, anchor_tolerance_m)
    new_valid = np.asarray(updated["valid"], dtype=bool)
    segment_ids = np.asarray(segments, dtype=np.int64)
    normal_active = np.asarray(normal_valid, dtype=bool)
    if not (changed.shape == new_valid.shape == segment_ids.shape == normal_active.shape):
        raise ValueError("Contact entry masks must share [T,N,2,K] shape")
    result: dict[str, np.ndarray] = {}
    for object_index in range(changed.shape[1]):
        for side, label in enumerate(("left", "right")):
            for segment in np.unique(segment_ids[:, object_index, side][
                new_valid[:, object_index, side]
            ]):
                segment_mask = (
                    new_valid[:, object_index, side] &
                    (segment_ids[:, object_index, side] == int(segment))
                )
                contact_name = contact_segment_constraint_name(
                    "contact", object_index, label, int(segment),
                )
                result[contact_name] = changed[:, object_index, side][segment_mask]
                segment_normal_mask = (
                    segment_mask & normal_active[:, object_index, side]
                )
                if segment_normal_mask.any():
                    normal_name = contact_segment_constraint_name(
                        "normal", object_index, label, int(segment),
                    )
                    result[normal_name] = changed[:, object_index, side][
                        segment_normal_mask
                    ]
    return result


def _normalize_np(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)


def anatomical_palm_frames(landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Construct the same anatomical palm frame for source BVH and SMPL-X.

    ``landmarks`` ends in ``[side, landmark, xyz]`` and contains wrist, palm,
    index/middle/pinky MCP landmarks.  Matrix columns are x, y and palm-normal.
    The right-hand normal is flipped together with x so both matrices remain
    right handed while their normal consistently points out of the palm.
    """
    values = np.asarray(landmarks, dtype=np.float64)
    if values.shape[-3:] != (2, 22, 3):
        raise ValueError(f"Palm landmarks must end in (2,22,3), got {values.shape}")
    origin = values[..., PALM_LANDMARKS["palm"], :]
    x = _normalize_np(
        values[..., PALM_LANDMARKS["index_mcp"], :] -
        values[..., PALM_LANDMARKS["pinky_mcp"], :]
    )
    y_raw = (
        values[..., PALM_LANDMARKS["middle_mcp"], :] -
        values[..., PALM_LANDMARKS["wrist"], :]
    )
    y = _normalize_np(y_raw - np.sum(y_raw * x, axis=-1, keepdims=True) * x)
    z = _normalize_np(np.cross(x, y))
    # Keep both frames right handed while giving left/right the same outward
    # palm-normal convention.
    side_sign = np.asarray([1.0, -1.0]).reshape((1,) * (z.ndim - 2) + (2, 1))
    x = x * side_sign
    z = z * side_sign
    matrix = np.stack([x, y, z], axis=-1)
    determinant = np.linalg.det(matrix)
    if not np.allclose(determinant, 1.0, atol=1e-5):
        raise ValueError("Degenerate anatomical palm frame")
    return origin.astype(np.float32), matrix.astype(np.float32)


def source_proxy_normal_support(
    landmarks_local: np.ndarray, source_points_local: np.ndarray,
    object_normals_local: np.ndarray, segments: np.ndarray,
    valid: np.ndarray, tolerance_degrees: float,
) -> np.ndarray:
    """Mask contacts whose source capsule supplies a meaningful face normal."""
    from .object_pipeline import _contact_segment_landmarks

    landmarks = np.asarray(landmarks_local, dtype=np.float64)
    points = np.asarray(source_points_local, dtype=np.float64)
    normals = np.asarray(object_normals_local, dtype=np.float64)
    segment_ids = np.asarray(segments, dtype=np.int64)
    active = np.asarray(valid, dtype=bool)
    if landmarks.shape[:-2] != active.shape[:-1] or landmarks.shape[-2:] != (22, 3):
        raise ValueError("Source normal support landmarks/contact shapes disagree")
    if points.shape != (*active.shape, 3) or normals.shape != points.shape:
        raise ValueError("Source normal support points/normals must match contact mask")
    if segment_ids.shape != active.shape:
        raise ValueError("Source normal support segments must match contact mask")
    result = np.zeros(active.shape, dtype=bool)
    threshold = float(np.cos(np.deg2rad(tolerance_degrees)))
    for index in np.argwhere(active):
        prefix = tuple(int(value) for value in index[:-1])
        slot = int(index[-1])
        first, last = _contact_segment_landmarks(int(segment_ids[prefix + (slot,)]))
        start = landmarks[prefix + (first,)]
        end = landmarks[prefix + (last,)]
        axis = end - start
        point = points[prefix + (slot,)]
        alpha = float(np.clip(
            np.dot(point - start, axis) / max(float(np.dot(axis, axis)), 1e-12),
            0.0, 1.0,
        ))
        radial = point - (start + alpha * axis)
        length = float(np.linalg.norm(radial))
        if length <= 1e-8:
            continue
        radial /= length
        opposition = -float(np.dot(radial, normals[prefix + (slot,)]))
        result[prefix + (slot,)] = opposition >= threshold
    return result


def robust_palm_frame_calibration(
    source_matrix: np.ndarray, target_matrix: np.ndarray,
    eligible: np.ndarray | None = None, maximum_frames: int = 90,
) -> np.ndarray:
    """Return one constant source-to-target palm-frame rotation per side."""
    source = np.asarray(source_matrix, dtype=np.float64)
    target = np.asarray(target_matrix, dtype=np.float64)
    if source.shape != target.shape or source.shape[-3:] != (2, 3, 3):
        raise ValueError("Palm calibration requires matching [T,...,2,3,3] frames")
    frames = source.shape[0]
    mask = np.ones((frames, 2), dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    if mask.shape != (frames, 2):
        raise ValueError("Palm calibration eligibility must have shape [T,2]")
    result = np.repeat(np.eye(3, dtype=np.float64)[None], 2, axis=0)
    for side in range(2):
        indices = np.flatnonzero(mask[:, side])[:maximum_frames]
        if not len(indices):
            indices = np.arange(min(frames, maximum_frames))
        relative = source[indices, side].transpose(0, 2, 1) @ target[indices, side]
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        median = np.median(rotvec, axis=0)
        result[side] = Rotation.from_rotvec(median).as_matrix()
    return result.astype(np.float32)


def calibrated_source_palm_targets(
    source_landmarks_local: np.ndarray, target_reference_local: np.ndarray,
    source_contact: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build object-local source palm SE(3) targets with a constant frame calibration."""
    source = np.asarray(source_landmarks_local, dtype=np.float32)
    target = np.asarray(target_reference_local, dtype=np.float32)
    if source.shape != target.shape or source.ndim != 5:
        raise ValueError("Palm targets require matching [T,N,2,22,3] landmarks")
    frames, objects = source.shape[:2]
    source_origin = np.zeros((frames, objects, 2, 3), dtype=np.float32)
    source_matrix = np.zeros((frames, objects, 2, 3, 3), dtype=np.float32)
    target_matrix = np.zeros_like(source_matrix)
    for object_index in range(objects):
        source_origin[:, object_index], source_matrix[:, object_index] = anatomical_palm_frames(
            source[:, object_index]
        )
        _, target_matrix[:, object_index] = anatomical_palm_frames(target[:, object_index])
    calibration = np.zeros((objects, 2, 3, 3), dtype=np.float32)
    desired_matrix = np.zeros_like(source_matrix)
    contact = np.asarray(source_contact, dtype=bool)
    for object_index in range(objects):
        calibration[object_index] = robust_palm_frame_calibration(
            source_matrix[:, object_index], target_matrix[:, object_index],
            eligible=~contact[:, object_index],
        )
        desired_matrix[:, object_index] = (
            source_matrix[:, object_index] @ calibration[object_index, None]
        )
    desired_quaternion = Rotation.from_matrix(
        desired_matrix.reshape(-1, 3, 3)
    ).as_quat().reshape(frames, objects, 2, 4)
    return source_origin, desired_quaternion.astype(np.float32), calibration


def task_jacobian_diagnostics(
    jacobian: np.ndarray, required_rank: int,
    relative_singular_value: float = 1e-4,
) -> dict[str, float | int | bool | np.ndarray]:
    """Measure whether the active variables can control the requested task."""
    value = np.asarray(jacobian, dtype=np.float64)
    if value.ndim != 2 or not value.size:
        raise ValueError("Task Jacobian must be a non-empty matrix")
    singular = np.linalg.svd(value, compute_uv=False)
    maximum = float(singular[0]) if len(singular) else 0.0
    threshold = maximum * float(relative_singular_value)
    rank = int(np.count_nonzero(singular > threshold)) if maximum > 0.0 else 0
    condition = (
        float(maximum / singular[rank - 1]) if rank else float("inf")
    )
    return {
        "rank": rank,
        "required_rank": int(required_rank),
        "full_rank": rank >= int(required_rank),
        "condition": condition,
        "singular_values": singular,
    }


def full_body_fallback_required(
    rank_failure_fraction: float, palm_p95_after_two: float,
    residual_improvement: float, rank_failure_limit: float = 0.05,
    palm_normalized_limit: float = 1.0, improvement_minimum: float = 0.30,
    hierarchy_failed: bool = False,
) -> tuple[bool, list[str]]:
    """Apply the locked v4 fallback rule without silently relaxing gates."""
    reasons = []
    if float(rank_failure_fraction) > float(rank_failure_limit):
        reasons.append("task_jacobian_rank")
    if float(palm_p95_after_two) > float(palm_normalized_limit):
        reasons.append("palm_se3_after_two_outer_iterations")
    if float(residual_improvement) < float(improvement_minimum):
        reasons.append("insufficient_residual_improvement")
    if bool(hierarchy_failed):
        reasons.append("upper_body_hierarchy_infeasible")
    return bool(reasons), reasons


def tier_constraint_p95(
    constraints: dict[str, dict[str, Any]], tier: dict[str, Any] | None,
) -> list[float]:
    """Return only constraints activated in the failed hierarchy tier."""
    if tier is None:
        return []
    return [
        float(constraints[name]["p95"])
        for name in tier.get("constraints", ())
        if name in constraints
    ]


def feasibility_projection_allowed(
    maximum_normalized_p95: float, correspondence_stable_rounds: int,
    required_stable_rounds: int,
) -> bool:
    """Allow exact-gate projection once the discrete task is fixed.

    A residual-magnitude activation threshold has no physical meaning and can
    reject an otherwise useful constraint projection (for example 1.123 while
    accepting 1.100).  Projection itself is bounded by parameter limits and
    the caller rolls back every step that does not strictly reduce the exact
    P95 violation score, so stable correspondence and finite residuals are the
    complete logical preconditions.  The physical acceptance gate remains
    exactly 1.0 for every individual constraint.
    """
    return bool(
        np.isfinite(maximum_normalized_p95) and
        int(correspondence_stable_rounds) >= int(required_stable_rounds)
    )


def normalized_gate_score(normalized_p95: dict[str, float]) -> float:
    """Score only physical-gate violations; zero means fully feasible."""
    if not normalized_p95 or not all(
        np.isfinite(value) for value in normalized_p95.values()
    ):
        return float("inf")
    return float(sum(
        max(float(value) - 1.0, 0.0) ** 2
        for value in normalized_p95.values()
    ))


def componentwise_gate_improvement_allowed(
    previous: dict[str, float], candidate: dict[str, float],
    epsilon: float = 1e-6,
) -> bool:
    """Accept a targeted projection only if no other hard gate regresses."""
    if previous.keys() != candidate.keys():
        return False
    if not all(np.isfinite(value) for value in (*previous.values(), *candidate.values())):
        return False
    monotone = all(
        float(candidate[name]) <= max(1.0, float(previous[name])) + float(epsilon)
        for name in previous
    )
    return bool(
        monotone and
        normalized_gate_score(candidate) + float(epsilon) < normalized_gate_score(previous)
    )


def constraint_gradient_diagnostics(
    values: dict[str, Any], normalized_p95: dict[str, float], parameters: list[Any],
    parameter_names: list[str], inequality_prefixes: tuple[str, ...],
    maximum_constraints: int = 8,
) -> dict[str, Any]:
    """Measure controllability and pairwise directions of failed hard gates.

    This is diagnostic only: it uses the exact P95 violation differentiated at
    the final fixed-correspondence state and never modifies a parameter.
    """
    import torch

    failed = [
        name for name, p95 in sorted(
            normalized_p95.items(), key=lambda item: (-item[1], item[0]),
        )
        if p95 > 1.0 and name in values
    ][:int(maximum_constraints)]
    gradients: dict[str, list[Any]] = {}
    result: dict[str, Any] = {"constraints": {}, "cosine": {}}
    for name in failed:
        loss = quantile_constraint_violation_loss(
            values[name], name.startswith(inequality_prefixes), 0.95, 1.0,
        )
        raw = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True,
        )
        grads = [
            torch.zeros_like(parameter) if gradient is None else gradient
            for parameter, gradient in zip(parameters, raw)
        ]
        gradients[name] = grads
        result["constraints"][name] = {
            "p95": float(normalized_p95[name]),
            "loss": float(loss.detach().cpu()),
            "gradient_norm": {
                label: float(torch.linalg.vector_norm(gradient).detach().cpu())
                for label, gradient in zip(parameter_names, grads)
            },
        }
    for left_index, left_name in enumerate(failed):
        left_grads = gradients[left_name]
        left_norm = torch.sqrt(sum(
            (gradient * gradient).sum() for gradient in left_grads
        ))
        for right_name in failed[left_index + 1:]:
            right_grads = gradients[right_name]
            right_norm = torch.sqrt(sum(
                (gradient * gradient).sum() for gradient in right_grads
            ))
            denominator = left_norm * right_norm
            cosine = (
                sum(
                    (left * right).sum()
                    for left, right in zip(left_grads, right_grads)
                ) / torch.clamp(denominator, min=1e-20)
            )
            result["cosine"][f"{left_name}|{right_name}"] = float(
                cosine.detach().cpu()
            )
    return result


def fixed_state_confirmation_allowed(
    normalized_p95: dict[str, float], consecutive_passes: int,
    required_passes: int, correspondence_stable_rounds: int,
    required_correspondence_rounds: int,
) -> bool:
    """Confirm an unchanged feasible state without perturbing it again."""
    return bool(
        normalized_p95 and
        0 < int(consecutive_passes) < int(required_passes) and
        int(correspondence_stable_rounds) >= int(required_correspondence_rounds) and
        all(np.isfinite(value) and float(value) <= 1.0 for value in normalized_p95.values())
    )


def failed_tier_is_safe_fallback_seed(
    normalized_p95: list[float] | tuple[float, ...],
    initial_normalized_p95: list[float] | tuple[float, ...] = (),
    maximum: float = 2.0, minimum_improvement: float = 0.10,
) -> bool:
    """Allow a near-feasible failed tier to seed the expanded variable space."""
    values = np.asarray(normalized_p95, dtype=np.float64)
    initial = np.asarray(initial_normalized_p95, dtype=np.float64)
    if not values.size or not np.isfinite(values).all() or values.max() > float(maximum):
        return False
    if not initial.size or not np.isfinite(initial).all():
        return True
    improvement = (initial.max() - values.max()) / max(float(initial.max()), 1e-8)
    return bool(improvement >= float(minimum_improvement))


def fallback_seed_sources(use_failed_tier: bool) -> dict[str, str]:
    """Choose fallback seeds without carrying failed-tier MANO drift.

    A near-feasible failed tier can contain useful arm/chest progress, but its
    hand coefficients are not a safe seed: contact and normal tiers have not
    yet activated the source finger-motion constraints, so MANO is a large
    unconstrained nullspace. Always restart both hands from the last feasible
    hierarchy checkpoint and let the expanded full-body solve re-establish the
    failed contact tier from there.
    """
    return {
        "body": "failed_tier" if use_failed_tier else "last_feasible_tier",
        "hands": "last_feasible_tier",
    }


@dataclass
class AugmentedConstraint:
    multiplier: Any
    penalty: float = 1.0
    previous_p95: float = float("inf")
    stable_rounds: int = 0
    inequality: bool = False


def _constraint_violation(residual, inequality: bool, tolerance: float = 1.0):
    """Map a normalized physical gate to its differentiable violation.

    Equality-style vector targets are tolerance tubes, not requests to drive
    morphology-dependent retarget residuals to exactly zero. Inequalities use
    the same normalized upper gate. A satisfied high-priority constraint stays
    in the hierarchy with zero gradient and reactivates immediately if a lower
    tier pushes it outside the gate.
    """
    import torch

    limit = float(tolerance)
    if inequality:
        return torch.relu(residual - limit)
    if residual.ndim > 1 and residual.shape[-1] in {2, 3, 4, 6}:
        norm = torch.linalg.vector_norm(residual, dim=-1, keepdim=True)
        excess = torch.relu(norm - limit)
        return residual * (excess / torch.clamp(norm, min=1e-8))
    return torch.sign(residual) * torch.relu(residual.abs() - limit)


def active_violation_mean(component, violation):
    """Average an AL term over violating physical samples only.

    Acceptance is defined by per-constraint P95 gates. Averaging over every
    already-satisfied frame diluted the remaining contact tail by up to two
    orders of magnitude and made a visibly localized failure look converged to
    the optimizer. Vector residuals keep their physical sample together while
    retaining the prior per-component scale.
    """
    import torch

    if violation.ndim > 1 and violation.shape[-1] in {2, 3, 4, 6}:
        active = torch.linalg.vector_norm(violation, dim=-1) > 0.0
        active = active.unsqueeze(-1).expand_as(violation)
    else:
        active = violation.abs() > 0.0
    if bool(active.any()):
        return component[active].mean()
    return component.sum() * 0.0


def tail_constraint_violation_loss(
    residual, inequality: bool, tail_fraction: float = 0.10,
    tolerance: float = 1.0,
):
    """Penalize the worst physical-gate violations used by the P95 test.

    The hierarchy accepts constraints by their P95 physical residual. A mean
    over every frame can therefore improve many already-safe samples while
    making the small failing tail worse. This fixed-correspondence polish
    reduces the worst fraction instead. Vector residuals are reduced to one
    magnitude per sample before selecting the tail, matching ``constraint_p95``.
    """
    import math
    import torch

    fraction = float(tail_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    value = _constraint_violation(residual, inequality, tolerance)
    if value.ndim > 1 and value.shape[-1] in {2, 3, 4, 6}:
        measured = torch.linalg.vector_norm(value, dim=-1).reshape(-1)
    else:
        measured = value.abs().reshape(-1)
    if measured.numel() == 0:
        return residual.sum() * 0.0
    count = max(1, int(math.ceil(fraction * int(measured.numel()))))
    return torch.topk(measured.square(), count, sorted=False).values.mean()


def quantile_constraint_violation_loss(
    residual, inequality: bool, quantile: float = 0.95,
    tolerance: float = 1.0,
):
    """Squared violation of the exact quantile used by the acceptance gate."""
    import torch

    q = float(quantile)
    if not 0.0 <= q <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    measured = torch.relu(residual) if inequality else residual
    if measured.ndim > 1 and measured.shape[-1] in {2, 3, 4, 6}:
        measured = torch.linalg.vector_norm(measured, dim=-1).reshape(-1)
    else:
        measured = measured.abs().reshape(-1)
    if measured.numel() == 0:
        return residual.sum() * 0.0
    gate = torch.quantile(measured, q)
    return torch.relu(gate - float(tolerance)).square()


def feasibility_projection_loss(
    residual, inequality: bool, quantile: float = 0.95,
    tail_fraction: float = 0.10, tolerance: float = 1.0,
):
    """Give a P95 projector both exact and dense tail gradients.

    ``torch.quantile`` differentiates through only the one or two samples
    bracketing the requested quantile. On a long motion window that leaves the
    rest of the violating P95 tail untouched and caused strong-Wolfe L-BFGS to
    stall. The tail term supplies gradients to the worst physical residuals;
    the caller still accepts or rolls back a step exclusively with the exact
    P95 gate, so this does not relax feasibility.
    """
    return (
        quantile_constraint_violation_loss(
            residual, inequality, quantile, tolerance,
        ) +
        tail_constraint_violation_loss(
            residual, inequality, tail_fraction, tolerance,
        )
    )


def feasibility_projection_uses_dense_tail(
    normalized_p95: float, activation: float = 1.01,
) -> bool:
    """Use dense tail gradients only outside the exact-quantile near field."""
    return bool(
        np.isfinite(normalized_p95) and
        float(normalized_p95) > float(activation)
    )


def exact_projection_retry_required(dense_used: bool, accepted: bool) -> bool:
    """Retry the exact P95 direction when a dense-tail step is rejected."""
    return bool(dense_used and not accepted)


def minimum_change_backtracking_fractions(steps: int = 14) -> tuple[float, ...]:
    """Deterministic largest-first steps for a lexicographic prior projection."""
    if int(steps) < 1:
        raise ValueError("steps must be positive")
    return tuple(0.5 ** index for index in range(int(steps)))


def parameter_limit_diagnostics(parameter, limit: float) -> dict[str, Any]:
    """Report whether axis-angle spline controls are clamped at their bound."""
    import torch

    norms = torch.linalg.vector_norm(parameter.detach(), dim=-1).reshape(-1)
    if not norms.numel():
        return {"maximum_fraction": 0.0, "saturated_fraction": 0.0, "top": []}
    normalized = norms / max(float(limit), 1e-8)
    count = min(5, int(normalized.numel()))
    top_values, top_indices = torch.topk(normalized, count)
    joint_count = int(parameter.shape[-2]) if parameter.ndim >= 2 else 1
    return {
        "maximum_fraction": float(normalized.max().cpu()),
        "saturated_fraction": float((normalized >= 0.99).float().mean().cpu()),
        "top": [
            {
                "control": int(index) // joint_count,
                "joint": int(index) % joint_count,
                "fraction": float(value),
            }
            for value, index in zip(top_values.cpu(), top_indices.cpu())
        ],
    }


@dataclass
class AugmentedLagrangian:
    initial_penalty: float = 1.0
    penalty_multiplier: float = 4.0
    maximum_penalty: float = 4096.0
    required_reduction: float = 0.25
    states: dict[str, AugmentedConstraint] = field(default_factory=dict)

    def clone(self) -> "AugmentedLagrangian":
        """Clone hierarchy state when expanding the optimization variables."""
        cloned = AugmentedLagrangian(
            initial_penalty=float(self.initial_penalty),
            penalty_multiplier=float(self.penalty_multiplier),
            maximum_penalty=float(self.maximum_penalty),
            required_reduction=float(self.required_reduction),
        )
        cloned.states = {
            name: AugmentedConstraint(
                multiplier=state.multiplier.detach().clone(),
                penalty=float(state.penalty),
                previous_p95=float(state.previous_p95),
                stable_rounds=int(state.stable_rounds),
                inequality=bool(state.inequality),
            )
            for name, state in self.states.items()
        }
        return cloned

    def restart_for_primal_reseed(self) -> "AugmentedLagrangian":
        """Restart dual state after replacing only part of the primal seed.

        A full-body fallback uses the failed tier's body together with the
        last feasible tier's hands. The failed tier's multipliers and raised
        penalties were accumulated at a different hand pose, so carrying them
        into that hybrid seed is not an AL continuation. Preserve only the
        solver policy and let the expanded variable space establish fresh
        dual state.
        """
        return AugmentedLagrangian(
            initial_penalty=float(self.initial_penalty),
            penalty_multiplier=float(self.penalty_multiplier),
            maximum_penalty=float(self.maximum_penalty),
            required_reduction=float(self.required_reduction),
        )

    def restart_for_hierarchy_expansion(self) -> "AugmentedLagrangian":
        """Re-establish a balanced AL problem when adding a constraint tier.

        The primal pose entering a new hierarchy level already satisfies all
        earlier physical gates.  Their old multipliers and penalties encode
        the path taken by the *smaller* problem, however, and can be thousands
        of times larger than the newly introduced constraints.  Carrying that
        state into the expanded problem makes the result depend on tier order
        and can prevent the MANO constraints from moving at all.  Keep the
        feasible primal pose and solver policy, but initialize every active
        constraint on the same dual/penalty footing for the expanded solve.

        Earlier constraints are not removed or softened: ``active_names``
        still contains them and they are jointly enforced from the first step
        of the new tier.
        """
        return AugmentedLagrangian(
            initial_penalty=float(self.initial_penalty),
            penalty_multiplier=float(self.penalty_multiplier),
            maximum_penalty=float(self.maximum_penalty),
            required_reduction=float(self.required_reduction),
        )

    def reset_duals(self, names: list[str] | tuple[str, ...]) -> list[str]:
        """Reset dual history when a discrete constraint target changes.

        The penalty is retained because the physical gate and its scale have
        not changed.  Lambda, previous residual, and stable-round history are
        tied to the old face/anchor correspondence and cannot be transferred
        to a newly selected surface task.
        """
        import torch

        reset: list[str] = []
        for name in names:
            state = self.states.get(name)
            if state is None:
                continue
            state.multiplier = torch.zeros_like(state.multiplier)
            state.previous_p95 = float("inf")
            state.stable_rounds = 0
            reset.append(name)
        return reset

    def reset_dual_entries(
        self, entries: dict[str, np.ndarray],
    ) -> dict[str, dict[str, int]]:
        """Reset lambda only where a discrete target row changed.

        P95 history belongs to the segment task and is invalidated as a whole,
        while unchanged per-frame/slot multipliers still encode the same
        equality constraint and must survive the correspondence update.
        """
        import torch

        reset: dict[str, dict[str, int]] = {}
        for name, raw_mask in entries.items():
            state = self.states.get(name)
            if state is None:
                continue
            mask = torch.as_tensor(
                np.asarray(raw_mask, dtype=bool), device=state.multiplier.device,
            )
            total = int(mask.numel())
            if not state.multiplier.ndim or state.multiplier.shape[0] != total:
                state.multiplier = torch.zeros_like(state.multiplier)
                count = int(state.multiplier.shape[0]) if state.multiplier.ndim else 1
            else:
                state.multiplier[mask] = 0.0
                count = int(mask.sum().detach().cpu())
            state.previous_p95 = float("inf")
            state.stable_rounds = 0
            if count:
                reset[name] = {"entries": count, "total": total}
        return reset

    def term(self, name: str, residual, inequality: bool = False):
        import torch

        value = _constraint_violation(residual, inequality, 1.0)
        state = self.states.get(name)
        if state is None or tuple(state.multiplier.shape) != tuple(value.shape):
            state = AugmentedConstraint(
                multiplier=torch.zeros_like(value),
                penalty=float(self.initial_penalty),
                inequality=bool(inequality),
            )
            self.states[name] = state
        component = (
            state.multiplier * value +
            0.5 * float(state.penalty) * value.square()
        )
        return active_violation_mean(component, value)

    def update(self, name: str, residual, tolerance: float = 1.0) -> dict[str, float | bool]:
        import torch

        state = self.states[name]
        raw = residual.detach()
        value = _constraint_violation(raw, state.inequality, tolerance)
        multiplier_reset = tuple(state.multiplier.shape) != tuple(value.shape)
        if multiplier_reset:
            # A hard-feasibility update can remove contact rows.  Old
            # multipliers have no trustworthy row-to-row correspondence with
            # the reduced set, so retain rho but restart lambda and convergence
            # history for this named constraint.
            state = AugmentedConstraint(
                multiplier=torch.zeros_like(value),
                penalty=float(state.penalty),
                inequality=bool(state.inequality),
            )
            self.states[name] = state
        measured = torch.relu(raw) if state.inequality else raw
        absolute = (
            torch.linalg.vector_norm(measured, dim=-1).reshape(-1)
            if measured.ndim > 1 and measured.shape[-1] in {2, 3, 4, 6}
            else measured.abs().reshape(-1)
        )
        p95 = float(torch.quantile(absolute, 0.95).cpu()) if absolute.numel() else 0.0
        with torch.no_grad():
            updated = state.multiplier + float(state.penalty) * value
            state.multiplier = torch.clamp(updated, min=0.0) if state.inequality else updated
        sufficient = (
            np.isfinite(state.previous_p95) and
            p95 <= state.previous_p95 * (1.0 - float(self.required_reduction))
        )
        if p95 <= tolerance:
            state.stable_rounds += 1
        else:
            state.stable_rounds = 0
            if not sufficient:
                state.penalty = min(
                    float(self.maximum_penalty),
                    float(state.penalty) * float(self.penalty_multiplier),
                )
        state.previous_p95 = p95
        return {
            "p95": p95,
            "penalty": float(state.penalty),
            "stable_rounds": int(state.stable_rounds),
            "passed": bool(p95 <= tolerance),
            "multiplier_reset": bool(multiplier_reset),
        }


@dataclass(frozen=True)
class RemappedSurfacePoint:
    point: np.ndarray
    face: int
    barycentric: np.ndarray
    geodesic_m: float


class WeldedSurfaceTopology:
    """Geodesic patches over an immutable triangle mesh.

    Vertices are spatially welded only in the topology copy.  Returned faces
    and barycentric coordinates always refer to the released mesh.
    """

    def __init__(self, mesh, weld_tolerance_m: float = 5e-5):
        import trimesh

        self.vertices = np.asarray(mesh.vertices, dtype=np.float64)
        self.faces = np.asarray(mesh.faces, dtype=np.int64)
        self.triangles = self.vertices[self.faces]
        self._trimesh = trimesh
        tolerance = float(weld_tolerance_m)
        if tolerance <= 0.0:
            raise ValueError("Topology weld tolerance must be positive")
        keys = np.round(self.vertices / tolerance).astype(np.int64)
        _, welded = np.unique(keys, axis=0, return_inverse=True)
        welded_position = np.zeros((int(welded.max()) + 1, 3), dtype=np.float64)
        counts = np.zeros(len(welded_position), dtype=np.int64)
        np.add.at(welded_position, welded, self.vertices)
        np.add.at(counts, welded, 1)
        welded_position /= counts[:, None]
        self.welded_position = welded_position
        self.welded_faces = welded[self.faces]
        graph: list[dict[int, float]] = [dict() for _ in range(len(welded_position))]
        for face in self.welded_faces:
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                if a == b:
                    continue
                length = float(np.linalg.norm(welded_position[a] - welded_position[b]))
                old = graph[int(a)].get(int(b), float("inf"))
                graph[int(a)][int(b)] = min(old, length)
                graph[int(b)][int(a)] = min(graph[int(b)].get(int(a), float("inf")), length)
        self.graph = graph
        self._distance_cache: dict[tuple[int, int], np.ndarray] = {}

    def _vertex_distances(self, source_face: int, source_point: np.ndarray, radius_m: float) -> np.ndarray:
        quantized = tuple(np.round(np.asarray(source_point) * 1e5).astype(np.int64).tolist())
        cache_key = (int(source_face), int(round(float(radius_m) * 1e6)), *quantized)
        cached = self._distance_cache.get(cache_key)
        if cached is not None:
            return cached.copy()
        distances = np.full(len(self.graph), np.inf, dtype=np.float64)
        queue: list[tuple[float, int]] = []
        for vertex in self.welded_faces[int(source_face)]:
            distance = float(np.linalg.norm(self.welded_position[vertex] - source_point))
            if distance < distances[vertex]:
                distances[vertex] = distance
                heapq.heappush(queue, (distance, int(vertex)))
        while queue:
            distance, vertex = heapq.heappop(queue)
            if distance != distances[vertex] or distance > radius_m:
                continue
            for neighbor, length in self.graph[vertex].items():
                candidate = distance + length
                if candidate < distances[neighbor] and candidate <= radius_m:
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        self._distance_cache[cache_key] = distances.copy()
        return distances

    def project(
        self, source_point: np.ndarray, source_face: int, query_point: np.ndarray,
        maximum_geodesic_m: float,
    ) -> RemappedSurfacePoint:
        source_point = np.asarray(source_point, dtype=np.float64)
        query_point = np.asarray(query_point, dtype=np.float64)
        distances = self._vertex_distances(source_face, source_point, maximum_geodesic_m)
        face_distance = np.min(distances[self.welded_faces], axis=1)
        candidate_faces = np.flatnonzero(face_distance <= maximum_geodesic_m)
        if int(source_face) not in candidate_faces:
            candidate_faces = np.r_[candidate_faces, int(source_face)]
        triangles = self.triangles[candidate_faces]
        repeated = np.broadcast_to(query_point, (len(triangles), 3))
        closest = self._trimesh.triangles.closest_point(triangles, repeated)
        barycentric = self._trimesh.triangles.points_to_barycentric(triangles, closest)
        geodesic = np.empty(len(candidate_faces), dtype=np.float64)
        for index, face_index in enumerate(candidate_faces):
            if int(face_index) == int(source_face):
                geodesic[index] = np.linalg.norm(closest[index] - source_point)
            else:
                vertices = self.welded_faces[face_index]
                geodesic[index] = np.min(
                    distances[vertices] +
                    np.linalg.norm(self.welded_position[vertices] - closest[index], axis=1)
                )
        permitted = geodesic <= float(maximum_geodesic_m) + 1e-9
        cost = np.linalg.norm(closest - query_point, axis=1)
        cost[~permitted] = np.inf
        # ``source_point`` is itself on ``source_face`` and therefore always
        # belongs to the allowed geodesic patch.  On a large triangle the
        # unconstrained closest point can lie outside the patch; rejecting the
        # candidate in that case incorrectly treats a non-empty patch as
        # empty.  Keep the original anchor as the conservative feasible point.
        source_cost = float(np.linalg.norm(source_point - query_point))
        if not permitted.any() or source_cost <= float(np.min(cost)):
            source_triangle = self.triangles[[int(source_face)]]
            source_barycentric = self._trimesh.triangles.points_to_barycentric(
                source_triangle, source_point[None],
            )[0]
            return RemappedSurfacePoint(
                point=source_point.astype(np.float32),
                face=int(source_face),
                barycentric=source_barycentric.astype(np.float32),
                geodesic_m=0.0,
            )
        choice = int(np.argmin(cost))
        return RemappedSurfacePoint(
            point=closest[choice].astype(np.float32),
            face=int(candidate_faces[choice]),
            barycentric=barycentric[choice].astype(np.float32),
            geodesic_m=float(geodesic[choice]),
        )


def stable_correspondence_path(cost: np.ndarray, transition_cost: float) -> np.ndarray:
    """Viterbi path used to prevent frame-to-frame correspondence chatter."""
    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("Correspondence cost must have shape [T,C]")
    frames, candidates = values.shape
    cumulative = values[0].copy()
    previous = np.zeros((frames, candidates), dtype=np.int32)
    indices = np.arange(candidates)
    for frame in range(1, frames):
        transition = cumulative[:, None] + float(transition_cost) * (indices[:, None] != indices[None, :])
        previous[frame] = np.argmin(transition, axis=0)
        cumulative = values[frame] + transition[previous[frame], indices]
    path = np.zeros(frames, dtype=np.int32)
    path[-1] = int(np.argmin(cumulative))
    for frame in range(frames - 1, 0, -1):
        path[frame - 1] = previous[frame, path[frame]]
    return path


def constant_proxy_vertex_path(cost: np.ndarray) -> np.ndarray:
    """Choose one target vertex for one contiguous source-proxy run.

    A source capsule sample is a persistent anatomical identity.  Rolling and
    finger switches change which source proxy is active; they do not justify
    remapping the *same* proxy to a different target mesh vertex every frame.
    Object face/barycentric anchors remain fully dynamic along the run.
    """
    values = np.asarray(cost, dtype=np.float64)
    if values.ndim != 2 or not values.shape[1]:
        raise ValueError("Proxy correspondence cost must have shape [T,C]")
    feasible = np.isfinite(values).all(axis=0)
    if not feasible.any():
        raise ValueError("No target vertex is feasible for the complete proxy run")
    total = np.full(values.shape[1], np.inf, dtype=np.float64)
    total[feasible] = values[:, feasible].sum(axis=0)
    choice = int(np.argmin(total))
    return np.full(values.shape[0], choice, dtype=np.int32)


def source_proxy_run_boundaries(
    frame_indices: np.ndarray, segment: np.ndarray, proxy_index: np.ndarray,
) -> np.ndarray:
    """Boundaries of one exact source proxy identity's contiguous runs."""
    frame = np.asarray(frame_indices, dtype=np.int64)
    segment_values = np.asarray(segment, dtype=np.int64)
    proxy_values = np.asarray(proxy_index, dtype=np.int64)
    if frame.ndim != 1 or frame.shape != segment_values.shape or frame.shape != proxy_values.shape:
        raise ValueError("Proxy run fields must be matching vectors")
    return np.flatnonzero(
        (np.diff(frame) != 1) |
        (np.diff(segment_values) != 0) |
        (np.diff(proxy_values) != 0)
    ) + 1


def contact_vertex_owners(
    target_vertex: np.ndarray, valid: np.ndarray,
) -> list[dict[int, int]]:
    """Return the current source-proxy owner of each target vertex per frame.

    Correspondence slots are persistent source proxy identities.  Sequential
    updates must not let an earlier slot steal the current target vertex of a
    later slot, otherwise the later slot is forced to move and the outer loop
    can oscillate through pure identity permutations.
    """
    vertices = np.asarray(target_vertex, dtype=np.int64)
    mask = np.asarray(valid, dtype=bool)
    if vertices.shape != mask.shape or vertices.ndim != 2:
        raise ValueError("Contact vertex ownership requires matching [T,K] arrays")
    owners: list[dict[int, int]] = []
    for frame in range(len(vertices)):
        frame_owners: dict[int, int] = {}
        for slot in np.flatnonzero(mask[frame]):
            vertex = int(vertices[frame, slot])
            if vertex < 0:
                continue
            if vertex in frame_owners:
                raise ValueError("Two source proxy identities own the same target vertex")
            frame_owners[vertex] = int(slot)
        owners.append(frame_owners)
    return owners


def surface_switch_is_material(
    previous_cost_m: float, candidate_cost_m: float, minimum_improvement_m: float,
) -> bool:
    """Whether a discrete surface update improves its metric cost enough."""
    return bool(
        float(previous_cost_m) - float(candidate_cost_m) > float(minimum_improvement_m)
    )


def surface_remap_metric_cost(
    surface_gap_m: float, normal_cost_m: float, geodesic_offset_m: float,
    geodesic_scale: float = 1.0, reference_gap_m: float = 0.0,
    reference_scale: float = 1.0,
) -> float:
    """Metric correspondence cost anchored to the released object trajectory.

    Gap, normal proxy cost, geodesic offset, and immutable-reference gap are
    all metres.  Charging the actual surface offset prevents the allowed 5 mm
    patch from chasing the current optimized hand; charging the source-driven
    reference gap prevents two otherwise feasible target vertices from
    swapping whenever the current continuous pose crosses their bisector.
    """
    values = np.asarray(
        [surface_gap_m, normal_cost_m, geodesic_offset_m, geodesic_scale,
         reference_gap_m, reference_scale],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values < 0.0):
        return float("inf")
    return float(
        surface_gap_m + normal_cost_m + geodesic_scale * geodesic_offset_m +
        reference_scale * reference_gap_m
    )


def persistent_switch_mask(candidate: np.ndarray, minimum_frames: int) -> np.ndarray:
    """Keep only temporally persistent discrete surface updates."""
    values = np.asarray(candidate, dtype=bool)
    if values.ndim != 1:
        raise ValueError("Persistent switch candidates must be a vector")
    required = max(1, int(minimum_frames))
    if required == 1 or not values.any():
        return values.copy()
    padded = np.r_[False, values, False]
    boundaries = np.flatnonzero(padded[1:] != padded[:-1])
    result = np.zeros_like(values)
    for start, end in boundaries.reshape(-1, 2):
        if end - start >= required:
            result[start:end] = True
    return result


def target_vertex_candidate_allowed(
    previous_vertex: int, candidate_vertex: int, allow_vertex_switch: bool,
) -> bool:
    """Lock anatomical proxy-to-vertex identity after a tier is activated."""
    return bool(allow_vertex_switch or int(previous_vertex) == int(candidate_vertex))


def target_vertex_switch_within_local_patch(
    previous_position: np.ndarray, candidate_position: np.ndarray,
    maximum_distance_m: float,
) -> bool:
    """Restrict a target-mesh rebind to a local posed-surface neighbourhood."""
    previous = np.asarray(previous_position, dtype=np.float64)
    candidate = np.asarray(candidate_position, dtype=np.float64)
    if previous.shape != (3,) or candidate.shape != (3,):
        raise ValueError("Target vertex positions must be 3D vectors")
    return bool(
        np.isfinite(previous).all() and np.isfinite(candidate).all() and
        np.linalg.norm(candidate - previous) <= float(maximum_distance_m)
    )


def reference_gap_candidate_feasible(
    initial_gap_m: float, candidate_gap_m: float, slack_m: float,
) -> bool:
    """Keep a rebind compatible with the immutable dynamic-hand reference.

    The current optimized hand surface is deliberately not part of this gate:
    otherwise a drifted pose can choose a new vertex which merely makes that
    same drift self-consistent.  A candidate may improve the original binding,
    or worsen its reference-surface gap only by the explicit physical slack.
    """
    values = np.asarray(
        [initial_gap_m, candidate_gap_m, slack_m], dtype=np.float64,
    )
    return bool(
        np.isfinite(values).all() and float(slack_m) >= 0.0 and
        float(candidate_gap_m) <= float(initial_gap_m) + float(slack_m)
    )


def target_vertex_reserved_by_other(
    owner_slot: int | None, current_slot: int, allow_vertex_switch: bool,
) -> bool:
    """Never let one source proxy steal another proxy's target vertex.

    ``allow_vertex_switch`` permits movement to an unowned vertex.  It does
    not release another anatomical proxy's identity: doing that sequentially
    can leave duplicate owners when the displaced proxy has no feasible
    alternative in the current pose.
    """
    del allow_vertex_switch
    return bool(owner_slot is not None and int(owner_slot) != int(current_slot))


def normal_candidate_feasible(
    opposition: float, minimum_opposition: float, enforce: bool,
) -> bool:
    """Hard normal feasibility used from the normal/collision tier onward."""
    return bool(not enforce or float(opposition) >= float(minimum_opposition))


def normalized_normal_opposition_residual(opposition, minimum_opposition: float):
    """Normalize a facing-normal inequality at its actual angular gate.

    The augmented-Lagrangian convention treats a normalized inequality as
    satisfied when its P95 is at most one. ``1 - dot`` is zero for perfectly
    opposing contact normals and equals ``1 - cos(tolerance)`` exactly at the
    configured angular tolerance. The previous ``minimum - dot`` numerator
    accidentally moved the unit gate away from that physical threshold.
    """
    return (1.0 - opposition) / max(1e-6, 1.0 - float(minimum_opposition))


def surface_correspondence_policy(
    tier_index: int, vertex_lock_after_tier: int, outer_iteration: int,
    vertex_switch_tiers: tuple[int, ...] | list[int] | None = None,
    fallback: bool = False,
) -> tuple[bool, bool]:
    """Return vertex-switch and hard-normal policies for a hierarchy tier.

    Tier indices are zero based.  Enabled tiers alternate a discrete
    target-mesh update with their continuous constrained solve until the
    correspondence convergence gate has passed.  This is required because a
    MANO bend can rotate the posed surface normal after the tier-entry update;
    freezing the vertex at that point leaves a stale, sometimes infeasible
    normal task.  Oscillation is prevented by temporal DP, the same-proxy
    identity gate, immutable-reference gap gate, 5 mm local patch, hysteresis,
    and the explicit two-round correspondence-stability gate.

    Every update remains inside the source proxy's anatomical
    radial/longitudinal identity (enforced by the update routine), and vertex
    changes stop completely after ``vertex_lock_after_tier``. OBJ anchors may
    still slide inside their 5 mm patch. Normal feasibility becomes a hard
    candidate gate only from the normal/collision tier onward, after palm 6D
    and surface contact have already produced a physically meaningful pose.
    """
    tier = int(tier_index)
    switch_enabled = (
        tier < int(vertex_lock_after_tier)
        if vertex_switch_tiers is None else
        tier in {int(value) for value in vertex_switch_tiers}
    )
    del outer_iteration
    allow_vertex_switch = switch_enabled and not bool(fallback)
    # Proxy radial/longitudinal identity remains the first hard discrete gate.
    # At the normal tier the previous contact tier is already feasible, so the
    # posed normal is no longer the circular, uninitialized signal that caused
    # the rejected surface-side flip in the old solver.
    enforce_normal_feasibility = tier >= 2
    return allow_vertex_switch, enforce_normal_feasibility


def initial_correspondence_stable_rounds(
    required_stable_rounds: int, freeze_correspondence: bool,
) -> int:
    """Seed discrete convergence without mutating a fallback's fixed task."""
    return int(required_stable_rounds) if bool(freeze_correspondence) else 0


def correspondence_update_budget(
    maximum_outer_rounds: int, performed_preupdate: bool,
) -> int:
    """Count the tier-entry preupdate outside the outer alternation budget.

    ``maximum_outer_rounds`` describes continuous/discrete outer rounds.  A
    pre-continuous correspondence initialization is an extra setup operation;
    charging it against that budget can prevent the final outer iteration from
    performing the second unchanged update required by the stability gate.
    """
    maximum = int(maximum_outer_rounds)
    if maximum < 1:
        raise ValueError("maximum_outer_rounds must be positive")
    return maximum + int(bool(performed_preupdate))


def fixed_pose_correspondence_confirmation_allowed(
    stable_rounds: int, required_stable_rounds: int,
    correspondence_frozen: bool,
) -> bool:
    """Confirm one near-converged discrete map without another AL step."""
    return bool(
        not correspondence_frozen and
        0 < int(stable_rounds) < int(required_stable_rounds)
    )


def fallback_tier_correspondence_frozen(
    tier_index: int, freeze_correspondence_through_tier: int,
) -> bool:
    """Freeze only evidence tiers already activated before fallback."""
    return int(tier_index) <= int(freeze_correspondence_through_tier)


def fallback_vertex_rebind_locked(
    fallback: bool, tier_index: int, fallback_start_tier: int,
) -> bool:
    """Lock completed tiers, but keep the failed fallback tier alternating.

    The failed tier is precisely the task being re-solved in the expanded
    variable space, so its target-human vertex and OBJ anchor must continue the
    same bounded discrete/continuous alternation.  Earlier feasible tiers keep
    their established correspondence.  All candidate hard gates remain in
    force, so this does not broaden the physical task.
    """
    return bool(fallback and int(tier_index) < int(fallback_start_tier))


def correspondence_is_frozen_for_tier(
    tier_index: int, freeze_correspondence_through_tier: int,
    vertex_lock_after_tier: int, active_tiers: tuple[int, ...] | list[int] | None = None,
) -> bool:
    """Lock correspondence outside configured alternating tiers or in fallback."""
    configured_active = (
        int(tier_index) < int(vertex_lock_after_tier)
        if active_tiers is None else int(tier_index) in {int(value) for value in active_tiers}
    )
    return bool(
        fallback_tier_correspondence_frozen(
            tier_index, freeze_correspondence_through_tier,
        ) or not configured_active
    )


def use_constant_proxy_vertex_path(
    configured_constant: bool, enforce_normal_feasibility: bool,
) -> bool:
    """Keep one target vertex for one exact source-proxy run.

    A real roll or contact switch changes the source capsule sample identity
    and therefore starts another run.  Enabling the normal gate does not change
    that anatomy: letting the same proxy rebind per frame merely transfers
    posed-normal noise into correspondence chatter.  OBJ face/barycentric
    anchors remain frame-dynamic, and normal feasibility still filters the
    run's candidate set.
    """
    del enforce_normal_feasibility
    return bool(configured_constant)


def surface_correspondence_changed(
    old_vertex: int, new_vertex: int, old_anchor: np.ndarray,
    new_anchor: np.ndarray, anchor_tolerance_m: float,
    old_face: int | None = None, new_face: int | None = None,
) -> bool:
    """Ignore a triangle-edge face flip when its physical anchor is unchanged."""
    if int(old_vertex) != int(new_vertex):
        return True
    if old_face is None or new_face is None or int(old_face) == int(new_face):
        return False
    return bool(
        np.linalg.norm(
            np.asarray(new_anchor, dtype=np.float64) -
            np.asarray(old_anchor, dtype=np.float64)
        ) > float(anchor_tolerance_m)
    )


def posed_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted outward vertex normals for a batched posed mesh."""
    values = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if values.ndim != 3 or values.shape[-1] != 3 or triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("Posed normals require [T,V,3] vertices and [F,3] faces")
    result = np.zeros_like(values)
    for frame in range(len(values)):
        points = values[frame, triangles]
        face_normal = np.cross(
            points[:, 1] - points[:, 0], points[:, 2] - points[:, 0],
        )
        for corner in range(3):
            np.add.at(result[frame], triangles[:, corner], face_normal)
    return _normalize_np(result)


def support_foot_mask(
    positions: np.ndarray, fps: float, maximum_height_m: float, maximum_speed_m_s: float,
) -> np.ndarray:
    """Detect left/right support from heel/toe pairs without changing the motion."""
    points = np.asarray(positions, dtype=np.float64)
    if points.ndim != 4 or points.shape[1:] != (2, 2, 3):
        raise ValueError("Foot positions must have shape [T,2,2,3]")
    center = points.mean(axis=2)
    velocity = np.zeros(center.shape[:2], dtype=np.float64)
    if len(center) > 1:
        velocity[1:] = np.linalg.norm(np.diff(center, axis=0), axis=2) * float(fps)
        velocity[0] = velocity[1]
    height = points[..., 2].min(axis=2)
    return (height <= float(maximum_height_m)) & (velocity <= float(maximum_speed_m_s))


def derivative_jump_metrics(knots: np.ndarray, coefficients: np.ndarray, degree: int = 5) -> dict[str, float]:
    """Numerically audit derivative jumps through order three at internal knots."""
    from scipy.interpolate import BSpline

    knots = np.asarray(knots, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    spline = BSpline(knots, coefficients, int(degree), axis=0)
    internal = np.unique(knots[degree + 1:-(degree + 1)])
    metrics: dict[str, float] = {}
    for order in range(4):
        derivative = spline.derivative(order)
        maximum = 0.0
        for knot in internal:
            # This is a continuity audit, not a finite-difference curvature
            # estimate. A macroscopic epsilon reports legitimate derivative
            # evolution as a discontinuity. Adjacent representable floats
            # select the two polynomial pieces at their actual one-sided limits.
            left = derivative(np.nextafter(knot, -np.inf))
            right = derivative(np.nextafter(knot, np.inf))
            maximum = max(maximum, float(np.max(np.abs(right - left))))
        metrics[f"order_{order}_maximum_jump"] = maximum
    return metrics


def contact_constraint_coverage_failures(
    source_contact: np.ndarray,
    constraint_valid: np.ndarray,
    contact_region: np.ndarray,
    frame_mask: np.ndarray | None = None,
    minimum_points: int = 3,
    minimum_regions: int = 2,
) -> list[dict[str, int]]:
    """Return source-contact frames whose hard target constraints were thinned."""
    source = np.asarray(source_contact, dtype=bool)
    valid = np.asarray(constraint_valid, dtype=bool)
    regions = np.asarray(contact_region, dtype=np.int64)
    if source.ndim != 3 or valid.shape[:3] != source.shape or regions.shape != valid.shape:
        raise ValueError("Contact coverage requires [T,O,2] source and [T,O,2,C] constraints")
    selected = np.ones(source.shape[0], dtype=bool) if frame_mask is None else np.asarray(frame_mask, dtype=bool)
    if selected.shape != (source.shape[0],):
        raise ValueError("Contact coverage frame mask must contain one value per frame")
    failures: list[dict[str, int]] = []
    for frame, object_index, side in np.argwhere(source & selected[:, None, None]):
        active = valid[frame, object_index, side]
        active_regions = np.unique(regions[frame, object_index, side, active])
        region_count = int(np.count_nonzero(active_regions >= 0))
        point_count = int(np.count_nonzero(active))
        if point_count < int(minimum_points) or region_count < int(minimum_regions):
            failures.append({
                "frame": int(frame), "object": int(object_index), "side": int(side),
                "points": point_count, "regions": region_count,
            })
    return failures


def contact_segment_residual_summary(
    gap_m: np.ndarray, valid: np.ndarray, segment: np.ndarray,
    evaluation_frames: np.ndarray,
) -> dict[str, Any]:
    """Report contact distance by side and source proxy segment.

    Aggregating both hands or all phalanges hid the rejected right-hand mode.
    This diagnostic keeps every anatomical segment separate and only evaluates
    frames/slots which the independent source constraint made active.
    """
    gap = np.asarray(gap_m, dtype=np.float64)
    active = np.asarray(valid, dtype=bool)
    labels = np.asarray(segment, dtype=np.int64)
    frame_mask = np.asarray(evaluation_frames, dtype=bool)
    if gap.shape != active.shape or gap.shape != labels.shape or gap.ndim != 4:
        raise ValueError("Contact segment diagnostics require matching [T,N,2,K] arrays")
    if frame_mask.shape != (gap.shape[0],):
        raise ValueError("Contact segment evaluation mask has wrong shape")
    result: dict[str, Any] = {"left": {}, "right": {}}
    for side, side_name in enumerate(("left", "right")):
        side_active = active[:, :, side] & frame_mask[:, None, None]
        for segment_id in np.unique(labels[:, :, side][side_active]):
            mask = side_active & (labels[:, :, side] == int(segment_id))
            values = gap[:, :, side][mask]
            if not len(values):
                continue
            result[side_name][str(int(segment_id))] = {
                "count": int(len(values)),
                "median_m": float(np.median(values)),
                "p95_m": float(np.percentile(values, 95)),
                "maximum_m": float(np.max(values)),
            }
    return result


def _torch_palm_frames(landmarks):
    import torch

    normalize = lambda value: torch.nn.functional.normalize(value, dim=-1, eps=1e-8)
    origin = landmarks[..., PALM_LANDMARKS["palm"], :]
    x = normalize(
        landmarks[..., PALM_LANDMARKS["index_mcp"], :] -
        landmarks[..., PALM_LANDMARKS["pinky_mcp"], :]
    )
    y_raw = (
        landmarks[..., PALM_LANDMARKS["middle_mcp"], :] -
        landmarks[..., PALM_LANDMARKS["wrist"], :]
    )
    y = normalize(y_raw - (y_raw * x).sum(dim=-1, keepdim=True) * x)
    z = normalize(torch.linalg.cross(x, y, dim=-1))
    sign = torch.tensor([1.0, -1.0], dtype=landmarks.dtype, device=landmarks.device)
    sign = sign.reshape((1,) * (z.ndim - 2) + (2, 1))
    return origin, torch.stack([x * sign, y, z * sign], dim=-1)


def _rotation_residual(current, desired):
    from .object_pipeline import _matrix_to_axis_angle

    relative = desired.transpose(-1, -2) @ current
    return _matrix_to_axis_angle(relative)


def stable_vector_angle(first, second, epsilon: float = 1e-7):
    """Differentiable vector angle without the singular ``acos`` gradient."""
    import torch

    first_unit = torch.nn.functional.normalize(first, dim=-1, eps=1e-8)
    second_unit = torch.nn.functional.normalize(second, dim=-1, eps=1e-8)
    cross = torch.linalg.cross(first_unit, second_unit, dim=-1)
    sine = torch.sqrt(cross.square().sum(dim=-1) + float(epsilon) ** 2)
    cosine = torch.clamp((first_unit * second_unit).sum(dim=-1), -1.0, 1.0)
    return torch.atan2(sine, cosine)


def _compose(base, delta):
    from .object_pipeline import _compose_axis_angle

    return _compose_axis_angle(base, delta)


def _constraint_windows(contact: np.ndarray, frames: int, window: int) -> list[tuple[int, int]]:
    active = np.asarray(contact, dtype=bool).any(axis=(1, 2))
    if not active.any():
        return []
    padding = window // 4
    padded = np.r_[False, active, False]
    runs = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    intervals: list[tuple[int, int]] = []
    for start, end in runs:
        start = max(0, int(start) - padding)
        end = min(frames, int(end) + padding)
        cursor = start
        while cursor < end:
            right = min(end, cursor + window)
            left = max(start, right - window)
            intervals.append((left, right))
            if right == end:
                break
            cursor += max(1, window - 10)
    result: list[tuple[int, int]] = []
    for interval in intervals:
        if not result or interval != result[-1]:
            result.append(interval)
    return result


def deterministic_gold_windows(contact: np.ndarray, frames: int, window: int) -> list[tuple[int, int]]:
    """Dual-hand acquisition, stable carry, and right-release gold windows."""
    values = np.asarray(contact, dtype=bool)
    left = values[:, :, 0].any(axis=1)
    right = values[:, :, 1].any(axis=1)
    both = left & right
    starts = np.flatnonzero(both & np.r_[True, ~both[:-1]])
    ends = np.flatnonzero(both & np.r_[~both[1:], True]) + 1
    if not len(starts):
        return []
    lengths = ends - starts
    choice = int(np.argmax(lengths))
    acquisition = int(starts[choice])
    stable_end = int(ends[choice])
    stable = int((acquisition + stable_end - 1) // 2)
    release_candidates = np.flatnonzero(right[:-1] & ~right[1:] & left[1:]) + 1
    if len(release_candidates):
        release_lengths = []
        for candidate in release_candidates:
            end = int(candidate)
            while end < frames and left[end] and not right[end]:
                end += 1
            release_lengths.append(end - int(candidate))
        release = int(release_candidates[int(np.argmax(release_lengths))])
    else:
        release = stable_end
    centers = [
        acquisition,
        stable,
        release,
    ]
    result = []
    for center in centers:
        start = min(max(0, center - window // 2), max(0, frames - window))
        pair = (start, min(frames, start + window))
        if pair not in result:
            result.append(pair)
    return result


def _motion_vertices_and_landmarks(config, motion, frame_indices: np.ndarray):
    import torch
    from .fitting import _device_lock
    from .object_pipeline import _device, _model, _target_hand_landmarks

    device, use_cuda, device_index = _device(config)
    model = _model(config, motion.profile, device)
    indices = np.asarray(frame_indices, dtype=np.int64)
    batch = int(config.raw["object_pilot"]["forward_batch_frames"])
    vertex_chunks, landmark_chunks = [], []
    with _device_lock(config, use_cuda, device_index), torch.no_grad():
        for start in range(0, len(indices), batch):
            selected = indices[start:start + batch]
            count = len(selected)
            zeros3 = torch.zeros((count, 3), device=device)
            output = model(
                global_orient=torch.tensor(motion.global_orient[selected], device=device),
                body_pose=torch.tensor(motion.body_pose[selected], device=device),
                left_hand_pose=torch.tensor(motion.left_hand_pose[selected], device=device),
                right_hand_pose=torch.tensor(motion.right_hand_pose[selected], device=device),
                betas=torch.tensor(motion.profile.betas, device=device).reshape(1, -1).expand(count, -1),
                jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                expression=torch.zeros((count, model.num_expression_coeffs), device=device),
                return_verts=True,
            )
            vertex_chunks.append(
                output.vertices.cpu().numpy() * motion.scale + motion.transl[selected, None]
            )
            landmarks = _target_hand_landmarks(output.joints, output.vertices).cpu().numpy()
            landmark_chunks.append(
                landmarks * motion.scale + motion.transl[selected, None, None]
            )
    return model, np.concatenate(vertex_chunks), np.concatenate(landmark_chunks)


def prepare_surface_correspondence(
    config, motion, source, meshes: list[Any], object_translation: np.ndarray,
    object_quaternion: np.ndarray, target_vertex: np.ndarray,
    constraint_valid: np.ndarray,
) -> dict[str, np.ndarray]:
    """Project fresh target-hand correspondences into source-centred 5 mm patches."""
    from .source_hands import align_normal_axis_to_reference, hand_vertex_segments

    frames, objects, sides, contacts = source.contact_valid.shape
    remapped_anchor = np.asarray(source.contact_anchor_local, dtype=np.float32).copy()
    remapped_face = np.asarray(source.contact_face, dtype=np.int32).copy()
    remapped_barycentric = np.asarray(source.contact_barycentric, dtype=np.float32).copy()
    remapped_geodesic = np.zeros((frames, objects, sides, contacts), dtype=np.float32)
    remapped_valid = np.asarray(constraint_valid, dtype=bool).copy()
    remapped_normal = np.asarray(source.contact_normal_local, dtype=np.float32).copy()
    settings = config.raw["object_pilot"]
    topologies = [
        WeldedSurfaceTopology(mesh, float(settings["topology_weld_tolerance_m"]))
        for mesh in meshes
    ]
    active_frames = np.flatnonzero(remapped_valid.any(axis=(1, 2, 3)))
    model, active_vertices, _ = _motion_vertices_and_landmarks(config, motion, active_frames)
    active_normals = posed_vertex_normals(
        active_vertices, np.asarray(model.faces, dtype=np.int64),
    )
    segments = [hand_vertex_segments(model, side) for side in ("left", "right")]
    rotations = Rotation.from_quat(object_quaternion.reshape(-1, 4)).as_matrix().reshape(
        frames, objects, 3, 3,
    )
    maximum = float(settings["surface_remap_max_geodesic_m"])
    for active_index, frame in enumerate(active_frames):
        for object_index in range(objects):
            rotation = rotations[frame, object_index]
            center = object_translation[frame, object_index]
            local_vertices = (active_vertices[active_index] - center) @ rotation
            local_normals = active_normals[active_index] @ rotation
            for side in range(2):
                used: set[int] = set()
                for slot in np.flatnonzero(remapped_valid[frame, object_index, side]):
                    segment = int(source.contact_segment[frame, object_index, side, slot])
                    candidates = segments[side].get(segment, np.empty(0, dtype=np.int64))
                    if not len(candidates):
                        remapped_valid[frame, object_index, side, slot] = False
                        continue
                    if bool(settings.get("surface_remap_preserve_proxy_identity", False)):
                        identity = int(target_vertex[frame, object_index, side, slot])
                        if identity not in set(np.asarray(candidates, dtype=np.int64).tolist()):
                            remapped_valid[frame, object_index, side, slot] = False
                            continue
                        candidates = np.asarray([identity], dtype=np.int64)
                    source_anchor = source.contact_anchor_local[frame, object_index, side, slot]
                    source_face = int(source.contact_face[frame, object_index, side, slot])
                    best = None
                    for candidate in candidates:
                        if int(candidate) in used:
                            continue
                        try:
                            projected = topologies[object_index].project(
                                source_anchor, source_face, local_vertices[candidate], maximum,
                            )
                        except ValueError:
                            continue
                        distance = float(np.linalg.norm(local_vertices[candidate] - projected.point))
                        object_normal = align_normal_axis_to_reference(
                            np.asarray(
                                meshes[object_index].face_normals[projected.face],
                                dtype=np.float64,
                            ),
                            np.asarray(source.contact_normal_local[
                                frame, object_index, side, slot
                            ], dtype=np.float64),
                        )
                        opposition = -float(np.dot(local_normals[candidate], object_normal))
                        normal_cost = maximum * np.clip(
                            0.5 * (1.0 - opposition), 0.0, 1.0,
                        )
                        score = (
                            distance + normal_cost, projected.geodesic_m, int(candidate),
                        )
                        if best is None or score < best[0]:
                            best = (score, int(candidate), projected)
                    if best is None:
                        remapped_valid[frame, object_index, side, slot] = False
                        continue
                    _, candidate, projected = best
                    used.add(candidate)
                    target_vertex[frame, object_index, side, slot] = candidate
                    remapped_anchor[frame, object_index, side, slot] = projected.point
                    remapped_face[frame, object_index, side, slot] = projected.face
                    remapped_barycentric[frame, object_index, side, slot] = projected.barycentric
                    remapped_geodesic[frame, object_index, side, slot] = projected.geodesic_m
                    remapped_normal[frame, object_index, side, slot] = align_normal_axis_to_reference(
                        np.asarray(meshes[object_index].face_normals[projected.face], dtype=np.float32),
                        np.asarray(source.contact_normal_local[
                            frame, object_index, side, slot
                        ], dtype=np.float32),
                    )
    return {
        "target_vertex": target_vertex,
        "valid": remapped_valid,
        "anchor": remapped_anchor,
        "face": remapped_face,
        "barycentric": remapped_barycentric,
        "geodesic": remapped_geodesic,
        "normal": remapped_normal,
    }


def cached_surface_correspondence(
    config, motion, source, meshes: list[Any], object_translation: np.ndarray,
    object_quaternion: np.ndarray, target_vertex: np.ndarray,
    constraint_valid: np.ndarray, pilot_windows: bool,
) -> dict[str, np.ndarray]:
    """Cache only deterministic pre-solve geometry, never optimized state."""
    input_hash = str(motion.quality.get(
        "source_contact_dependency_hash",
        motion.quality.get("input_hash", "missing_input_hash"),
    ))
    mode = "pilot" if pilot_windows else "full"
    cache_dir = config.output_root / "cache/surface_correspondence_v4"
    cache_dir.mkdir(parents=True, exist_ok=True)
    settings = config.raw["object_pilot"]
    geometry_key = hashlib.sha256(json.dumps({
        "schema": "surface_correspondence_v4_proxy_identity_v2",
        "input_hash": input_hash,
        "model_signature": str(motion.quality.get("smplx_model_signature", "")),
        "topology_weld_tolerance_m": float(settings["topology_weld_tolerance_m"]),
        "surface_remap_max_geodesic_m": float(settings["surface_remap_max_geodesic_m"]),
        "target_vertex_sha256": hashlib.sha256(
            np.asarray(target_vertex, dtype=np.int32).tobytes()
        ).hexdigest(),
        "constraint_valid_sha256": hashlib.sha256(
            np.asarray(constraint_valid, dtype=np.uint8).tobytes()
        ).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    cache_path = cache_dir / (
        f"{motion.sequence.sequence_id}.{geometry_key}.{mode}.npz"
    )
    fields = ("target_vertex", "valid", "anchor", "face", "barycentric", "geodesic", "normal")
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached = {name: np.asarray(archive[name]).copy() for name in fields}
        if cached["target_vertex"].shape == target_vertex.shape and cached["valid"].shape == constraint_valid.shape:
            return cached
        cache_path.unlink(missing_ok=True)
    result = prepare_surface_correspondence(
        config, motion, source, meshes, object_translation, object_quaternion,
        target_vertex, constraint_valid,
    )
    handle, temporary = tempfile.mkstemp(
        prefix=cache_path.name + ".", suffix=".partial", dir=cache_dir,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **{name: result[name] for name in fields})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return result


def update_surface_correspondence_window(
    config, source, meshes: list[Any], model, topologies: list[WeldedSurfaceTopology],
    object_translation: np.ndarray, object_quaternion: np.ndarray,
    frame_indices: np.ndarray, vertices: np.ndarray, landmarks: np.ndarray,
    current: dict[str, np.ndarray],
    immutable_surface_vertices: np.ndarray,
    immutable_surface_landmarks: np.ndarray,
    target_reference_landmarks_local: np.ndarray,
    reference_target_vertex: np.ndarray,
    normal_constraint_valid: np.ndarray | None = None,
    allow_vertex_switch: bool = True,
    enforce_normal_feasibility: bool = False,
) -> tuple[dict[str, np.ndarray], float, dict[str, float]]:
    """Update discrete contact correspondences from the current posed mesh.

    Candidate target vertices never leave the source anatomical segment, and
    each candidate surface point is projected through ``WeldedSurfaceTopology``
    so it cannot cross a disconnected thin wall.  Viterbi selection is applied
    independently to each source proxy identity over contiguous frame runs.
    The released OBJ is never modified; welding exists only inside topology.
    """
    from .object_pipeline import (
        _contact_segment_landmarks, _proxy_identity_candidate_order,
        _transport_segment_surface_to_reference,
    )
    from .source_hands import align_normal_axis_to_reference, hand_vertex_segments

    selected = np.asarray(frame_indices, dtype=np.int64)
    posed = np.asarray(vertices, dtype=np.float64)
    if posed.shape[0] != len(selected):
        raise ValueError("Correspondence vertices and frame indices disagree")
    posed_landmarks = np.asarray(landmarks, dtype=np.float64)
    if posed_landmarks.shape != (len(selected), 2, 22, 3):
        raise ValueError("Correspondence landmarks must have shape [T,2,22,3]")
    immutable_vertices = np.asarray(immutable_surface_vertices, dtype=np.float64)
    immutable_landmarks = np.asarray(immutable_surface_landmarks, dtype=np.float64)
    if immutable_vertices.shape != posed.shape:
        raise ValueError("Immutable correspondence vertices must match posed vertices")
    if immutable_landmarks.shape != posed_landmarks.shape:
        raise ValueError("Immutable correspondence landmarks must match posed landmarks")
    reference_landmarks = np.asarray(
        target_reference_landmarks_local, dtype=np.float64,
    )
    if reference_landmarks.shape != (
        len(selected), object_translation.shape[1], 2, 22, 3,
    ):
        raise ValueError("Target reference landmarks must have shape [T,N,2,22,3]")
    result = {name: np.asarray(value).copy() for name, value in current.items()}
    valid = np.asarray(result["valid"], dtype=bool)
    immutable_target_vertex = np.asarray(reference_target_vertex, dtype=np.int64)
    if immutable_target_vertex.shape != result["target_vertex"].shape:
        raise ValueError("Reference target vertices must match correspondence shape")
    normal_supported = (
        np.ones_like(valid, dtype=bool) if normal_constraint_valid is None
        else np.asarray(normal_constraint_valid, dtype=bool)
    )
    if normal_supported.shape != valid.shape:
        raise ValueError("Normal-support mask must match contact correspondence")
    segments = [hand_vertex_segments(model, side) for side in ("left", "right")]
    posed_normals = posed_vertex_normals(posed, np.asarray(model.faces, dtype=np.int64))
    rotations = Rotation.from_quat(
        object_quaternion[selected].reshape(-1, 4)
    ).as_matrix().reshape(len(selected), object_translation.shape[1], 3, 3)
    maximum = float(config.raw["object_pilot"]["surface_remap_max_geodesic_m"])
    transition = float(config.raw["object_pilot"]["surface_remap_transition_m"])
    identity_switch = float(config.raw["object_pilot"].get(
        "surface_remap_proxy_identity_switch_m",
        config.raw["object_pilot"]["surface_remap_identity_switch_m"],
    ))
    target_vertex_patch = float(config.raw["object_pilot"].get(
        "target_vertex_remap_max_distance_m", maximum,
    ))
    anchor_switch = float(config.raw["object_pilot"]["surface_remap_anchor_switch_m"])
    geodesic_cost_scale = float(config.raw["object_pilot"].get(
        "surface_remap_geodesic_cost_scale", 1.0,
    ))
    reference_gap_slack = float(config.raw["object_pilot"].get(
        "surface_remap_reference_gap_slack_m", anchor_switch,
    ))
    reference_cost_scale = float(config.raw["object_pilot"].get(
        "surface_remap_reference_cost_scale", 1.0,
    ))
    minimum_opposition = float(np.cos(np.deg2rad(
        config.raw["object_pilot"]["contact_normal_tolerance_degrees"]
    )))
    anchor_stability = float(
        config.raw["object_pilot"]["acceptance"]["object_translation_m"]
    )
    discrete_total = 0
    discrete_changed = 0
    vertex_changed = 0
    anchor_changed = 0

    for object_index in range(object_translation.shape[1]):
        _, source_palm_matrix = anatomical_palm_frames(
            source.landmarks_local[selected, object_index]
        )
        _, target_palm_matrix = anatomical_palm_frames(posed_landmarks)
        _, immutable_palm_matrix = anatomical_palm_frames(immutable_landmarks)
        _, reference_palm_matrix = anatomical_palm_frames(
            reference_landmarks[:, object_index],
        )
        local_vertices = np.einsum(
            "tvi,tij->tvj",
            posed - object_translation[selected, object_index, None],
            rotations[:, object_index],
        )
        local_normals = np.einsum(
            "tvi,tij->tvj", posed_normals, rotations[:, object_index],
        )
        for side in range(2):
            occupied = [set() for _ in selected]
            owners = contact_vertex_owners(
                result["target_vertex"][:, object_index, side],
                valid[:, object_index, side],
            )
            for slot in range(valid.shape[-1]):
                active = np.flatnonzero(valid[:, object_index, side, slot])
                if not len(active):
                    continue
                # Split at time gaps, anatomical-segment changes, or the exact
                # capsule-axis proxy sample changing.  Slot number alone is
                # only storage order and is not a persistent proxy identity.
                segment_values = source.contact_segment[
                    selected[active], object_index, side, slot
                ]
                proxy_values = source.contact_proxy_index[
                    selected[active], object_index, side, slot
                ]
                boundaries = source_proxy_run_boundaries(
                    selected[active], segment_values, proxy_values,
                )
                for run in np.split(active, boundaries):
                    if not len(run):
                        continue
                    segment = int(source.contact_segment[
                        selected[run[0]], object_index, side, slot
                    ])
                    candidates = np.asarray(
                        segments[side].get(segment, np.empty(0, dtype=np.int64)),
                        dtype=np.int64,
                    )
                    if not len(candidates):
                        continue
                    costs = np.full((len(run), len(candidates)), np.inf, dtype=np.float64)
                    projected: dict[tuple[int, int], RemappedSurfacePoint] = {}
                    anchor_alternatives: dict[
                        tuple[int, int], tuple[RemappedSurfacePoint, float, float]
                    ] = {}
                    for row, local_frame in enumerate(run):
                        source_frame = int(selected[local_frame])
                        source_anchor = source.contact_anchor_local[
                            source_frame, object_index, side, slot
                        ]
                        source_face = int(source.contact_face[
                            source_frame, object_index, side, slot
                        ])
                        first, last = _contact_segment_landmarks(segment)
                        reference_geometry = _transport_segment_surface_to_reference(
                            immutable_vertices[int(local_frame), candidates],
                            immutable_landmarks[int(local_frame), side, first],
                            immutable_landmarks[int(local_frame), side, last],
                            immutable_palm_matrix[int(local_frame), side],
                            reference_landmarks[
                                int(local_frame), object_index, side, first
                            ],
                            reference_landmarks[
                                int(local_frame), object_index, side, last
                            ],
                            reference_palm_matrix[int(local_frame), side],
                        )
                        initial_vertex = int(immutable_target_vertex[
                            int(local_frame), object_index, side, slot
                        ])
                        initial_columns = np.flatnonzero(candidates == initial_vertex)
                        if not len(initial_columns):
                            continue
                        initial_reference_gap = float(np.linalg.norm(
                            reference_geometry[int(initial_columns[0])] - source_anchor,
                        ))
                        _, identity_feasible, _ = _proxy_identity_candidate_order(
                            source.contact_source_local[
                                source_frame, object_index, side, slot
                            ],
                            source.landmarks_local[
                                source_frame, object_index, side, first
                            ],
                            source.landmarks_local[
                                source_frame, object_index, side, last
                            ],
                            source_palm_matrix[int(local_frame), side],
                            posed[int(local_frame), candidates],
                            posed_landmarks[int(local_frame), side, first],
                            posed_landmarks[int(local_frame), side, last],
                            target_palm_matrix[int(local_frame), side],
                            float(config.raw["object_pilot"].get(
                                "target_proxy_radial_tolerance_degrees", 30.0,
                            )),
                            float(config.raw["object_pilot"].get(
                                "target_proxy_longitudinal_tolerance", 0.34,
                            )),
                        )
                        for column, candidate in enumerate(candidates):
                            if not bool(identity_feasible[column]):
                                continue
                            candidate_reference_gap = float(np.linalg.norm(
                                reference_geometry[column] - source_anchor,
                            ))
                            if (
                                int(candidate) != initial_vertex and
                                not reference_gap_candidate_feasible(
                                    initial_reference_gap,
                                    candidate_reference_gap,
                                    reference_gap_slack,
                                )
                            ):
                                continue
                            previous_vertex = int(result["target_vertex"][
                                local_frame, object_index, side, slot
                            ])
                            row_normal_supported = bool(normal_supported[
                                local_frame, object_index, side, slot
                            ])
                            if (
                                enforce_normal_feasibility and
                                not row_normal_supported and
                                int(candidate) != previous_vertex
                            ):
                                continue
                            if not target_vertex_candidate_allowed(
                                previous_vertex, int(candidate), allow_vertex_switch,
                            ):
                                continue
                            if (
                                enforce_normal_feasibility and
                                int(candidate) != previous_vertex and
                                not target_vertex_switch_within_local_patch(
                                    posed[int(local_frame), previous_vertex],
                                    posed[int(local_frame), int(candidate)],
                                    target_vertex_patch,
                                )
                            ):
                                continue
                            if int(candidate) in occupied[int(local_frame)]:
                                continue
                            owner = owners[int(local_frame)].get(int(candidate))
                            if target_vertex_reserved_by_other(
                                owner, int(slot), allow_vertex_switch,
                            ):
                                # A source proxy may keep or relinquish its own
                                # vertex, but another proxy cannot steal that
                                # anatomical identity. ``occupied`` additionally
                                # enforces uniqueness within the new assignment.
                                continue
                            try:
                                point = topologies[object_index].project(
                                    source_anchor, source_face,
                                    local_vertices[local_frame, candidate], maximum,
                                )
                            except ValueError:
                                continue
                            def surface_cost(
                                surface_point: RemappedSurfacePoint,
                            ) -> tuple[float, float]:
                                object_normal = align_normal_axis_to_reference(
                                    np.asarray(
                                        meshes[object_index].face_normals[surface_point.face],
                                        dtype=np.float64,
                                    ),
                                    np.asarray(source.contact_normal_local[
                                        source_frame, object_index, side, slot
                                    ], dtype=np.float64),
                                )
                                opposition = -float(np.dot(
                                    local_normals[local_frame, candidate], object_normal,
                                ))
                                # The 5 mm patch radius is also the physical
                                # cost scale for choosing a correctly facing
                                # hand vertex over a back-facing one.
                                normal_cost = (
                                    maximum * np.clip(
                                        0.5 * (1.0 - opposition), 0.0, 1.0,
                                    )
                                    if not enforce_normal_feasibility or row_normal_supported
                                    else 0.0
                                )
                                cost = surface_remap_metric_cost(
                                    float(np.linalg.norm(
                                        local_vertices[local_frame, candidate] -
                                        surface_point.point
                                    )),
                                    float(normal_cost),
                                    float(surface_point.geodesic_m),
                                    geodesic_cost_scale,
                                    candidate_reference_gap,
                                    reference_cost_scale,
                                )
                                return cost, opposition

                            candidate_cost, candidate_opposition = surface_cost(point)
                            if int(candidate) == previous_vertex:
                                old_point = RemappedSurfacePoint(
                                    point=np.asarray(result["anchor"][
                                        local_frame, object_index, side, slot
                                    ], dtype=np.float32),
                                    face=int(result["face"][
                                        local_frame, object_index, side, slot
                                    ]),
                                    barycentric=np.asarray(result["barycentric"][
                                        local_frame, object_index, side, slot
                                    ], dtype=np.float32),
                                    geodesic_m=float(result["geodesic"][
                                        local_frame, object_index, side, slot
                                    ]),
                                )
                                old_cost, old_opposition = surface_cost(old_point)
                                if (
                                    not normal_candidate_feasible(
                                        candidate_opposition, minimum_opposition,
                                        enforce_normal_feasibility and row_normal_supported,
                                    )
                                ):
                                    if not normal_candidate_feasible(
                                        old_opposition, minimum_opposition,
                                        enforce_normal_feasibility and row_normal_supported,
                                    ):
                                        continue
                                    point = old_point
                                    candidate_cost = old_cost
                                    candidate_opposition = old_opposition
                                # Keep the current real surface point unless
                                # sliding inside the allowed patch improves the
                                # distance+normal fit beyond the configured
                                # contact-median physical resolution.
                                if (
                                    not enforce_normal_feasibility or
                                    old_opposition >= minimum_opposition
                                ):
                                    improvement = old_cost - candidate_cost
                                    anchor_alternatives[(row, column)] = (
                                        old_point, old_cost, improvement,
                                    )
                                    if not surface_switch_is_material(
                                        old_cost, candidate_cost, anchor_switch,
                                    ):
                                        point = old_point
                                        candidate_cost = old_cost
                            elif (
                                not normal_candidate_feasible(
                                    candidate_opposition, minimum_opposition,
                                    enforce_normal_feasibility and row_normal_supported,
                                )
                            ):
                                continue
                            costs[row, column] = candidate_cost
                            if int(candidate) != previous_vertex:
                                # A new target surface identity must improve by
                                # more than the configured contact-P95 physical
                                # resolution; sub-gate nearest-vertex chasing is
                                # not a meaningful correspondence change.
                                costs[row, column] += identity_switch
                            projected[(row, column)] = point
                    minimum_anchor_frames = int(config.raw["object_pilot"].get(
                        "surface_remap_anchor_minimum_frames", 3,
                    ))
                    for column in range(len(candidates)):
                        proposed = np.zeros(len(run), dtype=bool)
                        for row in range(len(run)):
                            alternative = anchor_alternatives.get((row, column))
                            if alternative is not None:
                                proposed[row] = alternative[2] > anchor_switch
                        accepted = persistent_switch_mask(
                            proposed, minimum_anchor_frames,
                        )
                        for row in np.flatnonzero(proposed & ~accepted):
                            old_point, old_cost, _ = anchor_alternatives[(int(row), column)]
                            projected[(int(row), column)] = old_point
                            costs[int(row), column] = old_cost
                    feasible = np.isfinite(costs).any(axis=1)
                    if not feasible.all() and not enforce_normal_feasibility:
                        # Earlier tiers keep the previous complete run rather
                        # than widening anatomy or the 5 mm patch.
                        continue
                    # A back-facing target in the current pose is not proof
                    # that the source contact is absent: arm and MANO variables
                    # are precisely what must rotate that surface into place.
                    # Keep the prior correspondence for infeasible rows. If the
                    # normal remains infeasible, the tier fails explicitly
                    # instead of shrinking the contact-recall denominator.
                    feasible_rows = np.flatnonzero(feasible)
                    if not len(feasible_rows):
                        continue
                    row_boundaries = np.flatnonzero(np.diff(feasible_rows) != 1) + 1
                    for row_group in np.split(feasible_rows, row_boundaries):
                        group_costs = costs[row_group]
                        if use_constant_proxy_vertex_path(
                            bool(config.raw["object_pilot"].get(
                                "surface_remap_constant_vertex_per_proxy_run", False,
                            )),
                            enforce_normal_feasibility,
                        ):
                            try:
                                path = constant_proxy_vertex_path(group_costs)
                            except ValueError:
                                continue
                        else:
                            path = stable_correspondence_path(group_costs, transition)
                        for group_index, row in enumerate(row_group):
                            local_frame = int(run[row])
                            column = int(path[group_index])
                            point = projected[(int(row), column)]
                            candidate = int(candidates[column])
                            old_vertex = int(result["target_vertex"][
                                local_frame, object_index, side, slot
                            ])
                            old_anchor = np.asarray(
                                result["anchor"][local_frame, object_index, side, slot],
                                dtype=np.float64,
                            )
                            old_face = int(result["face"][
                                local_frame, object_index, side, slot
                            ])
                            discrete_total += 1
                            changed_vertex = int(old_vertex != candidate)
                            changed_anchor = int(
                                not changed_vertex and old_face != int(point.face) and
                                np.linalg.norm(old_anchor - point.point) > anchor_stability
                            )
                            vertex_changed += changed_vertex
                            anchor_changed += changed_anchor
                            discrete_changed += int(surface_correspondence_changed(
                                old_vertex, candidate, old_anchor, point.point, anchor_stability,
                                old_face, point.face,
                            ))
                            occupied[local_frame].add(candidate)
                            result["target_vertex"][local_frame, object_index, side, slot] = candidate
                            result["anchor"][local_frame, object_index, side, slot] = point.point
                            result["face"][local_frame, object_index, side, slot] = point.face
                            result["barycentric"][local_frame, object_index, side, slot] = point.barycentric
                            result["geodesic"][local_frame, object_index, side, slot] = point.geodesic_m
                            result["normal"][local_frame, object_index, side, slot] = align_normal_axis_to_reference(
                                np.asarray(meshes[object_index].face_normals[point.face], dtype=np.float32),
                                np.asarray(source.contact_normal_local[
                                    int(selected[local_frame]), object_index, side, slot
                                ], dtype=np.float32),
                            )
    denominator = max(1, discrete_total)
    fraction = float(discrete_changed / denominator)
    return result, fraction, {
        "total": int(discrete_total),
        "vertex_change_fraction": float(vertex_changed / denominator),
        "anchor_change_fraction": float(anchor_changed / denominator),
    }


def _support_baseline(config, motion, model, device):
    import torch

    settings = config.raw["object_pilot"]
    frames = len(motion.timestamps)
    result = np.zeros((frames, 2, 2, 3), dtype=np.float32)
    batch = int(settings["forward_batch_frames"])
    pairs = np.asarray([[7, 10], [8, 11]], dtype=np.int64)
    with torch.no_grad():
        for start in range(0, frames, batch):
            end = min(frames, start + batch)
            count = end - start
            zeros3 = torch.zeros((count, 3), device=device)
            output = model(
                global_orient=torch.tensor(motion.global_orient[start:end], device=device),
                body_pose=torch.tensor(motion.body_pose[start:end], device=device),
                left_hand_pose=torch.tensor(motion.left_hand_pose[start:end], device=device),
                right_hand_pose=torch.tensor(motion.right_hand_pose[start:end], device=device),
                betas=torch.tensor(motion.profile.betas, device=device).reshape(1, -1).expand(count, -1),
                jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                expression=torch.zeros((count, model.num_expression_coeffs), device=device),
                return_verts=False,
            )
            joints = output.joints.cpu().numpy() * motion.scale + motion.transl[start:end, None]
            result[start:end] = joints[:, pairs]
    support = support_foot_mask(
        result, motion.fps, float(settings["support_foot_height_m"]),
        float(settings["support_foot_speed_m_s"]),
    )
    return result, support


def _window_task_jacobian(
    config, motion, model, frame: int, active_sides: np.ndarray,
    body_joints: np.ndarray, device,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from .object_pipeline import _target_hand_landmarks

    sides = np.flatnonzero(active_sides)
    if not len(sides):
        return np.zeros((0, 3 * len(body_joints)), dtype=np.float32), sides
    body0 = torch.tensor(
        motion.body_pose[frame].reshape(21, 3), device=device,
    )
    root0 = torch.tensor(motion.global_orient[frame:frame + 1], device=device)
    left0 = torch.tensor(motion.left_hand_pose[frame:frame + 1], device=device)
    right0 = torch.tensor(motion.right_hand_pose[frame:frame + 1], device=device)
    translation = torch.tensor(motion.transl[frame:frame + 1], device=device)
    betas = torch.tensor(motion.profile.betas, device=device).reshape(1, -1)
    zeros3 = torch.zeros((1, 3), device=device)

    def task(flat_delta):
        delta = flat_delta.reshape(1, len(body_joints), 3)
        current = body0[None].clone()
        current[:, body_joints] = _compose(body0[None, body_joints], delta)
        output = model(
            global_orient=root0, body_pose=current.reshape(1, 63),
            left_hand_pose=left0, right_hand_pose=right0, betas=betas,
            jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
            expression=torch.zeros((1, model.num_expression_coeffs), device=device),
            return_verts=True,
        )
        landmarks = _target_hand_landmarks(output.joints, output.vertices) * motion.scale
        landmarks = landmarks + translation[:, None, None]
        origin, matrix = _torch_palm_frames(landmarks)
        from .object_pipeline import _matrix_to_axis_angle
        rotation = _matrix_to_axis_angle(matrix)
        return torch.cat([
            origin[0, sides] / float(config.raw["object_pilot"]["palm_position_tolerance_m"]),
            rotation[0, sides] / float(np.deg2rad(
                config.raw["object_pilot"]["palm_orientation_tolerance_degrees"]
            )),
        ], dim=-1).reshape(-1)

    zero = torch.zeros(3 * len(body_joints), device=device, requires_grad=True)
    # SMPL-X contains internal ``view`` operations that are not compatible
    # with functorch's batched tangents on non-contiguous vmap tensors.  The
    # task is at most 12-dimensional, so the unvectorized reverse sweeps are
    # deterministic and inexpensive relative to the window solve.
    jacobian = torch.autograd.functional.jacobian(task, zero, vectorize=False)
    return jacobian.detach().cpu().numpy(), sides


def _window_task_rank(
    config, motion, model, frame: int, active_sides: np.ndarray,
    body_joints: np.ndarray, device,
) -> dict[str, Any]:
    jacobian, sides = _window_task_jacobian(
        config, motion, model, frame, active_sides, body_joints, device,
    )
    required = 6 * len(sides)
    if not required:
        return {"rank": 0, "required_rank": 0, "full_rank": True, "condition": 1.0}
    return task_jacobian_diagnostics(
        jacobian, required,
        float(config.raw["object_pilot"]["jacobian_relative_singular_value"]),
    )


def _contact_window_rank_series(
    config, motion, model, start: int, window_contact: np.ndarray,
    body_joints: np.ndarray, device,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Evaluate one joint task Jacobian per actual contact frame.

    The combined Jacobian is sliced into per-side 6D tasks, avoiding three
    separate SMPL-X autodiff passes per frame while retaining the specified
    per-side rank and condition-number audit.
    """
    count = len(window_contact)
    rank = np.zeros((count, 2), dtype=np.int16)
    condition = np.ones((count, 2), dtype=np.float32)
    failures = 0
    active_frames = np.flatnonzero(window_contact.any(axis=(1, 2)))
    combined_reports = []
    for local_frame in active_frames:
        active_sides = window_contact[local_frame].any(axis=0)
        jacobian, sides = _window_task_jacobian(
            config, motion, model, start + int(local_frame), active_sides,
            body_joints, device,
        )
        combined = task_jacobian_diagnostics(
            jacobian, 6 * len(sides),
            float(config.raw["object_pilot"]["jacobian_relative_singular_value"]),
        )
        failures += int(not combined["full_rank"])
        combined_reports.append(combined)
        for order, side in enumerate(sides):
            side_report = task_jacobian_diagnostics(
                jacobian[6 * order:6 * (order + 1)], 6,
                float(config.raw["object_pilot"]["jacobian_relative_singular_value"]),
            )
            rank[local_frame, side] = int(side_report["rank"])
            condition[local_frame, side] = float(side_report["condition"])
    fraction = float(failures / max(1, len(active_frames)))
    summary = {
        "contact_frames": int(len(active_frames)),
        "rank_failure_frames": int(failures),
        "rank_failure_fraction": fraction,
        "maximum_condition": float(max(
            (item["condition"] for item in combined_reports), default=1.0,
        )),
    }
    return rank, condition, fraction, summary


def cached_contact_window_rank_series(
    config, motion, model, start: int, window_contact: np.ndarray,
    body_joints: np.ndarray, device,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """Cache deterministic task-rank diagnostics, never solver state."""
    contact_bytes = np.ascontiguousarray(window_contact, dtype=np.uint8).tobytes()
    payload = {
        "schema": "palm_task_rank_v4_v1",
        "input_hash": str(motion.quality.get(
            "source_contact_dependency_hash",
            motion.quality.get("input_hash", "missing_input_hash"),
        )),
        "model_signature": str(motion.quality.get("smplx_model_signature", "")),
        "start": int(start),
        "frames": int(len(window_contact)),
        "body_joints": np.asarray(body_joints, dtype=np.int64).tolist(),
        "relative_singular_value": float(
            config.raw["object_pilot"]["jacobian_relative_singular_value"]
        ),
        "contact_sha256": hashlib.sha256(contact_bytes).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cache_dir = config.output_root / "cache/task_rank_v4"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{motion.sequence.sequence_id}.{key}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as archive:
            return (
                np.asarray(archive["rank"], dtype=np.int16),
                np.asarray(archive["condition"], dtype=np.float32),
                float(archive["failure_fraction"]),
                json.loads(str(archive["summary_json"].item())),
            )
    result = _contact_window_rank_series(
        config, motion, model, start, window_contact, body_joints, device,
    )
    rank, condition, fraction, summary = result
    handle, temporary = tempfile.mkstemp(
        prefix=cache_path.name + ".", suffix=".partial", dir=cache_dir,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(
                stream, rank=np.asarray(rank, dtype=np.int16),
                condition=np.asarray(condition, dtype=np.float32),
                failure_fraction=np.float64(fraction),
                summary_json=np.asarray(json.dumps(summary, sort_keys=True)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return result


def solve_constrained_object_motion(
    config, motion, source, meshes: list[Any], grids: list[Any],
    object_translation: np.ndarray, object_quaternion: np.ndarray,
    initial_reference: dict[str, np.ndarray], target_reference_local: np.ndarray,
    pilot_windows: bool = False,
) -> dict[str, Any]:
    """Direct-C4, per-side constrained human/object retarget.

    The released object pose is read-only.  Upper-body chains and MANO are
    optimized first; root/lower-body coefficients are introduced only for a
    window whose palm task is rank-deficient or remains outside its 6D gate.
    """
    import torch
    from .fitting import _device_lock
    from .object_pipeline import (
        _device, _forward, _incident_face_table, _matrix_to_axis_angle, _model,
        _selected_vertex_normals, _target_hand_landmarks, _torch_grid_sample,
        _vertex_sets,
    )
    from .trajectory import c3_archive_payload, c4_bspline_basis

    settings = config.raw["object_pilot"]
    al_settings = settings["augmented_lagrangian"]
    device, use_cuda, device_index = _device(config)
    model = _model(config, motion.profile, device)
    frames = len(motion.timestamps)
    objects = object_translation.shape[1]
    degree = int(config.raw["temporal_continuity"]["spline_degree"])
    basis_np, knots = c4_bspline_basis(
        motion.timestamps, float(settings["c4_knot_seconds"]), degree,
    )
    controls = basis_np.shape[1]
    body_coeff = np.zeros((controls, 21, 3), dtype=np.float32)
    root_coeff = np.zeros((controls, 3), dtype=np.float32)
    translation_coeff = np.zeros((controls, 3), dtype=np.float32)
    left_coeff = np.zeros((controls, 15, 3), dtype=np.float32)
    right_coeff = np.zeros((controls, 15, 3), dtype=np.float32)
    body_base = np.asarray(motion.body_pose, dtype=np.float32).reshape(frames, 21, 3).copy()
    root_base = np.asarray(motion.global_orient, dtype=np.float32).copy()
    translation_base = np.asarray(motion.transl, dtype=np.float32).copy()
    left_base = np.asarray(motion.left_hand_pose, dtype=np.float32).reshape(frames, 15, 3).copy()
    right_base = np.asarray(motion.right_hand_pose, dtype=np.float32).reshape(frames, 15, 3).copy()
    upper = np.asarray(settings["upper_body_joint_indices"], dtype=np.int64)
    lower = np.asarray(settings["lower_body_fallback_joint_indices"], dtype=np.int64)
    upper_limits = np.deg2rad(np.asarray(settings["upper_body_max_delta_degrees"], dtype=np.float32))
    lower_limits = np.deg2rad(np.asarray(settings["lower_body_max_delta_degrees"], dtype=np.float32))
    hand_limit = float(np.deg2rad(settings["mano_max_delta_degrees"]))
    root_limit = float(np.deg2rad(settings["fallback_root_rotation_degrees"]))
    translation_limit = float(settings["fallback_root_translation_m"])
    face_table_np, face_mask_np = _incident_face_table(
        np.asarray(model.faces), model.lbs_weights.shape[0],
    )
    face_table = torch.tensor(face_table_np, dtype=torch.long, device=device)
    face_mask = torch.tensor(face_mask_np, dtype=torch.bool, device=device)
    _, _, collision_vertex_ids_np = _vertex_sets(
        model, int(settings["hand_vertex_samples"]), int(settings["body_vertex_samples"]),
    )
    collision_vertex_ids = torch.tensor(collision_vertex_ids_np, dtype=torch.long, device=device)

    source_palm_position, source_palm_quaternion, calibration = calibrated_source_palm_targets(
        source.landmarks_local, target_reference_local, source.contact,
    )
    desired_palm_matrix = Rotation.from_quat(
        source_palm_quaternion.reshape(-1, 4),
    ).as_matrix().reshape(frames, objects, 2, 3, 3).astype(np.float32)
    baseline_feet, support = _support_baseline(config, motion, model, device)
    reference_target_vertex = np.asarray(
        initial_reference["dynamic_contact_target_vertex"], dtype=np.int32,
    ).copy()
    target_vertex = reference_target_vertex.copy()
    constraint_valid = np.asarray(
        initial_reference["dynamic_contact_constraint_valid"], dtype=bool,
    ).copy()
    contact = np.asarray(source.contact, dtype=bool)
    source_normal_valid = source_proxy_normal_support(
        source.landmarks_local, source.contact_source_local,
        source.contact_normal_local, source.contact_segment,
        source.contact_valid,
        float(settings["contact_normal_tolerance_degrees"]),
    )
    windows = (
        deterministic_gold_windows(contact, frames, int(settings["ik_window_frames"]))
        if pilot_windows else
        _constraint_windows(contact, frames, int(settings["ik_window_frames"]))
    )
    pilot_frame_mask = np.ones(frames, dtype=bool)
    if pilot_windows:
        selected_frames = np.zeros(frames, dtype=bool)
        for start, end in windows:
            selected_frames[start:end] = True
        pilot_frame_mask = selected_frames.copy()
        constraint_valid &= selected_frames[:, None, None, None]
    correspondence = cached_surface_correspondence(
        config, motion, source, meshes, object_translation, object_quaternion,
        target_vertex, constraint_valid, pilot_windows,
    )
    target_vertex = correspondence["target_vertex"]
    constraint_valid = correspondence["valid"]
    minimum_points = int(settings["acceptance"]["contact_patch_active_vertices_p05"])
    minimum_regions = int(settings["acceptance"]["contact_patch_minimum_regions"])
    coverage_failures = contact_constraint_coverage_failures(
        contact, constraint_valid, source.contact_region,
        pilot_frame_mask, minimum_points, minimum_regions,
    )
    if coverage_failures:
        preview = coverage_failures[:8]
        raise ValueError(
            "Initial contact correspondence cannot preserve the hard "
            f"{minimum_points}-point/{minimum_regions}-region grasp coverage; "
            f"failed_frames={len(coverage_failures)} preview={preview}"
        )
    correspondence_topologies = [
        WeldedSurfaceTopology(mesh, float(settings["topology_weld_tolerance_m"]))
        for mesh in meshes
    ]
    rank = np.zeros((frames, 2), dtype=np.int16)
    condition = np.ones((frames, 2), dtype=np.float32)
    fallback_mask = np.zeros(frames, dtype=bool)
    palm_multiplier = np.zeros((frames, objects, 2, 6), dtype=np.float32)
    contact_multiplier = np.zeros((*constraint_valid.shape, 3), dtype=np.float32)
    layer_residual = np.zeros((5, 2), dtype=np.float32)
    layer_penalty = np.ones((5, 2), dtype=np.float32)
    layer_multiplier = np.zeros((5, 2), dtype=np.float32)
    layer_converged = np.zeros((5, 2), dtype=bool)
    root_fallback_translation = np.zeros((frames, 3), dtype=np.float32)
    root_fallback_rotvec = np.zeros((frames, 3), dtype=np.float32)
    window_reports: list[dict[str, Any]] = []

    def clamp_parameter(parameter, limits):
        with torch.no_grad():
            norm = torch.linalg.vector_norm(parameter, dim=-1, keepdim=True)
            maximum = torch.as_tensor(limits, dtype=parameter.dtype, device=parameter.device)
            if maximum.ndim == 1 and parameter.ndim == 3:
                maximum = maximum[None, :, None]
            parameter.mul_(torch.minimum(
                torch.ones_like(norm), maximum / torch.clamp(norm, min=1e-8),
            ))

    with _device_lock(config, use_cuda, device_index):
        for start, end in windows:
            count = end - start
            window_contact = contact[start:end]
            active_sides = window_contact.any(axis=(0, 1))
            if not active_sides.any():
                continue
            window_rank, window_condition, rank_failure_fraction, joint_rank = (
                cached_contact_window_rank_series(
                    config, motion, model, start, window_contact, upper, device,
                )
            )
            rank[start:end] = np.maximum(rank[start:end], window_rank)
            condition[start:end] = np.maximum(condition[start:end], window_condition)
            side_rank = []
            for side in range(2):
                mask = window_contact[:, :, side].any(axis=1)
                side_rank.append({
                    "minimum_rank": int(window_rank[mask, side].min(initial=6)) if mask.any() else 0,
                    "maximum_condition": float(window_condition[mask, side].max(initial=1.0)) if mask.any() else 1.0,
                    "full_rank": bool((window_rank[mask, side] >= 6).all()) if mask.any() else True,
                })
            initial_fallback = (
                rank_failure_fraction > float(settings["jacobian_rank_failure_fraction"])
            )
            active_control = np.flatnonzero(np.abs(basis_np[start:end]).sum(axis=0) > 0)
            basis = torch.tensor(basis_np[start:end, active_control], device=device)
            immutable_vertex_chunks = []
            immutable_landmark_chunks = []
            reference_batch = int(settings["forward_batch_frames"])
            with torch.no_grad():
                for reference_start in range(start, end, reference_batch):
                    reference_end = min(end, reference_start + reference_batch)
                    output = _forward(
                        model, motion, slice(reference_start, reference_end), device,
                    )
                    immutable_vertex_chunks.append(
                        output.vertices.detach().cpu().numpy() * motion.scale +
                        motion.transl[reference_start:reference_end, None]
                    )
                    immutable_landmark_chunks.append(
                        _target_hand_landmarks(
                            output.joints, output.vertices,
                        ).detach().cpu().numpy() * motion.scale +
                        motion.transl[reference_start:reference_end, None, None]
                    )
            immutable_surface_vertices = np.concatenate(
                immutable_vertex_chunks, axis=0,
            ).astype(np.float32, copy=False)
            immutable_surface_landmarks = np.concatenate(
                immutable_landmark_chunks, axis=0,
            ).astype(np.float32, copy=False)

            def optimize(
                fallback: bool, freeze_correspondence_through_tier: int = -1,
                start_tier: int = 0,
                seed_al: AugmentedLagrangian | None = None,
            ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
                body_joints = np.unique(np.r_[upper, lower] if fallback else upper)
                body_parameter = torch.nn.Parameter(torch.tensor(
                    body_coeff[active_control][:, body_joints], device=device,
                ).contiguous())
                left_parameter = torch.nn.Parameter(
                    torch.tensor(left_coeff[active_control], device=device).contiguous()
                )
                right_parameter = torch.nn.Parameter(
                    torch.tensor(right_coeff[active_control], device=device).contiguous()
                )
                parameters = [body_parameter, left_parameter, right_parameter]
                if fallback:
                    root_parameter = torch.nn.Parameter(
                        torch.tensor(root_coeff[active_control], device=device).contiguous()
                    )
                    translation_parameter = torch.nn.Parameter(torch.tensor(
                        translation_coeff[active_control], device=device,
                    ).contiguous())
                    parameters.extend([root_parameter, translation_parameter])
                else:
                    root_parameter = torch.zeros((len(active_control), 3), device=device)
                    translation_parameter = torch.zeros((len(active_control), 3), device=device)
                # Every optimize() call is an expansion/refinement of its
                # incoming global C4 state. Priors must regularize the change
                # from that state, not pull all coefficients back to zero.
                # This is essential when fallback opens new DoFs after MANO
                # has already approached the dynamic finger target, and for
                # shared controls in overlapping temporal windows.
                body_prior_delta = torch.einsum(
                    "tc,cjd->tjd", basis, body_parameter.detach(),
                )
                left_prior_delta = torch.einsum(
                    "tc,cjd->tjd", basis, left_parameter.detach(),
                )
                right_prior_delta = torch.einsum(
                    "tc,cjd->tjd", basis, right_parameter.detach(),
                )
                root_prior_delta = torch.einsum(
                    "tc,cd->td", basis, root_parameter.detach(),
                )
                translation_prior_delta = torch.einsum(
                    "tc,cd->td", basis, translation_parameter.detach(),
                )
                body0 = torch.tensor(body_base[start:end], device=device)
                root0 = torch.tensor(root_base[start:end], device=device)
                translation0 = torch.tensor(translation_base[start:end], device=device)
                left0 = torch.tensor(left_base[start:end], device=device)
                right0 = torch.tensor(right_base[start:end], device=device)
                betas = torch.tensor(motion.profile.betas, device=device).reshape(1, -1).expand(count, -1)
                zeros3 = torch.zeros((count, 3), device=device)
                object_rotation = torch.tensor(
                    Rotation.from_quat(object_quaternion[start:end].reshape(-1, 4)).as_matrix().reshape(
                        count, objects, 3, 3,
                    ), dtype=body0.dtype, device=device,
                )
                object_center = torch.tensor(object_translation[start:end], device=device)
                desired_position = torch.tensor(source_palm_position[start:end], device=device)
                desired_matrix = torch.tensor(desired_palm_matrix[start:end], device=device)
                target_reference = torch.tensor(target_reference_local[start:end], device=device)
                support_window = torch.tensor(support[start:end], device=device)
                baseline_feet_window = torch.tensor(baseline_feet[start:end], device=device)
                contact_valid_window = torch.tensor(constraint_valid[start:end], device=device)
                source_normal_valid_window = torch.tensor(
                    source_normal_valid[start:end], device=device,
                )
                target_vertex_window = torch.tensor(
                    target_vertex[start:end], dtype=torch.long, device=device,
                )
                remapped_anchor = torch.tensor(correspondence["anchor"][start:end], device=device)
                remapped_normal = torch.tensor(correspondence["normal"][start:end], device=device)
                source_contact_window = torch.tensor(window_contact, device=device)
                source_segment_window = torch.tensor(
                    source.contact_segment[start:end], dtype=torch.long, device=device,
                )
                correspondence_receipts: list[dict[str, Any]] = []
                correspondence_tier_converged: list[bool] = []

                def evaluate():
                    body_delta = torch.einsum("tc,cjd->tjd", basis, body_parameter)
                    left_delta = torch.einsum("tc,cjd->tjd", basis, left_parameter)
                    right_delta = torch.einsum("tc,cjd->tjd", basis, right_parameter)
                    root_delta = torch.einsum("tc,cd->td", basis, root_parameter)
                    translation_delta = torch.einsum("tc,cd->td", basis, translation_parameter)
                    current_body = body0.clone()
                    current_body[:, body_joints] = _compose(body0[:, body_joints], body_delta)
                    current_root = _compose(root0, root_delta)
                    current_left = _compose(left0, left_delta)
                    current_right = _compose(right0, right_delta)
                    current_translation = translation0 + translation_delta
                    output = model(
                        global_orient=current_root, body_pose=current_body.reshape(count, 63),
                        left_hand_pose=current_left.reshape(count, 45),
                        right_hand_pose=current_right.reshape(count, 45), betas=betas,
                        jaw_pose=zeros3, leye_pose=zeros3, reye_pose=zeros3,
                        expression=torch.zeros((count, model.num_expression_coeffs), device=device),
                        return_verts=True,
                    )
                    vertices = output.vertices * motion.scale + current_translation[:, None]
                    landmarks = _target_hand_landmarks(output.joints, output.vertices) * motion.scale
                    landmarks = landmarks + current_translation[:, None, None]
                    palm_world, palm_matrix_world = _torch_palm_frames(landmarks)
                    residuals: dict[str, Any] = {}
                    for object_index in range(objects):
                        rotation = object_rotation[:, object_index]
                        center_value = object_center[:, object_index]
                        palm_local = torch.einsum(
                            "tsj,tjk->tsk", palm_world - center_value[:, None], rotation,
                        )
                        palm_matrix_local = rotation.transpose(-1, -2)[:, None] @ palm_matrix_world
                        landmarks_local = torch.einsum(
                            "tslj,tjk->tslk", landmarks - center_value[:, None, None], rotation,
                        )
                        for side in range(2):
                            mask = source_contact_window[:, object_index, side]
                            label = "left" if side == 0 else "right"
                            if mask.any():
                                position_residual = (
                                    palm_local[:, side] - desired_position[:, object_index, side]
                                ) / float(settings["palm_position_tolerance_m"])
                                orientation_residual = _rotation_residual(
                                    palm_matrix_local[:, side], desired_matrix[:, object_index, side],
                                ) / float(np.deg2rad(settings["palm_orientation_tolerance_degrees"]))
                                residuals[f"palm_position_{label}"] = position_residual[mask]
                                residuals[f"palm_orientation_{label}"] = orientation_residual[mask]
                                finger_position = (
                                    landmarks_local[:, side, 2:] - target_reference[:, object_index, side, 2:]
                                ) / float(settings["source_finger_position_tolerance_m"])
                                target_chains = []
                                source_chains = []
                                target_bends = []
                                source_bends = []
                                for base in (2, 6, 10, 14, 18):
                                    chain = [base, base + 1, base + 2, base + 3]
                                    target_bones = torch.diff(landmarks_local[:, side, chain], dim=1)
                                    source_bones = torch.diff(target_reference[:, object_index, side, chain], dim=1)
                                    target_unit = torch.nn.functional.normalize(target_bones, dim=-1, eps=1e-8)
                                    source_unit = torch.nn.functional.normalize(source_bones, dim=-1, eps=1e-8)
                                    target_chains.append(target_unit)
                                    source_chains.append(source_unit)
                                    target_bends.append(stable_vector_angle(
                                        target_unit[:, :-1], target_unit[:, 1:],
                                    ))
                                    source_bends.append(stable_vector_angle(
                                        source_unit[:, :-1], source_unit[:, 1:],
                                    ))
                                # Preserve the finger axis. Each finger owns a
                                # separate augmented-Lagrangian state below;
                                # concatenation here previously allowed one
                                # failed finger to be hidden by the other four.
                                target_direction = torch.stack(target_chains, dim=1)
                                source_direction = torch.stack(source_chains, dim=1)
                                direction_angle = stable_vector_angle(
                                    target_direction, source_direction,
                                ) / float(np.deg2rad(
                                    settings["source_finger_direction_tolerance_degrees"]
                                ))
                                bend_difference = (
                                    torch.stack(target_bends, dim=1) -
                                    torch.stack(source_bends, dim=1)
                                ) / float(np.deg2rad(settings["source_finger_bend_tolerance_degrees"]))
                                target_opposition = torch.stack([
                                    torch.linalg.vector_norm(
                                        landmarks_local[:, side, 21] - landmarks_local[:, side, 2], dim=-1,
                                    ),
                                    torch.linalg.vector_norm(
                                        landmarks_local[:, side, 21] - landmarks_local[:, side, 5], dim=-1,
                                    ),
                                ], dim=-1)
                                source_opposition = torch.stack([
                                    torch.linalg.vector_norm(
                                        target_reference[:, object_index, side, 21] - target_reference[:, object_index, side, 2], dim=-1,
                                    ),
                                    torch.linalg.vector_norm(
                                        target_reference[:, object_index, side, 21] - target_reference[:, object_index, side, 5], dim=-1,
                                    ),
                                ], dim=-1)
                                opposition = (
                                    target_opposition - source_opposition
                                ) / float(settings["source_thumb_opposition_tolerance_m"])
                                # Physical quantities and individual fingers
                                # need independent multipliers and P95 gates.
                                # One bad joint landmark is only 5% of the old
                                # pooled position samples and was therefore
                                # invisible to a whole-hand P95 statistic.
                                masked_position = finger_position[mask]
                                masked_direction = direction_angle[mask]
                                masked_bend = bend_difference[mask]
                                for finger_index, finger_name in enumerate(FINGER_NAMES):
                                    landmark_slice = slice(
                                        4 * finger_index, 4 * (finger_index + 1),
                                    )
                                    residuals[
                                        f"finger_position_{label}_{finger_name}"
                                    ] = masked_position[:, landmark_slice].reshape(-1, 3)
                                    residuals[
                                        f"finger_direction_{label}_{finger_name}"
                                    ] = masked_direction[:, finger_index].reshape(-1)
                                    residuals[
                                        f"finger_bend_{label}_{finger_name}"
                                    ] = masked_bend[:, finger_index].reshape(-1)
                                residuals[f"finger_opposition_{label}"] = (
                                    opposition[mask].reshape(-1)
                                )
                            valid = contact_valid_window[:, object_index, side]
                            if valid.any():
                                ids = torch.clamp(target_vertex_window[:, object_index, side], min=0)
                                row = torch.arange(count, device=device)[:, None]
                                predicted = vertices[row, ids]
                                anchor = torch.einsum(
                                    "tki,tji->tkj", remapped_anchor[:, object_index, side], rotation,
                                ) + center_value[:, None]
                                normal = torch.einsum(
                                    "tki,tji->tkj", remapped_normal[:, object_index, side], rotation,
                                )
                                desired_surface = anchor + float(settings["contact_surface_offset_m"]) * normal
                                contact_gap = torch.linalg.vector_norm(
                                    predicted - desired_surface, dim=-1,
                                )
                                residuals[
                                    f"diagnostic_contact_gap_{object_index}_{label}"
                                ] = torch.where(
                                    valid, contact_gap,
                                    torch.full_like(contact_gap, float("nan")),
                                )
                                contact_vector = (
                                    predicted - desired_surface
                                ) / float(settings["contact_gap_tolerance_m"])
                                human_normal = _selected_vertex_normals(
                                    vertices, ids, face_table, face_mask,
                                )
                                opposition = -(human_normal * normal).sum(dim=-1)
                                minimum = float(np.cos(np.deg2rad(settings["contact_normal_tolerance_degrees"])))
                                normal_valid = valid & source_normal_valid_window[:, object_index, side]
                                normal_residual = normalized_normal_opposition_residual(
                                    opposition, minimum,
                                )
                                for segment in torch.unique(
                                    source_segment_window[:, object_index, side][valid]
                                ).tolist():
                                    segment_mask = (
                                        valid &
                                        (source_segment_window[:, object_index, side] == int(segment))
                                    )
                                    name = contact_segment_constraint_name(
                                        "contact", object_index, label, int(segment),
                                    )
                                    residuals[name] = contact_vector[segment_mask]
                                    segment_normal_mask = segment_mask & normal_valid
                                    if segment_normal_mask.any():
                                        normal_name = contact_segment_constraint_name(
                                            "normal", object_index, label, int(segment),
                                        )
                                        residuals[normal_name] = normal_residual[
                                            segment_normal_mask
                                        ]
                        local_collision = torch.einsum(
                            "tvi,tij->tvj",
                            vertices[:, collision_vertex_ids] - center_value[:, None], rotation,
                        )
                        sdf = _torch_grid_sample(
                            local_collision.reshape(-1, 3), grids[object_index], device,
                        ).reshape(count, -1)
                        residuals[f"collision_{object_index}"] = (
                            float(settings["penetration_margin_m"]) - sdf
                        ) / float(settings["collision_tolerance_m"])
                    if fallback and support_window.any():
                        foot_pairs = torch.tensor([[7, 10], [8, 11]], dtype=torch.long, device=device)
                        feet = output.joints[:, foot_pairs] * motion.scale + current_translation[:, None, None]
                        for side, label in enumerate(("left", "right")):
                            mask = support_window[:, side]
                            if mask.any():
                                position = (
                                    feet[:, side] - baseline_feet_window[:, side]
                                )[mask].reshape(int(mask.sum()), -1)
                                current_direction = torch.nn.functional.normalize(
                                    feet[:, side, 1] - feet[:, side, 0], dim=-1, eps=1e-8,
                                )
                                baseline_direction = torch.nn.functional.normalize(
                                    baseline_feet_window[:, side, 1] - baseline_feet_window[:, side, 0],
                                    dim=-1, eps=1e-8,
                                )
                                direction_angle = stable_vector_angle(
                                    current_direction, baseline_direction,
                                )[mask, None]
                                residuals[f"support_{label}"] = torch.cat([
                                    position / float(settings["support_foot_position_tolerance_m"]),
                                    direction_angle / float(np.deg2rad(
                                        settings["support_foot_orientation_tolerance_degrees"]
                                    )),
                                ], dim=-1)
                    pose_scale = float(np.deg2rad(settings["pose_prior_tolerance_degrees"]))
                    body_deviation = body_delta - body_prior_delta
                    left_deviation = left_delta - left_prior_delta
                    right_deviation = right_delta - right_prior_delta
                    root_deviation = root_delta - root_prior_delta
                    translation_deviation = translation_delta - translation_prior_delta
                    rotational_blocks = [body_deviation, left_deviation, right_deviation]
                    soft_terms = [
                        (body_deviation / pose_scale).square().mean(),
                        (left_deviation / pose_scale).square().mean(),
                        (right_deviation / pose_scale).square().mean(),
                    ]
                    if fallback:
                        soft_terms.extend([
                            (root_deviation / root_limit).square().mean(),
                            (translation_deviation / max(translation_limit, 1e-8)).square().mean(),
                        ])
                        rotational_blocks.append(root_deviation)
                    dt = 1.0 / float(motion.fps)
                    velocity_scale = float(np.deg2rad(
                        settings["source_velocity_tolerance_degrees_s"]
                    ))
                    acceleration_scale = float(np.deg2rad(
                        settings["source_acceleration_tolerance_degrees_s2"]
                    ))
                    for block in rotational_blocks:
                        if len(block) > 1:
                            soft_terms.append((
                                (block[1:] - block[:-1]) / (dt * velocity_scale)
                            ).square().mean())
                        if len(block) > 2:
                            soft_terms.append((
                                (block[2:] - 2.0 * block[1:-1] + block[:-2]) /
                                (dt * dt * acceleration_scale)
                            ).square().mean())
                    if fallback and len(translation_deviation) > 1:
                        soft_terms.append((
                            (translation_deviation[1:] - translation_deviation[:-1]) /
                            (dt * float(settings["root_translation_velocity_tolerance_m_s"]))
                        ).square().mean())
                    if fallback and len(translation_deviation) > 2:
                        soft_terms.append((
                            (translation_deviation[2:] - 2.0 * translation_deviation[1:-1] +
                             translation_deviation[:-2]) /
                            (dt * dt * float(
                                settings["root_translation_acceleration_tolerance_m_s2"]
                            ))
                        ).square().mean())
                    residuals["soft"] = torch.stack(soft_terms).mean()
                    residuals["state"] = (
                        body_delta, left_delta, right_delta, root_delta, translation_delta,
                    )
                    residuals["correspondence_vertices"] = vertices
                    residuals["correspondence_landmarks"] = landmarks
                    return residuals

                def contact_segment_diagnostics(values) -> dict[str, Any]:
                    gap = np.full(
                        constraint_valid[start:end].shape, np.nan, dtype=np.float32,
                    )
                    for object_index in range(objects):
                        for side, label in enumerate(("left", "right")):
                            name = f"diagnostic_contact_gap_{object_index}_{label}"
                            if name in values:
                                gap[:, object_index, side] = (
                                    values[name].detach().cpu().numpy()
                                )
                    return contact_segment_residual_summary(
                        gap,
                        constraint_valid[start:end],
                        source.contact_segment[start:end],
                        np.ones(count, dtype=bool),
                    )

                contact_names = tuple(
                    name
                    for label in ("left", "right")
                    for name in contact_segment_constraint_names(
                        "contact", source.contact_segment[start:end],
                        constraint_valid[start:end], label,
                    )
                )
                normal_names = tuple(
                    name
                    for label in ("left", "right")
                    for name in contact_segment_constraint_names(
                        "normal", source.contact_segment[start:end],
                        constraint_valid[start:end] & source_normal_valid[start:end], label,
                    )
                )
                hierarchy = [
                    [name for name in (
                        "palm_position_left", "palm_orientation_left",
                        "palm_position_right", "palm_orientation_right",
                    )],
                    list(contact_names),
                    [name for name in (*normal_names, *(
                        f"collision_{index}" for index in range(objects)
                    ))],
                    [
                        name
                        for label in ("left", "right")
                        for name in finger_constraint_names(label)
                    ],
                    [name for name in ("support_left", "support_right")],
                ]
                inequality_prefixes = ("normal_", "collision_")
                al = (
                    seed_al.clone() if seed_al is not None else
                    AugmentedLagrangian(
                        initial_penalty=float(al_settings["initial_penalty"]),
                        penalty_multiplier=float(al_settings["penalty_multiplier"]),
                        maximum_penalty=float(al_settings["maximum_penalty"]),
                        required_reduction=float(al_settings["required_reduction"]),
                    )
                )
                active_names: list[str] = []
                tier_receipts: list[dict[str, Any]] = []
                last_feasible_state = {
                    "body": body_parameter.detach().clone(),
                    "left": left_parameter.detach().clone(),
                    "right": right_parameter.detach().clone(),
                    "root": root_parameter.detach().clone(),
                    "translation": translation_parameter.detach().clone(),
                }
                last_feasible_tier = int(start_tier) - 1
                vertex_lock_after_tier = int(settings.get(
                    "surface_remap_vertex_lock_after_tier", 2,
                ))
                correspondence_active_tiers = tuple(settings.get(
                    "surface_remap_active_tiers", range(vertex_lock_after_tier),
                ))

                current_tier = -1

                def constraint_p95(name: str, value) -> float:
                    measured = torch.relu(value) if name.startswith(inequality_prefixes) else value
                    if value.ndim > 1 and value.shape[-1] in {2, 3, 4, 6}:
                        measured = torch.linalg.vector_norm(measured, dim=-1).reshape(-1)
                    else:
                        measured = measured.abs().reshape(-1)
                    return (
                        float(torch.quantile(measured, 0.95).detach().cpu())
                        if measured.numel() else 0.0
                    )

                def objective():
                    values = evaluate()
                    total = torch.zeros((), device=device)
                    for name in active_names:
                        if name in values:
                            total = total + al.term(
                                name, values[name], name.startswith(inequality_prefixes),
                            )
                    if current_tier >= 4:
                        total = total + values["soft"]
                    return total, values

                def remap_correspondence(values, tier_index: int, policy_outer: int):
                    nonlocal target_vertex_window, contact_valid_window
                    nonlocal remapped_anchor, remapped_normal
                    window_correspondence = {
                        "target_vertex": target_vertex_window.detach().cpu().numpy(),
                        "valid": contact_valid_window.detach().cpu().numpy(),
                        "anchor": remapped_anchor.detach().cpu().numpy(),
                        "face": correspondence["face"][start:end],
                        "barycentric": correspondence["barycentric"][start:end],
                        "geodesic": correspondence["geodesic"][start:end],
                        "normal": remapped_normal.detach().cpu().numpy(),
                    }
                    allow_vertex_switch, enforce_normal_feasibility = (
                        surface_correspondence_policy(
                            tier_index, vertex_lock_after_tier, policy_outer,
                            settings.get("surface_remap_vertex_switch_tiers"),
                            fallback=fallback_vertex_rebind_locked(
                                fallback, tier_index, start_tier,
                            ),
                        )
                    )
                    updated, change_fraction, change_breakdown = update_surface_correspondence_window(
                        config, source, meshes, model, correspondence_topologies,
                        object_translation, object_quaternion,
                        np.arange(start, end, dtype=np.int64),
                        values["correspondence_vertices"].detach().cpu().numpy(),
                        values["correspondence_landmarks"].detach().cpu().numpy(),
                        window_correspondence,
                        immutable_surface_vertices,
                        immutable_surface_landmarks,
                        target_reference_local[start:end],
                        reference_target_vertex[start:end],
                        normal_constraint_valid=source_normal_valid[start:end],
                        allow_vertex_switch=allow_vertex_switch,
                        enforce_normal_feasibility=enforce_normal_feasibility,
                    )
                    changed_constraint_entries = changed_contact_constraint_entries(
                        window_correspondence, updated,
                        source.contact_segment[start:end],
                        source_normal_valid[start:end],
                        float(settings["acceptance"]["object_translation_m"]),
                    )
                    dual_resets = (
                        al.reset_dual_entries(changed_constraint_entries)
                        if change_fraction > 0.0 else []
                    )
                    for name in (
                        "target_vertex", "valid", "anchor", "face",
                        "barycentric", "geodesic", "normal",
                    ):
                        correspondence[name][start:end] = updated[name]
                    target_vertex[start:end] = updated["target_vertex"]
                    constraint_valid[start:end] = updated["valid"]
                    coverage_failures = contact_constraint_coverage_failures(
                        contact, constraint_valid, source.contact_region,
                        pilot_frame_mask, minimum_points, minimum_regions,
                    )
                    if coverage_failures:
                        preview = coverage_failures[:8]
                        raise RuntimeError(
                            "Correspondence update removed immutable source-contact "
                            f"coverage; failed_frames={len(coverage_failures)} "
                            f"preview={preview}"
                        )
                    target_vertex_window = torch.tensor(
                        updated["target_vertex"], dtype=torch.long, device=device,
                    )
                    contact_valid_window = torch.tensor(updated["valid"], device=device)
                    remapped_anchor = torch.tensor(updated["anchor"], device=device)
                    remapped_normal = torch.tensor(updated["normal"], device=device)
                    return evaluate(), change_fraction, {
                        **change_breakdown,
                        "dual_resets": dual_resets,
                    }

                for tier_index, names in enumerate(hierarchy):
                    current_tier = tier_index
                    if tier_index < int(start_tier):
                        inherited_values = evaluate()
                        inherited_names = [
                            name for name in names if name in inherited_values
                        ]
                        active_names.extend(
                            name for name in inherited_names if name not in active_names
                        )
                        tier_receipts.append({
                            "tier": tier_index + 1,
                            "constraints": list(active_names),
                            "initial_p95": {
                                name: constraint_p95(name, inherited_values[name])
                                for name in active_names if name in inherited_values
                            },
                            "outer": [],
                            "converged": True,
                            "inherited_for_fallback": True,
                            "feasibility_projection": [],
                        })
                        correspondence_tier_converged.append(True)
                        continue
                    # Each newly activated level deforms the continuous hand
                    # surface.  Palm-only correspondence stability therefore
                    # cannot be carried into contact, normal, finger, or
                    # fallback/support optimization.  Re-establish the two
                    # stable alternating rounds independently for every tier.
                    # A seeded full-body fallback keeps correspondence for
                    # every completed evidence tier.  Its failed tier must,
                    # however, continue alternating the OBJ anchor inside the
                    # immutable 5 mm source patch: freezing the upper-body
                    # anchor here makes the added root/lower-body DoFs solve
                    # against stale geometry. Target vertex identity remains
                    # independently locked by the vertex-switch policy.
                    tier_correspondence_frozen = correspondence_is_frozen_for_tier(
                        tier_index, freeze_correspondence_through_tier,
                        vertex_lock_after_tier, correspondence_active_tiers,
                    )
                    correspondence_stable = initial_correspondence_stable_rounds(
                        int(settings["surface_remap_required_stable_rounds"]),
                        tier_correspondence_frozen,
                    )
                    correspondence_round = 0
                    initial_values = evaluate()
                    tier_entry_parameters = [
                        parameter.detach().clone() for parameter in parameters
                    ]
                    active_names.extend(name for name in names if name in initial_values)
                    # This is a new constrained problem, not an AL continuation
                    # of the previous tier: the feasible primal seed is kept,
                    # while all active constraints restart with balanced dual
                    # and penalty state.  In particular, palm/contact penalties
                    # accumulated before MANO activation must not be allowed to
                    # dominate the newly added finger constraints by 10^3--10^4.
                    al = al.restart_for_hierarchy_expansion()
                    receipt = {
                        "tier": tier_index + 1,
                        "constraints": list(active_names),
                        "initial_p95": {
                            name: constraint_p95(name, initial_values[name])
                            for name in active_names if name in initial_values
                        },
                        "outer": [],
                    }
                    if not active_names:
                        tier_receipts.append(receipt)
                        continue
                    # Establish the newly activated tier's discrete surface
                    # task from the previous tier's feasible pose *before*
                    # optimizing the new continuous residuals. Optimizing a
                    # normal tier against stale back-facing vertices first can
                    # destroy the already feasible palm/contact state before
                    # correspondence gets a chance to update.
                    preupdated = False
                    if (
                        not tier_correspondence_frozen and
                        (
                            surface_correspondence_policy(
                                tier_index, vertex_lock_after_tier, 0,
                                settings.get("surface_remap_vertex_switch_tiers"),
                                fallback=fallback_vertex_rebind_locked(
                                    fallback, tier_index, start_tier,
                                ),
                            )[0] or fallback
                        )
                    ):
                        initial_values, change_fraction, change_breakdown = remap_correspondence(
                            initial_values, tier_index, 0,
                        )
                        correspondence_round += 1
                        if change_fraction < float(settings["surface_remap_stable_fraction"]):
                            correspondence_stable += 1
                        else:
                            correspondence_stable = 0
                        correspondence_receipts.append({
                            "tier": int(tier_index + 1),
                            "round": int(correspondence_round),
                            "phase": "pre_continuous",
                            "change_fraction": float(change_fraction),
                            **change_breakdown,
                            "stable_rounds": int(correspondence_stable),
                        })
                        print(json.dumps({
                            "event": "v4_correspondence_preupdate",
                            "window": [int(start), int(end)],
                            "fallback": bool(fallback),
                            "tier": int(tier_index + 1),
                            "change_fraction": float(change_fraction),
                            **change_breakdown,
                            "p95_before_continuous": {
                                name: constraint_p95(name, initial_values[name])
                                for name in active_names if name in initial_values
                            },
                        }, sort_keys=True), flush=True)
                        preupdated = True
                    stable = 0
                    best_gate_score = float("inf")
                    best_gate_state: list[Any] | None = None
                    for outer in range(int(al_settings["maximum_outer_iterations"])):
                        optimizer = torch.optim.Adam([
                            {"params": [body_parameter], "lr": float(al_settings["adam_learning_rate_body"])},
                            {"params": [left_parameter, right_parameter], "lr": float(al_settings["adam_learning_rate_hand"])},
                            *([{"params": [root_parameter, translation_parameter],
                                "lr": float(al_settings["adam_learning_rate_body"])}] if fallback else []),
                        ])
                        for _ in range(int(al_settings["adam_iterations"])):
                            optimizer.zero_grad(set_to_none=True)
                            loss, _ = objective()
                            if not torch.isfinite(loss):
                                raise FloatingPointError("Non-finite v4 augmented-Lagrangian objective")
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                            optimizer.step()
                            joint_limits = np.asarray([
                                upper_limits[np.flatnonzero(upper == joint)[0]]
                                if joint in upper else lower_limits[np.flatnonzero(lower == joint)[0]]
                                for joint in body_joints
                            ], dtype=np.float32)
                            clamp_parameter(body_parameter, joint_limits)
                            clamp_parameter(left_parameter, hand_limit)
                            clamp_parameter(right_parameter, hand_limit)
                            if fallback:
                                clamp_parameter(root_parameter, root_limit)
                                clamp_parameter(translation_parameter, translation_limit)
                        lbfgs = torch.optim.LBFGS(
                            parameters, lr=0.5, max_iter=int(al_settings["lbfgs_iterations"]),
                            history_size=int(al_settings["lbfgs_history"]),
                            line_search_fn="strong_wolfe",
                        )

                        def closure():
                            lbfgs.zero_grad(set_to_none=True)
                            loss, _ = objective()
                            loss.backward()
                            # PyTorch LBFGS flattens gradients with ``view``;
                            # indexed SMPL-X chains can produce strided leaf
                            # gradients, so make the optimizer contract
                            # explicit instead of relying on incidental layout.
                            for parameter in parameters:
                                if parameter.grad is not None and not parameter.grad.is_contiguous():
                                    parameter.grad = parameter.grad.contiguous()
                            return loss

                        lbfgs.step(closure)
                        joint_limits = np.asarray([
                            upper_limits[np.flatnonzero(upper == joint)[0]]
                            if joint in upper else lower_limits[np.flatnonzero(lower == joint)[0]]
                            for joint in body_joints
                        ], dtype=np.float32)
                        clamp_parameter(body_parameter, joint_limits)
                        clamp_parameter(left_parameter, hand_limit)
                        clamp_parameter(right_parameter, hand_limit)
                        if fallback:
                            clamp_parameter(root_parameter, root_limit)
                            clamp_parameter(translation_parameter, translation_limit)
                        values = evaluate()
                        if (
                            not tier_correspondence_frozen and
                            correspondence_stable < int(settings["surface_remap_required_stable_rounds"])
                            and correspondence_round < correspondence_update_budget(
                                int(settings["surface_remap_maximum_outer_rounds"]),
                                preupdated,
                            )
                        ):
                            values, change_fraction, change_breakdown = remap_correspondence(
                                values, tier_index, outer + int(preupdated),
                            )
                            correspondence_round += 1
                            if change_fraction < float(settings["surface_remap_stable_fraction"]):
                                correspondence_stable += 1
                            else:
                                correspondence_stable = 0
                            correspondence_receipts.append({
                                "tier": int(tier_index + 1),
                                "round": int(correspondence_round),
                                "phase": "post_continuous",
                                "change_fraction": float(change_fraction),
                                **change_breakdown,
                                "stable_rounds": int(correspondence_stable),
                            })
                        updates = {
                            name: al.update(name, values[name]) for name in active_names if name in values
                        }
                        correspondence_passed = (
                            correspondence_stable >= int(settings["surface_remap_required_stable_rounds"])
                        )
                        update_p95 = {
                            name: float(details["p95"])
                            for name, details in updates.items()
                        }
                        if correspondence_passed:
                            gate_score = normalized_gate_score(update_p95)
                            if gate_score < best_gate_score:
                                best_gate_score = gate_score
                                best_gate_state = [
                                    parameter.detach().clone() for parameter in parameters
                                ]
                        passed = (
                            bool(updates) and all(item["passed"] for item in updates.values())
                            and correspondence_passed
                        )
                        stable = stable + 1 if passed else 0
                        receipt["outer"].append({
                            "iteration": outer + 1,
                            "constraints": updates,
                            "passed": passed,
                            "correspondence_passed": bool(correspondence_passed),
                        })
                        print(json.dumps({
                            "event": "v4_constraint_iteration",
                            "window": [int(start), int(end)],
                            "fallback": bool(fallback),
                            "tier": int(tier_index + 1),
                            "outer": int(outer + 1),
                            "passed": bool(passed),
                            "correspondence_round": int(correspondence_round),
                            "correspondence_stable_rounds": int(correspondence_stable),
                            "correspondence_change_fraction": (
                                float(correspondence_receipts[-1]["change_fraction"])
                                if correspondence_receipts else 0.0
                            ),
                            "correspondence_vertex_change_fraction": (
                                float(correspondence_receipts[-1]["vertex_change_fraction"])
                                if correspondence_receipts else 0.0
                            ),
                            "correspondence_anchor_change_fraction": (
                                float(correspondence_receipts[-1]["anchor_change_fraction"])
                                if correspondence_receipts else 0.0
                            ),
                            "p95": {
                                name: float(details["p95"])
                                for name, details in updates.items()
                            },
                            "rho": {
                                name: float(details["penalty"])
                                for name, details in updates.items()
                            },
                            "parameter_limits": {
                                "left_hand": parameter_limit_diagnostics(
                                    left_parameter, hand_limit,
                                ),
                                "right_hand": parameter_limit_diagnostics(
                                    right_parameter, hand_limit,
                                ),
                            },
                            "finger_position_by_finger": {
                                label: {
                                    finger: constraint_p95(
                                        f"finger_position_{label}_{finger}",
                                        values[f"finger_position_{label}_{finger}"],
                                    )
                                    for finger in FINGER_NAMES
                                    if f"finger_position_{label}_{finger}" in values
                                }
                                for label in ("left", "right")
                                if any(
                                    f"finger_position_{label}_{finger}" in values
                                    for finger in FINGER_NAMES
                                )
                            },
                            "contact_gap_by_source_segment": (
                                contact_segment_diagnostics(values)
                            ),
                        }, sort_keys=True), flush=True)
                        if stable >= int(al_settings["stable_outer_iterations"]):
                            break
                    required_correspondence_stable = int(
                        settings["surface_remap_required_stable_rounds"]
                    )
                    if fixed_pose_correspondence_confirmation_allowed(
                        correspondence_stable,
                        required_correspondence_stable,
                        tier_correspondence_frozen,
                    ):
                        # The sixth AL round may be the first discrete round
                        # below the 1% gate. Confirm the exact same fixed human
                        # pose once more instead of spending a seventh
                        # continuous optimization round. Any changed mapping
                        # is installed first, then all physical constraints are
                        # re-evaluated below before feasibility can pass.
                        confirmation_values = evaluate()
                        (
                            confirmation_values,
                            confirmation_change,
                            confirmation_breakdown,
                        ) = remap_correspondence(
                            confirmation_values, tier_index,
                            correspondence_round + 1,
                        )
                        correspondence_round += 1
                        if confirmation_change < float(
                            settings["surface_remap_stable_fraction"]
                        ):
                            correspondence_stable += 1
                        else:
                            correspondence_stable = 0
                        correspondence_receipts.append({
                            "tier": int(tier_index + 1),
                            "round": int(correspondence_round),
                            "phase": "fixed_pose_confirmation",
                            "change_fraction": float(confirmation_change),
                            **confirmation_breakdown,
                            "stable_rounds": int(correspondence_stable),
                        })
                        print(json.dumps({
                            "event": "v4_correspondence_fixed_pose_confirmation",
                            "window": [int(start), int(end)],
                            "fallback": bool(fallback),
                            "tier": int(tier_index + 1),
                            "change_fraction": float(confirmation_change),
                            **confirmation_breakdown,
                            "stable_rounds": int(correspondence_stable),
                            "p95": {
                                name: constraint_p95(
                                    name, confirmation_values[name],
                                )
                                for name in active_names
                                if name in confirmation_values
                            },
                        }, sort_keys=True), flush=True)
                    required_stable = int(al_settings["stable_outer_iterations"])
                    polish_receipts: list[dict[str, Any]] = []
                    if stable < required_stable:
                        current_values = evaluate()
                        current_p95 = {
                            name: constraint_p95(name, current_values[name])
                            for name in active_names if name in current_values
                        }
                        current_gate_score = normalized_gate_score(current_p95)
                        if (
                            best_gate_state is not None and
                            best_gate_score + 1e-12 < current_gate_score
                        ):
                            with torch.no_grad():
                                for parameter, checkpoint in zip(
                                    parameters, best_gate_state,
                                ):
                                    parameter.copy_(checkpoint)
                            current_values = evaluate()
                            current_p95 = {
                                name: constraint_p95(name, current_values[name])
                                for name in active_names if name in current_values
                            }
                            if best_gate_score == 0.0:
                                stable = max(stable, 1)
                            polish_receipts.append({
                                "pass": "best_stable_correspondence_restore",
                                "p95": current_p95,
                                "passed": best_gate_score == 0.0,
                                "previous_score": current_gate_score,
                                "restored_score": normalized_gate_score(current_p95),
                            })
                            print(json.dumps({
                                "event": "v4_best_gate_restore",
                                "window": [int(start), int(end)],
                                "fallback": bool(fallback),
                                "tier": int(tier_index + 1),
                                "p95": current_p95,
                                "previous_score": current_gate_score,
                                "restored_score": normalized_gate_score(current_p95),
                            }, sort_keys=True), flush=True)
                        if fixed_state_confirmation_allowed(
                            current_p95, stable, required_stable,
                            correspondence_stable,
                            int(settings["surface_remap_required_stable_rounds"]),
                        ):
                            # ``evaluate`` is a second deterministic check of
                            # exactly the same frozen-correspondence state. Do
                            # not run another optimizer step after every hard
                            # gate has passed: that can only perturb a feasible
                            # boundary solution while trying to satisfy the
                            # bookkeeping requirement for consecutive checks.
                            stable = required_stable
                            polish_receipts.append({
                                "pass": "fixed_state_confirmation",
                                "p95": current_p95,
                                "passed": True,
                            })
                            print(json.dumps({
                                "event": "v4_fixed_state_confirmation",
                                "window": [int(start), int(end)],
                                "fallback": bool(fallback),
                                "tier": int(tier_index + 1),
                                "p95": current_p95,
                                "passed": True,
                            }, sort_keys=True), flush=True)
                        elif feasibility_projection_allowed(
                            max(current_p95.values(), default=float("inf")),
                            correspondence_stable,
                            int(settings["surface_remap_required_stable_rounds"]),
                        ):
                            projection_stable = 0
                            projection_score = normalized_gate_score(current_p95)
                            projection_reference_p95 = dict(current_p95)
                            for projection_index in range(int(settings.get(
                                "feasibility_projection_passes", 2,
                            ))):
                                if projection_stable:
                                    confirmed_values = evaluate()
                                    confirmed_p95 = {
                                        name: constraint_p95(name, confirmed_values[name])
                                        for name in active_names if name in confirmed_values
                                    }
                                    confirmed_passed = bool(confirmed_p95) and all(
                                        value <= 1.0 for value in confirmed_p95.values()
                                    )
                                    projection_stable = (
                                        projection_stable + 1 if confirmed_passed else 0
                                    )
                                    polish_receipts.append({
                                        "pass": projection_index + 1,
                                        "p95": confirmed_p95,
                                        "passed": confirmed_passed,
                                        "accepted": True,
                                        "fixed_state_confirmation": True,
                                    })
                                    print(json.dumps({
                                        "event": "v4_feasibility_projection_confirmation",
                                        "window": [int(start), int(end)],
                                        "fallback": bool(fallback),
                                        "tier": int(tier_index + 1),
                                        "pass": int(projection_index + 1),
                                        "p95": confirmed_p95,
                                        "passed": confirmed_passed,
                                    }, sort_keys=True), flush=True)
                                    if projection_stable >= required_stable:
                                        break
                                    continue
                                parameter_checkpoint = [
                                    parameter.detach().clone() for parameter in parameters
                                ]
                                dense_used = any(
                                    feasibility_projection_uses_dense_tail(
                                        projection_reference_p95.get(name, float("inf")),
                                        float(settings.get(
                                            "feasibility_projection_tail_activation", 1.01,
                                        )),
                                    )
                                    for name in active_names
                                    if name in current_values
                                )
                                projection = torch.optim.LBFGS(
                                    parameters, lr=float(settings.get(
                                        "feasibility_projection_learning_rate", 0.10,
                                    )),
                                    max_iter=int(al_settings["lbfgs_iterations"]),
                                    history_size=int(al_settings["lbfgs_history"]),
                                    line_search_fn="strong_wolfe",
                                )

                                def projection_closure():
                                    projection.zero_grad(set_to_none=True)
                                    values = evaluate()
                                    violations = []
                                    for name in active_names:
                                        if name not in values:
                                            continue
                                        inequality = name.startswith(inequality_prefixes)
                                        quantile = float(settings.get(
                                            "feasibility_projection_quantile", 0.95,
                                        ))
                                        if feasibility_projection_uses_dense_tail(
                                            projection_reference_p95.get(name, float("inf")),
                                            float(settings.get(
                                                "feasibility_projection_tail_activation", 1.01,
                                            )),
                                        ):
                                            violations.append(feasibility_projection_loss(
                                                values[name], inequality, quantile,
                                                float(settings.get(
                                                    "feasibility_projection_tail_fraction", 0.10,
                                                )), 1.0,
                                            ))
                                        else:
                                            violations.append(quantile_constraint_violation_loss(
                                                values[name], inequality, quantile, 1.0,
                                            ))
                                    loss = torch.stack(violations).sum()
                                    loss.backward()
                                    for parameter in parameters:
                                        if parameter.grad is not None and not parameter.grad.is_contiguous():
                                            parameter.grad = parameter.grad.contiguous()
                                    return loss

                                projection.step(projection_closure)
                                joint_limits = np.asarray([
                                    upper_limits[np.flatnonzero(upper == joint)[0]]
                                    if joint in upper else lower_limits[np.flatnonzero(lower == joint)[0]]
                                    for joint in body_joints
                                ], dtype=np.float32)
                                clamp_parameter(body_parameter, joint_limits)
                                clamp_parameter(left_parameter, hand_limit)
                                clamp_parameter(right_parameter, hand_limit)
                                if fallback:
                                    clamp_parameter(root_parameter, root_limit)
                                    clamp_parameter(translation_parameter, translation_limit)
                                projected_values = evaluate()
                                projected_p95 = {
                                    name: constraint_p95(name, projected_values[name])
                                    for name in active_names if name in projected_values
                                }
                                projected_score = normalized_gate_score(projected_p95)
                                accepted = bool(
                                    np.isfinite(projected_score) and
                                    projected_score < projection_score
                                )
                                projection_mode = "dense_tail" if dense_used else "exact_p95"
                                if exact_projection_retry_required(dense_used, accepted):
                                    with torch.no_grad():
                                        for parameter, checkpoint in zip(
                                            parameters, parameter_checkpoint,
                                        ):
                                            parameter.copy_(checkpoint)
                                    exact_projection = torch.optim.LBFGS(
                                        parameters, lr=float(settings.get(
                                            "feasibility_projection_learning_rate", 0.10,
                                        )),
                                        max_iter=int(al_settings["lbfgs_iterations"]),
                                        history_size=int(al_settings["lbfgs_history"]),
                                        line_search_fn="strong_wolfe",
                                    )

                                    def exact_projection_closure():
                                        exact_projection.zero_grad(set_to_none=True)
                                        values = evaluate()
                                        quantile = float(settings.get(
                                            "feasibility_projection_quantile", 0.95,
                                        ))
                                        violations = [
                                            quantile_constraint_violation_loss(
                                                values[name],
                                                name.startswith(inequality_prefixes),
                                                quantile, 1.0,
                                            )
                                            for name in active_names if name in values
                                        ]
                                        loss = torch.stack(violations).sum()
                                        loss.backward()
                                        for parameter in parameters:
                                            if (
                                                parameter.grad is not None and
                                                not parameter.grad.is_contiguous()
                                            ):
                                                parameter.grad = parameter.grad.contiguous()
                                        return loss

                                    exact_projection.step(exact_projection_closure)
                                    clamp_parameter(body_parameter, joint_limits)
                                    clamp_parameter(left_parameter, hand_limit)
                                    clamp_parameter(right_parameter, hand_limit)
                                    if fallback:
                                        clamp_parameter(root_parameter, root_limit)
                                        clamp_parameter(translation_parameter, translation_limit)
                                    exact_values = evaluate()
                                    exact_p95 = {
                                        name: constraint_p95(name, exact_values[name])
                                        for name in active_names if name in exact_values
                                    }
                                    exact_score = normalized_gate_score(exact_p95)
                                    if (
                                        np.isfinite(exact_score) and
                                        exact_score < projection_score
                                    ):
                                        projected_values = exact_values
                                        projected_p95 = exact_p95
                                        projected_score = exact_score
                                        accepted = True
                                        projection_mode = "exact_p95_retry"
                                if not accepted:
                                    with torch.no_grad():
                                        for parameter, checkpoint in zip(
                                            parameters, parameter_checkpoint,
                                        ):
                                            parameter.copy_(checkpoint)
                                    projected_values = evaluate()
                                    projected_p95 = {
                                        name: constraint_p95(name, projected_values[name])
                                        for name in active_names if name in projected_values
                                    }
                                else:
                                    projection_score = projected_score
                                    projection_reference_p95 = dict(projected_p95)
                                projected_passed = bool(projected_p95) and all(
                                    value <= 1.0 for value in projected_p95.values()
                                )
                                projection_stable = projection_stable + 1 if projected_passed else 0
                                polish_receipts.append({
                                    "pass": projection_index + 1,
                                    "p95": projected_p95,
                                    "passed": projected_passed,
                                    "accepted": accepted,
                                    "mode": projection_mode,
                                })
                                print(json.dumps({
                                    "event": "v4_feasibility_projection",
                                    "window": [int(start), int(end)],
                                    "fallback": bool(fallback),
                                    "tier": int(tier_index + 1),
                                    "pass": int(projection_index + 1),
                                    "p95": projected_p95,
                                    "passed": projected_passed,
                                    "accepted": accepted,
                                    "mode": projection_mode,
                                }, sort_keys=True), flush=True)
                            if projection_stable < required_stable:
                                # A summed projection can hide an independently
                                # controllable residual behind the conditioning
                                # of many already-near-feasible constraints.
                                # Project the current worst gate alone, but
                                # accept it only under a componentwise monotone
                                # filter over *all* active hard constraints.
                                for targeted_index in range(int(settings.get(
                                    "targeted_feasibility_projection_passes", 6,
                                ))):
                                    targeted_values = evaluate()
                                    targeted_before = {
                                        name: constraint_p95(
                                            name, targeted_values[name],
                                        )
                                        for name in active_names
                                        if name in targeted_values
                                    }
                                    failed_names = [
                                        name for name, value in targeted_before.items()
                                        if value > 1.0
                                    ]
                                    if not failed_names:
                                        projection_stable = required_stable
                                        break
                                    target_name = max(
                                        failed_names,
                                        key=lambda name: targeted_before[name],
                                    )
                                    parameter_checkpoint = [
                                        parameter.detach().clone()
                                        for parameter in parameters
                                    ]
                                    targeted_optimizer = torch.optim.LBFGS(
                                        parameters,
                                        lr=float(settings.get(
                                            "feasibility_projection_learning_rate", 0.10,
                                        )),
                                        max_iter=int(al_settings["lbfgs_iterations"]),
                                        history_size=int(al_settings["lbfgs_history"]),
                                        line_search_fn="strong_wolfe",
                                    )

                                    def targeted_closure():
                                        targeted_optimizer.zero_grad(set_to_none=True)
                                        values = evaluate()
                                        loss = feasibility_projection_loss(
                                            values[target_name],
                                            target_name.startswith(inequality_prefixes),
                                            float(settings.get(
                                                "feasibility_projection_quantile", 0.95,
                                            )),
                                            float(settings.get(
                                                "feasibility_projection_tail_fraction", 0.10,
                                            )),
                                            1.0,
                                        )
                                        loss.backward()
                                        for parameter in parameters:
                                            if (
                                                parameter.grad is not None and
                                                not parameter.grad.is_contiguous()
                                            ):
                                                parameter.grad = parameter.grad.contiguous()
                                        return loss

                                    targeted_optimizer.step(targeted_closure)
                                    clamp_parameter(body_parameter, joint_limits)
                                    clamp_parameter(left_parameter, hand_limit)
                                    clamp_parameter(right_parameter, hand_limit)
                                    if fallback:
                                        clamp_parameter(root_parameter, root_limit)
                                        clamp_parameter(translation_parameter, translation_limit)
                                    proposed_parameters = [
                                        parameter.detach().clone()
                                        for parameter in parameters
                                    ]
                                    targeted_after_values = evaluate()
                                    targeted_after = {
                                        name: constraint_p95(
                                            name, targeted_after_values[name],
                                        )
                                        for name in active_names
                                        if name in targeted_after_values
                                    }
                                    accepted = componentwise_gate_improvement_allowed(
                                        targeted_before, targeted_after,
                                    )
                                    accepted_fraction = 1.0 if accepted else 0.0
                                    if not accepted:
                                        # The single-constraint L-BFGS line
                                        # search only sees its target. Backtrack
                                        # the resulting parameter displacement
                                        # against the complete hard-gate vector
                                        # instead of discarding a useful
                                        # direction merely because its full
                                        # step perturbs another boundary gate.
                                        for fraction in minimum_change_backtracking_fractions(14)[1:]:
                                            with torch.no_grad():
                                                for parameter, checkpoint, proposed in zip(
                                                    parameters, parameter_checkpoint,
                                                    proposed_parameters,
                                                ):
                                                    parameter.copy_(
                                                        checkpoint + float(fraction) *
                                                        (proposed - checkpoint)
                                                    )
                                            backtracked_values = evaluate()
                                            backtracked_p95 = {
                                                name: constraint_p95(
                                                    name, backtracked_values[name],
                                                )
                                                for name in active_names
                                                if name in backtracked_values
                                            }
                                            if componentwise_gate_improvement_allowed(
                                                targeted_before, backtracked_p95,
                                            ):
                                                accepted = True
                                                accepted_fraction = float(fraction)
                                                targeted_after = backtracked_p95
                                                break
                                        if not accepted:
                                            with torch.no_grad():
                                                for parameter, checkpoint in zip(
                                                    parameters, parameter_checkpoint,
                                                ):
                                                    parameter.copy_(checkpoint)
                                            targeted_after = dict(targeted_before)
                                    targeted_passed = bool(targeted_after) and all(
                                        value <= 1.0
                                        for value in targeted_after.values()
                                    )
                                    projection_stable = (
                                        projection_stable + 1
                                        if targeted_passed else 0
                                    )
                                    polish_receipts.append({
                                        "pass": targeted_index + 1,
                                        "target": target_name,
                                        "p95": targeted_after,
                                        "passed": targeted_passed,
                                        "accepted": accepted,
                                        "accepted_fraction": accepted_fraction,
                                        "mode": "componentwise_targeted",
                                    })
                                    print(json.dumps({
                                        "event": "v4_targeted_feasibility_projection",
                                        "window": [int(start), int(end)],
                                        "fallback": bool(fallback),
                                        "tier": int(tier_index + 1),
                                        "pass": int(targeted_index + 1),
                                        "target": target_name,
                                        "p95": targeted_after,
                                        "passed": targeted_passed,
                                        "accepted": accepted,
                                        "accepted_fraction": accepted_fraction,
                                    }, sort_keys=True), flush=True)
                                    if projection_stable >= required_stable:
                                        break
                            if projection_stable >= required_stable:
                                stable = required_stable
                    receipt["feasibility_projection"] = polish_receipts
                    receipt["converged"] = stable >= int(al_settings["stable_outer_iterations"])
                    if not receipt["converged"]:
                        diagnostic_values = evaluate()
                        diagnostic_p95 = {
                            name: constraint_p95(name, diagnostic_values[name])
                            for name in active_names if name in diagnostic_values
                        }
                        gradient_diagnostics = constraint_gradient_diagnostics(
                            diagnostic_values, diagnostic_p95, parameters,
                            ["body", "left_hand", "right_hand"] + (
                                ["root", "translation"] if fallback else []
                            ),
                            inequality_prefixes,
                        )
                        receipt["failed_constraint_gradients"] = gradient_diagnostics
                        print(json.dumps({
                            "event": "v4_failed_constraint_gradients",
                            "window": [int(start), int(end)],
                            "fallback": bool(fallback),
                            "tier": int(tier_index + 1),
                            **gradient_diagnostics,
                        }, sort_keys=True), flush=True)
                    minimum_change_receipts: list[dict[str, Any]] = []
                    if receipt["converged"]:
                        # Hard gates have lexical priority. Only now move each
                        # parameter block back toward the state that entered
                        # this tier, accepting the largest deterministic step
                        # whose exact active-constraint P95 values all remain
                        # <= 1. This removes nullspace drift without putting a
                        # pose-prior weight in competition with contact.
                        group_names = (
                            ["body", "left_hand", "right_hand"] +
                            (["root", "translation"] if fallback else [])
                        )
                        for parameter_index in (
                            [1, 2, 0] + ([3, 4] if fallback else [])
                        ):
                            parameter = parameters[parameter_index]
                            target = tier_entry_parameters[parameter_index]
                            origin = parameter.detach().clone()
                            accepted_fraction = 0.0
                            current_projection_values = evaluate()
                            accepted_p95 = {
                                name: constraint_p95(
                                    name, current_projection_values[name],
                                )
                                for name in active_names
                                if name in current_projection_values
                            }
                            if not torch.equal(origin, target):
                                for fraction in minimum_change_backtracking_fractions():
                                    with torch.no_grad():
                                        parameter.copy_(
                                            origin + float(fraction) * (target - origin)
                                        )
                                    trial_values = evaluate()
                                    trial_p95 = {
                                        name: constraint_p95(name, trial_values[name])
                                        for name in active_names if name in trial_values
                                    }
                                    if trial_p95 and all(
                                        value <= 1.0 for value in trial_p95.values()
                                    ):
                                        accepted_fraction = float(fraction)
                                        accepted_p95 = trial_p95
                                        break
                                if accepted_fraction == 0.0:
                                    with torch.no_grad():
                                        parameter.copy_(origin)
                            minimum_change_receipts.append({
                                "group": group_names[parameter_index],
                                "fraction": accepted_fraction,
                                "maximum_p95": max(accepted_p95.values(), default=0.0),
                            })
                        receipt["minimum_change_projection"] = minimum_change_receipts
                        print(json.dumps({
                            "event": "v4_minimum_change_projection",
                            "window": [int(start), int(end)],
                            "fallback": bool(fallback),
                            "tier": int(tier_index + 1),
                            "groups": minimum_change_receipts,
                        }, sort_keys=True), flush=True)
                    correspondence_tier_converged.append(
                        correspondence_stable >= int(settings["surface_remap_required_stable_rounds"])
                    )
                    tier_receipts.append(receipt)
                    if receipt["converged"]:
                        last_feasible_state = {
                            "body": body_parameter.detach().clone(),
                            "left": left_parameter.detach().clone(),
                            "right": right_parameter.detach().clone(),
                            "root": root_parameter.detach().clone(),
                            "translation": translation_parameter.detach().clone(),
                        }
                        last_feasible_tier = int(tier_index)
                    if not receipt["converged"] and tier_index < 4:
                        break
                final = evaluate()
                return {
                    "fallback": fallback,
                    "tiers": tier_receipts,
                    "constraints": {
                        name: {
                            "p95": constraint_p95(name, value),
                            "penalty": float(al.states[name].penalty) if name in al.states else 0.0,
                            "multiplier_norm": float(torch.linalg.vector_norm(
                                al.states[name].multiplier,
                            ).detach().cpu()) if name in al.states else 0.0,
                        }
                        for name, value in final.items()
                        if not name.startswith("diagnostic_") and name not in {
                            "soft", "state", "correspondence_vertices",
                            "correspondence_landmarks",
                        }
                    },
                    "correspondence": correspondence_receipts,
                    "correspondence_converged": bool(
                        correspondence_tier_converged and all(correspondence_tier_converged)
                    ),
                    "al": al,
                    "body_joints": body_joints,
                    "active_control": active_control,
                }, {
                    "body": body_parameter.detach().cpu().numpy(),
                    "left": left_parameter.detach().cpu().numpy(),
                    "right": right_parameter.detach().cpu().numpy(),
                    "root": root_parameter.detach().cpu().numpy(),
                    "translation": translation_parameter.detach().cpu().numpy(),
                    "checkpoint_body": last_feasible_state["body"].cpu().numpy(),
                    "checkpoint_left": last_feasible_state["left"].cpu().numpy(),
                    "checkpoint_right": last_feasible_state["right"].cpu().numpy(),
                    "checkpoint_root": last_feasible_state["root"].cpu().numpy(),
                    "checkpoint_translation": last_feasible_state["translation"].cpu().numpy(),
                    "checkpoint_tier": int(last_feasible_tier),
                }

            fallback_reasons = ["task_jacobian_rank"] if initial_fallback else []
            report, values = optimize(initial_fallback)
            if not initial_fallback:
                palm_after_two = max((
                    details["p95"] for name, details in report["constraints"].items()
                    if name.startswith("palm_")
                ), default=float("inf"))
                failed_tier = next((
                    tier for tier in report["tiers"] if not tier.get("converged", False)
                ), None)
                improvement = 1.0
                if failed_tier is not None:
                    names = list(failed_tier.get("initial_p95", {}))
                    initial_p95 = max(
                        (failed_tier["initial_p95"][name] for name in names),
                        default=float("inf"),
                    )
                    second_index = min(1, len(failed_tier.get("outer", [])) - 1)
                    after_two_residual = (
                        max((
                            failed_tier["outer"][second_index]["constraints"][name]["p95"]
                            for name in names
                            if name in failed_tier["outer"][second_index]["constraints"]
                        ), default=float("inf"))
                        if second_index >= 0 else float("inf")
                    )
                    improvement = (
                        (initial_p95 - after_two_residual) / max(initial_p95, 1e-8)
                        if np.isfinite(initial_p95) and np.isfinite(after_two_residual) else 0.0
                    )
                fallback_needed, fallback_reasons = full_body_fallback_required(
                    rank_failure_fraction, palm_after_two, improvement,
                    float(settings["jacobian_rank_failure_fraction"]), 1.0,
                    float(settings["fallback_residual_improvement"]),
                    hierarchy_failed=failed_tier is not None,
                )
                if fallback_needed:
                    # Expand from the solved upper-body/MANO state.  Starting
                    # fallback from zero discards the feasible palm/contact
                    # progress and turns the bounded root into a substitute
                    # for the arm chain rather than a reachability aid.
                    seed_joints = np.asarray(report["body_joints"], dtype=np.int64)
                    final_p95 = tier_constraint_p95(
                        report["constraints"], failed_tier,
                    )
                    use_failed_tier = failed_tier_is_safe_fallback_seed(
                        final_p95,
                        list((failed_tier or {}).get("initial_p95", {}).values()),
                        float(settings.get("fallback_seed_maximum_normalized_p95", 2.0)),
                        float(settings.get("fallback_seed_minimum_improvement", 0.10)),
                    )
                    seed_sources = fallback_seed_sources(use_failed_tier)
                    body_seed_prefix = (
                        "" if seed_sources["body"] == "failed_tier" else "checkpoint_"
                    )
                    body_coeff[active_control[:, None], seed_joints[None, :]] = values[
                        f"{body_seed_prefix}body"
                    ]
                    # The failed contact/normal tier has not constrained
                    # dynamic finger motion yet. Its MANO result is therefore
                    # nullspace drift, not progress. Preserve the last
                    # feasible hand checkpoint even when the near-feasible
                    # failed body state is useful to the full-body fallback.
                    left_coeff[active_control] = values["checkpoint_left"]
                    right_coeff[active_control] = values["checkpoint_right"]
                    print(json.dumps({
                        "event": "v4_fallback_seed",
                        "window": [int(start), int(end)],
                        "source": (
                            "mixed_failed_body_last_feasible_hands"
                            if use_failed_tier else "last_feasible_tier"
                        ),
                        "body_source": seed_sources["body"],
                        "hand_source": seed_sources["hands"],
                        "maximum_normalized_p95": max(final_p95, default=float("inf")),
                        "resume_tier": int(len(report["tiers"])),
                    }, sort_keys=True), flush=True)
                    failed_tier_index = len(report["tiers"]) - 1
                    # The fallback seed is deliberately hybrid: failed-tier
                    # body progress plus the last feasible hands. Reusing the
                    # failed iterate's lambda/rho here made the right hand
                    # diverge (rho had already reached 4096 at a different
                    # MANO state). Keep the AL policy, but restart its dual
                    # state for this new primal point and expanded DoF space.
                    fallback_al_seed = report["al"].restart_for_primal_reseed()
                    report, values = optimize(
                        True,
                        freeze_correspondence_through_tier=failed_tier_index - 1,
                        start_tier=failed_tier_index,
                        seed_al=fallback_al_seed,
                    )
            report["fallback_reasons"] = fallback_reasons
            report["rank_failure_fraction"] = float(rank_failure_fraction)
            fallback = bool(report["fallback"])
            fallback_mask[start:end] |= fallback
            joints = np.asarray(report["body_joints"], dtype=np.int64)
            body_coeff[active_control[:, None], joints[None, :]] = values["body"]
            left_coeff[active_control] = values["left"]
            right_coeff[active_control] = values["right"]
            if fallback:
                root_coeff[active_control] = values["root"]
                translation_coeff[active_control] = values["translation"]
            al = report.pop("al")
            for side, label in enumerate(("left", "right")):
                palm_names = (
                    f"palm_position_{label}", f"palm_orientation_{label}",
                )
                if all(name in al.states for name in palm_names):
                    value = np.concatenate([
                        al.states[name].multiplier.detach().cpu().numpy()
                        for name in palm_names
                    ], axis=-1)
                    mask = contact[start:end, :, side]
                    cursor = 0
                    for object_index in range(objects):
                        count_active = int(mask[:, object_index].sum())
                        palm_multiplier[start:end, object_index, side][mask[:, object_index]] = value[cursor:cursor + count_active]
                        cursor += count_active
                side_contact_names: list[str] = []
                side_normal_names: list[str] = []
                for object_index in range(objects):
                    segment_values = source.contact_segment[
                        start:end, object_index, side
                    ]
                    object_valid = constraint_valid[start:end, object_index, side]
                    for segment in np.unique(segment_values[object_valid]):
                        contact_name = contact_segment_constraint_name(
                            "contact", object_index, label, int(segment),
                        )
                        normal_name = contact_segment_constraint_name(
                            "normal", object_index, label, int(segment),
                        )
                        side_contact_names.append(contact_name)
                        side_normal_names.append(normal_name)
                        if contact_name not in al.states:
                            continue
                        mask = object_valid & (segment_values == int(segment))
                        value = al.states[
                            contact_name
                        ].multiplier.detach().cpu().numpy()
                        destination = contact_multiplier[
                            start:end, object_index, side
                        ]
                        destination[mask] = value
                        contact_multiplier[
                            start:end, object_index, side
                        ] = destination
                layer_names = (
                    palm_names,
                    tuple(side_contact_names),
                    tuple(side_normal_names),
                    finger_constraint_names(label),
                    (f"support_{label}",),
                )
                for tier_index, names in enumerate(layer_names):
                    details = [report["constraints"][name] for name in names if name in report["constraints"]]
                    if details:
                        layer_residual[tier_index, side] = max(
                            layer_residual[tier_index, side], max(item["p95"] for item in details),
                        )
                        layer_penalty[tier_index, side] = max(
                            layer_penalty[tier_index, side], max(item["penalty"] for item in details),
                        )
                        layer_multiplier[tier_index, side] = max(
                            layer_multiplier[tier_index, side],
                            max(item["multiplier_norm"] for item in details),
                        )
                        layer_converged[tier_index, side] = all(
                            item["p95"] <= 1.0 for item in details
                        )
            window_reports.append({
                "start": int(start), "end": int(end), "joint_rank": {
                    key: value for key, value in joint_rank.items() if key != "singular_values"
                }, "side_rank": [
                    {key: value for key, value in item.items() if key != "singular_values"}
                    for item in side_rank
                ], **report,
            })

    body_delta = np.einsum("tc,cjd->tjd", basis_np, body_coeff)
    root_delta = np.einsum("tc,cd->td", basis_np, root_coeff)
    translation_delta = np.einsum("tc,cd->td", basis_np, translation_coeff)
    left_delta = np.einsum("tc,cjd->tjd", basis_np, left_coeff)
    right_delta = np.einsum("tc,cjd->tjd", basis_np, right_coeff)

    def compose_numpy(base: np.ndarray, delta: np.ndarray) -> np.ndarray:
        base_matrix = Rotation.from_rotvec(base.reshape(-1, 3)).as_matrix()
        delta_matrix = Rotation.from_rotvec(delta.reshape(-1, 3)).as_matrix()
        return Rotation.from_matrix(base_matrix @ delta_matrix).as_rotvec().reshape(base.shape)

    motion.body_pose = compose_numpy(body_base, body_delta).reshape(frames, 63).astype(np.float32)
    motion.global_orient = compose_numpy(root_base, root_delta).astype(np.float32)
    motion.transl = (translation_base + translation_delta).astype(np.float32)
    motion.left_hand_pose = compose_numpy(left_base, left_delta).reshape(frames, 45).astype(np.float32)
    motion.right_hand_pose = compose_numpy(right_base, right_delta).reshape(frames, 45).astype(np.float32)
    motion.continuous_spline = c3_archive_payload(
        motion.timestamps, motion.global_orient, motion.body_pose,
        motion.left_hand_pose, motion.right_hand_pose, motion.transl, degree,
    )
    _, target_vertices_world, target_landmarks_world = _motion_vertices_and_landmarks(
        config, motion, np.arange(frames),
    )
    target_feet, _ = _support_baseline(config, motion, model, device)
    target_landmarks_local = np.zeros_like(source.landmarks_local)
    target_palm_position = np.zeros_like(source_palm_position)
    target_palm_matrix = np.zeros((frames, objects, 2, 3, 3), dtype=np.float32)
    rotations = Rotation.from_quat(object_quaternion.reshape(-1, 4)).as_matrix().reshape(
        frames, objects, 3, 3,
    )
    for object_index in range(objects):
        target_landmarks_local[:, object_index] = np.einsum(
            "tslj,tjk->tslk",
            target_landmarks_world - object_translation[:, object_index, None, None],
            rotations[:, object_index],
        )
        target_palm_position[:, object_index], target_palm_matrix[:, object_index] = anatomical_palm_frames(
            target_landmarks_local[:, object_index],
        )
    contact_gap_m = np.full(constraint_valid.shape, np.nan, dtype=np.float32)
    frame_rows = np.arange(frames)[:, None]
    for object_index in range(objects):
        local_vertices = np.einsum(
            "tvi,tij->tvj",
            target_vertices_world - object_translation[:, object_index, None],
            rotations[:, object_index],
        )
        for side in range(2):
            ids = np.maximum(target_vertex[:, object_index, side], 0)
            predicted = local_vertices[frame_rows, ids]
            desired = (
                correspondence["anchor"][:, object_index, side] +
                float(settings["contact_surface_offset_m"]) *
                correspondence["normal"][:, object_index, side]
            )
            distance = np.linalg.norm(predicted - desired, axis=-1)
            active = constraint_valid[:, object_index, side]
            contact_gap_m[:, object_index, side][active] = distance[active]
    segment_residual_summary = contact_segment_residual_summary(
        contact_gap_m, constraint_valid, source.contact_segment, pilot_frame_mask,
    )
    print(json.dumps({
        "event": "v4_contact_segment_residual_summary",
        "summary": segment_residual_summary,
    }, sort_keys=True), flush=True)
    target_palm_quaternion = Rotation.from_matrix(
        target_palm_matrix.reshape(-1, 3, 3),
    ).as_quat().reshape(frames, objects, 2, 4).astype(np.float32)
    root_fallback_translation[:] = translation_delta
    root_fallback_rotvec[:] = root_delta
    return {
        "solver_version": "constrained_se3_v4",
        "pilot_windows": bool(pilot_windows),
        "pilot_frame_mask": pilot_frame_mask,
        "windows": window_reports,
        "source_palm_position_local": source_palm_position,
        "source_palm_quat_xyzw_local": source_palm_quaternion,
        "target_palm_position_local": target_palm_position,
        "target_palm_quat_xyzw_local": target_palm_quaternion,
        "target_hand_landmarks_local": target_landmarks_local,
        "remapped_contact_anchor_local": correspondence["anchor"],
        "remapped_contact_face": correspondence["face"],
        "remapped_contact_barycentric": correspondence["barycentric"],
        "remapped_contact_geodesic_m": correspondence["geodesic"],
        "remapped_contact_normal_local": correspondence["normal"],
        "contact_target_vertex": target_vertex,
        "contact_constraint_valid": constraint_valid,
        "contact_normal_constraint_valid": source_normal_valid & constraint_valid,
        "contact_gap_m": contact_gap_m,
        "contact_segment_residual_summary": segment_residual_summary,
        "palm_constraint_multiplier": palm_multiplier,
        "contact_constraint_multiplier": contact_multiplier,
        "solver_layer_residual_p95": layer_residual,
        "solver_layer_penalty": layer_penalty,
        "solver_layer_multiplier_norm": layer_multiplier,
        "solver_layer_converged": layer_converged,
        "palm_jacobian_rank": rank,
        "palm_jacobian_condition": condition,
        "full_body_fallback": fallback_mask,
        "support_foot": support,
        "support_foot_baseline_world": baseline_feet,
        "support_foot_target_world": target_feet,
        "root_fallback_translation": root_fallback_translation,
        "root_fallback_rotvec": root_fallback_rotvec,
        "spline_knots": knots,
        "spline_degree": degree,
        "spline_body_delta_coeff": body_coeff,
        "spline_root_delta_coeff": root_coeff,
        "spline_translation_delta_coeff": translation_coeff,
        "spline_left_hand_delta_coeff": left_coeff,
        "spline_right_hand_delta_coeff": right_coeff,
        "spline_rotation_basis": "baseline_composed_right_SO3_rotvec_delta",
        "calibration": calibration,
    }
