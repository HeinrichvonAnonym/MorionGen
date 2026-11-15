from __future__ import annotations

import shutil
import sys
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from utils.motion_retarget.adapters import (
    _COMFI_URDF_FROM_SMPLX, comfi_biomechanical_body_pose, inventory,
    inventory_andy, inventory_comfi, inventory_hiphi,
)
from utils.motion_retarget.bvh import parse_bvh
from utils.motion_retarget.config import load_config
from utils.motion_retarget.gpu import assigned_gpu_index, physical_gpu_count, physical_gpu_token
from utils.motion_retarget.hiphi_stream import STAGE_SENTINEL, cleanup_stale_hiphi_stages
from utils.motion_retarget.canonical import canonicalize_bvh_root
from utils.motion_retarget.hands import coverage_quality, empty_hand_track, load_wilor_cache, observed_hand_track
from utils.motion_retarget.fitting import summarize_fit_errors
from utils.motion_retarget.joints import bvh_to_smplx_pose
from utils.motion_retarget.jsonutil import json_safe
from utils.motion_retarget.ledger import Ledger
from utils.motion_retarget.profiles import normalize_gender, profile_scale, profiles_for_reference
from utils.motion_retarget.quality import _wrist_basis_alignment
from utils.motion_retarget.pipeline import (
    _canonicalize_z_up, _estimate_motion_time_offset, _video_cache_path,
    input_dependency_hash, run_config_hash,
)
from utils.motion_retarget.rotations import interpolate_pose_gaps
from utils.motion_retarget.schema import MotionRecord, RetargetProfile, SequenceRef
from utils.motion_retarget.schema import ObjectTrackRecord
from utils.motion_retarget.storage import read_motion_npz, write_motion_npz
from utils.motion_retarget.objects import (
    build_signed_distance_grid, canonicalize_object_track, contact_hysteresis,
    parse_object_track_csv, TriangleSurfaceQuery,
)
from utils.motion_retarget.object_pipeline import (
    _adapt_hand_landmark_lengths,
    _contact_segment_landmarks,
    _compose_axis_angle, _contact_priority_gradients,
    _hierarchical_priority_gradients, _continuation_scale,
    _cosine_learning_rate_scale, _incident_face_table,
    _hand_dynamic_shape_errors,
    _selected_vertex_normals, _stable_contact_patches,
    _penetration_metrics,
    _proxy_identity_candidate_order,
    _segment_longitudinal_coordinate,
    _transport_segment_surface_to_reference,
    source_contact_dependency_hash,
)
from utils.motion_retarget.constrained_object_v4 import (
    AugmentedLagrangian, FINGER_LANDMARK_SLICES, FINGER_NAMES,
    active_violation_mean,
    WeldedSurfaceTopology, anatomical_palm_frames,
    constant_proxy_vertex_path, contact_constraint_coverage_failures,
    changed_contact_constraint_entries, changed_contact_constraint_names,
    contact_segment_constraint_name, contact_segment_constraint_names,
    contact_segment_residual_summary,
    contact_vertex_owners,
    componentwise_gate_improvement_allowed,
    constraint_gradient_diagnostics,
    correspondence_update_budget,
    derivative_jump_metrics, deterministic_gold_windows,
    feasibility_projection_allowed, fallback_tier_correspondence_frozen,
    fallback_vertex_rebind_locked,
    fixed_pose_correspondence_confirmation_allowed,
    failed_tier_is_safe_fallback_seed, fixed_state_confirmation_allowed,
    exact_projection_retry_required, feasibility_projection_loss,
    feasibility_projection_uses_dense_tail,
    finger_constraint_names,
    fallback_seed_sources, full_body_fallback_required,
    minimum_change_backtracking_fractions,
    normalized_gate_score,
    initial_correspondence_stable_rounds,
    stable_correspondence_path,
    normal_candidate_feasible, normalized_normal_opposition_residual,
    persistent_switch_mask, posed_vertex_normals, source_proxy_run_boundaries,
    reference_gap_candidate_feasible,
    source_proxy_normal_support,
    surface_correspondence_changed, support_foot_mask,
    surface_correspondence_policy,
    surface_remap_metric_cost,
    surface_switch_is_material, target_vertex_candidate_allowed,
    target_vertex_switch_within_local_patch,
    tier_constraint_p95,
    quantile_constraint_violation_loss, stable_vector_angle,
    parameter_limit_diagnostics,
    tail_constraint_violation_loss,
    target_vertex_reserved_by_other, task_jacobian_diagnostics,
    correspondence_is_frozen_for_tier,
    use_constant_proxy_vertex_path, _window_task_rank,
)
from utils.motion_retarget.trajectory import (
    c3_archive_payload, c3_smooth_relative_rotvec, c3_smooth_rotvec,
    c4_bspline_basis,
)
from utils.motion_retarget.smplx_model import (
    HAND_POSE_CONVENTION, HAND_TIP_VERTEX_IDS, create_smplx_model,
    smplx_hand_landmarks,
)
from utils.motion_retarget.source_hands import (
    SourceHandContacts, _bridge_annotation_prior, _hand_proxy,
    align_normal_axis_to_reference, contact_facing_normals,
    hand_vertex_segments, read_source_hand_contacts, write_source_hand_contacts,
)
from utils.motion_retarget.validation import run_mujoco_validation, run_smplx_mesh_render, validate_archive
from python.scripts.extract_wilor_mano import (
    normalize_keypoints_2d, reflect_left_mano_matrices, robust_triangulate_and_score,
)
from python.scripts.render_smplx_mesh import nearest_frame_indices
from python.scripts.audit_object_penetration_regions import classify_penetration


BVH = """HIERARCHY
ROOT Hips
{
 OFFSET 0 0 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT LeftHand
 {
  OFFSET 1 0 0
  CHANNELS 3 Yrotation Xrotation Zrotation
  JOINT LeftHandIndex1
  {
   OFFSET 1 0 0
   CHANNELS 3 Xrotation Yrotation Zrotation
   JOINT LeftHandIndex2
   {
    OFFSET 1 0 0
    CHANNELS 3 Xrotation Yrotation Zrotation
    JOINT LeftHandIndex3
    {
     OFFSET 1 0 0
     CHANNELS 3 Xrotation Yrotation Zrotation
     End Site { OFFSET 1 0 0 }
    }
   }
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.01
1 2 3 10 20 30 4 5 6 7 8 9 10 11 12 13 14 15
2 3 4 11 21 31 5 6 7 8 9 10 11 12 13 14 15 16
"""


class MotionRetargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "config/motion_retarget.yaml")
        cls.run_dir = ROOT / "develop_logs/test_runs/motion_retarget_unit"
        shutil.rmtree(cls.run_dir, ignore_errors=True)
        cls.run_dir.mkdir(parents=True)

    def test_bvh_channel_order_and_fk(self):
        motion = parse_bvh(BVH)
        self.assertEqual(motion.frame_count, 2)
        expected = Rotation.from_euler("ZXY", [[10, 20, 30]], degrees=True).as_rotvec()[0]
        np.testing.assert_allclose(motion.local_rotvecs()[0, 0], expected, atol=1e-6)
        positions, rotations = motion.forward_kinematics()
        np.testing.assert_allclose(positions[0, 0], [1, 2, 3])
        self.assertTrue(np.allclose(np.linalg.det(rotations), 1.0))

    def test_v4_three_dof_cannot_control_six_dof_task(self):
        jacobian = np.asarray([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
        ])
        result = task_jacobian_diagnostics(jacobian, required_rank=6)
        self.assertEqual(result["rank"], 3)
        self.assertFalse(result["full_rank"])
        full = task_jacobian_diagnostics(np.eye(6), required_rank=6)
        self.assertTrue(full["full_rank"])

    def test_v4_one_bad_finger_landmark_cannot_hide_in_whole_hand_p95(self):
        # One joint contributes exactly one of the twenty dynamic landmarks.
        # Its per-finger gate must fail even though it sits exactly in the 5%
        # tail that the old whole-hand P95 could hide.
        residual = np.zeros((100, 20, 3), dtype=np.float64)
        middle = FINGER_LANDMARK_SLICES["middle"]
        residual[:, middle.start + 1, 0] = 1.35
        pooled = np.quantile(
            np.linalg.norm(residual.reshape(-1, 3), axis=-1), 0.95,
        )
        per_finger = np.quantile(
            np.linalg.norm(residual[:, middle].reshape(-1, 3), axis=-1), 0.95,
        )
        self.assertGreater(per_finger, 1.0)
        self.assertLess(pooled, 1.0)
        names = finger_constraint_names("right")
        self.assertEqual(len(names), 16)
        self.assertEqual(len(set(names)), len(names))
        self.assertIn("finger_position_right_middle", names)
        self.assertEqual(FINGER_NAMES, ("index", "middle", "pinky", "ring", "thumb"))

    def test_v4_full_body_fallback_rule_is_deterministic(self):
        required, reasons = full_body_fallback_required(0.051, 0.5, 0.8)
        self.assertTrue(required)
        self.assertEqual(reasons, ["task_jacobian_rank"])
        required, reasons = full_body_fallback_required(0.0, 1.01, 0.29)
        self.assertTrue(required)
        self.assertEqual(
            reasons,
            ["palm_se3_after_two_outer_iterations", "insufficient_residual_improvement"],
        )
        self.assertEqual(full_body_fallback_required(0.05, 1.0, 0.30), (False, []))
        self.assertEqual(
            full_body_fallback_required(0.0, 0.9, 0.8, hierarchy_failed=True),
            (True, ["upper_body_hierarchy_infeasible"]),
        )

    def test_v4_fallback_seed_ignores_unactivated_diagnostics(self):
        constraints = {
            "normal_right": {"p95": 1.004},
            "contact_right": {"p95": 0.99},
            "finger_position_right_index": {"p95": 7.0},
        }
        tier = {"constraints": ["contact_right", "normal_right"]}
        self.assertEqual(tier_constraint_p95(constraints, tier), [0.99, 1.004])

    def test_v4_fallback_never_seeds_hands_from_failed_pre_finger_tier(self):
        self.assertEqual(fallback_seed_sources(True), {
            "body": "failed_tier",
            "hands": "last_feasible_tier",
        })
        self.assertEqual(fallback_seed_sources(False), {
            "body": "last_feasible_tier",
            "hands": "last_feasible_tier",
        })

    def test_v4_fallback_primal_reseed_restarts_dual_state(self):
        import torch

        solver = AugmentedLagrangian(
            initial_penalty=2.0,
            penalty_multiplier=3.0,
            maximum_penalty=99.0,
            required_reduction=0.2,
        )
        residual = torch.tensor([2.0, -2.0])
        solver.term("right_contact", residual)
        solver.update("right_contact", residual)
        self.assertTrue(solver.states)
        restarted = solver.restart_for_primal_reseed()
        self.assertFalse(restarted.states)
        self.assertEqual(restarted.initial_penalty, 2.0)
        self.assertEqual(restarted.penalty_multiplier, 3.0)
        self.assertEqual(restarted.maximum_penalty, 99.0)
        self.assertEqual(restarted.required_reduction, 0.2)

    def test_v4_minimum_change_projection_uses_largest_first_steps(self):
        fractions = minimum_change_backtracking_fractions(5)
        self.assertEqual(fractions, (1.0, 0.5, 0.25, 0.125, 0.0625))
        with self.assertRaises(ValueError):
            minimum_change_backtracking_fractions(0)

    def test_v4_feasibility_projection_requires_finite_stable_correspondence(self):
        self.assertTrue(feasibility_projection_allowed(1.01, 2, 2))
        self.assertTrue(feasibility_projection_allowed(3.0, 2, 2))
        self.assertFalse(feasibility_projection_allowed(float("inf"), 2, 2))
        self.assertFalse(feasibility_projection_allowed(1.01, 1, 2))

    def test_v4_gate_score_prefers_best_stable_physical_state(self):
        self.assertEqual(normalized_gate_score({"left": 0.9, "right": 1.0}), 0.0)
        self.assertLess(
            normalized_gate_score({"contact": 1.06, "collision": 0.8}),
            normalized_gate_score({"contact": 1.02, "collision": 1.15}),
        )
        self.assertTrue(np.isinf(normalized_gate_score({"contact": np.nan})))

    def test_v4_targeted_projection_requires_componentwise_monotone_hard_gates(self):
        self.assertTrue(componentwise_gate_improvement_allowed(
            {"normal": 1.08, "contact": 1.03, "palm": 0.9},
            {"normal": 1.04, "contact": 1.02, "palm": 0.95},
        ))
        self.assertFalse(componentwise_gate_improvement_allowed(
            {"normal": 1.08, "contact": 1.03, "palm": 0.9},
            {"normal": 1.04, "contact": 1.04, "palm": 0.95},
        ))
        self.assertFalse(componentwise_gate_improvement_allowed(
            {"normal": 1.08}, {"other": 1.0},
        ))

    def test_v4_failed_constraint_gradient_diagnostics_reports_control_blocks(self):
        import torch

        body = torch.tensor([2.0], requires_grad=True)
        hand = torch.tensor([3.0], requires_grad=True)
        values = {
            "contact": body + hand,
            "normal_surface": body - hand,
        }
        diagnostics = constraint_gradient_diagnostics(
            values, {"contact": 5.0, "normal_surface": 1.5},
            [body, hand], ["body", "hand"], ("normal_",),
        )
        self.assertGreater(
            diagnostics["constraints"]["contact"]["gradient_norm"]["hand"], 0.0,
        )
        self.assertIn("contact|normal_surface", diagnostics["cosine"])

    def test_v4_fixed_feasible_state_is_confirmed_without_perturbation(self):
        self.assertTrue(fixed_state_confirmation_allowed(
            {"left": 0.97, "right": 1.0}, 1, 2, 2, 2,
        ))
        self.assertFalse(fixed_state_confirmation_allowed(
            {"left": 0.97, "right": 1.001}, 1, 2, 2, 2,
        ))
        self.assertFalse(fixed_state_confirmation_allowed(
            {"left": 0.97, "right": 1.0}, 0, 2, 2, 2,
        ))
        self.assertFalse(fixed_state_confirmation_allowed(
            {"left": 0.97, "right": 1.0}, 1, 2, 1, 2,
        ))

    def test_v4_correspondence_change_resets_dual_but_retains_penalty(self):
        import torch

        solver = AugmentedLagrangian()
        residual = torch.tensor([[1.5, 0.0, 0.0]])
        solver.term("contact_right", residual)
        solver.update("contact_right", residual)
        state = solver.states["contact_right"]
        state.penalty = 64.0
        self.assertGreater(float(torch.linalg.vector_norm(state.multiplier)), 0.0)
        self.assertEqual(solver.reset_duals(("contact_right", "missing")), ["contact_right"])
        self.assertEqual(state.penalty, 64.0)
        self.assertEqual(float(torch.linalg.vector_norm(state.multiplier)), 0.0)
        self.assertTrue(np.isinf(state.previous_p95))
        self.assertEqual(state.stable_rounds, 0)

    def test_v4_al_tail_is_not_diluted_by_satisfied_samples(self):
        import torch

        violation = torch.tensor([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
        ])
        component = violation.square()
        self.assertAlmostEqual(
            float(active_violation_mean(component, violation)),
            float(component[-1].mean()),
        )
        self.assertEqual(
            float(active_violation_mean(torch.zeros_like(component), torch.zeros_like(violation))),
            0.0,
        )

    def test_v4_full_body_fallback_clones_hierarchy_state(self):
        import torch

        solver = AugmentedLagrangian(maximum_penalty=4096.0)
        residual = torch.tensor([[1.5, 0.0, 0.0]])
        solver.term("contact_right", residual)
        solver.update("contact_right", residual)
        solver.states["contact_right"].penalty = 4096.0
        cloned = solver.clone()
        self.assertIsNot(cloned, solver)
        self.assertIsNot(cloned.states["contact_right"], solver.states["contact_right"])
        self.assertEqual(cloned.states["contact_right"].penalty, 4096.0)
        np.testing.assert_allclose(
            cloned.states["contact_right"].multiplier.numpy(),
            solver.states["contact_right"].multiplier.numpy(),
        )
        cloned.reset_duals(("contact_right",))
        self.assertGreater(
            float(torch.linalg.vector_norm(solver.states["contact_right"].multiplier)), 0.0,
        )

    def test_v4_hierarchy_expansion_keeps_policy_but_rebalances_duals(self):
        import torch

        solver = AugmentedLagrangian(
            initial_penalty=2.0,
            penalty_multiplier=3.0,
            maximum_penalty=729.0,
            required_reduction=0.2,
        )
        residual = torch.tensor([[1.5, 0.0, 0.0]])
        solver.term("palm_right", residual)
        solver.update("palm_right", residual)
        solver.states["palm_right"].penalty = 729.0

        expanded = solver.restart_for_hierarchy_expansion()

        self.assertEqual(expanded.states, {})
        self.assertEqual(expanded.initial_penalty, 2.0)
        self.assertEqual(expanded.penalty_multiplier, 3.0)
        self.assertEqual(expanded.maximum_penalty, 729.0)
        self.assertEqual(expanded.required_reduction, 0.2)
        # The previous problem remains an immutable diagnostic receipt.
        self.assertEqual(solver.states["palm_right"].penalty, 729.0)
        self.assertGreater(
            float(torch.linalg.vector_norm(solver.states["palm_right"].multiplier)), 0.0,
        )

    def test_v4_near_feasible_failed_tier_seeds_fallback(self):
        self.assertTrue(failed_tier_is_safe_fallback_seed(
            [0.98, 1.42], [1.1, 4.67], 2.0, 0.10,
        ))
        self.assertFalse(failed_tier_is_safe_fallback_seed(
            [0.98, 2.01], [1.1, 4.67], 2.0, 0.10,
        ))
        self.assertFalse(failed_tier_is_safe_fallback_seed(
            [0.98, 1.42], [1.1, 1.50], 2.0, 0.10,
        ))
        self.assertFalse(failed_tier_is_safe_fallback_seed(
            [0.98, np.nan], [1.1, 4.67], 2.0, 0.10,
        ))

    def test_v4_feasibility_projection_targets_p95_tail(self):
        import torch

        residual = torch.tensor([0.5] * 8 + [1.2, 1.4], requires_grad=True)
        loss = tail_constraint_violation_loss(
            residual, inequality=False, tail_fraction=0.20,
        )
        self.assertAlmostEqual(
            float(loss.detach()), (0.2 ** 2 + 0.4 ** 2) / 2.0, places=6,
        )
        loss.backward()
        np.testing.assert_allclose(residual.grad.detach().numpy()[:8], 0.0)
        self.assertGreater(float(residual.grad.detach().numpy()[8]), 0.0)
        self.assertGreater(float(residual.grad.detach().numpy()[9]), 0.0)

        vector = torch.tensor([[1.1, 0.0, 0.0], [0.0, 1.3, 0.0]])
        vector_loss = tail_constraint_violation_loss(
            vector, inequality=False, tail_fraction=0.50,
        )
        self.assertAlmostEqual(float(vector_loss), 0.3 ** 2, places=6)

    def test_v4_projection_loss_matches_acceptance_quantile(self):
        import torch

        residual = torch.tensor([0.5] * 18 + [1.2, 1.4], requires_grad=True)
        loss = quantile_constraint_violation_loss(
            residual, inequality=False, quantile=0.95,
        )
        expected_gate = float(torch.quantile(residual.detach(), 0.95))
        self.assertAlmostEqual(float(loss.detach()), (expected_gate - 1.0) ** 2, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(residual.grad).all())

        safe = quantile_constraint_violation_loss(
            torch.tensor([0.5] * 19 + [1.1]), inequality=False, quantile=0.95,
        )
        self.assertEqual(float(safe), 0.0)

    def test_v4_projection_combines_exact_p95_with_dense_tail_gradient(self):
        import torch

        residual = torch.tensor(
            [0.5] * 90 + [1.01 + 0.01 * index for index in range(10)],
            requires_grad=True,
        )
        loss = feasibility_projection_loss(
            residual, inequality=False, quantile=0.95, tail_fraction=0.10,
        )
        loss.backward()
        # A bare interpolated quantile reaches at most two samples; the search
        # term must move the full violating tail while safe samples stay fixed.
        self.assertGreater(int((residual.grad > 0).sum()), 2)
        self.assertTrue(torch.equal(
            residual.grad[:90], torch.zeros_like(residual.grad[:90]),
        ))
        self.assertFalse(feasibility_projection_uses_dense_tail(1.0009, 1.001))
        self.assertTrue(feasibility_projection_uses_dense_tail(1.0011, 1.001))
        self.assertTrue(exact_projection_retry_required(True, False))
        self.assertFalse(exact_projection_retry_required(True, True))
        self.assertFalse(exact_projection_retry_required(False, False))

    def test_v4_vector_angle_has_finite_gradient_at_alignment(self):
        import torch

        first = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 1e-8, 0.0]], requires_grad=True,
        )
        second = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        angles = stable_vector_angle(first, second)
        self.assertTrue(torch.isfinite(angles).all())
        angles.sum().backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertLess(float(angles.max()), 1e-5)

    def test_v4_parameter_limit_diagnostics_detects_saturation(self):
        import torch

        value = torch.tensor([[[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]]])
        diagnostics = parameter_limit_diagnostics(value, 1.0)
        self.assertAlmostEqual(diagnostics["maximum_fraction"], 1.0)
        self.assertAlmostEqual(diagnostics["saturated_fraction"], 0.5)
        self.assertEqual(diagnostics["top"][0]["control"], 1)
        self.assertEqual(diagnostics["top"][0]["joint"], 0)

    def test_v4_gold_windows_cover_acquire_stable_and_right_release(self):
        contact = np.zeros((320, 1, 2), dtype=bool)
        contact[80:240, 0, 0] = True
        contact[100:220, 0, 1] = True
        windows = deterministic_gold_windows(contact, len(contact), 40)
        self.assertEqual(windows, [(80, 120), (139, 179), (200, 240)])
        self.assertTrue(contact[windows[0][1] - 1, 0].all())
        self.assertTrue(contact[(windows[1][0] + windows[1][1]) // 2, 0].all())
        self.assertFalse(contact[windows[2][1] - 1, 0, 1])

    def test_v4_real_smplx_upper_chain_jacobian_executes_without_vmap(self):
        from utils.motion_retarget.object_pipeline import _device, _model

        config = load_config(ROOT / "config/motion_retarget_hiphi_object_v4.yaml")
        profile = RetargetProfile("test", "female", 1.69, (0.0,) * 16)
        record = MotionRecord(
            SequenceRef("hiphi", "jacobian", Path("source.bvh")), profile, 90.0,
            np.asarray([0.0]), np.zeros((1, 3), np.float32),
            np.zeros((1, 63), np.float32), np.zeros((1, 45), np.float32),
            np.zeros((1, 45), np.float32), np.zeros((1, 3), np.float32), 1.0,
            np.ones(1, bool), np.ones(1, bool), np.ones(1), np.ones(1),
            np.zeros(1, bool), np.zeros(1, bool), np.eye(4),
        )
        device, _, _ = _device(config)
        model = _model(config, profile, device)
        result = _window_task_rank(
            config, record, model, 0, np.asarray([True, False]),
            np.asarray(config.raw["object_pilot"]["upper_body_joint_indices"]), device,
        )
        self.assertGreaterEqual(result["rank"], 6)

    def test_v4_anatomical_palm_frames_are_rigid_and_side_consistent(self):
        landmarks = np.zeros((1, 2, 22, 3), dtype=np.float32)
        for side in range(2):
            landmarks[0, side, 0] = [0.0, 0.0, 0.0]
            landmarks[0, side, 1] = [0.0, 1.0, 0.0]
            landmarks[0, side, 2] = [1.0, 1.0, 0.0]
            landmarks[0, side, 6] = [0.0, 2.0, 0.0]
            landmarks[0, side, 10] = [-1.0, 1.0, 0.0]
        origin, matrix = anatomical_palm_frames(landmarks)
        np.testing.assert_allclose(origin[:, :, 1], 1.0)
        np.testing.assert_allclose(
            matrix.transpose(0, 1, 3, 2) @ matrix,
            np.broadcast_to(np.eye(3), matrix.shape), atol=1e-6,
        )
        np.testing.assert_allclose(np.linalg.det(matrix), 1.0, atol=1e-6)

    def test_v4_surface_remap_stays_on_connected_patch_and_five_mm(self):
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray([
                [0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.0, 0.004, 0.0],
                [0.004, 0.004, 0.0], [0.0001, 0.0001, 0.001],
                [0.0041, 0.0001, 0.001], [0.0001, 0.0041, 0.001],
            ]),
            faces=np.asarray([[0, 1, 2], [1, 3, 2], [4, 5, 6]]), process=False,
        )
        topology = WeldedSurfaceTopology(mesh, weld_tolerance_m=1e-6)
        result = topology.project(
            np.asarray([0.001, 0.001, 0.0]), 0,
            np.asarray([0.0035, 0.0035, 0.0]), 0.005,
        )
        self.assertIn(result.face, {0, 1})
        self.assertLessEqual(result.geodesic_m, 0.005 + 1e-9)
        self.assertNotEqual(result.face, 2)
        self.assertAlmostEqual(float(result.barycentric.sum()), 1.0, places=6)

    def test_v4_surface_patch_keeps_source_anchor_as_feasible_fallback(self):
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=np.asarray([
                [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            ]),
            faces=np.asarray([[0, 1, 2]]), process=False,
        )
        topology = WeldedSurfaceTopology(mesh, weld_tolerance_m=1e-6)
        source = np.asarray([0.01, 0.01, 0.0])
        result = topology.project(source, 0, np.asarray([0.8, 0.1, 0.02]), 0.005)
        np.testing.assert_allclose(result.point, source, atol=1e-8)
        self.assertEqual(result.face, 0)
        self.assertEqual(result.geodesic_m, 0.0)
        self.assertAlmostEqual(float(result.barycentric.sum()), 1.0, places=6)

    def test_v4_correspondence_path_penalizes_chatter(self):
        cost = np.asarray([
            [0.0, 0.1], [0.1, 0.0], [0.0, 0.1], [0.1, 0.0],
        ])
        path = stable_correspondence_path(cost, transition_cost=0.2)
        self.assertTrue(np.all(path == path[0]))

    def test_v4_source_proxy_identity_has_one_target_vertex_per_run(self):
        cost = np.asarray([
            [0.0, 0.2], [0.3, 0.0], [0.0, 0.2], [0.3, 0.0],
        ])
        path = constant_proxy_vertex_path(cost)
        self.assertTrue(np.all(path == path[0]))
        self.assertEqual(int(path[0]), 1)

    def test_v4_proxy_identity_uses_radial_side_and_longitudinal_position(self):
        source_frame = Rotation.from_euler("z", 37.0, degrees=True).as_matrix()
        target_frame = Rotation.from_euler("y", -51.0, degrees=True).as_matrix()
        source_start_local = np.asarray([0.0, 0.0, 0.0])
        source_end_local = np.asarray([1.0, 0.0, 0.0])
        source_surface_local = np.asarray([0.25, 0.0, 0.01])
        target_vertices_local = np.asarray([
            [0.25, 0.0, 0.01],
            [0.25, 0.0, -0.01],
            [0.90, 0.0, 0.01],
        ])
        order, feasible, violation = _proxy_identity_candidate_order(
            source_surface_local @ source_frame.T,
            source_start_local @ source_frame.T,
            source_end_local @ source_frame.T,
            source_frame,
            target_vertices_local @ target_frame.T,
            source_start_local @ target_frame.T,
            source_end_local @ target_frame.T,
            target_frame,
            30.0, 0.34,
        )
        self.assertEqual(int(order[0]), 0)
        np.testing.assert_array_equal(feasible, np.asarray([True, False, False]))
        self.assertLess(float(violation[0]), 1.0)

    def test_v4_proxy_segment_landmark_mapping_covers_all_capsules(self):
        self.assertEqual(_contact_segment_landmarks(0), (0, 1))
        self.assertEqual(_contact_segment_landmarks(1), (1, 2))
        self.assertEqual(_contact_segment_landmarks(5), (1, 18))
        self.assertEqual(_contact_segment_landmarks(6), (2, 3))
        self.assertEqual(_contact_segment_landmarks(20), (20, 21))
        with self.assertRaises(ValueError):
            _contact_segment_landmarks(21)
        np.testing.assert_allclose(
            _segment_longitudinal_coordinate(
                np.asarray([[0.25, 1.0, 0.0], [2.0, 0.0, 0.0]]),
                np.zeros(3), np.asarray([1.0, 0.0, 0.0]),
            ),
            np.asarray([0.25, 1.0]),
        )

    def test_v4_proxy_run_splits_when_exact_proxy_identity_changes(self):
        boundaries = source_proxy_run_boundaries(
            np.asarray([10, 11, 12, 14, 15, 16]),
            np.asarray([6, 6, 6, 6, 6, 7]),
            np.asarray([24, 24, 25, 25, 25, 28]),
        )
        np.testing.assert_array_equal(boundaries, np.asarray([2, 3, 5]))

    def test_v4_contact_proxy_ownership_prevents_slot_vertex_theft(self):
        owners = contact_vertex_owners(
            np.asarray([[101, 202, -1], [303, 404, 505]]),
            np.asarray([[True, True, False], [True, True, True]]),
        )
        self.assertEqual(owners[0], {101: 0, 202: 1})
        self.assertEqual(owners[1], {303: 0, 404: 1, 505: 2})
        with self.assertRaises(ValueError):
            contact_vertex_owners(
                np.asarray([[101, 101]]), np.asarray([[True, True]]),
            )

    def test_v4_contact_residual_diagnostics_keep_hands_and_segments_separate(self):
        gap = np.asarray([[[[0.001, 0.002], [0.010, 0.020]]]])
        valid = np.ones_like(gap, dtype=bool)
        segment = np.asarray([[[[6, 7], [6, 6]]]])
        summary = contact_segment_residual_summary(
            gap, valid, segment, np.asarray([True]),
        )
        self.assertAlmostEqual(summary["left"]["6"]["p95_m"], 0.001)
        self.assertAlmostEqual(summary["left"]["7"]["p95_m"], 0.002)
        self.assertAlmostEqual(summary["right"]["6"]["p95_m"], 0.0195)

    def test_v4_contact_al_states_are_separate_per_object_hand_and_segment(self):
        segments = np.zeros((2, 2, 2, 3), dtype=np.int16)
        valid = np.zeros_like(segments, dtype=bool)
        segments[0, 0, 1, :2] = [6, 9]
        valid[0, 0, 1, :2] = True
        segments[1, 1, 1, :2] = [9, 16]
        valid[1, 1, 1, :2] = True
        names = contact_segment_constraint_names(
            "contact", segments, valid, "right",
        )
        self.assertEqual(names, (
            "contact_right_object_0_segment_6",
            "contact_right_object_0_segment_9",
            "contact_right_object_1_segment_9",
            "contact_right_object_1_segment_16",
        ))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            contact_segment_constraint_name("normal", 1, "left", 20),
            "normal_left_object_1_segment_20",
        )
        with self.assertRaises(ValueError):
            contact_segment_constraint_name("normal", 0, "right", 21)

    def test_v4_correspondence_change_resets_only_changed_segment_duals(self):
        shape = (2, 1, 2, 2)
        current = {
            "valid": np.ones(shape, dtype=bool),
            "target_vertex": np.arange(np.prod(shape)).reshape(shape),
            "face": np.zeros(shape, dtype=np.int32),
            "anchor": np.zeros((*shape, 3), dtype=np.float32),
        }
        updated = {name: value.copy() for name, value in current.items()}
        updated["face"][0, 0, 1, 1] = 7
        updated["anchor"][0, 0, 1, 1, 0] = 0.002
        segments = np.zeros(shape, dtype=np.int16)
        segments[:, 0, 0] = 6
        segments[:, 0, 1] = 9
        self.assertEqual(changed_contact_constraint_names(
            current, updated, segments, 0.0001,
        ), (
            "contact_right_object_0_segment_9",
            "normal_right_object_0_segment_9",
        ))
        entries = changed_contact_constraint_entries(
            current, updated, segments, np.ones(shape, dtype=bool), 0.0001,
        )
        np.testing.assert_array_equal(
            entries["contact_right_object_0_segment_9"],
            np.asarray([False, True, False, False]),
        )

    def test_v4_correspondence_entry_reset_preserves_unchanged_lambdas(self):
        import torch

        solver = AugmentedLagrangian()
        residual = torch.full((4, 3), 2.0)
        name = "contact_right_object_0_segment_9"
        solver.term(name, residual)
        solver.update(name, residual)
        before = solver.states[name].multiplier.detach().clone()
        receipt = solver.reset_dual_entries({
            name: np.asarray([False, True, False, False]),
        })
        self.assertEqual(receipt[name], {"entries": 1, "total": 4})
        self.assertEqual(float(torch.linalg.vector_norm(
            solver.states[name].multiplier[1]
        )), 0.0)
        torch.testing.assert_close(
            solver.states[name].multiplier[[0, 2, 3]], before[[0, 2, 3]],
        )

    def test_v4_initial_binding_transports_surface_to_dynamic_reference(self):
        current_start = np.asarray([10.0, -2.0, 3.0])
        current_end = np.asarray([12.0, -2.0, 3.0])
        current_palm = np.eye(3)
        vertices = np.asarray([
            [10.5, -1.9, 3.2],
            [11.5, -2.3, 3.1],
        ])
        reference_start = np.asarray([0.0, 1.0, 2.0])
        reference_end = np.asarray([0.0, 2.0, 2.0])
        reference_palm = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        transported = _transport_segment_surface_to_reference(
            vertices, current_start, current_end, current_palm,
            reference_start, reference_end, reference_palm,
        )
        alpha = np.asarray([0.25, 0.75])
        current_axis_points = current_start + alpha[:, None] * (
            current_end - current_start
        )
        radial = vertices - current_axis_points
        expected = (
            reference_start + alpha[:, None] * (reference_end - reference_start) +
            radial @ reference_palm.T
        )
        np.testing.assert_allclose(transported, expected, atol=1e-7)

    def test_v4_surface_switch_requires_metric_improvement(self):
        self.assertFalse(surface_switch_is_material(0.0100, 0.0096, 0.0005))
        self.assertTrue(surface_switch_is_material(0.0101, 0.0095, 0.0005))

    def test_v4_surface_remap_cost_charges_source_anchor_geodesic_drift(self):
        self.assertAlmostEqual(
            surface_remap_metric_cost(0.004, 0.001, 0.003), 0.008,
        )
        # A 2 mm gap improvement must not buy 3 mm of source-trajectory drift.
        self.assertGreater(
            surface_remap_metric_cost(0.002, 0.0, 0.003),
            surface_remap_metric_cost(0.004, 0.0, 0.0),
        )
        self.assertTrue(np.isinf(surface_remap_metric_cost(0.0, 0.0, -0.001)))
        self.assertGreater(
            surface_remap_metric_cost(
                0.004, 0.0, 0.0, reference_gap_m=0.003,
            ),
            surface_remap_metric_cost(
                0.004, 0.0, 0.0, reference_gap_m=0.001,
            ),
        )

    def test_v4_anchor_switch_requires_temporal_persistence(self):
        candidate = np.asarray([
            False, True, False, True, True, False, True, True, True, False,
        ])
        np.testing.assert_array_equal(
            persistent_switch_mask(candidate, 3),
            np.asarray([
                False, False, False, False, False,
                False, True, True, True, False,
            ]),
        )

    def test_v4_surface_switch_scales_match_contact_acceptance(self):
        config = load_config(ROOT / "config/motion_retarget_hiphi_object_v4.yaml")
        settings = config.raw["object_pilot"]
        acceptance = settings["acceptance"]
        self.assertEqual(int(settings["surface_remap_vertex_lock_after_tier"]), 4)
        self.assertTrue(correspondence_is_frozen_for_tier(
            0, -1, 4, settings["surface_remap_active_tiers"],
        ))
        self.assertEqual(surface_correspondence_policy(2, 4, 0), (True, True))
        self.assertFalse(correspondence_is_frozen_for_tier(
            2, -1, 4, settings["surface_remap_active_tiers"],
        ))
        self.assertEqual(surface_correspondence_policy(
            2, 4, 0, settings["surface_remap_vertex_switch_tiers"],
        ), (True, True))
        self.assertFalse(correspondence_is_frozen_for_tier(
            3, -1, 4, settings["surface_remap_active_tiers"],
        ))
        self.assertEqual(surface_correspondence_policy(
            3, 4, 0, settings["surface_remap_vertex_switch_tiers"],
        ), (True, True))
        self.assertEqual(float(settings["surface_remap_reference_gap_slack_m"]), 0.0015)
        self.assertEqual(float(settings["surface_remap_reference_cost_scale"]), 1.0)
        self.assertEqual(float(settings["surface_remap_anchor_switch_m"]), 0.003)
        self.assertEqual(float(settings["target_vertex_remap_max_distance_m"]), 0.005)
        self.assertEqual(int(settings["target_contact_points_per_hand"]), 4)
        self.assertEqual(float(settings["mano_max_delta_degrees"]), 180.0)
        self.assertNotIn("feasibility_projection_trigger", settings)
        self.assertTrue(settings["surface_remap_constant_vertex_per_proxy_run"])
        self.assertGreaterEqual(
            int(settings["target_contact_points_per_hand"]),
            int(acceptance["contact_patch_active_vertices_p05"]),
        )
        self.assertEqual(
            float(settings["surface_remap_identity_switch_m"]),
            float(acceptance["contact_p95_m"]),
        )
        self.assertGreater(
            float(settings["surface_remap_anchor_switch_m"]),
            float(acceptance["object_translation_m"]),
        )
        self.assertEqual(
            float(settings["surface_remap_anchor_switch_m"]),
            float(acceptance["contact_median_m"]),
        )
        self.assertEqual(float(settings["contact_surface_offset_m"]), 0.0)
        self.assertEqual(
            float(settings["contact_gap_tolerance_m"]),
            float(acceptance["contact_p95_m"]),
        )
        self.assertEqual(
            float(settings["contact_normal_tolerance_degrees"]),
            float(acceptance["normal_angle_p95_deg"]),
        )
        self.assertLess(
            float(acceptance["normal_angle_median_deg"]),
            float(settings["contact_normal_tolerance_degrees"]),
        )

    def test_v4_target_vertex_locks_after_tier_activation(self):
        self.assertTrue(target_vertex_candidate_allowed(101, 202, True))
        self.assertTrue(target_vertex_candidate_allowed(101, 101, False))
        self.assertFalse(target_vertex_candidate_allowed(101, 202, False))
        self.assertTrue(target_vertex_reserved_by_other(1, 0, True))
        self.assertTrue(target_vertex_reserved_by_other(1, 0, False))
        self.assertFalse(target_vertex_reserved_by_other(0, 0, True))
        self.assertFalse(target_vertex_reserved_by_other(None, 0, True))

    def test_v4_palm_bridge_vertices_blend_wrist_and_mcp(self):
        model = create_smplx_model(self.config.smplx_model_dir, "female", 16)
        weights = model.lbs_weights.detach().cpu().numpy()
        for side, wrist, starts in (
            ("left", 20, (25, 28, 31, 34, 37)),
            ("right", 21, (40, 43, 46, 49, 52)),
        ):
            segments = hand_vertex_segments(model, side)
            for finger, start in enumerate(starts):
                selected = segments[finger + 1]
                self.assertGreaterEqual(len(selected), 3)
                self.assertTrue(np.all(weights[selected, wrist] > 0.035))
                self.assertTrue(np.all(weights[selected, start] > 0.035))

    def test_v4_target_vertex_rebind_stays_inside_local_five_mm_patch(self):
        origin = np.zeros(3)
        self.assertTrue(target_vertex_switch_within_local_patch(
            origin, np.asarray([0.003, 0.004, 0.0]), 0.005,
        ))
        self.assertFalse(target_vertex_switch_within_local_patch(
            origin, np.asarray([0.0031, 0.004, 0.0]), 0.005,
        ))

    def test_v4_target_vertex_rebind_preserves_immutable_reference_gap(self):
        self.assertTrue(reference_gap_candidate_feasible(0.002, 0.001, 0.0015))
        self.assertTrue(reference_gap_candidate_feasible(0.002, 0.0035, 0.0015))
        self.assertFalse(reference_gap_candidate_feasible(0.002, 0.0036, 0.0015))
        self.assertFalse(reference_gap_candidate_feasible(np.nan, 0.001, 0.0015))

    def test_v4_normal_tier_rejects_back_facing_candidate(self):
        minimum = float(np.cos(np.deg2rad(30.0)))
        self.assertTrue(normal_candidate_feasible(0.0, minimum, False))
        self.assertTrue(normal_candidate_feasible(minimum, minimum, True))
        self.assertFalse(normal_candidate_feasible(0.0, minimum, True))

    def test_v4_normal_residual_unit_gate_is_configured_angle(self):
        minimum = float(np.cos(np.deg2rad(30.0)))
        self.assertAlmostEqual(
            float(normalized_normal_opposition_residual(1.0, minimum)), 0.0,
        )
        self.assertAlmostEqual(
            float(normalized_normal_opposition_residual(minimum, minimum)), 1.0,
        )
        self.assertGreater(
            float(normalized_normal_opposition_residual(0.0, minimum)), 1.0,
        )

    def test_v4_enabled_tiers_alternate_vertices_until_correspondence_converges(self):
        self.assertEqual(surface_correspondence_policy(0, 3, 0), (True, False))
        self.assertEqual(surface_correspondence_policy(0, 3, 1), (True, False))
        self.assertEqual(surface_correspondence_policy(1, 3, 0), (True, False))
        self.assertEqual(surface_correspondence_policy(1, 3, 1), (True, False))
        self.assertEqual(surface_correspondence_policy(2, 3, 0), (True, True))
        self.assertEqual(surface_correspondence_policy(2, 3, 5), (True, True))
        self.assertEqual(surface_correspondence_policy(3, 3, 0), (False, True))
        self.assertEqual(surface_correspondence_policy(2, 0, 0), (False, True))

    def test_v4_correspondence_preupdate_does_not_consume_outer_round_budget(self):
        self.assertEqual(correspondence_update_budget(6, False), 6)
        self.assertEqual(correspondence_update_budget(6, True), 7)
        with self.assertRaises(ValueError):
            correspondence_update_budget(0, True)

    def test_v4_fixed_pose_correspondence_confirmation_only_finishes_started_gate(self):
        self.assertTrue(fixed_pose_correspondence_confirmation_allowed(1, 2, False))
        self.assertFalse(fixed_pose_correspondence_confirmation_allowed(0, 2, False))
        self.assertFalse(fixed_pose_correspondence_confirmation_allowed(2, 2, False))
        self.assertFalse(fixed_pose_correspondence_confirmation_allowed(1, 2, True))

    def test_v4_fallback_keeps_failed_tier_alternating_but_locks_completed_tiers(self):
        self.assertEqual(initial_correspondence_stable_rounds(2, False), 0)
        self.assertEqual(initial_correspondence_stable_rounds(2, True), 2)
        self.assertTrue(fallback_tier_correspondence_frozen(0, 1))
        self.assertTrue(fallback_tier_correspondence_frozen(1, 1))
        self.assertFalse(fallback_tier_correspondence_frozen(2, 1))
        self.assertFalse(correspondence_is_frozen_for_tier(1, -1, 2))
        self.assertTrue(correspondence_is_frozen_for_tier(2, -1, 2))
        self.assertTrue(correspondence_is_frozen_for_tier(1, 1, 3))
        self.assertFalse(fallback_vertex_rebind_locked(True, 1, 1))
        self.assertTrue(fallback_vertex_rebind_locked(True, 0, 1))
        self.assertFalse(fallback_vertex_rebind_locked(True, 3, 1))
        self.assertFalse(fallback_vertex_rebind_locked(False, 1, 1))

    def test_v4_fallback_keeps_failed_finger_tier_anchor_active(self):
        settings = load_config(
            ROOT / "config/motion_retarget_hiphi_object_v4.yaml"
        ).raw["object_pilot"]
        failed_tier = 3
        # Completed tiers 0..2 are frozen. Tier 3 continues its bounded
        # target-human vertex and OBJ-anchor alternation in the expanded solve.
        self.assertFalse(correspondence_is_frozen_for_tier(
            failed_tier, failed_tier - 1,
            int(settings["surface_remap_vertex_lock_after_tier"]),
            settings["surface_remap_active_tiers"],
        ))
        self.assertEqual(
            surface_correspondence_policy(
                failed_tier,
                int(settings["surface_remap_vertex_lock_after_tier"]), 0,
                settings["surface_remap_vertex_switch_tiers"],
                fallback=fallback_vertex_rebind_locked(
                    True, failed_tier, failed_tier,
                ),
            )[0],
            True,
        )

    def test_v4_source_contact_cache_ignores_temporary_paths_and_solver_settings(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            roots = [Path(first), Path(second)]
            hashes = []
            for root in roots:
                motion = root / "motion_actor.bvh"
                mesh = root / "object.obj"
                trajectory = root / "track.csv"
                motion.write_bytes(b"same bvh")
                mesh.write_bytes(b"same mesh")
                trajectory.write_bytes(b"same trajectory")
                reference = SimpleNamespace(source=motion)
                source = SimpleNamespace(
                    object_id="rack", category="clothrack",
                    mesh_path=mesh, trajectory_path=trajectory,
                )
                hashes.append(source_contact_dependency_hash(
                    reference, [source], "both hands",
                    {
                        "mesh_scale_to_m": 1.0,
                        "source_contact_exit_m": 0.02,
                        "surface_remap_vertex_lock_after_tier": 2 + len(hashes),
                    },
                ))
            self.assertEqual(hashes[0], hashes[1])

            changed = source_contact_dependency_hash(
                SimpleNamespace(source=roots[1] / "motion_actor.bvh"),
                [SimpleNamespace(
                    object_id="rack", category="clothrack",
                    mesh_path=roots[1] / "object.obj",
                    trajectory_path=roots[1] / "track.csv",
                )],
                "both hands",
                {"mesh_scale_to_m": 1.0, "source_contact_exit_m": 0.03},
            )
            self.assertNotEqual(hashes[1], changed)

    def test_v4_contact_coverage_cannot_be_thinned_to_fake_recall(self):
        source = np.ones((2, 1, 2), dtype=bool)
        valid = np.ones((2, 1, 2, 4), dtype=bool)
        region = np.broadcast_to(
            np.asarray([0, 0, 1, 1], dtype=np.int16), valid.shape,
        ).copy()
        self.assertEqual(
            contact_constraint_coverage_failures(source, valid, region), [],
        )
        valid[1, 0, 1, 1:] = False
        failures = contact_constraint_coverage_failures(source, valid, region)
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0], {
            "frame": 1, "object": 0, "side": 1, "points": 1, "regions": 1,
        })

    def test_v4_normal_tier_uses_temporal_path_for_radial_rolling(self):
        self.assertTrue(use_constant_proxy_vertex_path(True, False))
        self.assertTrue(use_constant_proxy_vertex_path(True, True))
        self.assertFalse(use_constant_proxy_vertex_path(False, True))

    def test_v4_correspondence_ignores_shared_edge_face_flip(self):
        anchor = np.asarray([0.1, 0.2, 0.3])
        self.assertFalse(surface_correspondence_changed(
            42, 42, anchor, anchor + [0.00005, 0.0, 0.0], 0.0001, 10, 11,
        ))
        self.assertTrue(surface_correspondence_changed(
            42, 42, anchor, anchor + [0.00011, 0.0, 0.0], 0.0001, 10, 11,
        ))
        self.assertFalse(surface_correspondence_changed(
            42, 42, anchor, anchor + [0.01, 0.0, 0.0], 0.0001, 10, 10,
        ))
        self.assertTrue(surface_correspondence_changed(
            42, 43, anchor, anchor, 0.0001, 10, 10,
        ))

    def test_v4_posed_vertex_normals_preserve_surface_orientation(self):
        vertices = np.asarray([[
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
        ]])
        faces = np.asarray([[0, 1, 2], [0, 2, 3]])
        normals = posed_vertex_normals(vertices, faces)
        np.testing.assert_allclose(
            normals, np.broadcast_to(np.asarray([0.0, 0.0, 1.0]), normals.shape), atol=1e-7,
        )

    def test_v4_thin_surface_normal_is_oriented_toward_source_hand(self):
        normal = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
        anchor = np.zeros((2, 3))
        hand = np.asarray([[0.0, 0.0, -0.01], [0.0, 0.0, -0.01]])
        oriented = contact_facing_normals(normal, anchor, hand)
        expected = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
        np.testing.assert_allclose(oriented, expected, atol=1e-8)
        self.assertTrue(np.all(np.sum(oriented * (hand - anchor), axis=-1) > 0.0))

    def test_v4_penetrating_capsule_uses_interior_not_overshot_surface(self):
        normal = np.asarray([[0.0, 0.0, 1.0]])
        anchor = np.zeros((1, 3))
        capsule_axis = np.asarray([[0.0, 0.0, 0.005]])
        overshot_surface = np.asarray([[0.0, 0.0, -0.005]])
        toward_interior = contact_facing_normals(normal, anchor, capsule_axis)
        toward_overshoot = contact_facing_normals(normal, anchor, overshot_surface)
        np.testing.assert_allclose(toward_interior, [[0.0, 0.0, 1.0]])
        np.testing.assert_allclose(toward_overshoot, [[0.0, 0.0, -1.0]])

    def test_v4_remapped_face_keeps_trusted_source_normal_sign(self):
        face = np.asarray([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        source = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.8, 0.2]])
        aligned = align_normal_axis_to_reference(face, source)
        np.testing.assert_allclose(aligned, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        self.assertTrue(np.all(np.sum(aligned * source, axis=-1) >= 0.0))

    def test_v4_normal_constraint_requires_source_capsule_support(self):
        landmarks = np.zeros((1, 1, 2, 22, 3), dtype=np.float32)
        landmarks[..., 1, 0] = 1.0
        points = np.zeros((1, 1, 2, 1, 3), dtype=np.float32)
        points[..., 0, 1] = 1.0
        normals = np.zeros_like(points)
        normals[:, :, 0, :, 1] = -1.0
        normals[:, :, 1, :, 1] = 1.0
        segments = np.zeros((1, 1, 2, 1), dtype=np.int16)
        valid = np.ones_like(segments, dtype=bool)
        supported = source_proxy_normal_support(
            landmarks, points, normals, segments, valid, 60.0,
        )
        np.testing.assert_array_equal(
            supported, np.asarray([[[[True], [False]]]]),
        )

    def test_v4_augmented_lagrangian_keeps_side_states_separate(self):
        import torch

        solver = AugmentedLagrangian(required_reduction=0.25)
        left = torch.tensor([2.0], requires_grad=True)
        right = torch.tensor([0.5], requires_grad=True)
        loss = solver.term("left_palm", left) + solver.term("right_palm", right)
        loss.backward()
        left_result = solver.update("left_palm", left)
        right_result = solver.update("right_palm", right)
        self.assertFalse(left_result["passed"])
        self.assertTrue(right_result["passed"])
        self.assertIsNot(solver.states["left_palm"], solver.states["right_palm"])

    def test_v4_augmented_lagrangian_uses_physical_tolerance_tubes(self):
        import torch

        solver = AugmentedLagrangian()
        inside = torch.tensor([[0.6, 0.6]], requires_grad=True)
        inside_loss = solver.term("inside", inside)
        inside_loss.backward()
        torch.testing.assert_close(inside.grad, torch.zeros_like(inside.grad))
        inside_result = solver.update("inside", inside)
        self.assertTrue(inside_result["passed"])
        self.assertEqual(float(torch.linalg.vector_norm(
            solver.states["inside"].multiplier,
        )), 0.0)

        outside = torch.tensor([[1.2, 0.0]], requires_grad=True)
        outside_loss = solver.term("outside", outside)
        outside_loss.backward()
        self.assertGreater(float(torch.linalg.vector_norm(outside.grad)), 0.0)
        outside_result = solver.update("outside", outside)
        self.assertFalse(outside_result["passed"])
        self.assertGreater(float(torch.linalg.vector_norm(
            solver.states["outside"].multiplier,
        )), 0.0)

    def test_v4_augmented_lagrangian_resets_multiplier_for_reduced_feasible_set(self):
        import torch

        solver = AugmentedLagrangian()
        solver.term("contact_left", torch.tensor([[0.5, 0.0, 0.0], [0.2, 0.0, 0.0]]))
        solver.states["contact_left"].penalty = 16.0
        result = solver.update("contact_left", torch.tensor([[0.4, 0.0, 0.0]]))
        self.assertTrue(result["multiplier_reset"])
        self.assertEqual(solver.states["contact_left"].multiplier.shape, (1, 3))
        self.assertEqual(solver.states["contact_left"].penalty, 16.0)

    def test_v4_support_foot_mask_and_c3_continuity(self):
        feet = np.zeros((8, 2, 2, 3), dtype=np.float64)
        feet[:, 1, :, 2] = 0.2
        support = support_foot_mask(feet, 90.0, 0.08, 0.08)
        self.assertTrue(support[:, 0].all())
        self.assertFalse(support[:, 1].any())
        times = np.linspace(0.0, 1.0, 91)
        basis, knots = c4_bspline_basis(times, 0.04, degree=5)
        coefficients = np.linspace(0.0, 1.0, basis.shape[1])[:, None]
        metrics = derivative_jump_metrics(knots, coefficients, degree=5)
        self.assertLess(metrics["order_3_maximum_jump"], 1e-2)

    def test_bvh_preserves_end_site_and_fk(self):
        motion = parse_bvh(BVH)
        joint = motion.joint_index("LeftHandIndex3")
        self.assertEqual(len(motion.joints[joint].end_sites), 1)
        np.testing.assert_allclose(motion.joints[joint].end_sites[0], [1.0, 0.0, 0.0])
        positions, rotations = motion.forward_kinematics(include_nonroot_translation=False)
        expected = positions[:, joint] + np.einsum(
            "tij,j->ti", rotations[:, joint], np.asarray([1.0, 0.0, 0.0]),
        )
        actual = motion.end_site_positions(joint, include_nonroot_translation=False)
        np.testing.assert_allclose(actual[:, 0], expected, atol=1e-6)

    def test_comfi_uses_serial_urdf_axes_not_xyz_labels(self):
        names = (
            "Left_Hip_Flexion_Extension", "Left_Hip_Abduction_Adduction",
            "Left_Hip_Internal_External_Rotation", "Right_Hip_Flexion_Extension",
            "Right_Hip_Abduction_Adduction", "Right_Hip_Internal_External_Rotation",
            "Left_Knee_Flexion_Extension", "Right_Knee_Flexion_Extension",
            "Left_Ankle_Plantarflexion_Dorsiflexion", "Left_Ankle_Inversion_Eversion",
            "Right_Ankle_Plantarflexion_Dorsiflexion", "Right_Ankle_Inversion_Eversion",
            "Lumbar_Flexion_Extension", "Lumbar_Lateral_Bending",
            "Thoracic_Flexion_Extension", "Thoracic_Lateral_Bending",
            "Thoracic_Internal_External_Rotation",
            "Cervical_Flexion_Extension", "Cervical_Lateral_Bending",
            "Cervical_Internal_External_Rotation", "Left_Clavicle_Elevation_Depression",
            "Right_Clavicle_Elevation_Depression", "Left_Shoulder_Flexion_Extension",
            "Left_Shoulder_Abduction_Adduction", "Left_Shoulder_Internal_External_Rotation",
            "Right_Shoulder_Flexion_Extension", "Right_Shoulder_Abduction_Adduction",
            "Right_Shoulder_Internal_External_Rotation", "Left_Elbow_Flexion_Extension",
            "Left_Elbow_Pronation_Supination", "Right_Elbow_Flexion_Extension",
            "Right_Elbow_Pronation_Supination", "Left_Wrist_Flexion_Extension",
            "Left_Wrist_Radial_Ulnar_Deviation", "Right_Wrist_Flexion_Extension",
            "Right_Wrist_Radial_Ulnar_Deviation",
        )
        row = {name + "[rad]": "0" for name in names}
        row["Left_Knee_Flexion_Extension[rad]"] = "0.4"
        row["Left_Clavicle_Elevation_Depression[rad]"] = "0.2"
        row["Right_Clavicle_Elevation_Depression[rad]"] = "0.2"
        row["Left_Elbow_Flexion_Extension[rad]"] = "0.3"
        row["Left_Elbow_Pronation_Supination[rad]"] = "0.1"
        pose = comfi_biomechanical_body_pose([row]).reshape(21, 3)
        basis = np.asarray([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        np.testing.assert_allclose(_COMFI_URDF_FROM_SMPLX, basis, atol=0.0)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(basis)), 1.0)
        knee_urdf = Rotation.from_rotvec([0.0, 0.0, -0.4]).as_matrix()
        np.testing.assert_allclose(
            Rotation.from_rotvec(pose[3]).as_matrix(), basis.T @ knee_urdf @ basis, atol=1e-6,
        )
        left_clavicle = Rotation.from_rotvec([0.2, 0.0, 0.0]).as_matrix()
        right_clavicle = Rotation.from_rotvec([-0.2, 0.0, 0.0]).as_matrix()
        np.testing.assert_allclose(Rotation.from_rotvec(pose[12]).as_matrix(), basis.T @ left_clavicle @ basis, atol=1e-6)
        np.testing.assert_allclose(Rotation.from_rotvec(pose[13]).as_matrix(), basis.T @ right_clavicle @ basis, atol=1e-6)
        expected_urdf = (
            Rotation.from_rotvec([0.0, 0.0, 0.3]).as_matrix() @
            Rotation.from_rotvec([0.0, -0.1, 0.0]).as_matrix()
        )
        np.testing.assert_allclose(
            Rotation.from_rotvec(pose[17]).as_matrix(), basis.T @ expected_urdf @ basis, atol=1e-6,
        )

    def test_hiphi_inventory_uses_release_metadata_without_tar_scan(self):
        root = self.run_dir / "hiphi_metadata_inventory"
        shutil.rmtree(root, ignore_errors=True)
        (root / "data").mkdir(parents=True)
        (root / "metadata").mkdir()
        (root / ".download_verified").touch()
        archive_name = "HiPHI_data_part_0001_of_0032.tar.zst"
        (root / "data" / archive_name).touch()
        (root / "data/motion_to_part.csv").write_text(
            "motion_id,mirrored_motion_id,archive_name,uncompressed_bytes\n"
            f"Posture-sit_0001,Posture-sit_0001__mirror,{archive_name},42\n"
        )
        (root / "metadata/hiphi_metadata.csv").write_text(
            "motion_id,frame,lu,mirrored\n"
            "Posture-sit_0001,Posture,sit,false\n"
            "Posture-sit_0001__mirror,Posture,sit,true\n"
        )
        config = SimpleNamespace(
            raw={"datasets": {"hiphi": {"require_download_verified": True}},
                 "batch": {"cache_subdir": "cache"}},
            data_root=self.run_dir,
            dataset_path=lambda dataset: root,
        )
        with mock.patch("utils.motion_retarget.adapters._list_tar_members", side_effect=AssertionError):
            rows = inventory_hiphi(config)
        self.assertEqual([row.sequence_id for row in rows], ["Posture-sit_0001"])
        self.assertEqual(rows[0].member, "HiPHI/data/Posture/sit/Posture-sit_0001/motion_actor.bvh")

    def test_mano_order_and_shapes(self):
        motion = parse_bvh(BVH)
        root, body, left, right = bvh_to_smplx_pose(motion)
        self.assertEqual(root.shape, (2, 3))
        self.assertEqual(body.shape, (2, 63))
        self.assertEqual(left.shape, (2, 45))
        self.assertFalse(np.allclose(left[:, :9], 0))
        self.assertTrue(np.allclose(right, 0))

    def test_left_mano_reflection_preserves_so3_and_flips_yz_axes(self):
        rotvec = np.asarray([[0.2, 0.3, -0.4]])
        reflected = reflect_left_mano_matrices(Rotation.from_rotvec(rotvec).as_matrix())
        converted = Rotation.from_matrix(reflected).as_rotvec()
        np.testing.assert_allclose(converted, [[0.2, -0.3, 0.4]], atol=1e-7)
        np.testing.assert_allclose(np.linalg.det(reflected), 1.0, atol=1e-7)

    def test_wilor_keypoints_drop_singleton_opencv_axes(self):
        expected = np.arange(42, dtype=np.float32).reshape(21, 2)
        np.testing.assert_array_equal(normalize_keypoints_2d(expected[:, None, :]), expected)
        np.testing.assert_array_equal(normalize_keypoints_2d(expected[None, :, :]), expected)
        with self.assertRaises(ValueError):
            normalize_keypoints_2d(np.zeros((2, 21, 2)))

    def test_wilor_multiview_triangulation_rejects_outlier_camera(self):
        points = np.stack([
            np.linspace(-0.2, 0.2, 21), np.linspace(-0.1, 0.1, 21), np.full(21, 4.0),
        ], axis=1)
        translations = ([0.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [-1.0, -1.0, 0.0])
        calibrations, rows = [], []
        for view, translation in enumerate(translations):
            projection = np.concatenate([np.eye(3), np.asarray(translation)[:, None]], axis=1)
            homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
            image = (projection @ homogeneous.T).T
            image = image[:, :2] / image[:, 2:3]
            if view == 3:
                image += 2.0
            calibrations.append({"K": np.eye(3), "D": np.zeros(5), "P": projection})
            rows.append({"view": view, "keypoints": image, "img_size": np.asarray([10.0, 10.0])})
        _, errors, inliers = robust_triangulate_and_score(rows, calibrations, 0.03)
        self.assertEqual(inliers, [0, 1, 2])
        self.assertLess(max(errors[index] for index in inliers), 0.03)
        self.assertGreater(errors[3], 0.03)

    def test_so3_short_and_long_gaps(self):
        pose = np.zeros((8, 45), dtype=np.float32)
        pose[3, 0] = 0.6
        pose[7, 0] = 1.2
        valid = np.asarray([True, False, False, True, False, False, False, True])
        result, imputed = interpolate_pose_gaps(pose, valid, fps=10, max_gap_seconds=0.2)
        self.assertAlmostEqual(float(result[1, 0]), 0.2, places=5)
        self.assertAlmostEqual(float(result[2, 0]), 0.4, places=5)
        np.testing.assert_allclose(result[4:7, 0], [0.69375, 0.9, 1.10625], atol=1e-5)
        self.assertTrue(imputed[1])

    def test_y_up_root_maps_template_up_to_world_z(self):
        root, translation, transform = canonicalize_bvh_root(
            np.zeros((2, 3)), np.zeros((2, 3)), 1.8, 1.8,
        )
        matrix = Rotation.from_rotvec(root[0]).as_matrix()
        np.testing.assert_allclose(matrix @ np.asarray([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(translation, 0.0)

    def test_source_transform_reproduces_scaled_root_translation(self):
        source_cm = np.asarray([[100.0, 200.0, 300.0], [200.0, 300.0, 400.0]])
        _, translation, transform = canonicalize_bvh_root(
            np.zeros((2, 3)), source_cm, target_height_m=1.0, source_height_m=2.0,
        )
        source_m_scaled = source_cm / 100.0 * 0.5
        reconstructed = source_m_scaled @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(reconstructed, translation, atol=1e-6)

    def test_native_root_position_converts_to_smplx_translation_with_pivot(self):
        source_cm = np.asarray([[20.0, 105.0, -30.0], [25.0, 110.0, -28.0]])
        _, canonical_root_position, transform = canonicalize_bvh_root(
            np.zeros((2, 3)), source_cm, target_height_m=1.65, source_height_m=1.80,
        )
        target_pelvis_pivot = np.asarray([0.001, -0.352, 0.014])
        smplx_translation = canonical_root_position - target_pelvis_pivot
        source_m_scaled = source_cm / 100.0 * (1.65 / 1.80)
        transformed_source_root = source_m_scaled @ transform[:3, :3].T + transform[:3, 3]
        np.testing.assert_allclose(
            transformed_source_root, smplx_translation + target_pelvis_pivot, atol=1e-6,
        )

    def test_andy_gvhmr_motion_sync_recovers_video_to_body_offset(self):
        body_fps, video_fps = 60.0, 30.0
        true_offset = 0.7
        body_times = np.arange(int(12 * body_fps)) / body_fps
        video_times = np.arange(int(10 * video_fps)) / video_fps

        def angle(times):
            return (0.5 * np.exp(-((times - 2.1) / 0.25) ** 2) +
                    0.8 * np.exp(-((times - 6.4) / 0.45) ** 2) -
                    0.35 * np.exp(-((times - 8.2) / 0.18) ** 2))

        body_pose = np.zeros((len(body_times), 63), dtype=np.float32)
        video_pose = np.zeros((len(video_times), 63), dtype=np.float32)
        body_pose[:, 9] = angle(body_times)
        video_pose[:, 9] = angle(video_times + true_offset)
        offset, correlation, overlap = _estimate_motion_time_offset(
            np.zeros((len(body_times), 3)), body_pose, body_fps,
            np.zeros((len(video_times), 3)), video_pose, video_fps,
            max_abs_offset_s=2.0, min_overlap_s=2.0,
        )
        self.assertAlmostEqual(offset, true_offset, delta=1.0 / video_fps)
        self.assertGreater(correlation, 0.95)
        self.assertGreaterEqual(overlap, int(2 * video_fps))

    def test_fitted_root_canonicalization_accounts_for_smplx_pelvis_pivot(self):
        root = np.asarray([[-np.pi / 2.0, 0.0, 0.0]], dtype=np.float32)
        translation = np.asarray([[0.2, 0.3, 1.0]], dtype=np.float32)
        pivot = np.asarray([0.0, -0.35, 0.0])
        _, canonical_translation, transform = _canonicalize_z_up(
            root, translation, 1.0, pivot,
        )
        source_root_joint = translation[0] + pivot
        transformed_root_joint = transform[:3, :3] @ source_root_joint + transform[:3, 3]
        canonical_root_joint = canonical_translation[0] + pivot
        np.testing.assert_allclose(transformed_root_joint, canonical_root_joint, atol=1e-6)

    def test_npz_roundtrip_and_atomic_path(self):
        count = 10
        profile = RetargetProfile("test", "male", 1.78, (0.0,) * 16)
        sequence = SequenceRef("synthetic", "one", Path("source.bvh"))
        timestamps = np.arange(count) / 30.0
        continuous = c3_archive_payload(
            timestamps, np.zeros((count, 3)), np.zeros((count, 63)),
            np.zeros((count, 45)), np.zeros((count, 45)), np.zeros((count, 3)),
        )
        record = MotionRecord(
            sequence, profile, 30.0, timestamps,
            np.zeros((count, 3)), np.zeros((count, 63)), np.zeros((count, 45)), np.zeros((count, 45)),
            np.zeros((count, 3)), 1.0,
            np.ones(count, bool), np.ones(count, bool), np.ones(count), np.ones(count),
            np.zeros(count, bool), np.zeros(count, bool), np.eye(4),
            {
                "test": True, "source_motion_scale": 1.0, "config_hash": "config-one",
                "input_hash": "a" * 64, "model_dependency_hash": "b" * 64,
                "profile_height_error_m": 0.0,
                "target_root_pivot_m": [0.0, -0.35, 0.0],
            }, continuous,
        )
        target = self.run_dir / "roundtrip.npz"
        write_motion_npz(target, record)
        data = read_motion_npz(target)
        self.assertEqual(data["poses"].shape, (count, 165))
        self.assertEqual(data["pose_aa_smplsim"].shape, (count, 156))
        self.assertEqual(str(data["profile_source"]), "configured")
        self.assertEqual(str(data["betas_source"]), "configured")
        self.assertEqual(str(data["hand_pose_convention"]), HAND_POSE_CONVENTION)
        self.assertEqual(float(data["source_motion_scale"]), 1.0)
        self.assertEqual(str(data["config_hash"].item()), "config-one")
        self.assertEqual(str(data["input_hash"].item()), "a" * 64)
        self.assertEqual(str(data["model_dependency_hash"].item()), "b" * 64)
        self.assertEqual(
            str(data["quality_json"].item()),
            '{"config_hash": "config-one", "input_hash": "' + "a" * 64 +
            '", "model_dependency_hash": "' + "b" * 64 +
            '", "profile_height_error_m": 0.0, "source_motion_scale": 1.0, '
            '"target_root_pivot_m": [0.0, -0.35, 0.0], "test": true}',
        )
        self.assertFalse(list(self.run_dir.glob("*.partial")))
        archive_report = validate_archive(target, "config-one", 0.002)
        self.assertTrue(archive_report["passed"])
        self.assertTrue(all(type(value) is bool for value in archive_report["checks"].values()))
        self.assertFalse(validate_archive(target, "different")["passed"])

    def test_object_csv_quaternion_and_time_contract(self):
        payload = (
            "frame,time_sec,px,py,pz,qx,qy,qz,qw\n"
            "0,0.000000,1,2,3,0,0,0,1\n"
            "1,0.011111,2,3,4,0,0,0,-1\n"
        )
        timestamps, translation, quaternion = parse_object_track_csv(payload, expected_frames=2)
        np.testing.assert_allclose(timestamps, [0.0, 0.011111])
        np.testing.assert_allclose(translation[1], [2.0, 3.0, 4.0])
        self.assertGreater(float(np.dot(quaternion[0], quaternion[1])), 0.999)

    def test_object_canonical_transform_is_rigid_without_scale(self):
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
        transform[:3, 3] = [0.25, -0.5, 1.0]
        source = np.asarray([[2.0, 3.0, 4.0]])
        translation, quaternion = canonicalize_object_track(
            source, np.asarray([[0.0, 0.0, 0.0, 1.0]]), transform,
        )
        np.testing.assert_allclose(translation, source @ transform[:3, :3].T + transform[:3, 3], atol=1e-7)
        np.testing.assert_allclose(
            Rotation.from_quat(quaternion).as_matrix()[0], transform[:3, :3], atol=1e-6,
        )

    def test_contact_hysteresis_rejects_single_frame_flash(self):
        distance = np.asarray([0.08, 0.04, 0.08, 0.04, 0.045, 0.06, 0.08])
        mask = contact_hysteresis(distance, 0.05, 0.07, minimum_frames=3)
        np.testing.assert_array_equal(mask, [False, False, False, True, True, True, False])

    def test_signed_distance_grid_has_negative_inside(self):
        import trimesh
        mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        grid = build_signed_distance_grid(mesh, pitch=0.02, padding=0.04, sample_count=2000)
        values = grid.sample(np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]]))
        self.assertLess(values[0], 0.0)
        self.assertGreater(values[1], 0.0)
        self.assertGreaterEqual(grid.sign_agreement, 0.55)
        distance, normal = grid.query_surface(np.asarray([[0.25, 0.02, -0.03]]))
        self.assertAlmostEqual(float(distance[0]), 0.15, places=6)
        self.assertEqual(normal.shape, (1, 3))
        self.assertIsNotNone(grid.triangle_query)

    def test_triangle_surface_query_returns_face_and_barycentric(self):
        import trimesh
        mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
        query = TriangleSurfaceQuery(mesh)
        closest, distance, faces, barycentric = query.query(
            np.asarray([[0.25, 0.02, -0.03], [0.0, 0.0, 0.0]]),
        )
        np.testing.assert_allclose(closest[0], [0.1, 0.02, -0.03], atol=1e-6)
        self.assertAlmostEqual(float(distance[0]), 0.15, places=6)
        self.assertTrue(np.all(faces >= 0))
        np.testing.assert_allclose(barycentric.sum(axis=1), 1.0, atol=1e-6)

    def test_dynamic_hand_proxy_has_palm_fingers_and_actor_scale(self):
        frames = 3
        landmarks = np.zeros((frames, 22, 3), dtype=np.float32)
        landmarks[:, 0] = [0.0, 0.0, 0.0]
        landmarks[:, 1] = [0.0, 0.05, 0.0]
        for finger in range(5):
            base = 2 + finger * 4
            x = (finger - 2) * 0.02
            for segment in range(4):
                landmarks[:, base + segment] = [x, 0.07 + 0.025 * segment, 0.0]
        points, radii, regions, segments = _hand_proxy(landmarks)
        self.assertEqual(points.shape[0], frames)
        self.assertEqual(points.shape[1], len(radii))
        self.assertEqual(set(regions.tolist()), set(range(6)))
        self.assertEqual(set(segments.tolist()), set(range(21)))
        self.assertEqual(len(segments), len(radii))
        self.assertGreater(float(radii[regions == 0].median() if hasattr(radii, "median") else np.median(radii[regions == 0])), 0.0)

    def test_target_hand_reference_uses_source_directions_and_target_lengths(self):
        raw = np.zeros((3, 1, 2, 22, 3), dtype=np.float32)
        target = np.zeros_like(raw)
        raw[..., 1, :] = [0.0, 1.0, 0.0]
        target[..., 1, :] = [0.0, 1.0, 0.0]
        for finger in range(5):
            for step in range(4):
                index = 2 + finger * 4 + step
                raw[..., index, :] = [0.0, 2.0 + 2.0 * step, 0.0]
                target[..., index, :] = [0.5 + 0.5 * step, 1.0, 0.0]
        adapted = _adapt_hand_landmark_lengths(
            raw, target, np.ones((3, 1, 2), dtype=bool),
        )
        for finger in range(5):
            chain = [1, *(2 + finger * 4 + step for step in range(4))]
            adapted_lengths = np.linalg.norm(np.diff(adapted[0, 0, 0, chain], axis=0), axis=1)
            target_lengths = np.linalg.norm(np.diff(target[0, 0, 0, chain], axis=0), axis=1)
            np.testing.assert_allclose(adapted_lengths, target_lengths, atol=1e-7)
            # Palm-to-MCP is immutable target hand shape; all three
            # controllable phalanges follow the released +Y BVH direction.
            internal = np.diff(adapted[0, 0, 0, chain[1:]], axis=0)
            np.testing.assert_allclose(internal[:, 0], 0.0, atol=1e-7)
            self.assertTrue(np.all(internal[:, 1] > 0.0))

    def test_target_hand_reference_applies_palm_calibration_to_fingers(self):
        raw = np.zeros((1, 1, 2, 22, 3), dtype=np.float32)
        target = np.zeros_like(raw)
        raw[..., 1, :] = [0.0, 0.0, 0.0]
        target[..., 1, :] = [0.0, 0.0, 0.0]
        for finger in range(5):
            base = 2 + 4 * finger
            raw[..., base, :] = [0.02 * (finger - 2), 0.04, 0.0]
            target[..., base, :] = [0.02 * (finger - 2), 0.04, 0.0]
            for step in range(1, 4):
                raw[..., base + step, :] = raw[..., base, :] + [0.02 * step, 0.0, 0.0]
                target[..., base + step, :] = target[..., base, :] + [0.02 * step, 0.0, 0.0]
        raw[..., 0, :] = [0.0, -0.04, 0.0]
        target[..., 0, :] = [0.0, -0.04, 0.0]
        source_frame = np.broadcast_to(
            np.eye(3, dtype=np.float32), (1, 1, 2, 3, 3),
        ).copy()
        desired_rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix().astype(np.float32)
        desired_frame = np.broadcast_to(
            desired_rotation, (1, 1, 2, 3, 3),
        ).copy()
        adapted = _adapt_hand_landmark_lengths(
            raw, target, np.ones((1, 1, 2), dtype=bool),
            source_frame, desired_frame, source_frame,
        )
        # Source +X phalanges must become calibrated target +Y phalanges.
        for finger in range(5):
            base = 2 + 4 * finger
            internal = np.diff(adapted[0, 0, 0, base:base + 4], axis=0)
            np.testing.assert_allclose(internal[:, 0], 0.0, atol=1e-7)
            self.assertTrue(np.all(internal[:, 1] > 0.0))
        # Target-shape wrist/MCP offsets use the identical calibrated frame.
        np.testing.assert_allclose(adapted[0, 0, 0, 0], [0.04, 0.0, 0.0], atol=1e-7)

    def test_penetration_p95_uses_positive_depth_samples_not_frame_maxima(self):
        metrics = _penetration_metrics({
            "penetration": np.asarray([[0.005], [0.002]]),
            "penetration_samples": np.asarray([[[0.0, 0.001, 0.005]], [[0.0, 0.0015, 0.002]]]),
        })
        self.assertAlmostEqual(metrics["penetration_p95_m"], 0.00455)
        self.assertAlmostEqual(metrics["penetration_frame_max_p95_m"], 0.00485)
        self.assertEqual(metrics["penetration_positive_sample_count"], 4)
        histogram = _penetration_metrics({
            "penetration": np.asarray([[0.005], [0.002]]),
            "penetration_histogram_edges": np.asarray([0.0, 0.001, 0.002, 0.003]),
            "penetration_histogram_counts": np.asarray([[5, 3, 2]]),
        })
        self.assertAlmostEqual(histogram["penetration_p95_m"], 0.00275)
        self.assertEqual(histogram["penetration_positive_sample_count"], 10)

    def test_soft_tissue_penetration_never_excuses_nonhand_overlap(self):
        metrics = {
            "all": {"penetration_p95_m": 0.0047, "penetration_max_m": 0.006,
                    "positive_sample_count": 10000},
            "contact_hand": {"penetration_p95_m": 0.0047, "penetration_max_m": 0.006,
                             "positive_sample_count": 9960},
            "noncontact_hand": {"penetration_p95_m": 0.0048, "penetration_max_m": 0.006,
                                "positive_sample_count": 40},
            "nonhand": {"penetration_p95_m": 0.0, "penetration_max_m": 0.0,
                        "positive_sample_count": 0},
        }
        strict, soft, fraction = classify_penetration(metrics, 0.003, 0.010)
        self.assertFalse(strict)
        self.assertTrue(soft)
        self.assertAlmostEqual(fraction, 0.004)
        metrics["nonhand"]["penetration_max_m"] = 0.004
        self.assertFalse(classify_penetration(metrics, 0.003, 0.010)[1])

    def test_annotation_prior_only_bridges_geometrically_eligible_frames(self):
        contact = np.asarray([False, True, True, False, False, True, False, False])
        eligible = np.asarray([False, True, True, True, False, True, True, False])
        result, assisted = _bridge_annotation_prior(contact, eligible, maximum_gap_frames=2)
        self.assertTrue(result[3])
        self.assertFalse(result[4])
        self.assertTrue(assisted[3])
        self.assertFalse(assisted[4])

    def test_v4_source_contact_cache_roundtrip_is_source_only(self):
        frames, objects, contacts = 3, 1, 2
        value = SourceHandContacts(
            np.asarray([f"l{i}" for i in range(22)]),
            np.asarray([f"r{i}" for i in range(6)]),
            np.asarray([f"s{i}" for i in range(21)]),
            np.zeros((frames, objects, 2, 22, 3)),
            np.zeros((frames, objects, 2, 22, 3)),
            np.ones((frames, objects, 2), bool),
            np.ones((frames, objects, 2), bool),
            np.zeros((frames, objects, 2), bool),
            np.zeros((frames, objects, 2)),
            np.zeros((frames, objects, 2, contacts, 3)),
            np.zeros((frames, objects, 2, contacts, 3)),
            np.zeros((frames, objects, 2, contacts), np.int32),
            np.full((frames, objects, 2, contacts, 3), 1.0 / 3.0),
            np.zeros((frames, objects, 2, contacts, 3)),
            np.zeros((frames, objects, 2, contacts), np.int16),
            np.zeros((frames, objects, 2, contacts), np.int16),
            np.zeros((frames, objects, 2, contacts), np.int16),
            np.ones((frames, objects, 2, contacts), bool),
            np.full((2, 6), 0.01),
        )
        path = self.run_dir / "source_contacts.npz"
        write_source_hand_contacts(path, value)
        loaded = read_source_hand_contacts(path)
        np.testing.assert_array_equal(loaded.contact, value.contact)
        np.testing.assert_allclose(loaded.contact_barycentric, value.contact_barycentric)

    def test_canonical_smplx_factory_has_one_hand_mean_convention(self):
        model_a = create_smplx_model(self.config.smplx_model_dir, "male", 16)
        model_b = create_smplx_model(self.config.smplx_model_dir, "male", 16)
        import torch
        pose = torch.zeros((1, 45)); pose[0, 0] = 0.35
        with torch.no_grad():
            a = model_a(left_hand_pose=pose, right_hand_pose=torch.zeros_like(pose)).vertices
            b = model_b(left_hand_pose=pose, right_hand_pose=torch.zeros_like(pose)).vertices
        torch.testing.assert_close(a, b, atol=1e-7, rtol=0.0)

    def test_smplx_hand_landmarks_use_true_fingertip_vertices(self):
        model = create_smplx_model(self.config.smplx_model_dir, "female", 16)
        import torch
        with torch.no_grad():
            output = model(betas=torch.zeros((1, 16)))
            landmarks = smplx_hand_landmarks(output.joints, output.vertices)
        for side_index, side in enumerate(("left", "right")):
            expected = output.vertices[0, list(HAND_TIP_VERTEX_IDS[side])]
            actual = landmarks[0, side_index, [5, 9, 13, 17, 21]]
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0.0)

    def test_hand_dynamic_shape_gate_is_source_relative_not_delta_clamp(self):
        source = np.zeros((4, 1, 22, 3), dtype=np.float64)
        source[..., 0, :] = [0.0, 0.0, 0.0]
        source[..., 1, :] = [0.0, 0.02, 0.0]
        for finger in range(5):
            base = 2 + 4 * finger
            for joint in range(4):
                source[..., base + joint, :] = [0.01 * finger, 0.03 + 0.01 * joint, 0.0]
        target = source.copy()
        exact = _hand_dynamic_shape_errors(source, target, np.ones((4, 1), bool))
        self.assertLess(exact["bone_direction_p95_deg"], 1e-6)
        self.assertLess(exact["finger_bend_p95_deg"], 1e-6)
        target[..., 21, :] += [0.04, 0.0, 0.0]
        wrong = _hand_dynamic_shape_errors(source, target, np.ones((4, 1), bool))
        self.assertGreater(wrong["bone_direction_p95_deg"], 2.0)
        self.assertGreater(wrong["thumb_opposition_p95_m"], 0.005)

    def test_schema_v4_dynamic_hand_multi_object_roundtrip(self):
        count, objects = 10, 2
        profile = RetargetProfile("test", "female", 1.65, (0.0,) * 16)
        timestamps = np.arange(count) / 30.0
        record = MotionRecord(
            SequenceRef("hiphi", "objects", Path("source.bvh")), profile, 30.0, timestamps,
            np.zeros((count, 3)), np.zeros((count, 63)), np.zeros((count, 45)), np.zeros((count, 45)),
            np.zeros((count, 3)), 1.0, np.ones(count, bool), np.ones(count, bool),
            np.ones(count), np.ones(count), np.zeros(count, bool), np.zeros(count, bool), np.eye(4),
            {"source_motion_scale": 1.0, "config_hash": "c", "input_hash": "a" * 64,
             "model_dependency_hash": "b" * 64},
            c3_archive_payload(timestamps, np.zeros((count, 3)), np.zeros((count, 63)),
                               np.zeros((count, 45)), np.zeros((count, 45)), np.zeros((count, 3))),
        )
        quaternion = np.zeros((count, objects, 4)); quaternion[..., 3] = 1.0
        record.objects = ObjectTrackRecord(
            np.asarray(["one", "two"]), np.asarray(["box", "table"]),
            np.zeros((count, objects, 3)), quaternion, np.ones((count, objects), bool),
            np.zeros((count, objects, 3)), quaternion.copy(), timestamps, timestamps.copy(),
            np.asarray(["a.obj", "b.obj"]), np.asarray(["a" * 64, "b" * 64]), np.full(objects, 0.01),
            np.zeros((count, objects), bool), np.zeros((count, objects), bool),
            np.zeros((count, objects), bool), np.ones((count, objects)), np.ones((count, objects)),
            np.ones((count, objects)), np.zeros((count, objects)),
            np.zeros((count, objects, 3)), np.zeros((count, objects, 3)),
            np.full((count, objects), -1), np.full((count, objects), -1),
        )
        contacts = 3
        dynamic_shape = (count, objects, 2, contacts)
        record.objects.source_hand_landmark_names = np.asarray([f"landmark_{i}" for i in range(22)])
        record.objects.source_hand_region_names = np.asarray([f"region_{i}" for i in range(6)])
        record.objects.source_hand_segment_names = np.asarray([f"segment_{i}" for i in range(21)])
        record.objects.source_hand_landmarks_local = np.zeros((count, objects, 2, 22, 3))
        record.objects.source_hand_landmark_velocity_local = np.zeros((count, objects, 2, 22, 3))
        record.objects.target_hand_landmarks_local = np.zeros((count, objects, 2, 22, 3))
        record.objects.target_reference_hand_landmarks_local = np.zeros((count, objects, 2, 22, 3))
        record.objects.source_hand_observed = np.ones((count, objects, 2), bool)
        record.objects.source_hand_contact = np.ones((count, objects, 2), bool)
        record.objects.annotation_assisted = np.zeros((count, objects, 2), bool)
        record.objects.source_hand_proxy_gap_m = np.zeros((count, objects, 2))
        record.objects.contact_source_local = np.zeros((*dynamic_shape, 3))
        record.objects.contact_anchor_local = np.zeros((*dynamic_shape, 3))
        record.objects.contact_face = np.zeros(dynamic_shape, dtype=np.int32)
        record.objects.contact_barycentric = np.full((*dynamic_shape, 3), 1.0 / 3.0)
        record.objects.contact_normal_local = np.zeros((*dynamic_shape, 3))
        record.objects.contact_region = np.zeros(dynamic_shape, dtype=np.int16)
        record.objects.contact_segment = np.zeros(dynamic_shape, dtype=np.int16)
        record.objects.contact_proxy_index = np.zeros(dynamic_shape, dtype=np.int16)
        record.objects.contact_valid = np.ones(dynamic_shape, bool)
        record.objects.contact_constraint_valid = np.ones(dynamic_shape, bool)
        record.objects.contact_target_vertex = np.zeros(dynamic_shape, dtype=np.int32)
        record.objects.source_hand_proxy_radii_m = np.full((2, 6), 0.01)
        target = self.run_dir / "objects_v4.npz"
        write_motion_npz(target, record)
        data = read_motion_npz(target)
        self.assertEqual(int(data["schema_version"]), 4)
        self.assertEqual(data["object_transl"].shape, (count, objects, 3))
        self.assertEqual(data["dynamic_contact_proxy_index"].shape, dynamic_shape)
        record.objects.solver_version = "constrained_se3_v4"
        record.objects.solver_pilot_windows = True
        record.objects.solver_pilot_frame_mask = np.ones(count, dtype=bool)
        record.objects.source_palm_position_local = np.zeros((count, objects, 2, 3))
        record.objects.target_palm_position_local = np.zeros((count, objects, 2, 3))
        palm_quaternion = np.zeros((count, objects, 2, 4)); palm_quaternion[..., 3] = 1.0
        record.objects.source_palm_quat_xyzw_local = palm_quaternion
        record.objects.target_palm_quat_xyzw_local = palm_quaternion.copy()
        record.objects.remapped_contact_anchor_local = np.zeros((*dynamic_shape, 3))
        record.objects.remapped_contact_face = np.zeros(dynamic_shape, dtype=np.int32)
        record.objects.remapped_contact_barycentric = np.full((*dynamic_shape, 3), 1.0 / 3.0)
        record.objects.remapped_contact_geodesic_m = np.zeros(dynamic_shape)
        record.objects.palm_constraint_multiplier = np.zeros((count, objects, 2, 6))
        record.objects.contact_constraint_multiplier = np.zeros((*dynamic_shape, 3))
        record.objects.solver_layer_residual_p95 = np.zeros((5, 2))
        record.objects.solver_layer_penalty = np.ones((5, 2))
        record.objects.solver_layer_multiplier_norm = np.zeros((5, 2))
        record.objects.solver_layer_converged = np.ones((5, 2), bool)
        record.objects.palm_jacobian_rank = np.full((count, 2), 6, dtype=np.int16)
        record.objects.palm_jacobian_condition = np.ones((count, 2))
        record.objects.full_body_fallback = np.zeros(count, bool)
        record.objects.support_foot = np.ones((count, 2), bool)
        record.objects.support_foot_baseline_world = np.zeros((count, 2, 2, 3))
        record.objects.support_foot_target_world = np.zeros((count, 2, 2, 3))
        record.objects.root_fallback_translation = np.zeros((count, 3))
        record.objects.root_fallback_rotvec = np.zeros((count, 3))
        degree = 5
        controls = 8
        record.objects.solver_spline_knots = np.r_[
            np.zeros(degree + 1), [1.0 / 3.0, 2.0 / 3.0], np.ones(degree + 1),
        ]
        record.objects.solver_spline_degree = degree
        record.objects.solver_spline_body_delta_coeff = np.zeros((controls, 21, 3))
        record.objects.solver_spline_root_delta_coeff = np.zeros((controls, 3))
        record.objects.solver_spline_translation_delta_coeff = np.zeros((controls, 3))
        record.objects.solver_spline_left_hand_delta_coeff = np.zeros((controls, 15, 3))
        record.objects.solver_spline_right_hand_delta_coeff = np.zeros((controls, 15, 3))
        record.objects.solver_spline_rotation_basis = "baseline_composed_right_SO3_rotvec_delta"
        target_v5 = self.run_dir / "objects_v5.npz"
        write_motion_npz(target_v5, record)
        data_v5 = read_motion_npz(target_v5)
        self.assertEqual(int(data_v5["schema_version"]), 5)
        self.assertEqual(str(data_v5["object_solver_version"]), "constrained_se3_v4")
        self.assertEqual(data_v5["palm_constraint_multiplier"].shape, (count, objects, 2, 6))
        self.assertEqual(data_v5["solver_spline_body_delta_coeff"].shape, (controls, 21, 3))
        self.assertFalse(list(self.run_dir.glob("objects_v4.npz.*.partial")))

    def test_object_config_inherits_base_config(self):
        config = load_config(ROOT / "config/motion_retarget_hiphi_object.yaml")
        self.assertIn("object_pilot", config.raw)
        self.assertTrue(str(config.output_root).endswith("motion_retarget_hiphi_object_pilot"))
        self.assertEqual(config.raw["datasets"]["hiphi"]["input_mode"], "bvh")

    def test_source_native_profile_uses_actor_metadata_and_zero_betas(self):
        reference = SequenceRef(
            "hiphi", "one", Path("source.bvh"), metadata={
                "actor_id": "A001", "gender": "M", "height_m": 1.73,
            },
        )
        profiles = profiles_for_reference(self.config, reference, ["source_native"])
        self.assertEqual(len(profiles), 1)
        profile = profiles[0]
        self.assertEqual((profile.name, profile.gender, profile.height_m), ("source_native", "male", 1.73))
        self.assertEqual(profile.actor_id, "A001")
        self.assertEqual(profile.betas, (0.0,) * 16)
        self.assertEqual(profile.betas_source, "zero_default_no_source_smplx_betas")

    def test_source_native_profile_reuses_complete_smplx_metadata(self):
        reference = SequenceRef(
            "hiphi", "one", Path("source.bvh"), metadata={
                "actor_id": "A001", "smplx": {
                    "gender": "female", "height_m": 1.64, "betas": [0.1] * 16,
                },
            },
        )
        profile = profiles_for_reference(self.config, reference, ["source_native"])[0]
        self.assertEqual(profile.profile_source, "source_smplx")
        self.assertEqual(profile.betas_source, "source_smplx")
        self.assertEqual(profile.betas, (0.1,) * 16)

    def test_source_native_missing_metadata_falls_back_to_two_profiles(self):
        reference = SequenceRef("hiphi", "one", Path("source.bvh"))
        profiles = profiles_for_reference(self.config, reference, ["source_native"])
        self.assertEqual([profile.name for profile in profiles], ["male_178", "female_165"])
        self.assertEqual(normalize_gender("F"), "female")

    def test_explicit_gpu_assignment_is_not_pid_dependent(self):
        with mock.patch.dict(os.environ, {"MOTION_RETARGET_GPU_INDEX": "1"}):
            self.assertEqual(assigned_gpu_index(2), 1)
        with mock.patch.dict(os.environ, {"MOTION_RETARGET_GPU_INDEX": "2"}):
            with self.assertRaises(ValueError):
                assigned_gpu_index(2)

    def test_physical_gpu_token_prefers_uuid(self):
        physical_gpu_token.cache_clear()

    def test_physical_gpu_count_uses_nvidia_inventory(self):
        physical_gpu_count.cache_clear()
        with mock.patch(
            "utils.motion_retarget.gpu.subprocess.check_output", return_value="0\n1\n",
        ):
            self.assertEqual(physical_gpu_count(), 2)
        physical_gpu_count.cache_clear()
        with mock.patch(
            "utils.motion_retarget.gpu.subprocess.check_output",
            return_value="0, GPU-zero\n1, GPU-one\n",
        ):
            self.assertEqual(physical_gpu_token(1), "GPU-one")
        physical_gpu_token.cache_clear()

    def test_stale_hiphi_cleanup_requires_sentinel_and_dead_pid(self):
        root = self.run_dir / "scratch/motion_retarget/hiphi"
        stale, unrelated = root / "stale", root / "unrelated"
        stale.mkdir(parents=True)
        unrelated.mkdir()
        (stale / STAGE_SENTINEL).write_text(
            '{"pid": 99999999, "created_at": 1, "archive": "x"}'
        )
        config = SimpleNamespace(
            data_root=self.run_dir, raw={"batch": {"scratch_subdir": "scratch"}},
        )
        removed = cleanup_stale_hiphi_stages(config, minimum_age_seconds=0)
        self.assertEqual(removed, [stale])
        self.assertFalse(stale.exists())
        self.assertTrue(unrelated.exists())

    def test_quintic_rotation_projection_and_archive_are_c3(self):
        count = 120
        times = np.arange(count, dtype=np.float64) / 40.0
        smooth = 0.35 * np.sin(2.0 * np.pi * 0.5 * times)
        noisy = smooth + 0.09 * np.where(np.arange(count) % 2, 1.0, -1.0)
        pose = np.zeros((count, 45), dtype=np.float32)
        pose[:, 0] = noisy
        filtered, metadata = c3_smooth_rotvec(times, pose, 0.16)
        self.assertGreaterEqual(metadata["continuity_order"], 3)
        self.assertLess(np.percentile(np.abs(np.diff(filtered[:, 0])), 95),
                        np.percentile(np.abs(np.diff(pose[:, 0])), 95))
        payload = c3_archive_payload(
            times, np.zeros((count, 3)), np.zeros((count, 63)), filtered, filtered,
            np.zeros((count, 3)),
        )
        self.assertGreaterEqual(int(payload["trajectory_continuity_order"]), 3)
        self.assertTrue(np.isfinite(payload["c3_left_hand_quat_coeffs"]).all())

    def test_ledger_resume_requires_existing_output_and_hash(self):
        ledger = Ledger(self.run_dir / "ledger.sqlite3")
        output = self.run_dir / "done.npz"
        output.touch()
        ledger.upsert(
            "d", "s", "p", "complete", "hash", str(output), {"ok": True},
            input_hash="input", model_hash="model",
        )
        self.assertTrue(ledger.completed("d", "s", "p", "hash"))
        self.assertTrue(ledger.completed("d", "s", "p", "hash", "input", "model"))
        self.assertFalse(ledger.completed("d", "s", "p", "hash", "changed", "model"))
        self.assertFalse(ledger.completed("d", "s", "p", "hash", "input", "changed"))
        self.assertEqual(ledger.metrics("d", "s", "p"), {"ok": True})
        self.assertFalse(ledger.completed("d", "s", "p", "different"))
        ledger.upsert("d", "s", "p", "extracted", "hash", str(output), {"ok": True})
        self.assertTrue(ledger.completed("d", "s", "p", "hash"))
        ledger.close()

    def test_combined_input_hash_and_visual_cache_key_change_with_dependencies(self):
        first = self.run_dir / "input_a.bin"
        second = self.run_dir / "input_b.bin"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        reference = SequenceRef("andy", "one", first, metadata={"participant": "1"})
        ledger = Ledger(self.run_dir / "input_hash.sqlite3")
        with mock.patch(
            "utils.motion_retarget.pipeline.input_dependency_paths",
            return_value=[first, second],
        ):
            before = input_dependency_hash(self.config, ledger, reference)
            second.write_bytes(b"changed")
            after = input_dependency_hash(self.config, ledger, reference)
        ledger.close()
        self.assertNotEqual(before, after)
        self.assertIn(before[:16], _video_cache_path(self.config, reference, before).name)
        self.assertNotEqual(
            _video_cache_path(self.config, reference, before),
            _video_cache_path(self.config, reference, after),
        )

    def test_profile_height_scale(self):
        for name, expected in (("male_178", 1.78), ("female_165", 1.65)):
            with self.subTest(profile=name):
                scale, unscaled = profile_scale(self.config.smplx_model_dir, self.config.profiles[name])
                self.assertLess(abs(scale * unscaled - expected), 0.002)

    def test_inventory_contract(self):
        self.assertEqual(len(inventory_andy(self.config)), 195)
        self.assertEqual(len(inventory_comfi(self.config)), 422)
        exact = inventory(self.config, "comfi", sequence_ids={"1508_Static"})
        self.assertEqual([row.sequence_id for row in exact], ["1508_Static"])

    def test_input_mode_override_changes_run_hash(self):
        reference = SequenceRef("comfi", "one", Path("source.zip"))
        self.assertEqual(run_config_hash(self.config, reference, None), self.config.digest)
        explicit = run_config_hash(self.config, reference, "mocap_raw")
        self.assertNotEqual(explicit, self.config.digest)
        self.assertEqual(explicit, run_config_hash(self.config, reference, "mocap_raw"))

    def test_hand_quality_never_promotes_imputation(self):
        left = empty_hand_track(10)
        right = observed_hand_track(np.zeros((10, 45)))
        label, coverage = coverage_quality(left, right, 0.8, 0.3)
        self.assertEqual(label, "partial")
        self.assertEqual(coverage, 0.5)

    def test_wrist_basis_alignment_recovers_proper_rotation(self):
        source = np.asarray([
            [1.0, 0.2, 0.1], [0.7, 0.8, 0.1], [-0.4, 0.9, 0.2],
            [-0.8, 0.5, 0.1], [0.3, -0.2, 0.9],
        ])
        expected = Rotation.from_euler("xyz", [17.0, -11.0, 23.0], degrees=True).as_matrix()
        target = source @ expected
        actual = _wrist_basis_alignment(source * np.arange(1, 6)[:, None], target)
        self.assertAlmostEqual(float(np.linalg.det(actual)), 1.0, places=6)
        source_unit = source / np.linalg.norm(source, axis=1, keepdims=True)
        target_unit = target / np.linalg.norm(target, axis=1, keepdims=True)
        np.testing.assert_allclose(source_unit @ actual, target_unit, atol=1e-6)

    def test_contact_patch_keeps_vertex_and_oriented_anchor_stable(self):
        contact = np.asarray([[False], [True], [True], [True], [False]])
        candidates = np.asarray([7, 11, 13])
        distance = np.asarray([
            [[0.10, 0.10, 0.10]], [[0.01, 0.03, 0.04]],
            [[0.04, 0.01, 0.03]], [[0.01, 0.02, 0.04]], [[0.10, 0.10, 0.10]],
        ])
        nearest = np.asarray([
            [[0, 0, 0]], [[0, 1, 2]], [[1, 1, 2]], [[0, 1, 2]], [[0, 0, 0]],
        ])
        surface = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
        normals = np.tile([0.0, 0.0, 1.0], (3, 1))
        vertex, anchor, normal, valid = _stable_contact_patches(
            contact, candidates, distance, nearest, surface, normals, 2, 0.035,
        )
        np.testing.assert_array_equal(vertex[1:4, 0], np.tile(vertex[1, 0], (3, 1)))
        np.testing.assert_allclose(anchor[1:4, 0], np.tile(anchor[1, 0], (3, 1, 1)))
        np.testing.assert_allclose(normal[valid], np.tile([0.0, 0.0, 1.0], (valid.sum(), 1)))

    def test_contact_patch_segments_real_object_local_regrasp(self):
        contact = np.ones((6, 1), dtype=bool)
        candidates = np.asarray([7])
        distance = np.full((6, 1, 1), 0.001)
        nearest = np.asarray([0, 0, 0, 2, 2, 2], dtype=np.int32).reshape(6, 1, 1)
        surface = np.asarray([[0., 0., 0.], [0.01, 0., 0.], [0.03, 0., 0.]])
        normals = np.tile([0., 0., 1.], (3, 1))
        _, anchor, _, valid = _stable_contact_patches(
            contact, candidates, distance, nearest, surface, normals, 1, 0.01,
            segment_distance=0.02, segment_minimum_frames=2,
        )
        self.assertTrue(valid.all())
        np.testing.assert_allclose(anchor[:3, 0, 0], [surface[0]] * 3)
        np.testing.assert_allclose(anchor[3:, 0, 0], [surface[2]] * 3)

    def test_contact_patch_enforces_minimum_length_after_every_reanchor(self):
        contact = np.ones((8, 1), dtype=bool)
        candidates = np.asarray([7])
        distance = np.full((8, 1, 1), 0.001)
        nearest = np.asarray([0, 0, 0, 1, 2, 3, 4, 5], dtype=np.int32).reshape(8, 1, 1)
        surface = np.stack([np.arange(6) * 0.03, np.zeros(6), np.zeros(6)], axis=1)
        normals = np.tile([0., 0., 1.], (6, 1))
        _, anchor, _, valid = _stable_contact_patches(
            contact, candidates, distance, nearest, surface, normals, 1, 0.01,
            segment_distance=0.02, segment_minimum_frames=3,
            active_minimum_frames=1,
        )
        self.assertTrue(valid.all())
        changes = np.flatnonzero(np.linalg.norm(np.diff(anchor[:, 0, 0], axis=0), axis=1) > 1e-8)
        self.assertLessEqual(len(changes), 2)
        if len(changes) > 1:
            self.assertTrue(np.all(np.diff(changes) >= 3))

    def test_contact_patch_reanchors_when_curved_surface_normal_changes(self):
        contact = np.ones((6, 1), dtype=bool)
        candidates = np.asarray([7])
        distance = np.full((6, 1, 1), 0.001)
        nearest = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int32).reshape(6, 1, 1)
        surface = np.asarray([[0., 0., 0.], [0.001, 0., 0.]])
        normals = np.asarray([[0., 0., 1.], [0., 1., 0.]])
        _, anchor, normal, valid = _stable_contact_patches(
            contact, candidates, distance, nearest, surface, normals, 1, 0.01,
            segment_distance=0.02, segment_minimum_frames=3,
            segment_normal_degrees=30.0, active_minimum_frames=1,
        )
        self.assertTrue(valid.all())
        np.testing.assert_allclose(anchor[:3, 0, 0], [surface[0]] * 3)
        np.testing.assert_allclose(anchor[3:, 0, 0], [surface[1]] * 3)
        np.testing.assert_allclose(normal[:3, 0, 0], [normals[0]] * 3)
        np.testing.assert_allclose(normal[3:, 0, 0], [normals[1]] * 3)

    def test_contact_patch_does_not_activate_far_vertices_with_hand_contact(self):
        contact = np.ones((5, 1), dtype=bool)
        candidates = np.asarray([7, 11])
        distance = np.tile(np.asarray([[[0.005, 0.030]]]), (5, 1, 1))
        nearest = np.tile(np.asarray([[[0, 1]]], dtype=np.int32), (5, 1, 1))
        surface = np.asarray([[0., 0., 0.], [0.03, 0., 0.]])
        normals = np.tile([0., 0., 1.], (2, 1))
        vertex, _, _, valid = _stable_contact_patches(
            contact, candidates, distance, nearest, surface, normals, 2, 0.035,
            active_enter_distance=0.015, active_exit_distance=0.020,
            active_minimum_frames=1,
        )
        near = np.flatnonzero(vertex[0, 0] == 7)
        far = np.flatnonzero(vertex[0, 0] == 11)
        self.assertEqual(len(near), 1)
        self.assertTrue(valid[:, 0, near[0]].all())
        if len(far):
            self.assertFalse(valid[:, 0, far[0]].any())

    def test_contact_patch_greedily_covers_a_sliding_contact(self):
        contact = np.ones((10, 1), dtype=bool)
        candidates = np.asarray([7, 11, 13])
        distance = np.full((10, 1, 3), 0.10)
        distance[:6, 0, 0] = 0.010
        distance[:5, 0, 1] = 0.005
        distance[5:, 0, 2] = 0.010
        nearest = np.zeros((10, 1, 3), dtype=np.int32)
        surface = np.asarray([[0., 0., 0.]])
        normals = np.asarray([[0., 0., 1.]])
        vertex, _, _, valid = _stable_contact_patches(
            contact, candidates, distance, nearest, surface, normals, 2, 0.035,
            active_enter_distance=0.015, active_exit_distance=0.020,
            active_minimum_frames=1,
        )
        np.testing.assert_array_equal(vertex[0, 0], [7, 13])
        self.assertTrue(valid.any(axis=-1).all())

    def test_contact_ik_cosine_learning_rate_settles_at_final_scale(self):
        samples = np.asarray([
            _cosine_learning_rate_scale(progress, 0.03)
            for progress in np.linspace(0.0, 1.0, 9)
        ])
        self.assertAlmostEqual(samples[0], 1.0)
        self.assertAlmostEqual(samples[-1], 0.03)
        self.assertTrue(np.all(np.diff(samples) < 0.0))

    def test_penetration_continuation_reaches_full_weight_before_tail(self):
        self.assertEqual(_continuation_scale(0.34, 0.35, 0.65), 0.0)
        self.assertAlmostEqual(_continuation_scale(0.50, 0.35, 0.65), 0.5)
        self.assertEqual(_continuation_scale(0.65, 0.35, 0.65), 1.0)
        self.assertEqual(_continuation_scale(0.90, 0.35, 0.65), 1.0)

    def test_selected_vertex_normals_follow_posed_triangle(self):
        import torch
        vertices = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
        table, mask = _incident_face_table(np.asarray([[0, 1, 2]]), 3)
        normal = _selected_vertex_normals(
            vertices, torch.tensor([[0]]), torch.tensor(table), torch.tensor(mask),
        )
        np.testing.assert_allclose(normal.numpy(), [[[0.0, 0.0, 1.0]]], atol=1e-6)

    def test_axis_angle_perturbations_compose_on_so3(self):
        import torch
        base = np.asarray([[0.7, -0.4, 0.2], [-0.3, 1.1, 0.5]], dtype=np.float32)
        delta = np.asarray([[0.1, 0.2, -0.1], [0.05, -0.08, 0.03]], dtype=np.float32)
        actual = _compose_axis_angle(torch.tensor(base), torch.tensor(delta)).numpy()
        expected_matrix = (
            Rotation.from_rotvec(base).as_matrix() @ Rotation.from_rotvec(delta).as_matrix()
        )
        np.testing.assert_allclose(
            Rotation.from_rotvec(actual).as_matrix(), expected_matrix, atol=2e-6,
        )

    def test_axis_angle_composition_has_finite_identity_gradient(self):
        import torch
        base = torch.tensor([[0.7, -0.4, 0.2]], dtype=torch.float32)
        delta = torch.zeros_like(base, requires_grad=True)
        result = _compose_axis_angle(base, delta)
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.isfinite(delta.grad).all())

    def test_contact_priority_projects_conflicting_secondary_gradient(self):
        import torch
        parameter = torch.zeros(2)
        contact = torch.tensor([1.0, 0.0])
        secondary = torch.tensor([-1.0, 2.0])
        total = _contact_priority_gradients(
            [parameter], [contact], [secondary],
        )[0]
        np.testing.assert_allclose(total.numpy(), [1.0, 2.0], atol=1e-7)
        lower_after_projection = total - contact
        self.assertAlmostEqual(float(torch.dot(contact, lower_after_projection)), 0.0, places=7)

    def test_four_level_priority_preserves_higher_level_descent(self):
        import torch
        parameter = torch.zeros(3)
        levels = [
            [torch.tensor([1.0, 0.0, 0.0])],
            [torch.tensor([-2.0, 1.0, 0.0])],
            [torch.tensor([0.0, -2.0, 1.0])],
            [torch.tensor([-1.0, -1.0, -1.0])],
        ]
        total = _hierarchical_priority_gradients([parameter], levels)[0]
        self.assertGreaterEqual(float(torch.dot(total, levels[0][0])), 1.0 - 1e-6)
        self.assertTrue(torch.isfinite(total).all())

    def test_relative_rotation_smoothing_preserves_unchanged_baseline(self):
        times = np.arange(100, dtype=np.float64) / 50.0
        baseline = np.zeros((100, 6), dtype=np.float32)
        baseline[:, 1] = np.linspace(0.0, 0.5, 100)
        result, metadata = c3_smooth_relative_rotvec(times, baseline, baseline, 0.1)
        np.testing.assert_allclose(result, baseline, atol=1e-6)
        self.assertEqual(metadata["representation"], "baseline_composed_relative_SO3_delta")

    def test_relative_rotation_smoothing_honors_contact_weights(self):
        times = np.arange(120, dtype=np.float64) / 60.0
        baseline = np.zeros((120, 3), dtype=np.float32)
        optimized = baseline.copy()
        optimized[45:75, 1] = 0.35
        plain, _ = c3_smooth_relative_rotvec(times, baseline, optimized, 0.18)
        weights = np.ones(120, dtype=np.float64)
        weights[45:75] = 5.0
        weighted, metadata = c3_smooth_relative_rotvec(
            times, baseline, optimized, 0.18, sample_weights=weights,
        )
        plain_error = np.mean(np.abs(plain[45:75, 1] - optimized[45:75, 1]))
        weighted_error = np.mean(np.abs(weighted[45:75, 1] - optimized[45:75, 1]))
        self.assertLess(weighted_error, plain_error)
        self.assertTrue(metadata["weighted_samples"])

    def test_c4_bspline_basis_is_partition_of_unity(self):
        times = np.arange(180, dtype=np.float64) / 90.0
        basis, knots = c4_bspline_basis(times, 0.04, degree=5)
        self.assertEqual(basis.shape[0], len(times))
        self.assertEqual(basis.shape[1], len(knots) - 6)
        np.testing.assert_allclose(basis.sum(axis=1), 1.0, atol=1e-6)
        self.assertTrue(np.all(basis >= -1e-7))

    def test_marker_track_outlier_is_reported_and_excluded(self):
        errors = np.asarray([
            [0.01, 0.02, 0.20], [0.02, 0.03, 0.22], [0.01, 0.02, 0.24],
        ])
        valid = np.ones_like(errors, dtype=bool)
        metrics = summarize_fit_errors(
            errors, valid, "vertices",
            {"marker_track_outlier_m": 0.08, "min_marker_tracks": 2},
        )
        self.assertEqual(metrics["excluded_target_indices"], [2])
        self.assertLess(metrics["p95_error_m"], 0.04)
        self.assertGreater(metrics["raw_p95_error_m"], 0.2)

    def test_nonfinite_metrics_are_strict_json_safe(self):
        self.assertEqual(json_safe({"nan": np.nan, "inf": np.inf, "ok": np.float32(1.5)}), {
            "nan": None, "inf": None, "ok": 1.5,
        })

    def test_wilor_resampling_does_not_validate_outside_video(self):
        cache = self.run_dir / "wilor_synthetic.npz"
        count = 3
        np.savez_compressed(
            cache, timestamps=np.asarray([0.0, 0.1, 0.2]),
            left_hand_pose=np.zeros((count, 45)), right_hand_pose=np.zeros((count, 45)),
            left_valid=np.ones(count, bool), right_valid=np.ones(count, bool),
            left_confidence=np.ones(count), right_confidence=np.ones(count),
            left_reprojection_error=np.zeros(count), right_reprojection_error=np.zeros(count),
            metadata_json=np.asarray('{"reprojection_available": false}'),
        )
        target = np.asarray([-0.01, 0.025, 0.075, 0.21])
        left, right, metadata = load_wilor_cache(cache, target, 40.0, 0.5, 0.5, 0.03)
        np.testing.assert_array_equal(left.valid, [False, True, True, False])
        np.testing.assert_array_equal(right.valid, left.valid)
        self.assertTrue(left.imputed[[0, 3]].all())
        self.assertEqual(metadata["quality"]["left"]["target_valid_frames"], 2)

    def test_mujoco_quality_failure_preserves_structured_report(self):
        report = self.run_dir / "failed_quality.mujoco.json"

        def completed_quality_failure(*args, **kwargs):
            report.write_text('{"passed": false, "mapped_joint_fk_p95_m": 0.003}\n')
            return SimpleNamespace(returncode=1, stdout="quality gate failed")

        with mock.patch("utils.motion_retarget.validation.subprocess.run", side_effect=completed_quality_failure):
            payload = run_mujoco_validation(
                self.config, self.run_dir / "motion.npz", report,
            )
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["worker_returncode"], 1)
        self.assertAlmostEqual(payload["mapped_joint_fk_p95_m"], 0.003)

    def test_mesh_render_resampling_preserves_duration(self):
        timestamps = np.arange(53, dtype=np.float64) / 40.0
        indices = nearest_frame_indices(timestamps, 40.0, 30.0)
        self.assertEqual(len(indices), 40)
        self.assertEqual(indices[0], 0)
        self.assertLessEqual(indices[-1], 52)
        self.assertTrue(np.all(np.diff(indices) >= 0))

    def test_mesh_render_worker_receipt(self):
        output = self.run_dir / "mesh.mp4"

        def completed_render(command, **kwargs):
            output.write_bytes(b"mp4")
            return SimpleNamespace(
                returncode=0,
                stdout='{"renderer":"smplx_surface_mesh","frames":40,"fps":30.0}\n',
            )

        with mock.patch("utils.motion_retarget.validation.subprocess.run", side_effect=completed_render):
            payload = run_smplx_mesh_render(
                self.config, self.run_dir / "motion.npz", output,
            )
        self.assertEqual(payload["renderer"], "smplx_surface_mesh")
        self.assertEqual(payload["frames"], 40)


if __name__ == "__main__":
    unittest.main()
