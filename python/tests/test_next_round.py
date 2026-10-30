import unittest

import numpy as np

from motiogen.native_analysis import (boundary_kinematics, bone_length_cv, cluster_t_interval, common_step_boundary_jerk,
                                      common_step_jerk, facing_heading, holm, sign_flip_p, threshold_success)
from motiogen import task_geometry as tg


class CommonStepMetricTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.xyz = np.cumsum(rng.normal(size=(60, 4, 3)), axis=0)

    def test_unit_step_reproduces_native_jerk(self):
        native = np.diff(self.xyz, n=3, axis=0)*10.**3
        self.assertAlmostEqual(common_step_jerk(self.xyz, 10., .1), float(np.sqrt(np.mean(native**2))), places=9)

    def test_unit_step_reproduces_native_boundary_window(self):
        bounds = [8, 16, 24, 32]
        native = boundary_kinematics(self.xyz, 10., bounds)['boundary_jerk_rms']
        self.assertAlmostEqual(common_step_boundary_jerk(self.xyz, 10., bounds, .1), native, places=9)

    def test_cubic_motion_has_step_independent_jerk(self):
        t = np.arange(80)/20.
        xyz = np.zeros((80, 1, 3)); xyz[:, 0, 0] = t**3
        self.assertAlmostEqual(common_step_jerk(xyz, 20., .1), np.sqrt(36/3), places=6)
        self.assertAlmostEqual(common_step_boundary_jerk(xyz, 20., [40], .1), np.sqrt(36/3), places=6)

    def test_bone_cv_and_thresholds(self):
        xyz = np.zeros((5, 2, 3)); xyz[:, 1, 0] = [1, 1, 1, 1, 2]
        row = bone_length_cv(xyz, ['a', 'b'], [('a', 'b'), ('a', 'missing')])
        self.assertEqual(list(row), ['a-b'])
        self.assertAlmostEqual(row['a-b']['cv'], np.std([1, 1, 1, 1, 2])/1.2)
        self.assertEqual(threshold_success([.01, .03, .2]), {'success_2cm': 1/3, 'success_5cm': 2/3, 'success_10cm': 2/3})

    def test_facing_heading_in_report_frame(self):
        names = ['right_hip', 'left_hip', 'right_shoulder', 'left_shoulder']
        xyz = np.zeros((1, 4, 3))
        xyz[0, 0] = [-.1, 0, 1]; xyz[0, 1] = [.1, 0, 1]; xyz[0, 2] = [-.2, 0, 1.4]; xyz[0, 3] = [.2, 0, 1.4]
        forward, heading = facing_heading(xyz, names)
        np.testing.assert_allclose(forward, tg.BODY_FORWARD, atol=1e-12)
        self.assertAlmostEqual(heading, -90.)


class InferenceTests(unittest.TestCase):
    def test_cluster_t_interval(self):
        values = [1., 3., 2., 2., 5., 7.]
        result = cluster_t_interval(values, [0, 0, 1, 1, 2, 2])
        self.assertEqual(result['clusters'], 3)
        self.assertAlmostEqual(result['mean'], 10/3)
        half = 4.302652729911275*np.std([2, 2, 6], ddof=1)/np.sqrt(3)
        np.testing.assert_allclose(result['ci'], [10/3-half, 10/3+half], rtol=1e-9)

    def test_exact_sign_flip(self):
        self.assertEqual(sign_flip_p([1., 1., 1.])['p'], 2/8)
        self.assertEqual(sign_flip_p([1., -1.])['p'], 1.)
        self.assertEqual(sign_flip_p(np.ones(12))['patterns'], 4096)
        self.assertEqual(sign_flip_p(np.ones(30), draws=1000)['method'], 'monte_carlo')

    def test_holm(self):
        np.testing.assert_allclose(holm([.01, .04, .03, .5]), [.04, .09, .09, .5])


class TaskGeometryTests(unittest.TestCase):
    def test_body_frame_is_proper_rotation_of_report_axes(self):
        left = -tg.BODY_RIGHT
        self.assertAlmostEqual(np.linalg.det(np.stack([tg.BODY_FORWARD, left, tg.UP])), 1.)
        self.assertEqual(tg.body_to_world(1, 0, 0), [0., -1., 0.])
        np.testing.assert_allclose(tg.world_to_body(tg.body_to_world(.3, -.2, 1.1)), [.3, -.2, 1.1])

    def test_reach_task_tiers(self):
        tasks = tg.reach_tasks()
        self.assertEqual(len(tasks), 24)
        tiers = {t: [x for x in tasks if x['tier'] == t] for t in ['reachable', 'stepping', 'legacy', 'mirror']}
        self.assertEqual([len(v) for v in tiers.values()], [6, 6, 6, 6])
        for task in tiers['reachable']:
            self.assertAlmostEqual(task['nominal_shoulder_distance_ratio'], .8, places=3)
        for task in tiers['stepping']:
            self.assertAlmostEqual(task['nominal_shoulder_distance_ratio'], 1.3, places=3)
        for task in tiers['legacy']:
            self.assertLess(task['target_body_m'][0], 0)
            self.assertEqual(task['position_m'][1], .6)
        for task in tiers['mirror']:
            self.assertLess(task['nominal_shoulder_distance_ratio'], 1.)
            lateral = task['target_body_m'][1]
            self.assertLess(lateral if task['side'] == 'right' else -lateral, 0)
        self.assertIsNone(tg.reach_direction_prompt(next(t for t in tasks if t['height'] == 'mid')))
        self.assertIn('up', tg.reach_direction_prompt(next(t for t in tasks if t['height'] == 'high')))

    def test_ardy_directions_map_to_body_axes(self):
        to_world = lambda xz: np.array([xz[0], -xz[1], 0.])
        np.testing.assert_allclose(to_world(tg.native_direction(0)), tg.BODY_FORWARD, atol=1e-12)
        np.testing.assert_allclose(to_world(tg.native_direction(90)), tg.BODY_RIGHT, atol=1e-12)
        self.assertEqual(tg.native_direction(270), [1., 0.])
        paths = tg.ardy_path_tasks(); offsets = tg.ardy_offset_tasks()
        self.assertEqual((len(paths), len(offsets)), (24, 24))
        self.assertEqual(paths[0]['aligned_prompt'], paths[0]['conflict_prompt'])
        self.assertEqual(paths[0]['aligned_update'], paths[0]['conflict_update'])
        for task in offsets:
            self.assertAlmostEqual(float(np.dot(task['direction_xz'], task['lateral_xz'])), 0., places=12)

    def test_path_distance_matches_legacy_formula(self):
        for seconds in [.35, 4.95, 5., 5.35, 19.95]:
            legacy = .4*min(seconds, 5.)+-.2*max(seconds-5., 0.)
            self.assertEqual(float(tg.path_distance(seconds, .4, -.2)), legacy)


if __name__ == '__main__':
    unittest.main()
