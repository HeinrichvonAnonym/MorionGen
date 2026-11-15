"""Guard the reference audit against error cancellation and unequal clocks/masks."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'reference_audit', Path(__file__).resolve().parents[1] / 'scripts/evalkit_k_reference_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ReferenceAuditTests(unittest.TestCase):
    def test_cancellation_does_not_add_absolute_errors(self):
        d, k, e, valid = audit.decompose(np.array([0.]), np.array([10.]), np.array([0.]), [True])
        np.testing.assert_allclose(d + k, e)
        self.assertEqual(audit.summarize_signed(e, 1)['p95_abs'], 0.)
        self.assertEqual(abs(d[0]) + abs(k[0]), 20.)

    def test_all_stages_use_the_common_valid_frames(self):
        d, k, e, valid = audit.decompose(
            np.array([0., 1., 2., 3.]), np.array([1., np.nan, 3., 4.]),
            np.array([2., 3., np.nan, 5.]), [True, True, True, False])
        np.testing.assert_array_equal(valid, [True, False, False, False])
        for delta in (d, k, e):
            self.assertEqual(audit.summarize_signed(delta, 4)['coverage'], .25)

    def test_clock_does_not_extrapolate_endpoints(self):
        p = np.array([[[0., 0., 0.]], [[2., 4., 6.]]])
        result = audit.resample_points(np.array([0., 1.]), p, np.array([-.1, .5, 1.1]))
        np.testing.assert_allclose(result[1, 0], [1., 2., 3.])
        self.assertTrue(np.isnan(result[[0, 2]]).all())


if __name__ == '__main__':
    unittest.main()
