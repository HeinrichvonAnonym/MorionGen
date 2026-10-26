import unittest
import numpy as np
from motiogen.survey_motion import component_jerk_series


class MotionCurveTests(unittest.TestCase):
    def test_cubic_and_full_metric_reconciliation(self):
        fps=20
        xyz=np.zeros((40,2,3));xyz[:,:,0]=(np.arange(40)/fps)[:,None]**3
        values=component_jerk_series(xyz,fps)
        self.assertTrue(np.isnan(values[:3]).all())
        np.testing.assert_allclose(values[3:],np.sqrt(12),atol=1e-9)
        raw=np.random.default_rng(10).normal(size=(30,22,3))
        curve=component_jerk_series(raw,fps)
        expected=np.sqrt(np.mean((np.diff(raw,n=3,axis=0)*fps**3)**2))
        self.assertAlmostEqual(np.sqrt(np.mean(curve[3:]**2)),expected)

    def test_short_clip_and_invalid_input(self):
        self.assertTrue(np.isnan(component_jerk_series(np.zeros((3,2,3)),20)).all())
        for xyz,fps in [(np.zeros((4,2,3)),0),(np.full((4,2,3),np.nan),20),(np.zeros((4,2)),20)]:
            with self.assertRaises(ValueError):component_jerk_series(xyz,fps)

