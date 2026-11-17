"""Independent convention and intervention-boundary checks for knee diagnostics."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'python/scripts'),str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from evalkit_k4_knee_seed_probe import flexion,structure
from motiogen.degradation.metrics import angle


class KneeProbeTests(unittest.TestCase):
    def test_degenerate_segment_remains_unknown(self):
        self.assertTrue(np.isnan(flexion(np.zeros(3),np.zeros(3),np.ones(3))))

    def test_known_flexion_and_independent_scoring_formula(self):
        theta=np.deg2rad([0.,30.,90.,135.,180.])
        hip=np.zeros((5,3));knee=np.tile([1.,0.,0.],(5,1))
        ankle=knee+np.stack([np.cos(theta),np.sin(theta),np.zeros(5)],axis=-1)
        np.testing.assert_allclose(flexion(hip,knee,ankle),np.rad2deg(theta),atol=1e-10)
        np.testing.assert_allclose(flexion(hip,knee,ankle),180-angle(hip-knee,ankle-knee),atol=1e-10)

    def test_fixed_length_changes_and_world_frame_preserve_flexion(self):
        hip=np.array([0.,0.,0.]);knee=np.array([.4,0.,0.]);ankle=np.array([.2,.3,0.])
        expected=flexion(hip,knee,ankle)
        knee2=2.3*knee;ankle2=knee2+.7*(ankle-knee)
        self.assertAlmostEqual(flexion(hip,knee2,ankle2),expected)
        rotation=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
        shift=np.array([7.,-3.,2.])
        self.assertAlmostEqual(flexion(rotation@hip+shift,rotation@knee+shift,rotation@ankle+shift),expected)

    def test_structure_guard_allows_only_initialization_change(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as tmp:
            a=Path(tmp)/'a.xml';b=Path(tmp)/'b.xml'
            original='<Model><Coordinate name="knee"><default_value>0</default_value><range>0 2.44</range><locked>false</locked></Coordinate></Model>'
            a.write_text(original);b.write_text(original.replace('>0<','>1.8<'))
            self.assertEqual(structure(a),structure(b))
            b.write_text(original.replace('0 2.44','0 3.14'))
            self.assertNotEqual(structure(a),structure(b))
            b.write_text(original.replace('false','true'))
            self.assertNotEqual(structure(a),structure(b))


if __name__=='__main__':unittest.main()
