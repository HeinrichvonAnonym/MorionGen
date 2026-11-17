import tempfile,unittest
from pathlib import Path
import numpy as np
from motiogen.evalkit.k.trustworthy import directions,guard
from motiogen.evalkit.k.geometric_calibration import calibrate
from motiogen.evalkit.k.export import target_template
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.config import ROOT,digest

class TrustworthyTests(unittest.TestCase):
    def test_spatial_direction_detects_heading_even_with_same_incline(self):
        p=np.zeros((3,16,3));p[:,1]=[1,1,0];p[:,10]=[-.2,1,0];p[:,13]=[.2,1,0]
        r=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        q=p@r.T
        self.assertFalse(np.allclose(directions(p)['trunk_axis'],directions(q)['trunk_axis']))
        np.testing.assert_allclose(directions(p+np.array([5.,-2.,7.]))['trunk_axis'],directions(p)['trunk_axis'])
        self.assertAlmostEqual(directions(p)['trunk_axis'][0,1],directions(q)['trunk_axis'][0,1])

    def test_calibration_is_static_side_specific_and_copy_only(self):
        import opensim as osim
        import biomech
        osim.Logger.setLevelString('error')
        names,rest,_=target_template('kimodo');p=np.tile(rest,(3,1,1));p[:,names.index('left_knee'),1]-=.02
        source_hash=digest(biomech.MODEL)
        motion=Motion(p,names,np.arange(3)/30,'kimodo')
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as folder:
            rec=calibrate(biomech.MODEL,Path(folder)/'shape.osim',motion)
            self.assertEqual(digest(biomech.MODEL),source_hash)
            self.assertNotEqual(rec['measurements']['left_thigh']['ratio'],rec['measurements']['right_thigh']['ratio'])
            self.assertFalse(rec['measurements']['torso']['length_identified_from_motion'])
            self.assertEqual(rec['measurements']['torso']['source'],'audited_format_asset_prior')
            self.assertNotEqual(rec['scales']['torso'][1],rec['scales']['torso'][2])
            with self.assertRaises(ValueError):calibrate(biomech.MODEL,biomech.MODEL,motion)
