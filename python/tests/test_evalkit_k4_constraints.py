import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.constraints import couplers,synchronize
from motiogen.evalkit.k.k4_fitting import resample_native


class ConstraintTests(unittest.TestCase):
    def test_declared_coupling_changes_only_dependent_coordinates(self):
        model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
        cs.get('knee_angle_r').setValue(state,.7,False)
        cs.get('knee_angle_r_beta').setValue(state,-.3,False)
        independent={cs.get(i).getName():cs.get(i).getValue(state) for i in range(cs.getSize()) if not cs.get(i).isDependent(state)}
        synchronize(state,couplers(model,state))
        self.assertAlmostEqual(cs.get('knee_angle_r_beta').getValue(state),.7)
        for name,value in independent.items():self.assertEqual(cs.get(name).getValue(state),value)

    def test_heading_wrap_resampling_has_no_false_turn_and_retains_boundaries(self):
        times,data=resample_native(np.array([0.,.02]),['pelvis_rotation','pelvis_tx'],np.array([[179.,.2],[-179.,.4]]))
        np.testing.assert_allclose(times,[0.,.01,.02])
        np.testing.assert_allclose(data[:,0],[179.,180.,181.])
        np.testing.assert_allclose(data[:,1],[.2,.3,.4])


if __name__=='__main__':unittest.main()
