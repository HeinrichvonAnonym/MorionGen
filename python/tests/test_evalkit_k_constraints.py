"""Model constraint receipts are descriptive and cannot unlock or move a joint."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
import biomech
import numpy as np
from motiogen.evalkit.k.constraints import coordinate_contract,synchronize_couplers

class Constraints(unittest.TestCase):
    def test_authored_lock_and_ranges_are_read_without_state_changes(self):
        osim.Logger.setLevelString('error')
        model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
        hip=cs.get('hip_flexion_l');hip.setValue(state,.3,False)
        result=coordinate_contract(model,state,['hip_flexion_l','knee_angle_l','subtalar_angle_l'])['coordinates']
        self.assertAlmostEqual(hip.getValue(state),.3)
        self.assertAlmostEqual(result['knee_angle_l']['range_deg'][1],140.)
        self.assertEqual(result['subtalar_angle_l']['geometry_role'],'fixed_by_model')
        self.assertTrue(cs.get('subtalar_angle_l').getLocked(state))
        self.assertEqual(cs.get('subtalar_angle_l').getValue(state),0.)
        self.assertEqual(result['hip_flexion_l']['geometry_role'],'independent_coordinate')

    def test_coupled_coordinates_use_native_units_and_preserve_independent_motion(self):
        osim.Logger.setLevelString('error');model=osim.Model(str(biomech.MODEL));state=model.initSystem()
        labels=['knee_angle_l','knee_angle_l_beta','knee_angle_r','knee_angle_r_beta']
        q=np.array([[30.,0.,90.,0.],[60.,.1,120.,.2]])
        fixed,receipt=synchronize_couplers(biomech.MODEL,model,labels,q)
        np.testing.assert_array_equal(fixed[:,[0,2]],q[:,[0,2]])
        np.testing.assert_allclose(fixed[:,[1,3]],np.deg2rad(q[:,[0,2]]))
        np.testing.assert_array_equal(q[:,1],[0.,.1])
        again,_=synchronize_couplers(biomech.MODEL,model,labels,fixed)
        np.testing.assert_array_equal(again,fixed)
        self.assertEqual(len(receipt['couplers']),2)
        self.assertEqual(model.getCoordinateSet().get('knee_angle_l_beta').getMotionType(),3)

if __name__=='__main__':unittest.main()
