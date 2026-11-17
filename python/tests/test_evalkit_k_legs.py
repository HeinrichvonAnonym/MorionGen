import unittest
import numpy as np
from motiogen.evalkit.k.legs import accept_window
SETTINGS={'minimum_improvement_deg':1.,'knee_tolerance_deg':1.,'foot_tolerance_deg':5.,'foot_position_tolerance_m':.02,'step_tolerance_deg':5.,'velocity_tolerance_deg_s':10.}
class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.target=np.tile(np.array([[0,1,0],[0,.5,0],[0,0,0],[.2,0,0]]),(3,1,1));self.baseline=self.target.copy();self.baseline[:,1,0]=.1;self.times=np.arange(3)/10
    def test_geometric_improvement_accepted_without_reference(self):
        ok,r=accept_window(self.baseline,self.target,self.target,self.times,SETTINGS);self.assertTrue(ok);self.assertLess(r['measures']['proximal_after'],1e-5)
    def test_foot_cost_blocks_knee_improvement(self):
        candidate=self.target.copy();candidate[:,3,0]+=.1
        ok,r=accept_window(self.baseline,candidate,self.target,self.times,SETTINGS);self.assertFalse(ok);self.assertFalse(r['checks']['foot_position'])
    def test_unavailable_geometry_and_no_improvement_rejected(self):
        self.assertFalse(accept_window(self.baseline,self.baseline,self.target,self.times,SETTINGS)[0])
        candidate=self.target.copy();candidate[:,2]=np.nan
        self.assertFalse(accept_window(self.baseline,candidate,self.target,self.times,SETTINGS)[0])
