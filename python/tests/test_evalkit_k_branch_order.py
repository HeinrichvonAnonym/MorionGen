import unittest
import numpy as np
from motiogen.evalkit.k.branch_order import anchored_order
class BranchOrderTests(unittest.TestCase):
    def test_best_conditioned_plane_not_deepest_flexion(self):
        angles=np.deg2rad([0,175,90,40]);target=np.zeros((4,4,3));target[:,1]=[0,-1,0];target[:,2]=target[:,1]+np.stack([np.sin(angles),-np.cos(angles),np.zeros(4)],1)
        order,anchor,receipt=anchored_order(target);self.assertEqual(anchor,2);self.assertEqual(order.tolist(),[2,1,0,3]);self.assertTrue(receipt['qualified'])
    def test_straight_or_missing_does_not_claim_plane(self):
        target=np.tile([[0,0,0],[0,-1,0],[0,-2,0],[1,-2,0]],(3,1,1)).astype(float)
        order,anchor,r=anchored_order(target);self.assertFalse(r['qualified']);self.assertEqual(order.tolist(),[0,1,2])
        target[:]=np.nan;self.assertFalse(anchored_order(target)[2]['qualified'])
