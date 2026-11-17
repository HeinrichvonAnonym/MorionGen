import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.evalkit.k.refinement import refine


class RefinementTests(unittest.TestCase):
    def test_known_fk_pose_recovered_without_reference_or_geometry_change(self):
        m=osim.Model(str(biomech.MODEL));state=m.initSystem();cs=m.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(i).getValue(state)*(180/np.pi if cs.get(i).getMotionType()==1 else 1) for i in range(cs.getSize())])
        for side in ('l','r'):
            row[labels.index('hip_flexion_'+side)]=20
            row[labels.index('knee_angle_'+side)]=45
            row[labels.index('elbow_flex_'+side)]=30
        target=np.tile(row,(2,1));p,names,target=model_fk(biomech.MODEL,labels,target,return_coordinates=True)
        start=target.copy();start[:,labels.index('arm_flex_l')]+=8
        solved,receipt=refine(biomech.MODEL,labels,start,p,names,{'max_nfev':20})
        q,_,_=model_fk(biomech.MODEL,labels,solved,return_coordinates=True)
        before,_=model_fk(biomech.MODEL,labels,start)
        self.assertLess(np.linalg.norm(q-p),np.linalg.norm(before-p)*.2)
        self.assertTrue(all(r['after_loss']<=r['before_loss']+1e-9 for r in receipt['frames']))


if __name__=='__main__':unittest.main()
