import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.evalkit.k.root import compensate
from motiogen.evalkit.k.motion import Motion


class RootTests(unittest.TestCase):
    def test_model_fk_translation_and_unchanged_pose(self):
        model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(i).getValue(state)*(180/np.pi if cs.get(i).getMotionType()==1 else 1) for i in range(cs.getSize())])
        q=np.tile(row,(2,1));times=np.array([0.,.1])
        p,names=model_fk(biomech.MODEL,labels,q,assemble=False)
        shift=np.array([[.1,.2,-.3],[.2,.1,-.2]])
        motion=Motion(p+shift[:,None],names,times,'test_observed_input')
        changed,record=compensate(biomech.MODEL,times,labels,q,motion)
        np.testing.assert_allclose(record['after'],motion.positions,atol=1e-8)
        excluded=[labels.index(n) for n in ('pelvis_tx','pelvis_ty','pelvis_tz')]
        keep=[i for i in range(len(labels)) if i not in excluded]
        np.testing.assert_allclose(changed[:,keep],q[:,keep],atol=1e-8)


if __name__=='__main__':unittest.main()
