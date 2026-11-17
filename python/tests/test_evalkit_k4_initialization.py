import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.evalkit.k.initialization import seed_definitions,weighted_marker_loss,choose


class InitializationTests(unittest.TestCase):
    def fixture(self,path):
        m=osim.Model(str(biomech.MODEL));state=m.initSystem();cs=m.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(i).getValue(state)*(180/np.pi if cs.get(i).getMotionType()==1 else 1) for i in range(cs.getSize())])
        row[labels.index('pelvis_rotation')]=-60
        for s in ('l','r'):
            row[labels.index('knee_angle_'+s)]=110;row[labels.index('hip_flexion_'+s)]=40
        for name,(joint,body,_) in biomech.MARKERS.items():
            transform=m.getJointSet().get(joint).getChildFrame().findTransformInBaseFrame()
            location=osim.Vec3(*[transform.p().get(j) for j in range(3)])
            m.addMarker(osim.Marker(name,m.getBodySet().get(body),location))
        m.finalizeConnections();m.printToXML(str(path))
        q=np.tile(row,(3,1));p,names,q=model_fk(path,labels,q,direct_couplers=True,return_coordinates=True)
        return p,names,q,labels

    def test_independent_fk_seeds_and_loss(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as tmp:
            path=Path(tmp)/'m.osim';p,names,q,labels=self.fixture(path)
            seeds,receipt=seed_definitions(path,p,names)
            self.assertAlmostEqual(seeds['heading']['pelvis_rotation'],-60,places=5)
            for s in ('l','r'):self.assertAlmostEqual(seeds['knee']['knee_angle_'+s],110,places=5)
            weights={n:1. for n in biomech.MARKERS}
            loss=weighted_marker_loss(path,labels,q,p,names,weights)
            self.assertLess(loss['mean_weighted_squared_marker_error_m2'],1e-18)
            shifted=q.copy();shifted[:,labels.index('pelvis_tx')]+=.1
            shifted_loss=weighted_marker_loss(path,labels,shifted,p,names,weights)
            self.assertAlmostEqual(shifted_loss['mean_weighted_squared_marker_error_m2'],.01,places=10)
            bad=p.copy();bad[:,names.index('left_knee')]=bad[:,names.index('left_hip')]
            seeds,receipt=seed_definitions(path,bad,names)
            self.assertNotIn('knee',seeds);self.assertIn('left_knee',receipt['unavailable'])
            with self.assertRaises(ValueError):weighted_marker_loss(path,labels,q[:-1],p,names,weights)

    def test_clip_selection_retains_nominal_and_ignores_failed_candidates(self):
        row=lambda loss:{'status':'completed','input_loss':{'mean_weighted_squared_marker_error_m2':loss}}
        candidates={'nominal':row(1.),'knee':row(.1),'heading':{'status':'failed'},'heading_knee':row(.2)}
        self.assertEqual(choose(candidates),'knee')
        candidates['nominal']=row(.01);self.assertEqual(choose(candidates),'nominal')
        with self.assertRaises(ValueError):choose({'nominal':{'status':'failed'}})


if __name__=='__main__':unittest.main()
