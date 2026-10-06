import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from motiogen.config import ROOT
from motiogen.metrics import evaluate,so3_degrees
from motiogen.motion import HUMANML_NAMES,Y_TO_Z,canonicalize,load_motion,save_motion


class MetricsTest(unittest.TestCase):
    def motion(self,fps=20,n=12):
        return {'joints':np.zeros((n,22,3)),'joint_names':np.array(HUMANML_NAMES),'fps':fps,
                'metadata':{'units':'m','axes':'right-handed Z-up'}}

    def test_constraint_and_missing_rotation(self):
        motion=self.motion(); motion['joints'][5,21]=[1,2,3]
        scene={'name':'test','constraints':[{'time_seconds':0.25,'joint':'right_wrist','position_m':[1,2,3],'rotation_matrix':np.eye(3).tolist()}]}
        result=evaluate(motion,scene)['metrics']
        self.assertEqual(result['hand_position_error']['value'],0)
        self.assertEqual(result['position_success_rate']['value'],1)
        self.assertIsNone(result['hand_orientation_error']['value'])
        self.assertEqual(result['RULA']['status'],'unavailable')

    def test_rotation_and_basis(self):
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        self.assertAlmostEqual(float(so3_degrees(rot,np.eye(3))),90)
        p,r=canonicalize(np.array([[[1.,2.,3.]]]),'Y-up',rot[None,None])
        np.testing.assert_allclose(p,[[[1,-3,2]]])
        self.assertAlmostEqual(float(so3_degrees(r[0,0],Y_TO_Z)),90)
        self.assertAlmostEqual(np.linalg.det(Y_TO_Z),1)

    def test_position_unit_conversion(self):
        p,_=canonicalize(np.array([[[1000.,2000.,3000.]]]),'Y-up',units='mm')
        np.testing.assert_allclose(p,[[[1,-3,2]]])
        with self.assertRaises(ValueError):
            canonicalize(p,'left-handed')

    def test_native_fps_derivatives_and_boundary(self):
        for fps in (20,30):
            motion=self.motion(fps=fps,n=60)
            t=np.arange(60)/fps
            motion['joints'][:,:,0]=t[:,None]**3
            result=evaluate(motion,{'name':'cubic','constraints':[]})['metrics']
            self.assertAlmostEqual(result['jerk_rms']['value'],6/np.sqrt(3),places=7)
        motion=self.motion(); motion['joints'][6:,:,0]=1; motion['chunk_boundaries']=np.array([6])
        metrics=evaluate(motion,{'name':'jump','constraints':[]})['metrics']
        self.assertEqual(metrics['boundary_position_jump']['value'],1)
        self.assertGreater(metrics['boundary_jerk_rms']['value'],0)

    def test_invalid_constraint_fails(self):
        with self.assertRaises(ValueError):
            evaluate(self.motion(),{'name':'bad','constraints':[{'time_seconds':99,'joint':'right_wrist','position_m':[0,0,0]}]})

    def test_contact_f1_and_absent_positive(self):
        motion=self.motion(); motion['gt_contact']=np.array([1,0,1]); motion['pred_contact']=np.array([1,1,0])
        metrics=evaluate(motion,{'name':'contact'})['metrics']
        self.assertAlmostEqual(metrics['contact_f1']['value'],0.5)
        motion['gt_contact'][:]=0; motion['pred_contact'][:]=0
        self.assertIsNone(evaluate(motion,{'name':'contact'})['metrics']['contact_f1']['value'])

    def test_format_and_wrong_units(self):
        root=ROOT/'develop_logs/test_runs/tests'; root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as folder:
            path=Path(folder)/'motion.npz'; motion=self.motion()
            save_motion(path,motion['joints'],HUMANML_NAMES,20,motion['metadata'])
            np.testing.assert_allclose(load_motion(path)['joints'],motion['joints'])
            save_motion(path,motion['joints'],HUMANML_NAMES,20,{'units':'mm','axes':'right-handed Z-up'})
            with self.assertRaises(ValueError):
                load_motion(path)

    def test_geometric_recovery(self):
        motion=self.motion(n=50)
        motion['joints'][20:,21,0]=1
        scene={'name':'update','updates':[{'time_seconds':0.5,'joint':'right_wrist','position_m':[1,0,0]}]}
        metric=evaluate(motion,scene)['metrics']['goal_recovery_seconds']
        self.assertAlmostEqual(metric['value'],0.5)

    def test_recovery_does_not_overwrite_or_use_future_goal_window(self):
        motion=self.motion(n=60); motion['joints'][30:,21,0]=1
        scene={'name':'updates','updates':[{'time_seconds':0.5,'joint':'right_wrist','position_m':[1,0,0]},
                {'time_seconds':1.,'joint':'right_wrist','position_m':[1,0,0]}]}
        result=evaluate(motion,scene)['metrics']
        self.assertAlmostEqual(result['goal_recovery_rate']['value'],0.5)
        self.assertIsNone(result['goal_recovery_rate']['updates'][0]['recovery_seconds'])
        self.assertAlmostEqual(result['goal_recovery_seconds']['value'],0.5)

    def test_invalid_rotation_rejected(self):
        with self.assertRaises(ValueError):
            so3_degrees(np.diag([1,1,-1]),np.eye(3))


if __name__=='__main__':
    unittest.main()
