import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.fixed_geometry import build
from motiogen.evalkit.k.native_frames import read


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_data=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
        cls.input_config=ROOT/'config/evalkit_k4_20261117/input.json'
        cls.scratch=ROOT/'develop_logs/test_runs/20261117_k4'

    def test_fixed_geometry_uses_declared_rig_not_motion_phase(self):
        m=load(self.run_data/'inputs/074/kimodo/candidate','kimodo',config=self.input_config)
        other=copy.deepcopy(m);other.positions[:]=0
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            a=build(m,Path(temp)/'a.osim',75,True);b=build(other,Path(temp)/'b.osim',75,True)
        self.assertEqual(a['after'],b['after'])
        for key in ('shoulder_width','rest_hip_to_shoulder_midpoint'):
            self.assertAlmostEqual(a['after'][key],a['targets']['lengths_m'][key],places=5)
        self.assertFalse(a['force_eligible'])
        self.assertTrue(a['joint_domain']['subtalar_angle_l']['locked'])

    def test_native_head_transport_is_not_anatomical_or_model_fit(self):
        m=load(self.run_data/'inputs/000/closd/candidate','closd',config=self.input_config)
        native=read(m)
        self.assertTrue(native['head_relative_valid'].all())
        self.assertIn('no anatomical',native['scope'])
        p=load(self.run_data/'inputs/000/motionlcm/candidate','motionlcm',config=self.input_config)
        self.assertEqual(read(p)['status'],'unavailable')

    def test_owned_joint_markers_match_fk_after_serialization(self):
        import biomech
        from motiogen.evalkit.k.k4_fitting import model_fk
        from motiogen.evalkit.k.initialization import weighted_marker_loss
        m=load(self.run_data/'inputs/022/kimodo/candidate','kimodo',config=self.input_config)
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            path=Path(temp)/'owned.osim';receipt=build(m,path,75,True,'owned_joint_frame')
            self.assertLess(max(receipt['marker_joint_frame_offset_m'].values()),1e-9)
            model=osim.Model(str(path));state=model.initSystem();cs=model.getCoordinateSet()
            labels=[cs.get(i).getName() for i in range(cs.getSize())]
            row=np.array([cs.get(n).getDefaultValue()*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
            q=np.tile(row,(2,1))
            for s in ('l','r'):q[1,labels.index('knee_angle_'+s)]=110
            points,names,q=model_fk(path,labels,q,direct_couplers=True,return_coordinates=True)
            loss=weighted_marker_loss(path,labels,q,points,names,{n:1. for n in biomech.MARKERS})
            self.assertLess(loss['mean_weighted_squared_marker_error_m2'],1e-18)
            legacy=Path(temp)/'legacy.osim';old=build(m,legacy,75,True,'legacy')
            self.assertEqual(old['after'],receipt['after'])
            old_points,_=model_fk(legacy,labels,q,direct_couplers=True)
            np.testing.assert_allclose(old_points,points,atol=1e-12)
            self.assertGreater(old['marker_joint_frame_offset_m']['knee_r'],.001)


if __name__=='__main__':unittest.main()
