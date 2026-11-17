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


if __name__=='__main__':unittest.main()
