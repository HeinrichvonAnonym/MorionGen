import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.k.motion import load


class InputRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_run=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
        cls.config=ROOT/'config/evalkit_k4_20261117/input.json'
        cls.scratch=ROOT/'develop_logs/test_runs/20261117_k4'
        cls.scratch.mkdir(parents=True,exist_ok=True)

    def test_isolated_input_works_without_reference(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            source=Path(temp)/'input'
            shutil.copytree(self.data_run/'inputs/000/dart/candidate',source)
            m=load(source,'dart',config=self.config)
            self.assertEqual(m.provenance['input_geometry']['status'],'declared_fixed_rig')
            self.assertNotIn('model_bases',m.provenance)
            self.assertTrue(all(Path(p).is_relative_to(source) for p in m.provenance['inputs']))

    def test_positions_never_receive_reference_rest(self):
        for method in ('motionlcm','omnicontrol'):
            m=load(self.data_run/'inputs/074'/method/'candidate',method,config=self.config)
            self.assertIsNone(m.rotations)
            self.assertNotIn('rest_hip_to_shoulder_midpoint',m.provenance['geometry_targets']['lengths_m'])

    def test_legacy_positions_and_rotations_unchanged(self):
        p=self.data_run/'inputs/000/kimodo/candidate'
        old=load(p,'kimodo');new=load(p,'kimodo',config=self.config)
        np.testing.assert_array_equal(old.positions,new.positions)
        np.testing.assert_array_equal(old.rotations,new.rotations)

    def test_bad_hash_and_topology_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            source=Path(temp)/'input';shutil.copytree(self.data_run/'inputs/000/dart/candidate',source)
            path=source/'input_rig.json';rig=json.loads(path.read_text());rig['parents'][1]=1
            path.write_text(json.dumps(rig))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load(source,'dart',config=self.config)
            p=source/'k_provenance.json';d=json.loads(p.read_text());d['rig_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'rooted tree'):load(source,'dart',config=self.config)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            source=Path(temp)/'input';shutil.copytree(self.data_run/'inputs/000/dart/candidate',source)
            p=source/'k_provenance.json';d=json.loads(p.read_text());d['rig']='../reference.json';p.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'local input_rig'):load(source,'dart',config=self.config)


if __name__=='__main__':unittest.main()
