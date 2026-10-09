import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from motiogen.config import load_config
from motiogen.native_eval import NATIVE,full_preflight,smoke_result,stratified_preflight
from motiogen.cli import main


class NativeEvaluationTest(unittest.TestCase):
    def test_all_five_have_explicit_native_policy(self):
        self.assertEqual(set(NATIVE),{'motionlcm','omnicontrol','chois','ardy','kimodo'})
        self.assertIsNone(NATIVE['ardy']['command'])
        self.assertEqual(NATIVE['kimodo']['command'][-1][:2],['python','benchmark/evaluate_folder.py'])

    def test_full_profile_is_preflight_only(self):
        with tempfile.TemporaryDirectory() as folder:
            result=full_preflight(load_config(),'ardy',folder)
            self.assertFalse(result['execution_started'])
            self.assertIn(result['status'],('ready','missing_resources'))

    def test_kimodo_full_preflight_does_not_claim_dataset_ready_from_archive_only(self):
        with tempfile.TemporaryDirectory() as folder, patch('motiogen.native_eval.find_asset') as asset:
            asset.return_value=Path(folder)
            result=full_preflight(load_config(),'kimodo',folder)
            self.assertFalse(result['execution_started'])
            self.assertTrue(any(x['kind'].startswith('dataset') for x in result['missing']))

    def test_smoke_uses_three_configured_seeds(self):
        with patch('motiogen.cli.Run') as run, patch('motiogen.cli.environment_python') as python:
            python.return_value.exists.return_value=False
            run.return_value.finish.side_effect=lambda status,*args,**kwargs:{'status':status}
            self.assertEqual(main(['evaluate-native','--baseline','ardy','--profile','smoke']),1)
            self.assertEqual(run.call_count,3)

    def test_smoke_result_serializes_npz_scalar_fps(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); motion=root/'motion.npz'
            np.savez(motion,joints=np.zeros((2,1,3),dtype=np.float32),
                     joint_names=np.array(['pelvis']),fps=np.array(20.0),
                     metadata_json=np.array(json.dumps({'units':'m','axes':'right-handed Z-up'})))
            result=smoke_result({},'motionlcm',motion,root,{'metrics':{}})
            self.assertEqual(result['fps'],20.0)
            self.assertEqual(json.loads((root/'native_evaluation.json').read_text())['fps'],20.0)

    def test_stratified_preflight_requires_exact_ready_manifest(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('motiogen.datasets.stratified_root',return_value=Path(folder)), \
             patch('motiogen.native_eval.find_asset',return_value=Path(folder)):
            manifest={'seed':20261008,'group_count':58,'per_group':32,'case_count':1856,
                      'selection_sha256':'abc','extraction_status':'ready'}
            (Path(folder)/'selection_manifest.json').write_text(json.dumps(manifest))
            result=stratified_preflight(load_config())
            self.assertEqual(result['status'],'ready')
            manifest['case_count']=1855
            (Path(folder)/'selection_manifest.json').write_text(json.dumps(manifest))
            self.assertEqual(stratified_preflight(load_config())['status'],'missing_resources')

    def test_stratified_preflight_does_not_trust_stale_extraction_status(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as prepared, \
             patch('motiogen.datasets.stratified_root',return_value=Path(folder)), \
             patch('motiogen.datasets.prepared_root',return_value=Path(prepared)), \
             patch('motiogen.native_eval.find_asset',return_value=Path(folder)):
            root=Path(folder); case=root/'testsuite/content/group/0001'; case.mkdir(parents=True)
            (case/'seed_motion.json').write_text(json.dumps({'bvh_path':'BVH/day/motion.bvh'}))
            manifest={'seed':20261008,'group_count':58,'per_group':32,'case_count':1856,
                      'selection_sha256':'abc','extraction_status':'ready'}
            (root/'selection_manifest.json').write_text(json.dumps(manifest))
            result=stratified_preflight(load_config())
            self.assertEqual(result['status'],'missing_resources')
            self.assertTrue(any('selected BONES-SEED BVH' in x['name'] for x in result['missing']))


if __name__=='__main__':
    unittest.main()
