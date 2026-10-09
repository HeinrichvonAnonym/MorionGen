import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from motiogen.config import ROOT,load_config
from motiogen.runtime import Run,supervise


class RuntimeTest(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(); self.cfg['resources']['sample_seconds']=0.05
        self.budget={'ram_bytes':1024**3,'gpu_bytes':1024**3,'disk_ok':True,'available_ram_bytes':12*1024**3,'gpu':{},'free_disk_bytes':100*1024**3,'isolation':{}}

    def test_failure_and_unique_run(self):
        a=Run(self.cfg,'motionlcm','test'); b=Run(self.cfg,'motionlcm','test')
        self.assertNotEqual(a.path,b.path)
        with patch('motiogen.runtime.budgets',return_value=self.budget):
            result=supervise(a,[sys.executable,'-c','raise SystemExit(7)'],self.cfg)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['returncode'],7)
        self.assertTrue((a.path/'stdout.log').exists())
        self.assertEqual(json.loads((a.path/'manifest.json').read_text())['status'],'failed')
        b.finish('success','test-only run uniqueness')

    def test_missing_checkout_is_recorded_not_raised(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg={**self.cfg,'data_root':folder,'_data_root':folder}
            run=Run(cfg,'motionlcm','test')
            git=run.manifest['git']
            self.assertIsNone(git['commit']['returncode'])
            self.assertIn('FileNotFoundError',git['commit']['stderr'])

    def test_timeout(self):
        run=Run(self.cfg,'motionlcm','test')
        with patch('motiogen.runtime.budgets',return_value=self.budget):
            result=supervise(run,[sys.executable,'-c','import time; time.sleep(10)'],self.cfg,timeout=0.15)
        self.assertEqual(result['status'],'timeout')
        self.assertIn('termination_requested',(run.path/'events.jsonl').read_text())

    def test_supervised_external_command_can_opt_out_of_worker_protocol(self):
        run=Run(self.cfg,'motionlcm','external-smoke')
        with patch('motiogen.runtime.budgets',return_value=self.budget):
            result=supervise(run,[sys.executable,'-c','print("ok")'],self.cfg,require_protocol=False)
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['returncode'],0)

    def test_resource_fail_closed(self):
        run=Run(self.cfg,'motionlcm','test')
        with patch('motiogen.runtime.budgets',return_value={**self.budget,'ram_bytes':0}):
            result=supervise(run,[sys.executable,'-c','raise RuntimeError("must not launch")'],self.cfg)
        self.assertEqual(result['status'],'resource_terminated')
        self.assertNotIn('process_started',(run.path/'events.jsonl').read_text())

    def test_watchdog_terminates_running_process(self):
        run=Run(self.cfg,'motionlcm','test')
        with patch('motiogen.runtime.budgets',return_value={**self.budget,'ram_bytes':1024}):
            result=supervise(run,[sys.executable,'-c','import time; payload=bytearray(1000000); time.sleep(10)'],self.cfg)
        self.assertEqual(result['status'],'resource_terminated')
        self.assertIn('process_started',(run.path/'events.jsonl').read_text())
        self.assertIn('termination_requested',(run.path/'events.jsonl').read_text())


if __name__=='__main__':
    unittest.main()
