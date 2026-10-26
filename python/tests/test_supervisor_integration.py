"""Real CPU child processes exercise supervision; no GPU allocation or inference."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import psutil
from motiogen.config import ROOT
from motiogen.runtime import gpu_lock, supervise


class SupervisorIntegrationTests(unittest.TestCase):
    def config(self, directory, device=1):
        return {'data_root':str(directory),'gpu_index':device,'resources':{
            'timeout_seconds':2,'sample_seconds':.01,'ram_reserve_gib':0,
            'gpu_reserve_gib':0,'disk_reserve_gib':0}}

    def run_process(self, directory, code, timeout=2, protocol=False):
        run=SimpleNamespace(path=directory,manifest={'action':'test'},events=directory/'events.jsonl')
        run.finish=lambda status,reason=None,**fields:dict(status=status,reason=reason,**fields)
        budget={'disk_ok':True,'ram_bytes':100*1024**3,'gpu_bytes':1,'gpu':{}}
        with patch('motiogen.runtime.budgets',return_value=budget):
            return supervise(run,[sys.executable,'-c',code],self.config(directory),
                timeout=timeout,require_protocol=protocol)

    def test_child_receives_requested_physical_device_and_missing_protocol_fails(self):
        with TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as tmp:
            folder=Path(tmp)
            result=self.run_process(folder,'import os; print(os.environ["CUDA_VISIBLE_DEVICES"],flush=True)',protocol=True)
            self.assertEqual((folder/'stdout.log').read_text().strip(),'1')
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['reason'],'worker did not produce result protocol')

    def test_actual_timeout_terminates_spawned_process_group(self):
        with TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as tmp:
            folder=Path(tmp)
            code=('import subprocess,sys,time; '
                  'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); '
                  'print(child.pid,flush=True); time.sleep(30)')
            result=self.run_process(folder,code,timeout=.3)
            self.assertEqual(result['status'],'timeout')
            self.assertLess(result['elapsed_seconds'],5)
            child_pid=int((folder/'stdout.log').read_text().strip())
            try:
                child=psutil.Process(child_pid)
                self.assertTrue(not child.is_running() or child.status()==psutil.STATUS_ZOMBIE)
            except psutil.NoSuchProcess:
                pass
            events=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
            self.assertIn('termination_requested',[event['event'] for event in events])

    def test_device_leases_are_exclusive_but_independent(self):
        with TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as tmp:
            folder=Path(tmp)
            with gpu_lock({'run_root':str(folder),'gpu_index':0}):
                with (folder/'.gpu-0.lock').open('a') as same:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(same,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with (folder/'.gpu-1.lock').open('a') as other:
                    fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    fcntl.flock(other,fcntl.LOCK_UN)
            with (folder/'.gpu-0.lock').open('a') as released:
                fcntl.flock(released,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(released,fcntl.LOCK_UN)


if __name__=='__main__':unittest.main()
