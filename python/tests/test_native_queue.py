import os
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime,timedelta,timezone
from motiogen.runtime import gpu_snapshot
from motiogen.interventions import install


class DeviceTests(unittest.TestCase):
    def test_physical_device_selection(self):
        with patch('motiogen.runtime.command_output',return_value={'returncode':0,'stdout':'0, 16000, 12000, 4000\n1, 16000, 14000, 2000'}):
            self.assertEqual(gpu_snapshot(1)['index'],1)
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}):
                self.assertEqual(gpu_snapshot()['index'],1)
            self.assertFalse(gpu_snapshot(8)['available'])

    def test_omni_retains_arguments_and_counts_interventions(self):
        from types import SimpleNamespace
        calls=[]
        model=SimpleNamespace(cmdm_forward=lambda *a,**kw: calls.append(kw) or 5)
        diffusion=SimpleNamespace(guide=lambda x,*a,**kw:x+8)
        adapter=SimpleNamespace(name='omnicontrol',model=SimpleNamespace(model=model),diffusion=diffusion)
        receipt=install(adapter,{'realism':0,'spatial':0})
        self.assertEqual(model.cmdm_forward(y={'hint':123}),0)
        self.assertEqual(calls,[{'y':{'hint':123}}])
        self.assertEqual(diffusion.guide(3),3)
        self.assertEqual(receipt['calls'],{'realism':1,'spatial':1})


class QueueTests(unittest.TestCase):
    def test_failure_isolation_bounded_retry_and_resume_identity(self):
        from motiogen.experiment_queue import execute
        plan={'deadline':(datetime.now(timezone.utc)+timedelta(hours=8)).isoformat(),
              'devices':[0], 'jobs':[{'id':i,'method':'test','command':['test']} for i in ['a','b']]}
        outcomes=iter([{'status':'timeout'},{'status':'timeout'},{'status':'success'}])
        with TemporaryDirectory() as tmp:
            directory=Path(tmp)
            run=MagicMock(); run.path=directory/'worker'
            with patch('motiogen.experiment_queue.fingerprint',return_value='stable'), \
                 patch('motiogen.experiment_queue.load_config',return_value={'baselines':{'test':{}}}), \
                 patch('motiogen.experiment_queue.gpu_lock',side_effect=lambda cfg:nullcontext()) as lease, \
                 patch('motiogen.experiment_queue.Run',return_value=run), \
                 patch('motiogen.experiment_queue.supervise',side_effect=lambda *a,**k:next(outcomes)) as supervisor:
                state=execute(plan,directory)
                self.assertEqual(state['jobs']['a']['status'],'timeout')
                self.assertEqual(len(state['jobs']['a']['attempts']),2)
                self.assertEqual(state['jobs']['b']['status'],'success')
                self.assertEqual(supervisor.call_count,3)
                self.assertGreaterEqual(lease.call_count,3)
                execute(plan,directory,resume=True)
                self.assertEqual(supervisor.call_count,3)
            with patch('motiogen.experiment_queue.fingerprint',return_value='changed'):
                with self.assertRaises(ValueError): execute(plan,directory,resume=True)

    def test_report_reserve_prevents_admission(self):
        from motiogen.experiment_queue import execute
        plan={'deadline':(datetime.now(timezone.utc)+timedelta(hours=2)).isoformat(),
              'devices':[0], 'jobs':[{'id':'a','method':'test','command':['test']}]}
        with TemporaryDirectory() as tmp, \
             patch('motiogen.experiment_queue.fingerprint',return_value='stable'), \
             patch('motiogen.experiment_queue.load_config',return_value={}), \
             patch('motiogen.experiment_queue.gpu_lock',return_value=nullcontext()), \
             patch('motiogen.experiment_queue.supervise') as supervisor:
            state=execute(plan,Path(tmp))
            self.assertEqual(state['status'],'admission_closed')
            supervisor.assert_not_called()

if __name__=='__main__': unittest.main()
