import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.primal_interventions import NativeBranchCounter,NativePrimitiveClock

def inertialize(value):
    return value

class FakeNative:
    def _fwd_smplx_seq(self):
        return 7
    def snap_init_cond_to_ground(self):
        return 3
    def generate_perpetual_navigation(self,snap_to_ground,reproj_kpts,switch_on_inertialization):
        if snap_to_ground:
            self.snap_init_cond_to_ground()
            self._fwd_smplx_seq()
        if reproj_kpts:
            self._fwd_smplx_seq()
        if switch_on_inertialization:
            inertialize(2)
        return 11

class BranchTests(unittest.TestCase):
    def test_native_clock_preserves_scheduler_results(self):
        scheduler=SimpleNamespace(step=lambda x:x+1)
        model=SimpleNamespace(noise_scheduler=scheduler)
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
        with patch.dict(sys.modules,{'torch':fake_torch}):
            with NativePrimitiveClock(model,2,14,'paused') as clock:
                self.assertEqual([scheduler.step(i) for i in range(6)],[1,2,3,4,5,6])
        self.assertEqual([r['frame'] for r in clock.trace],[0,14,28])
        self.assertEqual(scheduler.step(9),10)
        self.assertEqual(clock.wait_seconds,0.)
        self.assertTrue(all(r['paced_wait_seconds']==0 for r in clock.trace))
    def test_realtime_wait_is_measured_not_inferred(self):
        scheduler=SimpleNamespace(step=lambda x:x+1)
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None))
        current=[0.]
        def now():current[0]+=.001;return current[0]
        def sleep(seconds):current[0]+=seconds+.002
        with patch.dict(sys.modules,{'torch':fake_torch}),patch('motiogen.primal_interventions.time.perf_counter',side_effect=now),patch('motiogen.primal_interventions.time.sleep',side_effect=sleep):
            with NativePrimitiveClock(SimpleNamespace(noise_scheduler=scheduler),2,14,'realtime') as clock:
                self.assertEqual([scheduler.step(i) for i in range(6)],[1,2,3,4,5,6])
        self.assertGreater(clock.wait_seconds,0.)
        self.assertAlmostEqual(clock.wait_seconds,sum(r['paced_wait_seconds'] for r in clock.trace))
    def test_factorial_counts_and_unchanged_values(self):
        for ground in [False,True]:
            for reproj in [False,True]:
                for inertia in [False,True]:
                    model=FakeNative();original=model._fwd_smplx_seq
                    with NativeBranchCounter(model) as counter:
                        self.assertEqual(model.generate_perpetual_navigation(ground,reproj,inertia),11)
                    counter.validate(ground,reproj,inertia)
                    self.assertEqual(model._fwd_smplx_seq,original)
    def test_wrong_claim_rejected(self):
        with NativeBranchCounter(FakeNative()) as counter:
            pass
        with self.assertRaises(RuntimeError):counter.validate(True,False,False)

if __name__=='__main__':unittest.main()
