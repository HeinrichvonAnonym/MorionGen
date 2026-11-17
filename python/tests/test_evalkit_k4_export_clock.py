import unittest
import numpy as np
from motiogen.evalkit.k.resampling.clock import native_multiple_clock
from motiogen.evalkit.k.k4_fitting import resample_native


class ExportClockTests(unittest.TestCase):
    def test_input_knots_and_endpoints_are_exact_and_clock_is_uniform(self):
        for frames,fps in [(102,30.),(150,30.),(301,60.),(15,120.)]:
            t=np.arange(frames)/fps;out,receipt=native_multiple_clock(t)
            np.testing.assert_array_equal(out[::receipt['subdivisions_per_native_interval']],t)
            np.testing.assert_allclose(np.diff(out),1/receipt['actual_rate_hz'],atol=1e-14)
            self.assertGreaterEqual(receipt['actual_rate_hz'],100-1e-10)
    def test_known_fk_readback_artifact_is_separate_from_native_pose(self):
        t=np.arange(10)/30.;q=np.zeros((10,1));q[2:,0]=90
        a,qa=resample_native(t,['hinge'],q);b,qb=resample_native(t,['hinge'],q,clock_policy='native_multiple')
        def fk(angles):return np.stack([np.cos(np.deg2rad(angles[:,0])),np.sin(np.deg2rad(angles[:,0]))],1)
        def readback(times,angles):return np.stack([np.interp(t,times,c) for c in fk(angles).T],1)
        self.assertGreater(np.linalg.norm(readback(a,qa)-fk(q),axis=-1).max(),.01)
        np.testing.assert_allclose(readback(b,qb),fk(q),atol=1e-12)
    def test_irregular_and_invalid_clocks_are_explicit(self):
        for t in [[0,.1,.3],[0,0,.1],[0,float('nan')],[0]]:
            with self.assertRaises(ValueError):native_multiple_clock(t)
        with self.assertRaises(ValueError):native_multiple_clock([0,.1],0)
    def test_preserve_domain_does_not_unlock_unrelated_joints(self):
        import tempfile
        from pathlib import Path
        import opensim as osim
        import biomech
        from motiogen.evalkit.k.diagnostic_domain import apply
        from motiogen.evalkit.k.resampling import audit_model
        with tempfile.TemporaryDirectory(dir=Path('develop_logs/test_runs/20261118_k4_native')) as tmp:
            path=Path(tmp)/'model.osim';model=osim.Model(str(biomech.MODEL));model.printToXML(str(path));before=audit_model(path)
            apply(path,{'base':'preserve_existing','scope':'independent_kinematic','coordinate_ranges_deg':{'arm_rot_r':[-180,180]}})
            after=audit_model(path)
            for name,row in before['joints'].items():
                for c,v in row['coordinates'].items():
                    if c!='arm_rot_r':self.assertEqual(v,after['joints'][name]['coordinates'][c])
    def test_legacy_clock_is_unchanged(self):
        t=np.arange(102)/30.;q=np.zeros((102,1));out,_=resample_native(t,['hinge'],q)
        np.testing.assert_array_equal(out,np.linspace(t[0],t[-1],338))


if __name__=='__main__':unittest.main()
