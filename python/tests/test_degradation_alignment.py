"""Independent geometric expectations for the R -> D experiment boundary."""
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.degradation.reference import read_bvh, sample
from motiogen.degradation.positions import SOURCE, project
from motiogen.degradation.metrics import compare
from motiogen.evalkit.k.motion import Motion


class DegradationTests(unittest.TestCase):
    def test_absolute_channels_and_intrinsic_rotation(self):
        text='''HIERARCHY
ROOT Hips { OFFSET 100 200 300 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
JOINT Head { OFFSET 0 10 0 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
End Site { OFFSET 0 1 0 } } }
MOTION
Frames: 2
Frame Time: 0.1
10 20 30 90 0 0 0 10 0 0 0 0
10 20 30 0 90 0 0 10 0 0 0 0
'''
        ref=read_bvh(text)
        np.testing.assert_allclose(ref.positions[:,0],[[.1,.2,.3],[.1,.2,.3]],atol=1e-12)
        np.testing.assert_allclose(ref.positions[:,1],[[0,.2,.3],[.1,.2,.4]],atol=1e-12)
        np.testing.assert_allclose(ref.rotations[0,0],[[0,-1,0],[1,0,0],[0,0,1]],atol=1e-12)

    def test_noncommuting_bvh_channel_order(self):
        text='''HIERARCHY
ROOT Hips { OFFSET 0 0 0 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
JOINT Head { OFFSET 0 10 0 CHANNELS 3 Zrotation Xrotation Yrotation } }
MOTION
Frames: 2
Frame Time: 0.1
0 0 0 90 90 0 0 0 0
0 0 0 90 90 0 0 0 0
'''
        ref=read_bvh(text)
        np.testing.assert_allclose(ref.rotations[0,0],[[0,0,1],[1,0,0],[0,1,0]],atol=1e-12)
        np.testing.assert_allclose(ref.positions[0,1],[0,0,.1],atol=1e-12)

    def fixture(self):
        names=list(dict.fromkeys(SOURCE.values()))
        rng=np.random.default_rng(51)
        p=rng.normal(size=(10,len(names),3))
        r=np.broadcast_to(np.eye(3),(10,len(names),3,3)).copy()
        return Motion(p,names,np.arange(10)/20,'reference',r,names)

    def test_positions_preserve_every_observed_node(self):
        ref=self.fixture();d=project(ref,'motionlcm')
        for j,n in enumerate(d.names):
            np.testing.assert_array_equal(d.positions[:,j],ref.positions[:,ref.names.index(SOURCE[n])])
        metrics=compare(ref,d)
        for region in metrics['positions'].values():self.assertLess(region['world_cm']['max'],1e-10)
        self.assertIsNone(d.rotations)

    def test_head_twist_is_actually_removed(self):
        ref=self.fixture();a=project(ref,'motionlcm')
        ref.rotations[:,ref.names.index('Head')]=Rotation.from_euler('Y',75,degrees=True).as_matrix()
        b=project(ref,'motionlcm')
        np.testing.assert_array_equal(a.positions,b.positions)
        self.assertIsNone(b.rotations)
        self.assertNotIn('rotations',b.provenance)

    def test_no_filter_or_extrapolation(self):
        ref=self.fixture()
        got=sample(ref,ref.times)
        np.testing.assert_allclose(ref.positions,got.positions,atol=1e-12)
        with self.assertRaises(ValueError):sample(ref,[-.1,.1])

    def test_world_translation_is_not_hidden_by_alignment(self):
        ref=self.fixture();d=project(ref,'motionlcm');d.positions+=np.array([.1,0,0])
        m=compare(ref,d)
        for region in m['positions'].values():
            self.assertAlmostEqual(region['world_cm']['p95'],10)
            self.assertLess(region['pelvis_relative_cm']['p95'],1e-10)

    def test_source_rig_neutral_pose_and_fixed_geometry(self):
        from motiogen.evalkit.k.export import target_template,retarget
        from motiogen.degradation.rig import aligned
        names,rest,parents=target_template('primal')
        source_names=[SOURCE[n] for n in names]
        p=np.broadcast_to(rest,(12,len(rest),3)).copy()
        p[:,:,0]+=np.arange(12)[:,None]*.01
        r=np.broadcast_to(np.eye(3),(12,len(rest),3,3)).copy()
        ref=Motion(p,source_names,np.arange(12)/30,'reference',r,source_names,
                   {'rest_positions':rest.tolist()})
        old,_=retarget(ref,'primal');new,_=aligned(ref,old,source_geometry=True)
        np.testing.assert_allclose(new.positions,p,atol=1e-10)
        fixed=np.asarray(new.provenance['neutral_joints'])
        self.assertEqual(fixed.shape,(22,3))
        for j,par in enumerate(parents):
            if par<0:continue
            np.testing.assert_allclose(np.linalg.norm(new.positions[:,j]-new.positions[:,par],axis=-1),
                                       np.linalg.norm(fixed[j]-fixed[par]),atol=1e-10)

    def test_humanml_heading_crosses_pi_without_fold(self):
        import tempfile
        from motiogen.evalkit.k.export import target_template
        from motiogen.degradation.positions import write
        from motiogen.degradation.codec import encode
        from motiogen.evalkit.k.motion import load
        names,rest,_=target_template('primal')
        yaw=np.linspace(150,210,80)
        r=Rotation.from_euler('Y',yaw[:,None],degrees=True).as_matrix()
        p=np.einsum('tij,kj->tki',r,rest)+np.array([2.,.3,-3.])
        motion=Motion(p,names,np.arange(80)/20,'omnicontrol')
        # Outputs remain in the prescribed scratch tree.
        scratch=Path(__file__).resolve().parents[2]/'develop_logs/test_runs/20261117_degradation'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            write(motion,folder);encode(Path(folder)/'motion.npz')
            decoded=load(folder,'omnicontrol')
            self.assertEqual(len(decoded.times),79)
            np.testing.assert_allclose(decoded.positions,p[:-1],atol=2e-5)
            with np.load(Path(folder)/'motion.npz') as z:
                self.assertNotIn('rotations',z.files)


if __name__=='__main__':unittest.main()
