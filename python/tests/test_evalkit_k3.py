import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.bones import normalize,lengths
from motiogen.evalkit.k.geometry import unit,chain
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.config import resolve_config

class BoneTests(unittest.TestCase):
    def test_directions_and_angles_survive_length_normalization(self):
        p=np.array([[[0.,0,0],[1.,0,0],[1.,2,0]],[[4.,0,0],[6.,0,0],[6.,3,0]]]);parents=[-1,0,1]
        q=normalize(p,np.array([np.nan,3.,4.]),parents)
        np.testing.assert_allclose(q[:,0],p[:,0])
        np.testing.assert_allclose(unit(np.diff(q,axis=1)),unit(np.diff(p,axis=1)))
        np.testing.assert_allclose(normalize(q,lengths(q,parents),parents),q)
    def test_missing_propagates_and_zero_bone_is_unknown(self):
        p=np.array([[[0.,0,0],[0.,0,0],[1.,0,0]],[[0.,0,0],[np.nan,0,0],[1.,0,0]]])
        q=normalize(p,[np.nan,1.,1.],[-1,0,1]);self.assertTrue(np.isnan(q[:,1:]).all())
    def test_independent_known_fk_rotation_order(self):
        rx=Rotation.from_euler('x',90,degrees=True).as_matrix();ry=Rotation.from_euler('y',90,degrees=True).as_matrix()
        r=chain(np.array([[rx,ry]]),[-1,0]);np.testing.assert_allclose(r[0,1]@np.array([0,0,1]),[1,0,0],atol=1e-12)
    def test_explicit_config_does_not_change_default(self):
        a=resolve_config();b=resolve_config('config/evalkit_k_v3.json');self.assertNotEqual(a['version'],b['version']);self.assertEqual(resolve_config()['version'],a['version'])
    def test_projection_keeps_independent_rotation_and_source(self):
        from motiogen.evalkit.k.export import target_template
        from motiogen.evalkit.k.bones import project_motion
        names,rest,_=target_template('primal');p=np.repeat(rest[None],3,axis=0);p[1,20]+=[0,.1,0]
        rot=np.broadcast_to(np.eye(3),(3,22,3,3)).copy();m=Motion(p,names,np.arange(3)/30,'primal',rot,names)
        q,receipt=project_motion(m)
        np.testing.assert_array_equal(m.positions,p);np.testing.assert_array_equal(q.rotations,rot)
        self.assertEqual(receipt['source'],'input_only');self.assertGreater(receipt['max_displacement_m'],0.)
    def test_score_normalization_separates_length_from_angles(self):
        from motiogen.evalkit.k.bones import common,NAMES
        from motiogen.evalkit.k.export import target_template
        from motiogen.evalkit.k.scoring import compare
        names,rest,_=target_template('primal')
        reference=common(np.repeat(rest[None],3,axis=0),names)
        degraded=normalize(reference,lengths(reference)*1.5)
        fits={'a':degraded.copy(),'b':degraded.copy()}
        scores,_=compare(reference,degraded,fits,fits,np.arange(3)/30,{'a':{},'b':{}})
        raw=scores['a']['raw']['positions']['left_wrist']['end_to_end']['world']['p95_abs']
        norm=scores['a']['normalized']['positions']['left_wrist']['end_to_end']['world']['p95_abs']
        self.assertGreater(raw,.01);self.assertLess(norm,1e-10)
        self.assertLess(scores['a']['raw']['angles']['left_elbow_flexion']['end_to_end']['p95_abs'],1e-5)
        fits['b'][0,NAMES.index('left_elbow')]=np.nan
        scores,_=compare(reference,degraded,{'a':degraded,'b':degraded},fits,np.arange(3)/30,{'a':{},'b':{}})
        for v in fits:
            self.assertEqual(scores[v]['raw']['angles']['left_elbow_flexion']['end_to_end']['valid'],2)
    def test_independent_closd_upright_and_world_handedness(self):
        from motiogen.evalkit.k.calibration import UPRIGHT
        from motiogen.motion import Y_TO_Z
        # The upstream upright start is a 120-degree cyclic x/y/z permutation.
        expected=Rotation.from_rotvec(np.ones(3)*(2*np.pi/3)/np.sqrt(3)).as_matrix()
        np.testing.assert_allclose(UPRIGHT,expected,atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(Y_TO_Z),1.)
        np.testing.assert_allclose(np.array([0.,1.,0.])@Y_TO_Z.T,[0,0,1])
    def test_prior_body_points_cannot_gain_support(self):
        from motiogen.evalkit.k.fitting3 import support
        from motiogen.evalkit.k.export import target_template
        names,rest,_=target_template('primal');p=np.repeat(rest[None],3,axis=0)
        result=support(p,p,names,prior=['left_elbow'])
        self.assertFalse(result['left_arm']['angle_valid'].any())
        self.assertTrue(result['right_arm']['angle_valid'].all())
    def test_support_thresholds_are_explicit(self):
        from motiogen.evalkit.k.fitting3 import support
        from motiogen.evalkit.k.export import target_template
        names,rest,_=target_template('primal');p=np.repeat(rest[None],3,axis=0);q=p.copy();q[:,:,0]+=.05
        loose=support(p,q,names,position_limit=.1);strict=support(p,q,names,position_limit=.01)
        self.assertTrue(loose['left_arm']['position_valid'].all())
        self.assertFalse(strict['left_arm']['position_valid'].any())
        np.testing.assert_array_equal(loose['left_arm']['angle_valid'],strict['left_arm']['angle_valid'])
    def test_declared_source_axes_must_match_selected_format(self):
        import tempfile,json
        from motiogen.evalkit.k.motion import load
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as folder:
            path=Path(folder)/'motion.npz'
            np.savez(path,joints=np.zeros((3,2,3)),joint_names=['a','b'],fps=20.,
                     metadata_json=json.dumps({'baseline':'motionlcm','axes':'Y-up','units':'m'}))
            with self.assertRaisesRegex(ValueError,'axes disagree'):load(path,config='config/evalkit_k_v3.json')
            np.savez(path,joints=np.zeros((3,2,3)),joint_names=['a','b'],fps=20.,
                     metadata_json=json.dumps({'baseline':'motionlcm','units':'cm'}))
            with self.assertRaisesRegex(ValueError,'expects metres'):load(path,config='config/evalkit_k_v3.json')
    def test_k3_muscle_primary_respects_spatial_support(self):
        from motiogen.evalkit.run import muscle_split
        frames={'times':np.array([0.,1.,2.]),'muscle_times':np.array([0.,1.,2.]),
            'muscle_effort':np.array([1.,5.,9.]),'residual_positive':np.array([False,False,True]),'residual_bw':np.array([0.,0.,2.])}
        self.assertEqual(muscle_split(frames)['primary_frames'],2)
        frames['k_force_eligible']=np.array([True,False,True]);r=muscle_split(frames)
        self.assertEqual(r['primary_frames'],1);self.assertEqual(r['primary_muscle_effort_sum_a2_mean'],1.)
        self.assertEqual(r['forced_frames'],3);self.assertEqual(r['forced_muscle_effort_sum_a2_mean'],5.)
    def test_memory_input_cache_distinguishes_geometry_and_observation(self):
        p=np.array([[[0.,0,0],[1.,0,0]],[[0.,0,0],[1.,0,0]]]);m=Motion(p,['a','b'],np.array([0.,1.]),'test')
        original=m.content_key();m.positions[1,1,1]=.01
        self.assertNotEqual(m.content_key(),original)
        m.positions[1,1,1]=0.;self.assertEqual(m.content_key(),original)
        m.provenance['degradation']={'prior_joints':['b']};self.assertNotEqual(m.content_key(),original)
if __name__=='__main__':unittest.main()
