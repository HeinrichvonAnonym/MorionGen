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
    def test_scoring_uses_original_window_lengths_independent_of_query_clock(self):
        from motiogen.evalkit.k.bones import common
        from motiogen.evalkit.k.export import target_template
        from motiogen.evalkit.k.scoring import compare
        names,rest,_=target_template('primal');reference=common(np.repeat(rest[None],3,axis=0),names)
        yaw=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        degraded=(reference-reference[:,:1])@yaw.T+reference[:,:1]
        args=(reference,degraded,{'a':degraded},{'a':degraded},np.arange(3)/30,{'a':{}})
        before,_=compare(*args);after,arrays=compare(*args,reference_lengths=2*lengths(reference))
        self.assertEqual(before['a']['raw'],after['a']['raw'])
        first=before['a']['normalized']['positions']['left_wrist']['end_to_end']['world']['p95_abs']
        second=after['a']['normalized']['positions']['left_wrist']['end_to_end']['world']['p95_abs']
        self.assertAlmostEqual(second,2*first)
        np.testing.assert_array_equal(arrays['reference_lengths'],2*lengths(reference))
    def test_prior_body_points_cannot_gain_support(self):
        from motiogen.evalkit.k.fitting3 import support
        from motiogen.evalkit.k.export import target_template
        names,rest,_=target_template('primal');p=np.repeat(rest[None],3,axis=0)
        result=support(p,p,names,prior=['left_elbow'])
        self.assertFalse(result['left_arm']['angle_valid'].any())
        self.assertTrue(result['right_arm']['angle_valid'].all())
    def test_extra_hand_prior_does_not_mask_observed_body_wrist(self):
        from motiogen.evalkit.k.fitting3 import support
        from motiogen.evalkit.k.export import target_template
        names,rest,_=target_template('chois');names=names.copy()
        names[20:]=['left_wrist_body','right_wrist_body'];names+=['left_wrist','right_wrist']
        p=np.repeat(np.concatenate([rest,rest[[20,21]]])[None],3,axis=0)
        m=Motion(p,names,np.arange(3)/30,'chois',provenance={'degradation':{'prior_joints':['left_wrist','right_wrist']}})
        points,legacy_names,_=m.legacy()
        self.assertEqual(legacy_names[20:24],['left_wrist','right_wrist','left_hand_point','right_hand_point'])
        q=support(points,points,legacy_names,prior=m.legacy_prior_joints())
        self.assertTrue(q['left_arm']['angle_valid'].all());self.assertTrue(q['right_arm']['angle_valid'].all())
        m.provenance['degradation']['prior_joints'].append('left_wrist_body')
        q=support(points,points,legacy_names,prior=m.legacy_prior_joints())
        self.assertFalse(q['left_arm']['angle_valid'].any());self.assertTrue(q['right_arm']['angle_valid'].all())
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
    def test_ablation_keeps_registered_config_after_runtime_selection(self):
        import tempfile,json
        from unittest.mock import patch
        sys.path.insert(0,str(ROOT/'python/scripts'))
        import evalkit_k3_experiment as experiment
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as folder:
            path=Path(folder);config=path/'config/evalkit_k3_20261116';config.mkdir(parents=True)
            registered={'fitting':{'strategy':'legacy','projection':False}}
            (config/'experiment_config.json').write_text(json.dumps(registered))
            selected={'fitting':{'per_method':{'motionlcm':{'strategy':'direction_refinement'}}}}
            (path/'config/evalkit_k_v3.json').write_text(json.dumps(selected))
            with patch.object(experiment,'ROOT',path):
                self.assertEqual(experiment.experiment_settings(),registered)
                (config/'experiment_config.json').write_text(json.dumps(selected))
                with self.assertRaisesRegex(ValueError,'unselected experiment'):experiment.experiment_settings()
    def test_experiment_resume_detects_native_sidecar_change(self):
        import tempfile
        sys.path.insert(0,str(ROOT/'python/scripts'))
        from evalkit_k3_experiment import source_hashes
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as folder:
            path=Path(folder);(path/'motion.npz').write_bytes(b'positions')
            native=path/'native_physics.npz';native.write_bytes(b'rotation1')
            first=source_hashes(path);native.write_bytes(b'rotation2')
            second=source_hashes(path)
            self.assertNotEqual(first,second)
            self.assertEqual(first[str((path/'motion.npz').resolve())],second[str((path/'motion.npz').resolve())])
    def test_cli_fit_keeps_explicit_method_for_generic_legacy_filename(self):
        import tempfile
        from unittest.mock import patch
        from motiogen.evalkit.k.export import target_template
        from motiogen.motion import Y_TO_Z
        sys.path.insert(0,str(ROOT/'python/scripts'))
        from evalkit_k import process
        names,rest,_=target_template('motionlcm');seen=[]
        def check_view(value,*args,**kwargs):
            self.assertIsInstance(value,Motion);seen.append(value.method)
            raise ValueError('test intercept before IK')
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as folder:
            path=Path(folder);source=path/'generic.npz'
            np.savez(source,joints=np.repeat((rest@Y_TO_Z.T)[None],3,axis=0),joint_names=names,fps=20.)
            with patch('motiogen.evalkit.k.fitting.fit',side_effect=check_view):
                result=process(source,'motionlcm',path/'output',fit=True,config='config/evalkit_k_v3.json')
            self.assertEqual(seen,['motionlcm']);self.assertEqual(result['fit']['reason'],'test intercept before IK')
if __name__=='__main__':unittest.main()
