import sys
from pathlib import Path
import unittest
import tempfile
import json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.geometry import frame,distance,chain,recover_short_gaps,valid_rotation
from motiogen.evalkit.k.motion import Motion,load
from motiogen.evalkit.k.posture import evaluate
from motiogen.evalkit.k.reference import template_fit,marker_frames


class KTests(unittest.TestCase):
    def test_rigid_frame(self):
        r=frame(np.array([[0.,1,0]]),np.array([[1.,0,0]]))
        np.testing.assert_allclose(r,np.eye(3)[None]);self.assertTrue(valid_rotation(r).all())

    def test_collinear(self):
        self.assertFalse(valid_rotation(frame(np.array([[0.,1,0]]),np.array([[0.,2,0]]))).any())

    def test_rotation_wrap(self):
        r=Rotation.from_euler('y',[[179],[-179]],degrees=True).as_matrix()
        self.assertAlmostEqual(float(distance(r[0],r[1])),2)

    def test_gap_is_prediction(self):
        times=np.arange(10)/30;r=Rotation.from_euler('y',(np.arange(10)*2)[:,None],degrees=True).as_matrix()
        observed=np.ones(10,bool);observed[3:5]=False
        g=r.copy();g[~observed]=np.nan
        p,mask=recover_short_gaps(times,g,observed,.1)
        np.testing.assert_allclose(p,r,atol=1e-10);self.assertEqual(mask.sum(),2)
        self.assertFalse(observed[3]);observed[:5]=False
        _,mask=recover_short_gaps(times,g,observed,.1);self.assertEqual(mask.sum(),0)

    def test_leaf_orientation_unknown(self):
        p=np.zeros((3,5,3));p[:,1]=[0,1,0];p[:,2]=[0,2,0];p[:,3]=[-1,1,0];p[:,4]=[1,1,0]
        m=Motion(p,['spine3','neck','head','left_shoulder','right_shoulder'],np.arange(3)/20,'motionlcm')
        result=evaluate(m)
        self.assertTrue(result['metrics']['thorax']['valid'].all())
        self.assertEqual(result['metrics']['neck']['status'],'unreliable')

    def test_native_is_not_anatomical(self):
        p=np.zeros((3,2,3));r=np.broadcast_to(np.eye(3),(3,2,3,3)).copy()
        m=Motion(p,['spine3','head'],np.arange(3)/30,'dart',r,['spine3','head'])
        self.assertEqual(evaluate(m)['metrics']['neck']['status'],'native_frame_only')
        self.assertEqual(evaluate(m,{'head':np.eye(3),'thorax':np.eye(3)})['metrics']['neck']['status'],'supported')

    def test_independent_template_and_missing_marker(self):
        template=np.array([[-1,0,1],[1,0,1],[-1,0,-1],[1,0,-1]])*.1
        r=Rotation.from_euler('xyz',np.arange(90).reshape(30,3),degrees=True).as_matrix()
        p=np.einsum('tij,kj->tki',r,template)+np.array([1.,2,3])
        p[10:,0]=np.nan
        recovered,res,cal=template_fit(p,r,5)
        self.assertLess(np.nanmax(distance(recovered,r)),1e-5)
        self.assertLess(np.nanmax(res),1e-10);self.assertEqual(len(cal),5)

    def test_bad_clock(self):
        with self.assertRaises(ValueError):Motion(np.zeros((2,1,3)),['head'],[0,0],'ardy')

    def test_position_profile_ignores_redundant_rotation(self):
        r=np.broadcast_to(np.eye(3),(3,2,3,3)).copy()
        m=Motion(np.zeros((3,2,3)),['spine3','head'],np.arange(3)/20,'motionlcm',r,['spine3','head'])
        self.assertEqual(evaluate(m,{'head':np.eye(3),'thorax':np.eye(3)})['metrics']['neck']['status'],'unreliable')

    def test_irregular_clock_not_silently_fitted(self):
        m=Motion(np.zeros((3,1,3)),['head'],[0,.02,.1],'ardy')
        with self.assertRaises(ValueError):m.legacy()

    def test_bent_and_seated_calibration(self):
        from motiogen.evalkit.k.fitting import upright_frames
        names=['left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_shoulder','right_shoulder']
        p=np.broadcast_to(np.array([[-.1,1,0],[.1,1,0],[-.1,.5,0],[.1,.5,0],[-.1,0,0],[.1,0,0],[-.2,1.5,0],[.2,1.5,0]]),(40,8,3)).copy()
        self.assertTrue(upright_frames(p,names,30)[1]['qualified'])
        p[:,6:,1:]=[1.,.5]
        self.assertFalse(upright_frames(p,names,30)[1]['qualified'])
        p[:,6:,1:]=[1.5,0];p[:,2:4,1:]=[1.,.5]
        self.assertFalse(upright_frames(p,names,30)[1]['qualified'])

    def test_long_and_edge_gaps_stay_unknown(self):
        t=np.arange(100)/30;r=np.broadcast_to(np.eye(3),(100,3,3)).copy();observed=np.ones(100,bool)
        observed[:5]=False;observed[10:40]=False;observed[-5:]=False;r[~observed]=np.nan
        recovered,predicted=recover_short_gaps(t,r,observed,.1)
        self.assertFalse(predicted.any());self.assertTrue(np.isnan(recovered[20]).all())

    def test_actual_eight_native_readers(self):
        evidence=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())
        for method,samples in evidence['baseline_samples'].items():
            with self.subTest(method=method):
                path=Path(samples[0]['path']);path=ROOT/path if not path.is_absolute() else path
                path=Path(str(path).replace('9bbdd73e-4410-4279-9639-6161f0c95839/','9bbdd73e-4410-4279-9639-6161f0c958391/'))
                if not path.exists():self.skipTest('Registered local baseline sample is not installed')
                m=load(path,method);result=evaluate(m)
                self.assertTrue(np.isfinite(m.positions).all())
                self.assertEqual(result['metrics']['neck']['status'],'unreliable' if method in ('motionlcm','omnicontrol') else 'model_convention')

    def test_invalid_quaternion_keeps_missing(self):
        from motiogen.motion import HUMANML_NAMES
        with tempfile.TemporaryDirectory() as d:
            np.savez(Path(d)/'motion.npz',joints=np.zeros((3,22,3)),joint_names=HUMANML_NAMES,fps=30)
            np.savez(Path(d)/'native_physics.npz',rotations_xyzw=np.zeros((3,2,4)),body_names=['Head','Chest'])
            m=load(d,'closd');self.assertTrue(np.isnan(m.rotations).all())
            self.assertEqual(evaluate(m)['metrics']['neck']['status'],'unreliable')

    def test_closd_prior_mask_uses_body_names(self):
        r=np.broadcast_to(np.eye(3),(3,2,3,3)).copy()
        m=Motion(np.zeros((3,2,3)),['spine3','head'],np.arange(3)/30,'closd',r,['Chest','Head'],
                 {'degradation':{'prior_joints':['spine3']}})
        self.assertEqual(evaluate(m,{'head':np.eye(3),'thorax':np.eye(3)})['metrics']['neck']['status'],'unreliable')

    def test_cli_prediction_keeps_observation_mask(self):
        sys.path.insert(0,str(ROOT/'python/scripts'))
        from evalkit_k import process
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);r=np.broadcast_to(np.eye(3),(10,2,3,3)).copy();r[3:5,1]=np.nan
            np.savez(folder/'motion.npz',joints=np.zeros((10,2,3)),joint_names=['Spine3','head'],global_rotations=r,fps=20)
            process(folder,'ardy',folder/'out',predict_short_gaps=True)
            with np.load(folder/'out/k.npz') as z:
                self.assertEqual(z['head_predicted'].sum(),2)
                self.assertFalse(z['head_valid'][3:5].any())
                self.assertTrue(np.isfinite(z['head_with_predictions'][3:5]).all())

    def test_extreme_world_offset_rejects_positions_keeps_rotation(self):
        r=np.broadcast_to(np.eye(3),(3,2,3,3)).copy()
        m=Motion(np.full((3,2,3),3e24),['spine3','head'],np.arange(3)/30,'dart',r,['spine3','head'])
        self.assertTrue(np.isnan(m.positions).all())
        self.assertEqual(m.provenance['coordinate_precision']['rejected_frames'],[0,1,2])
        self.assertEqual(evaluate(m)['metrics']['neck']['status'],'native_frame_only')

    def test_optical_neutral_and_forward_flexion_axes(self):
        names=['LFHD','RFHD','LBHD','RBHD','C7','T10','CLAV','STRN','RBAK']
        p=np.array([[-.1,2,-.1],[.1,2,-.1],[-.1,2,.1],[.1,2,.1],
                    [0,1,.1],[0,0,.1],[0,1,-.1],[0,0,-.1],[.05,.5,.1]])
        frames=marker_frames(p[None],names,'andy')
        np.testing.assert_allclose(frames['head'][0][0],np.eye(3),atol=1e-12)
        np.testing.assert_allclose(frames['thorax'][0][0],np.eye(3),atol=1e-12)
        rotation=Rotation.from_euler('x',-30,degrees=True).as_matrix()
        p[:4]=(p[:4]-[0,2,0])@rotation.T+[0,2,0]
        frames=marker_frames(p[None],names,'andy')
        relative=frames['thorax'][0][0].T@frames['head'][0][0]
        self.assertAlmostEqual(-Rotation.from_matrix(relative).as_euler('XYZ',degrees=True)[0],30)

    def test_failed_position_fit_preserves_native_k_output(self):
        sys.path.insert(0,str(ROOT/'python/scripts'))
        from evalkit_k import process
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);r=np.broadcast_to(np.eye(3),(3,2,3,3)).copy()
            np.savez(folder/'motion.npz',joints=np.full((3,2,3),3e24),joint_names=['Spine3','head'],global_rotations=r,fps=30,
                     metadata_json=json.dumps({'baseline':'ardy'}))
            result=process(folder,'ardy',folder/'out',fit=True)
            self.assertEqual(result['fit']['status'],'unavailable')
            self.assertEqual(result['metrics']['neck']['status'],'native_frame_only')
            self.assertTrue((folder/'out/k.npz').exists())

    def test_chois_body_wrist_and_extra_hand_priors_are_distinct(self):
        from motiogen.evalkit.k.validation import segment_fixture
        from motiogen.evalkit.k.export import retarget,serialize
        sys.path.insert(0,str(ROOT/'python/scripts'))
        from evalkit_k import process
        t=np.arange(40)/30;r=np.broadcast_to(np.eye(3),(40,3,3)).copy()
        target,parents=retarget(segment_fixture(t,r,r),'chois')
        with tempfile.TemporaryDirectory() as d:
            source=serialize(target,parents,Path(d)/'input');motion=load(source,'chois')
            prior=motion.provenance['degradation']['prior_joints']
            self.assertTrue({'left_wrist_body','right_wrist_body','left_wrist','right_wrist'}<=set(prior))
            process(source,'chois',Path(d)/'out')
            with np.load(Path(d)/'out/k.npz') as z:
                self.assertFalse(z['segment_valid'][:,20:22].any())
                self.assertFalse(z['position_valid'][:,20:24].any())
                self.assertTrue(z['head_valid'].all())

    def test_fast_rotation_resampling_preserves_clock_and_reports_loss(self):
        from motiogen.evalkit.k.export import resample
        from scipy.spatial.transform import Slerp
        t=50+np.arange(241)/120
        r=Rotation.from_euler('y',(90*np.sin(2*np.pi*8*(t-50)))[:,None],degrees=True).as_matrix()
        m=Motion(np.zeros((len(t),1,3)),['head'],t,'reference',r[:,None],['head'])
        low=resample(m,20)
        np.testing.assert_allclose(np.diff(low.times),.05,atol=1e-12)
        self.assertEqual(low.times[0],50)
        self.assertTrue(valid_rotation(low.rotations).all())
        truth=Slerp(t,Rotation.from_matrix(r))(low.times).as_matrix()
        self.assertGreater(float(np.median(distance(truth,low.rotations[:,0]))),5)


if __name__=='__main__':unittest.main()
