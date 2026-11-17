import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.head_neck import forward,interpolate_rotations,attach,swing
from motiogen.evalkit.k.motion import Motion


class HeadNeckTests(unittest.TestCase):
    def test_known_fk_joint_pivots_and_parent_translation(self):
        z=Rotation.from_euler('z',90,degrees=True).as_matrix()
        local=np.array([[z,np.eye(3)]])
        p,r=forward(np.array([[2.,3,4]]),np.eye(3)[None],[np.array([0,1,0]),np.array([0,2,0])],local)
        np.testing.assert_allclose(p[0],[[2,3,4],[2,4,4],[0,4,4]],atol=1e-14)
        np.testing.assert_allclose(r[0,2],z)
        shifted,_=forward(np.array([[12.,3,4]]),np.eye(3)[None],[[0,1,0],[0,2,0]],local)
        np.testing.assert_allclose(shifted-p,np.broadcast_to([10,0,0],p.shape))

    def test_no_extrapolation_or_missing_rotation_bridge(self):
        r=Rotation.from_euler('z',np.array([0,10,20,30])[:,None],degrees=True).as_matrix();r[1]=np.nan
        q=interpolate_rotations(np.arange(4.),r,[-.1,0,.5,1,1.5,2,2.5,3,3.1])
        self.assertEqual(np.isfinite(q).all((-1,-2)).tolist(),[False,True,False,False,False,True,True,True,False])
        self.assertAlmostEqual(Rotation.from_matrix(q[6]).as_euler('xyz',degrees=True)[2],25.)

    def test_native_chain_fk_preserves_independent_relative_target(self):
        names=['left_hip','right_hip','spine3','neck','Neck2','head'];parents=[-1,-1,-1,2,3,4]
        rest=np.array([[.1,0,0],[-.1,0,0],[0,1,0],[0,1.2,0],[0,1.3,0],[0,1.4,0]])
        t=np.arange(3.)
        rot=np.tile(np.eye(3),(3,6,1,1))
        rot[:,2]=Rotation.from_euler('z',np.array([10,20,30])[:,None],degrees=True).as_matrix()
        rot[:,3]=rot[:,2]@Rotation.from_euler('x',20,degrees=True).as_matrix()
        rot[:,4]=rot[:,3];rot[:,5]=rot[:,4]@Rotation.from_euler('y',40,degrees=True).as_matrix()
        positions=np.tile(rest,(3,1,1))
        for child in (3,4,5):positions[:,child]=positions[:,parents[child]]+np.einsum('tij,j->ti',rot[:,parents[child]],rest[child]-rest[parents[child]])
        rig={'names':names,'parents':parents,'neutral_joints':rest.tolist(),'segment_frame_basis':np.tile(np.eye(3),(6,1,1)).tolist(),
             'decoded_native_right_basis':np.tile(np.eye(3),(6,1,1)).tolist()}
        m=Motion(positions,names,t,'synthetic',rot,names,{'input_rig':rig})
        transport=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        torso=rot[:,2]@transport.T
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(positions[:,2],torso,positions[:,2])):
            definition,h=attach(m,None,[],np.empty((3,0)),t)
        self.assertEqual(list(h['names']),['spine3','neck','Neck2','head'])
        np.testing.assert_allclose(h['positions'],positions[:,2:],atol=1e-14)
        np.testing.assert_allclose(h['head_relative_thorax'],rot[:,2].transpose(0,2,1)@rot[:,5],atol=1e-14)
        replay,_=forward(h['base_origin'],h['base_rotation'],np.array(json.loads(json.dumps(definition))['offsets_m']),h['local_rotations'])
        np.testing.assert_allclose(replay,h['positions'])

    def test_position_only_twist_unknown_and_fixed_root_replay(self):
        t=np.arange(3.);names=['spine3','neck','head']
        positions=np.tile([[0,1,0],[0,1.2,0],[0,1.4,0]],(3,1,1)).astype(float)
        positions[1,2]=[.2,1.2,0]
        m=Motion(positions,names,t,'motionlcm');origin=np.tile([0,1,0],(3,1));rot=np.tile(np.eye(3),(3,1,1))
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(origin,rot,None)):
            definition,h=attach(m,None,[],np.empty((3,0)),t)
        np.testing.assert_allclose(h['positions'],positions,atol=1e-14)
        self.assertFalse(h['relative_rotation_valid'].any())
        self.assertTrue(np.isnan(h['head_relative_thorax']).all())
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(origin+[.5,0,0],rot,None)):
            other,shifted=attach(m,None,[],np.empty((3,0)),t,fixed_definition=definition)
        self.assertEqual(other['attachment_local_m'],definition['attachment_local_m'])
        np.testing.assert_allclose(shifted['positions']-h['positions'],np.broadcast_to([.5,0,0],h['positions'].shape))

    def test_antipodal_swing_and_missing_direction(self):
        r=swing(np.eye(3),np.array([0.,1,0]),np.array([0.,-1,0]))
        np.testing.assert_allclose(r@[0,1,0],[0,-1,0],atol=1e-14)
        self.assertTrue(np.isnan(swing(np.eye(3),np.zeros(3),np.ones(3))).all())


if __name__=='__main__':unittest.main()

class FixedNativeAttachmentTests(unittest.TestCase):
    def test_fixed_attachment_changes_origin_not_lengths_or_joint_targets(self):
        names=['left_hip','right_hip','spine3','neck','head'];rest=np.array([[.1,0,0],[-.1,0,0],[0,1,0],[0,1.2,0],[0,1.4,0]])
        rotations=np.tile(np.eye(3),(3,5,1,1));times=np.arange(3.)
        rig={'names':names,'parents':[-1,-1,-1,2,3],'neutral_joints':rest.tolist(),
             'decoded_native_right_basis':np.tile(np.eye(3),(5,1,1)).tolist(),'segment_frame_basis':np.tile(np.eye(3),(5,1,1)).tolist()}
        motion=Motion(np.tile(rest,(3,1,1)),names,times,'known',rotations,names,{'input_rig':rig})
        transport=np.array([[0,0,1],[0,1,0],[-1,0,0]])
        body_rotation=np.tile(transport.T,(3,1,1));body_origin=np.tile([.1,.9,.1],(3,1))
        registered=np.tile([.1,1,.1],(3,1))
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(body_origin,body_rotation,registered)):
            old,base=attach(motion,None,[],np.empty((3,0)),times)
            definition,fit=attach(motion,None,[],np.empty((3,0)),times,attachment_policy='input_fixed_offset')
        np.testing.assert_allclose(fit['positions'][:,1:],motion.positions[:,3:],atol=1e-14)
        np.testing.assert_allclose(fit['offsets'],base['offsets'])
        np.testing.assert_allclose(fit['local_rotations'],base['local_rotations'])
        self.assertEqual(definition['names'][0],'cervical_base')
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(body_origin+[.2,0,0],body_rotation,registered+[.2,0,0])):
            before,h=attach(motion,None,[],np.empty((3,0)),times,fixed_definition=definition)
        self.assertEqual(before['attachment_local_m'],definition['attachment_local_m'])
        np.testing.assert_allclose(h['positions']-fit['positions'],np.broadcast_to([.2,0,0],h['positions'].shape))
