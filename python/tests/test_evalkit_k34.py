import unittest
import numpy as np
from motiogen.evalkit.k.trunk_transfer import minimum_swing,transfer
from motiogen.evalkit.k.solver_seeds import knee_seed
from types import SimpleNamespace

class TransferTests(unittest.TestCase):
    def test_swing_including_antipode(self):
        for v in [[1,0,0],[-1,0,0],[0,1,0]]:
            q=minimum_swing(np.array([1.,0,0]),np.array(v,float))
            np.testing.assert_allclose(q@[1,0,0],v,atol=1e-12)
            np.testing.assert_allclose(q.T@q,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(q),1.)
    def test_fixed_rig_fk_and_endpoint_frames(self):
        names=['pelvis','spine1','spine3','left_hip','right_hip','left_shoulder','right_shoulder']
        rest=np.array([[0,0,0],[0,.2,0],[0,.6,0],[-.1,0,0],[.1,0,0],[-.2,.6,0],[.2,.6,0]])
        parents=[-1,0,1,0,0,2,2];r=np.tile(np.eye(3),(2,7,1,1));src=np.tile(rest,(2,1,1));src[:,5:7]+=[0,-.1,.2]
        p,q,receipt=transfer(SimpleNamespace(positions=src),names,names,names,rest,parents,r,np.zeros((2,3)))
        for j,par in enumerate(parents):
            if par>=0:np.testing.assert_allclose(p[:,j]-p[:,par],q[:,par]@(rest[j]-rest[par]),atol=1e-12)
        np.testing.assert_array_equal(q[:,[0,2,3,4,5,6]],r[:,[0,2,3,4,5,6]])
        np.testing.assert_allclose(np.cross(p[:,5]+p[:,6],src[:,5]+src[:,6]),0,atol=1e-12)
        self.assertTrue(all(receipt['corrected_frames']))
    def test_input_seed_inside_bounds(self):
        target=np.array([[0,0,0],[0,-1,0],[.5,-1.8660254038,0],[.6,-1.9,0]])
        seed=knee_seed(np.zeros(2),['hip_flexion_l','knee_angle_l'],target,np.array([-1.,0]),np.array([2.,2.4]))
        self.assertAlmostEqual(seed[1],np.pi/6,places=7)
        self.assertEqual(seed[0],0.)
    def test_infeasible_target_is_explicit(self):
        names=['pelvis','spine1','spine3','left_hip','right_hip','left_shoulder','right_shoulder']
        rest=np.array([[0,0,0],[0,2,0],[0,2.1,0],[-.1,0,0],[.1,0,0],[-.2,2.1,0],[.2,2.1,0]])
        parents=[-1,0,1,0,0,2,2];r=np.tile(np.eye(3),(1,7,1,1));src=rest[None].copy();src[:,5:7,1]=0;src[:,5:7,2]=1
        p,q,receipt=transfer(SimpleNamespace(positions=src),names,names,names,rest,parents,r,np.zeros((1,3)))
        self.assertEqual(receipt['corrected_frames'],[False]);np.testing.assert_array_equal(q,r)
    def test_world_rotation_equivariance(self):
        from scipy.spatial.transform import Rotation
        names=['pelvis','spine1','spine3','left_hip','right_hip','left_shoulder','right_shoulder']
        rest=np.array([[0,0,0],[0,.2,0],[0,.6,0],[-.1,0,0],[.1,0,0],[-.2,.6,0],[.2,.6,0]])
        parents=[-1,0,1,0,0,2,2];r=np.tile(np.eye(3),(1,7,1,1));src=rest[None].copy();src[:,5:7]+=[0,-.1,.2]
        p,q,_=transfer(SimpleNamespace(positions=src),names,names,names,rest,parents,r,np.zeros((1,3)))
        world=Rotation.from_rotvec([.3,-.5,.1]).as_matrix()
        p2,q2,_=transfer(SimpleNamespace(positions=src@world.T),names,names,names,rest,parents,world@r,np.zeros((1,3)))
        np.testing.assert_allclose(p2,p@world.T,atol=1e-12);np.testing.assert_allclose(q2,world@q,atol=1e-12)

class QualityTests(unittest.TestCase):
    def test_two_small_direction_errors_can_hide_joint_error(self):
        from motiogen.evalkit.k.quality import metric_fidelity,apply_metric_gates
        from motiogen.evalkit.k.bones import NAMES
        from motiogen.evalkit.k.fitting3 import support
        p=np.zeros((1,len(NAMES),3));ix={n:i for i,n in enumerate(NAMES)}
        for side,x in [('left',-.1),('right',.1)]:
            for node,y in [('hip',1.),('knee',.6),('ankle',.2),('foot',.1),('shoulder',1.6),('elbow',1.2),('wrist',.8)]:p[0,ix[side+'_'+node]]=[x,y,0]
        p[:,ix['pelvis_mid']]=[0,1,0];p[:,ix['shoulder_mid']]=[0,1.6,0]
        f=p.copy();theta=np.deg2rad(8)
        f[0,ix['left_knee']]=p[0,ix['left_hip']]+[.4*np.sin(theta),-.4*np.cos(theta),0]
        f[0,ix['left_ankle']]=f[0,ix['left_knee']]+[-.4*np.sin(theta),-.4*np.cos(theta),0]
        f[0,ix['left_foot']]=f[0,ix['left_ankle']]+[0,-.1,0]
        q=support(p,f,NAMES);self.assertTrue(q['left_leg']['angle_valid'][0])
        m=metric_fidelity(p,f,NAMES,10);self.assertTrue(m['left_knee_flexion']['observed'][0]);self.assertFalse(m['left_knee_flexion']['model_fit_valid'][0])
        apply_metric_gates(q,m);self.assertTrue(q['left_leg']['direction_valid'][0]);self.assertFalse(q['left_leg']['angle_valid'][0])
        self.assertEqual(m['left_knee_flexion']['reason'][0],3)
        # Missing foot observation must not erase an independently observed knee.
        m=metric_fidelity(p,f,NAMES,10,prior=['left_foot'])
        self.assertTrue(m['left_knee_flexion']['observed'][0]);self.assertFalse(m['left_foot_incline']['observed'][0])
