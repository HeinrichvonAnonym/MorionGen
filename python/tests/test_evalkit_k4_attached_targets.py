import sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.head_neck import attach
from motiogen.evalkit.k.attached_targets import compile_targets,AttachedTargets


class AttachedNodeTests(unittest.TestCase):
    def position_fixture(self):
        t=np.arange(3.)/30;origin=np.tile([0.,1.,0.],(3,1));r=np.tile(np.eye(3),(3,1,1))
        p=np.tile([[0.,1.,0.],[0.,1.2,0.],[0.,1.4,0.]],(3,1,1));p[1,2]=[.2,1.2,0.]
        m=Motion(p,['spine3','neck','head'],t,'motionlcm')
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(origin,r,None)):
            definition,_=attach(m,None,[],np.empty((3,0)),t)
        return m,origin,r,definition

    def test_position_chain_matches_full_fk_after_independent_body_transform(self):
        m,o,r,d=self.position_fixture()
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(o,r,None)):
            compiled=compile_targets(m,None,[],np.empty((3,0)),m.times,d)
        other_o=o+[2.,-.3,4.];other_r=Rotation.from_euler('xyz',[[80,10,0],[0,-60,30],[20,30,70]],degrees=True).as_matrix()
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(other_o,other_r,None)):
            _,full=attach(m,None,[],np.empty((3,0)),m.times,fixed_definition=d)
        actual=np.stack([compiled.positions(i,other_o[i],other_r[i]) for i in range(3)])
        np.testing.assert_allclose(actual,full['positions'][:,1:],atol=1e-14)
        self.assertFalse(full['relative_rotation_valid'].any())
        with self.assertRaises(ValueError):compiled.body_offset[0]=2

    def test_declared_internal_neck_chain_matches_full_fk_under_new_body_pose(self):
        t=np.arange(3.)/30;names=['left_hip','right_hip','spine3','neck','Neck2','head']
        rest=np.array([[.1,0,0],[-.1,0,0],[0,1,0],[0,1.2,0],[0,1.3,0],[0,1.4,0]])
        rotations=np.tile(np.eye(3),(3,6,1,1));rotations[:,3:]=Rotation.from_euler('z',np.array([10,30,70])[:,None],degrees=True).as_matrix()[:,None]
        p=np.tile(rest,(3,1,1)).astype(float)
        for j,parent in [(3,2),(4,3),(5,4)]:p[:,j]=p[:,parent]+np.einsum('tij,j->ti',rotations[:,parent],rest[j]-rest[parent])
        rig=dict(names=names,parents=[-1,-1,-1,2,3,4],neutral_joints=rest.tolist(),segment_frame_basis=np.tile(np.eye(3),(6,1,1)).tolist(),decoded_native_right_basis=np.tile(np.eye(3),(6,1,1)).tolist())
        m=Motion(p,names,t,'synthetic',rotations,names,{'input_rig':rig});o=p[:,2];r=np.tile(np.eye(3),(3,1,1))
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(o,r,o)):
            d,_=attach(m,None,[],np.empty((3,0)),t,attachment_policy='input_fixed_offset')
            compiled=compile_targets(m,None,[],np.empty((3,0)),t,d)
        other_o=o+[.4,.2,0];other_r=Rotation.from_euler('xyz',[[0,30,0],[20,0,80],[60,30,0]],degrees=True).as_matrix()
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(other_o,other_r,other_o)):
            _,full=attach(m,None,[],np.empty((3,0)),t,fixed_definition=d)
        np.testing.assert_allclose(np.stack([compiled.positions(i,other_o[i],other_r[i]) for i in range(3)]),full['positions'][:,[1,3]],atol=1e-14)

    def test_missing_input_retains_unknown_targets_without_reducing_other_support(self):
        m,o,r,d=self.position_fixture();m.positions[1,2]=np.nan
        with patch('motiogen.evalkit.k.head_neck.body_frames',return_value=(o,r,None)):
            c=compile_targets(m,None,[],np.empty((3,0)),m.times,d)
            _,full=attach(m,None,[],np.empty((3,0)),m.times,fixed_definition=d)
        self.assertEqual(c.receipt['supported_frames'],{'neck':3,'head':2})
        self.assertTrue(np.isnan(c.observed[1,1]).all());self.assertTrue(np.isnan(full['positions'][1,2]).all())
        self.assertTrue(np.isfinite(c.positions(1,o[1],r[1])).all())

    def test_wrong_clock_and_unfixed_geometry_rejected(self):
        m,o,r,d=self.position_fixture()
        with self.assertRaisesRegex(ValueError,'clock'):compile_targets(m,None,[],np.empty((3,0)),m.times+.1,d)
        with self.assertRaisesRegex(ValueError,'fixed'):compile_targets(m,None,[],np.empty((3,0)),m.times,{})

    def test_body_objective_addition_preserves_old_terms_and_known_model_fk(self):
        import opensim as osim
        import biomech
        from motiogen.evalkit.k.head_neck import body_frames
        from motiogen.evalkit.k.k4_fitting import model_fk
        from motiogen.evalkit.k.trajectory import problem
        osim.Logger.setLevelString('error');model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        q=np.tile([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels],(3,1));q[:,labels.index('lumbar_extension')]=[0,10,20]
        for name in ('knee_angle_l','knee_angle_r'):q[:,labels.index(name)]=40
        for name in ('elbow_flex_l','elbow_flex_r'):q[:,labels.index(name)]=30
        p,names,q=model_fk(biomech.MODEL,labels,q,direct_couplers=True,return_coordinates=True);t=np.arange(3.)/30
        o,r,_=body_frames(biomech.MODEL,labels,q);offset=np.array([.01,.4,.02]);world=np.tile([[0,0,0],[.1,.1,0]],(3,1,1));observed=o[:,None]+np.einsum('tij,j->ti',r,offset)[:,None]+world
        attached=AttachedTargets(('neck','head'),t,offset,world,observed,{'scope':'known FK'})
        settings={'version':'joint_trajectory_v1','target_violation_weight':4,'target_margin_fraction':.8}
        base,x,_,_,_=problem(biomech.MODEL,labels,q,p,names,t,settings)
        full,y,_,_,_=problem(biomech.MODEL,labels,q,p,names,t,settings,attached_targets=attached)
        np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(full.target[:,:base.k],base.target)
        np.testing.assert_allclose(full.residual(y),0,atol=1e-7)
        self.assertEqual(full.target.shape[1],base.target.shape[1]+6)
        np.testing.assert_allclose(full.jacobian(y).toarray(),np.stack([(full.residual(y+np.eye(len(y))[i]*1e-6)-full.residual(y-np.eye(len(y))[i]*1e-6))/(2e-6) for i in range(len(y))],1),atol=2e-3,rtol=2e-4)

if __name__=='__main__':unittest.main()
