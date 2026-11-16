"""Independent native fields with analytic rotations; no controlled serializer is used."""
import json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k import load,evaluate
from motiogen.evalkit.k.export import target_template
from motiogen.evalkit.k.calibration import CLOSD
from motiogen.motion import Y_TO_Z
from motion_retarget.bvh import parse_bvh


def rx(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[1.,0.,0.],[0.,c,-s],[0.,s,c]])


class Contracts(unittest.TestCase):
    def test_animated_bvh_translation_replaces_offset_with_known_fk(self):
        payload='''HIERARCHY
ROOT Hips
{
 OFFSET 0 100 0
 CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
 JOINT Child
 {
  OFFSET 0 20 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Leaf
  {
   OFFSET 0 10 0
   CHANNELS 3 Zrotation Xrotation Yrotation
   End Site { OFFSET 0 2 0 }
  }
 }
}
MOTION
Frames: 2
Frame Time: 0.05
0 5 0 90 0 0 0 20 0 0 0 0 0 0 0
0 6 0 0 0 0 0 21 0 0 0 0 0 0 0
'''
        b=parse_bvh(payload);p,_=b.forward_kinematics(True,translation_mode='absolute_channels',dtype=np.float64)
        np.testing.assert_allclose(p[0],[[0,5,0],[-20,5,0],[-30,5,0]],atol=1e-12)
        np.testing.assert_allclose(p[1],[[0,6,0],[0,27,0],[0,37,0]],atol=1e-12)
        legacy,_=b.forward_kinematics(True);np.testing.assert_allclose(legacy[1,0],[0,106,0])

    def test_primal_legacy_canonical_keeps_position_support_without_pose(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as directory:
            names,rest,_=target_template('primal');p=np.repeat(rest[None],3,0)
            path=Path(directory)/'old_canonical.npz'
            np.savez(path,joints=p@Y_TO_Z.T,joint_names=names,fps=30.,
                     metadata_json=json.dumps({'baseline':'primal','axes':'right-handed Z-up','units':'m'}))
            motion=load(path,config='config/evalkit_k_v3.json');result=evaluate(motion,config='config/evalkit_k_v3.json')
            np.testing.assert_allclose(motion.positions,p)
            self.assertIsNone(motion.rotations)
            self.assertTrue(result['body']['left_knee_flexion']['valid'].all())
            self.assertFalse(result['metrics']['neck']['valid'].any())
            self.assertIn('without_native_pose',motion.provenance['orientation_limitation'])

    def test_analytic_neutral_and_single_axis_in_eight_native_layouts(self):
        phases=np.deg2rad([0.,40.,90.]);expected=np.array([rx(a) for a in phases])
        for method in ['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']:
            with self.subTest(method=method),tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as directory:
                folder=Path(directory);names,rest,_=target_template(method);p=np.repeat(rest[None],3,0)
                local=np.broadcast_to(np.eye(3),(3,len(names),3,3)).copy();local[:,names.index('head')]=expected
                world=Y_TO_Z@local
                canonical={'joints':p@Y_TO_Z.T,'joint_names':names,'fps':30.,
                           'metadata_json':json.dumps({'baseline':method,'axes':'right-handed Z-up','units':'m'})}
                if method=='ardy':canonical['global_rotations']=world
                if method=='chois':
                    (folder/'raw').mkdir();np.savez(folder/'raw/native_prediction.npz',global_rotations=world[:,:22])
                if method=='closd':
                    q=np.zeros((3,len(CLOSD),4));q[:,:,0]=np.sin(np.pi/4);q[:,:,3]=np.cos(np.pi/4)
                    h=CLOSD.index('Head');q[:,h,0]=np.sin((phases+np.pi/2)/2);q[:,h,3]=np.cos((phases+np.pi/2)/2)
                    np.savez(folder/'native_physics.npz',rotations_xyzw=q,body_names=CLOSD)
                if method=='dart':np.savez(folder/'smplx_params.npz',global_orient=np.broadcast_to(Y_TO_Z,(3,3,3)),body_pose=local[:,1:])
                if method=='kimodo':canonical={'posed_joints':p,'global_rot_mats':local,'fps':30.}
                if method=='primal':
                    pose=np.zeros((1,3,69));pose[0,:,3+15*3]=phases
                    canonical={'pose':pose,'joints':p[None],'fps':30.}
                source=folder/'motion.npz';np.savez(source,**canonical)
                motion=load(source,method,config='config/evalkit_k_v3.json')
                if method in ['motionlcm','omnicontrol']:
                    self.assertIsNone(motion.rotations)
                    self.assertEqual(evaluate(motion,config='config/evalkit_k_v3.json')['metrics']['neck']['status'],'unreliable')
                else:
                    head='Head' if method=='closd' else 'head'
                    recovered=motion.rotations[:,motion.rotation_names.index(head)]
                    np.testing.assert_allclose(recovered,expected,atol=1e-7)
                    # Explicit vector expectation detects axis sign/order mistakes.
                    vectors=recovered@np.array([0.,0.,1.])
                    np.testing.assert_allclose(vectors,np.stack([np.zeros(3),-np.sin(phases),np.cos(phases)],1),atol=1e-7)


if __name__=='__main__':unittest.main()
