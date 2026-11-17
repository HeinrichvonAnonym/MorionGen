import json,sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.body_frames import targets
from motiogen.evalkit.k.motion import Motion,load


class BodyFrameTests(unittest.TestCase):
    def test_neutral_and_single_axis_known_world_directions(self):
        names=['pelvis','spine3','head','left_hip','right_hip']
        rest=np.array([[0,0,0],[0,1,0],[0,2,0],[.1,0,0],[-.1,0,0]])
        rotations=np.tile(np.eye(3),(2,5,1,1));z=Rotation.from_euler('z',90,degrees=True).as_matrix()
        rotations[1]=z
        rig={'names':names,'neutral_joints':rest.tolist(),'decoded_native_right_basis':np.tile(np.eye(3),(5,1,1)).tolist(),
             'segment_frame_basis':np.tile(np.eye(3),(5,1,1)).tolist()}
        motion=Motion(np.tile(rest,(2,1,1)),names,[0,1],'known',rotations,names,{'input_rig':rig})
        result,receipt=targets(motion)
        for body in ('pelvis','torso'):
            np.testing.assert_allclose(result[body][0]@[0,0,-1],[1,0,0],atol=1e-14)
            np.testing.assert_allclose(result[body][1]@[0,0,-1],[0,1,0],atol=1e-14)
            np.testing.assert_allclose(result[body][1]@[0,1,0],[-1,0,0],atol=1e-14)
        self.assertIn('no anatomical',receipt['scope'])

    def test_eight_frozen_formats_preserve_field_based_availability(self):
        run=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
        for method in ['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']:
            path=run/'inputs/015'/method/'candidate'
            motion=load(path/'factorial/reference.npz' if method=='primal' else path,method,config=ROOT/'config/evalkit_k4_20261117/input.json')
            result,receipt=targets(motion)
            if method in ('motionlcm','omnicontrol'):
                self.assertFalse(result);self.assertEqual(receipt['status'],'unavailable')
            else:
                self.assertEqual(set(result),{'pelvis','torso'})
                self.assertEqual(result['torso'].shape,(len(motion.times),3,3))
                np.testing.assert_allclose(np.linalg.det(result['torso']),1,atol=1e-6)


if __name__=='__main__':unittest.main()
