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
if __name__=='__main__':unittest.main()
