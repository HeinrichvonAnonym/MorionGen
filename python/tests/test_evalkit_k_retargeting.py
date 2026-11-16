"""Independent FK and topology checks for controlled spine conversion."""
import sys,unittest,tempfile,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.export import retarget,target_template,serialize
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.geometry import frame
from motiogen.evalkit.k.retargeting import spine_mapping

class Retargeting(unittest.TestCase):
    def test_four_node_fk_does_not_duplicate_proximal_rotation(self):
        names,rest,parents=target_template('ardy');source=list(names)
        replacements={'pelvis':'Hips','Spine':'Spine','spine1':'Spine1','spine2':'Spine2','Spine3':'Spine3','head':'Head'}
        source=[replacements.get(n,n) for n in source]
        rot=np.broadcast_to(np.eye(3),(4,len(names),3,3)).copy()
        for n,deg in [('Spine',10),('spine1',25),('spine2',40),('Spine3',55)]:rot[:,names.index(n)]=Rotation.from_euler('x',deg,degrees=True).as_matrix()
        p=np.zeros((4,len(names),3))
        for j,par in enumerate(parents):p[:,j]=rest[j] if par<0 else p[:,par]+rot[:,par]@(rest[j]-rest[par])
        basis=frame(np.array([0.,1,0]),rest[names.index('right_hip')]-rest[names.index('left_hip')])
        ref=Motion(p,source,np.arange(4)/20,'reference',rot,source,{'segment_basis':basis})
        new,_=retarget(ref,'ardy','topology_v2');old,_=retarget(ref,'ardy')
        np.testing.assert_allclose(new.positions,p,atol=1e-9)
        self.assertGreater(np.max(abs(old.positions-p)),.001)
        mapping=new.provenance['spine_mapping']['mapping'];self.assertEqual(len(set(mapping.values())),4)
        np.testing.assert_allclose(np.linalg.det(new.rotations),1.,atol=1e-9)
        q=Rotation.from_euler('y',37,degrees=True).as_matrix()
        rotated=Motion(p@q.T,source,ref.times,'reference',q@rot,source,{'segment_basis':basis})
        converted,_=retarget(rotated,'ardy','topology_v2')
        np.testing.assert_allclose(converted.positions,new.positions@q.T,atol=1e-9)
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs') as directory:
            serialize(new,parents,directory)
            meta=json.loads((Path(directory)/'k_provenance.json').read_text())
            self.assertEqual(meta['retarget_policy'],'topology_v2')
            self.assertNotIn('spine_mapping',meta)
            self.assertNotIn('retarget_bases',meta)

    def test_three_node_chain_compression_is_ordered_and_explicit(self):
        names=['pelvis','spine1','spine2','spine3'];rest=np.array([[0,0,0],[0,.25,0],[0,.75,0],[0,1,0]])
        src=['pelvis','spine1','spine2','spine_mid','spine3'];p=np.array([[0,i/4,0] for i in range(5)])[None]
        ref=Motion(np.repeat(p,2,axis=0),src,np.array([0.,.1]),'reference')
        mapped,receipt=spine_mapping(ref,names,rest,[-1,0,1,2],names,src)
        self.assertEqual(mapped,['pelvis','spine1','spine_mid','spine3'])
        self.assertEqual(receipt['status'],'derived_from_chain_arc_lengths')
        with self.assertRaisesRegex(ValueError,'additional_unobserved'):
            spine_mapping(ref,names,rest,[-1,0,1,2],names,['pelvis','spine1','head'])
        with self.assertRaisesRegex(ValueError,'degenerate'):
            spine_mapping(ref,names,np.zeros_like(rest),[-1,0,1,2],names,src)

    def test_unknown_policy_fails(self):
        names,rest,parents=target_template('kimodo');r=np.broadcast_to(np.eye(3),(2,len(names),3,3));m=Motion(np.repeat(rest[None],2,0),names,np.array([0.,.1]),'reference',r,names)
        with self.assertRaisesRegex(ValueError,'Unknown retarget policy'):retarget(m,'kimodo','typo')

if __name__=='__main__':unittest.main()
