import unittest
import numpy as np
from motiogen.evalkit.k.root_alignment import anchor_target, hip_midpoint, align_fitted_root
from motiogen.evalkit.k.config import digest


class RootAnchorTests(unittest.TestCase):
    def test_export_anchor_preserves_every_relative_point(self):
        names=['pelvis','left_hip','right_hip','head']
        points=np.array([[[2.,1,0],[1.8,.9,0],[2.2,.9,0],[2.,2,0]],
                         [[3.,1,0],[2.8,.8,.1],[3.2,.8,.1],[3.,2,0]]])
        original=points.copy();wanted=np.array([[2.,1.2,.1],[3.,1.4,.2]])
        fixed,shift=anchor_target(points,names,wanted)
        np.testing.assert_allclose(hip_midpoint(fixed,names),wanted)
        np.testing.assert_allclose(np.diff(fixed,axis=1),np.diff(points,axis=1))
        np.testing.assert_array_equal(points,original)
        self.assertFalse(np.allclose(shift[0],shift[1]))
        with self.assertRaises(ValueError):anchor_target(points,names,np.full((2,3),np.nan))

    def test_model_fk_translation_and_missing_observation(self):
        import opensim as osim
        import biomech
        from motiogen.evalkit.k.fitting3 import positions
        osim.Logger.setLevelString('error')
        path=biomech.MODEL;before=digest(path)
        model=osim.Model(str(path));state=model.initSystem();cs=model.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        values=[np.degrees(cs.get(i).getValue(state)) if cs.get(i).getMotionType()==1 else cs.get(i).getValue(state) for i in range(cs.getSize())]
        q=np.tile(values,(3,1));q[:,labels.index('pelvis_rotation')]=[-35.,10.,65.]
        times=np.arange(3)/100;fitted,names=positions(path,times,labels,q)
        delta=np.array([[.15,.25,-.1],[-.2,.1,.3],[.04,-.03,.07]])
        source=fitted+delta[:,None];source[1,names.index('left_hip')]=np.nan
        aligned,final,valid,shift,receipt=align_fitted_root(path,times,labels,q,source,names)
        np.testing.assert_array_equal(valid,[True,False,True]);np.testing.assert_array_equal(aligned[1],q[1])
        np.testing.assert_allclose(final[[0,2]],(fitted+delta[:,None])[[0,2]],atol=1e-8)
        angular=[i for i in range(len(labels)) if cs.get(labels[i]).getMotionType()==1]
        np.testing.assert_array_equal(aligned[:,angular],q[:,angular])
        self.assertEqual(receipt['valid_frames'],2);self.assertEqual(digest(path),before)
        prior_result=align_fitted_root(path,times,labels,q,source,names,prior=['left_hip'])
        self.assertFalse(prior_result[2].any());self.assertEqual(prior_result[4]['status'],'unavailable')
        np.testing.assert_array_equal(prior_result[0],q)
