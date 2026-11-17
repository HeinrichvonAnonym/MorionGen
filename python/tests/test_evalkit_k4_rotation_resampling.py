import sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.k4_fitting import model_fk,resample_native
from motiogen.evalkit.k.diagnostic_domain import apply
from motiogen.evalkit.k.rotation_resampling import hip_slerp


class RotationResamplingTests(unittest.TestCase):
    def test_equivalent_chart_poses_remain_identical_under_model_fk(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261117_k4') as temp:
            path=Path(temp)/'model.osim';model=osim.Model(str(biomech.MODEL));model.printToXML(str(path))
            ranges={n+s:[-360,360] for s in ('l','r') for n in ('hip_flexion_','hip_adduction_','hip_rotation_')}
            apply(path,{'base':'release_subtalar_original_range','scope':'independent_kinematic','coordinate_ranges_deg':ranges})
            model=osim.Model(str(path));state=model.initSystem();cs=model.getCoordinateSet()
            labels=[cs.get(i).getName() for i in range(cs.getSize())]
            row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
            q=np.tile(row,(2,1));cols=[labels.index(n+'_r') for n in ['hip_flexion','hip_adduction','hip_rotation']]
            q[0,cols]=[10,89,20];q[1,cols]=[190,91,200]
            t=np.array([0.,1.]);new,linear=resample_native(t,labels,q,10)
            corrected,receipt=hip_slerp(path,labels,q,t,new,linear)
            native,n=model_fk(path,labels,q,direct_couplers=True)
            bad,_=model_fk(path,labels,linear,direct_couplers=True)
            good,_=model_fk(path,labels,corrected,direct_couplers=True)
            np.testing.assert_allclose(native[0],native[1],atol=1e-10)
            np.testing.assert_allclose(good,np.broadcast_to(native[0],good.shape),atol=1e-10)
            self.assertGreater(np.linalg.norm(bad[len(bad)//2]-native[0]),.01)
            untouched=[i for i,n in enumerate(labels) if not n.startswith('hip_')]
            np.testing.assert_array_equal(corrected[:,untouched],linear[:,untouched])
            self.assertLess(receipt['joints']['hip_r']['native_model_matrix_error'],1e-10)

    def test_canonical_output_requires_declared_domain(self):
        m=osim.Model(str(biomech.MODEL));state=m.initSystem();cs=m.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
        q=np.tile(row,(2,1));q[:,labels.index('hip_rotation_r')]=70
        t=np.array([0.,1.]);new,linear=resample_native(t,labels,q,10)
        with self.assertRaisesRegex(ValueError,'declared independent domain'):
            hip_slerp(biomech.MODEL,labels,q,t,new,linear)


if __name__=='__main__':unittest.main()
