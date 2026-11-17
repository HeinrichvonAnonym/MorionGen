import sys,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.k4_fitting import model_fk,resample_native
from motiogen.evalkit.k.resampling import audit_model,interpolate_rotations
from motiogen.evalkit.k.resampling.model import _matrix
from motiogen.evalkit.k.constraints import couplers,synchronize


class WholeRotationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261118_k4_rotation')
        self.path=Path(self.temp.name)/'model.osim';m=osim.Model(str(biomech.MODEL))
        c=m.updCoordinateSet().get('pelvis_rotation');c.setRangeMin(-20);c.setRangeMax(20)
        m.updCoordinateSet().get('arm_flex_r').setRangeMax(np.deg2rad(160));m.printToXML(str(self.path))
        s=m.initSystem();cs=m.getCoordinateSet();self.labels=[cs.get(i).getName() for i in range(cs.getSize())]
        self.row=np.array([cs.get(n).getValue(s)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in self.labels])
    def tearDown(self):self.temp.cleanup()
    def run_case(self,q):
        t=np.linspace(0,1,len(q));new,linear=resample_native(t,self.labels,q,10)
        output,receipt=interpolate_rotations(self.path,self.labels,q,t,new,linear)
        return output,receipt,new,linear
    def local_rotations(self,q,name):
        model=osim.Model(str(self.path));state=model.initSystem();bindings=couplers(model,state);joint=model.getJointSet().get(name);result=[]
        for row in q:
            biomech._set(model,state,self.labels,row);synchronize(state,bindings);model.realizePosition(state)
            result.append(_matrix(joint.getParentFrame(),state).T@_matrix(joint.getChildFrame(),state))
        return Rotation.from_matrix(result)
    def test_every_joint_has_an_explicit_manifold_class(self):
        a=audit_model(self.path)['joints'];self.assertEqual(len(a),22)
        counts={k:sum(r['classification']==k for r in a.values()) for k in set(r['classification'] for r in a.values())}
        self.assertEqual(counts,{'three_axis_so3':6,'one_axis_geodesic_equivalent':6,'coordinate_coupled_manifold':4,'locked_fixed':6})
    def test_six_joints_match_independent_model_frame_geodesics(self):
        q=np.tile(self.row,(2,1));audit=audit_model(self.path)
        for name,r in audit['joints'].items():
            if r['classification']=='three_axis_so3':
                cols=[self.labels.index(n) for n in r['rotation_coordinates']];q[0,cols]=[5,-10,20];q[1,cols]=[-10,10,-5]
        for name in ('knee_angle_l','knee_angle_r'):q[:,self.labels.index(name)]=[20,100]
        q[:,self.labels.index('pelvis_tx')]=[1,3]
        out,r,new,linear=self.run_case(q)
        changed=set()
        for name,row in r['audit']['joints'].items():
            if row['classification']!='three_axis_so3':continue
            native=self.local_rotations(q,name);actual=self.local_rotations(out,name)
            wanted=Slerp([0,1],native)(new)
            self.assertLess((wanted.inv()*actual).magnitude().max(),1e-8)
            self.assertEqual(r['counts'][name]['fallback_intervals'],0)
            changed.update(row['rotation_coordinates'])
        untouched=[i for i,n in enumerate(self.labels) if n not in changed]
        np.testing.assert_array_equal(out[:,untouched],linear[:,untouched])
    def test_heading_wrap_is_a_short_rotation(self):
        q=np.tile(self.row,(2,1));q[:,self.labels.index('pelvis_rotation')]=[179,-179]
        out,r,new,_=self.run_case(q);actual=self.local_rotations(out,'ground_pelvis')
        self.assertLess(np.rad2deg((actual[:-1].inv()*actual[1:]).magnitude()).max(),.201)
        self.assertEqual(r['counts']['ground_pelvis']['fallback_intervals'],0)
    def test_shoulder_equivalent_charts_do_not_create_false_motion(self):
        q=np.tile(self.row,(2,1));cols=[self.labels.index(n+'_r') for n in ('arm_flex','arm_add','arm_rot')]
        q[0,cols]=[-80,-89,-90];q[1,cols]=[100,-91,90]
        out,r,_,linear=self.run_case(q);native=self.local_rotations(q,'acromial_r');actual=self.local_rotations(out,'acromial_r')
        self.assertLess((native[0].inv()*native[1]).magnitude(),1e-10)
        self.assertLess((native[0].inv()*actual).magnitude().max(),1e-8)
        self.assertGreater((native[0].inv()*self.local_rotations(linear,'acromial_r')).magnitude().max(),.01)
    def test_gimbal_gauge_is_found_inside_bounds(self):
        q=np.tile(self.row,(2,1));cols=[self.labels.index(n) for n in ('lumbar_extension','lumbar_bending','lumbar_rotation')]
        q[:,cols]=[80,90,80]
        out,r,_,_=self.run_case(q)
        self.assertEqual(r['counts']['back']['fallback_intervals'],0)
        self.assertLess((self.local_rotations(q,'back')[0].inv()*self.local_rotations(out,'back')).magnitude().max(),1e-8)
    def test_out_of_domain_geodesic_retains_entire_bounded_interval(self):
        model=osim.Model(str(self.path));c=model.updCoordinateSet().get('lumbar_extension');c.setRangeMin(np.deg2rad(-1));c.setRangeMax(np.deg2rad(1));model.printToXML(str(self.path))
        q=np.tile(self.row,(2,1));cols=[self.labels.index(n) for n in ('lumbar_extension','lumbar_bending','lumbar_rotation')]
        q[0,cols]=[0,30,30];q[1,cols]=[0,-30,-30]
        out,r,_,linear=self.run_case(q)
        self.assertEqual(r['counts']['back']['fallback_intervals'],1)
        np.testing.assert_array_equal(out[:,cols],linear[:,cols])
        self.assertTrue(np.isfinite(out).all())
    def test_missing_coordinate_is_rejected_without_filling(self):
        q=np.tile(self.row,(2,1));q[1,self.labels.index('hip_flexion_l')]=np.nan
        with self.assertRaisesRegex(ValueError,'Finite'):self.run_case(q)


if __name__=='__main__':unittest.main()
