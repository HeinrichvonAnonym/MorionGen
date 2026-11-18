import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.evalkit.k.refinement import refine


class RefinementTests(unittest.TestCase):
    def test_known_fk_pose_recovered_without_reference_or_geometry_change(self):
        m=osim.Model(str(biomech.MODEL));state=m.initSystem();cs=m.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(i).getValue(state)*(180/np.pi if cs.get(i).getMotionType()==1 else 1) for i in range(cs.getSize())])
        for side in ('l','r'):
            row[labels.index('hip_flexion_'+side)]=20
            row[labels.index('knee_angle_'+side)]=45
            row[labels.index('elbow_flex_'+side)]=30
        target=np.tile(row,(2,1));p,names,target=model_fk(biomech.MODEL,labels,target,return_coordinates=True)
        start=target.copy();start[:,labels.index('arm_flex_l')]+=8
        solved,receipt=refine(biomech.MODEL,labels,start,p,names,{'max_nfev':20})
        q,_,_=model_fk(biomech.MODEL,labels,solved,return_coordinates=True)
        before,_=model_fk(biomech.MODEL,labels,start)
        self.assertLess(np.linalg.norm(q-p),np.linalg.norm(before-p)*.2)
        self.assertTrue(all(r['after_loss']<=r['before_loss']+1e-9 for r in receipt['frames']))

    def test_pelvis_line_diagnostic_uses_existing_input_hip_positions(self):
        m=osim.Model(str(biomech.MODEL));state=m.initSystem();cs=m.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(i).getValue(state)*(180/np.pi if cs.get(i).getMotionType()==1 else 1) for i in range(cs.getSize())])
        for s in ('l','r'):
            row[labels.index('knee_angle_'+s)]=35;row[labels.index('elbow_flex_'+s)]=30
        p,names,q=model_fk(biomech.MODEL,labels,np.tile(row,(2,1)),return_coordinates=True)
        start=q.copy();start[:,labels.index('pelvis_rotation')]+=8
        solved,_=refine(biomech.MODEL,labels,start,p,names,{'max_nfev':20,'include_pelvis_line':True})
        final,_=model_fk(biomech.MODEL,labels,solved,direct_couplers=True)
        initial,_=model_fk(biomech.MODEL,labels,start,direct_couplers=True)
        from motiogen.degradation.metrics import angle
        line=lambda x:x[:,names.index('right_hip')]-x[:,names.index('left_hip')]
        self.assertLess(float(angle(line(final),line(p)).max()),float(angle(line(initial),line(p)).max())*.2)


if __name__=='__main__':unittest.main()

class JointTargetTests(unittest.TestCase):
    def test_known_deep_flexion_explicit_angle_objective(self):
        from motiogen.degradation.metrics import angle
        m=osim.Model(str(biomech.MODEL));state=m.initSystem();cs=m.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(i).getValue(state)*(180/np.pi if cs.get(i).getMotionType()==1 else 1) for i in range(cs.getSize())])
        for side in ('l','r'):
            row[labels.index('hip_flexion_'+side)]=70
            row[labels.index('knee_angle_'+side)]=110
            row[labels.index('elbow_flex_'+side)]=90
        goal=np.tile(row,(2,1));p,n,goal=model_fk(biomech.MODEL,labels,goal,direct_couplers=True,return_coordinates=True)
        start=goal.copy()
        for name in ('knee_angle_r','elbow_flex_l'):start[:,labels.index(name)]-=8
        q,receipt=refine(biomech.MODEL,labels,start,p,n,{'max_nfev':30,'include_pelvis_line':True,
                 'joint_angle_weight':1.,'target_violation_weight':3.,'target_margin_fraction':.8})
        f,_=model_fk(biomech.MODEL,labels,q,direct_couplers=True)
        for side,a,b,c in [('right','hip','knee','ankle'),('left','shoulder','elbow','wrist')]:
            i,j,k=[n.index(side+'_'+x) for x in (a,b,c)]
            delta=angle(f[:,i]-f[:,j],f[:,k]-f[:,j])-angle(p[:,i]-p[:,j],p[:,k]-p[:,j])
            self.assertLess(abs(delta).max(),.2)
        self.assertTrue(all(x['after_loss']<=x['before_loss'] for x in receipt['frames']))

class AbsoluteJacobianTests(unittest.TestCase):
    def test_known_derivatives_at_zero_and_bounds(self):
        from motiogen.evalkit.k.refinement import bounded_jacobian
        fun=lambda x:np.array([np.sin(x[0])+2*x[1],x[0]*x[1]])
        j=bounded_jacobian(fun,np.array([1e-9,2.]),np.array([0.,-3]),np.array([4.,2.]),1e-5)
        np.testing.assert_allclose(j,[[1,2],[2,1e-9]],atol=1e-9)

class RestartTests(unittest.TestCase):
    def test_restart_uses_observed_knee_geometry_and_retains_default(self):
        m=osim.Model(str(biomech.MODEL));state=m.initSystem();cs=m.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
        for side in ('l','r'):
            row[labels.index('knee_angle_'+side)]=100;row[labels.index('elbow_flex_'+side)]=45
        p,n,q=model_fk(biomech.MODEL,labels,np.tile(row,(2,1)),direct_couplers=True,return_coordinates=True)
        settings={'max_nfev':1,'include_pelvis_line':True,'extra_start':'neutral_knee_if_target_failed'}
        _,perfect=refine(biomech.MODEL,labels,q,p,n,settings)
        self.assertFalse(any(f['extra_start_triggered'] for f in perfect['frames']))
        start=q.copy();start[:,labels.index('knee_angle_l')]=10
        _,receipt=refine(biomech.MODEL,labels,start,p,n,settings)
        self.assertTrue(receipt['frames'][0]['extra_start_triggered'])
        self.assertIn('neutral_knee_start',[a['start'] for a in receipt['frames'][0]['attempts']])
        self.assertLess(abs(receipt['neutral_restart_knee_mapping']['left']['seed_deg']-100),1.)
        self.assertTrue(all(f['after_loss']<=f['before_loss']+1e-9 for f in receipt['frames']))

    def test_physical_direction_continuity_preserves_exact_fk_and_requires_clock(self):
        m=osim.Model(str(biomech.MODEL));state=m.initSystem();cs=m.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
        q=np.tile(row,(3,1));q[:,labels.index('hip_flexion_l')]=[20,22,24]
        q[:,labels.index('knee_angle_l')]=[60,62,64]
        p,n,q=model_fk(biomech.MODEL,labels,q,direct_couplers=True,return_coordinates=True)
        settings={'max_nfev':2,'direction_continuity_weight':.25,'continuity_weight':0,'include_pelvis_line':True}
        with self.assertRaisesRegex(ValueError,'clock'):refine(biomech.MODEL,labels,q,p,n,settings)
        result,receipt=refine(biomech.MODEL,labels,q,p,n,settings,times=np.arange(3)/30)
        actual,_=model_fk(biomech.MODEL,labels,result,direct_couplers=True)
        np.testing.assert_allclose(actual,p,atol=1e-8)
        self.assertTrue(all(x['selection_loss']<1e-10 for x in receipt['frames']))

    def test_reverse_traversal_preserves_fk_order_clock_and_missing_frame(self):
        model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
        q=np.tile(row,(3,1));q[:,labels.index('hip_flexion_l')]=[15,30,45]
        q[:,labels.index('knee_angle_l')]=[30,60,90]
        points,names,q=model_fk(biomech.MODEL,labels,q,direct_couplers=True,return_coordinates=True)
        original=q.copy();times=np.array([7.,7.03,7.1])
        settings={'max_nfev':2,'direction_continuity_weight':.25,'continuity_weight':0,
                  'include_pelvis_line':True,'traversal':'reverse'}
        fitted,receipt=refine(biomech.MODEL,labels,q,points,names,settings,times=times)
        actual,_=model_fk(biomech.MODEL,labels,fitted,direct_couplers=True)
        np.testing.assert_allclose(actual,points,atol=1e-8)
        np.testing.assert_array_equal(q,original)
        self.assertEqual(receipt['traversal']['output_order'],'original input order')
        missing=points.copy();missing[0]=np.nan
        _,masked=refine(biomech.MODEL,labels,q,missing,names,settings,times=times)
        self.assertEqual([r['status'] for r in masked['frames']],['input_missing','computed','computed'])
        with self.assertRaisesRegex(ValueError,'clock'):
            refine(biomech.MODEL,labels,q,points,names,settings,times=times[::-1])
