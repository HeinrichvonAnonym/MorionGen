"""Independent FK and temporal checks for K3.9 parent-frame transport."""
import tempfile,unittest
from pathlib import Path
import numpy as np
from motiogen.evalkit.k.config import ROOT,digest
from motiogen.evalkit.k.temporal import angular_increment,temporal
from motiogen.evalkit.k.limb_transport import transport

class TemporalTests(unittest.TestCase):
    def test_heading_change_and_antipodal_unknown_axis(self):
        a=np.deg2rad([0.,30.,60.]);v=np.stack([np.cos(a),np.zeros(3),np.sin(a)],1)
        step,rotation=angular_increment(v)
        np.testing.assert_allclose(step,30.)
        np.testing.assert_allclose(rotation[:,1],-30.)
        step,rotation=angular_increment(np.array([[1.,0,0],[-1.,0,0]]))
        self.assertAlmostEqual(step[0],180.)
        self.assertTrue(np.isnan(rotation).all())

class TransportTests(unittest.TestCase):
    def test_independent_fk_parent_change_preserves_distal_orientation(self):
        import opensim as osim
        import biomech
        osim.Logger.setLevelString('error')
        model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
        labels=[cs.get(i).getName() for i in range(cs.getSize())]
        q=np.array([[cs.get(n).getDefaultValue() if n.startswith('pelvis_t') and n in ['pelvis_tx','pelvis_ty','pelvis_tz'] else np.degrees(cs.get(n).getDefaultValue()) for n in labels]]*4)
        for side in ('l','r'):
            q[:,labels.index('hip_flexion_'+side)]=25.;q[:,labels.index('knee_angle_'+side)]=55.
        changed=q.copy();changed[:,labels.index('pelvis_rotation')]=[10.,15.,20.,25.]
        before=digest(biomech.MODEL)
        output,receipt=transport(biomech.MODEL,biomech.MODEL,labels,q,changed)
        self.assertEqual(before,digest(biomech.MODEL))
        np.testing.assert_allclose(output[:,labels.index('pelvis_rotation')],changed[:,labels.index('pelvis_rotation')])
        def rotation(data,body):
            values=[]
            for row in data:
                biomech._set(model,state,labels,row);r=model.getBodySet().get(body).getTransformInGround(state).R()
                values.append([[r.get(i,j) for j in range(3)] for i in range(3)])
            return np.array(values)
        for body in ['femur_l','femur_r','tibia_l','tibia_r','calcn_l','calcn_r']:
            np.testing.assert_allclose(rotation(output,body),rotation(q,body),atol=1e-6)
        self.assertLess(receipt['limbs']['legs_r']['orientation_error_max_deg'],1e-4)
        # Pre-root provenance matters although translation cannot alter rotations.
        from motiogen.evalkit.k.fitting3 import positions
        from motiogen.evalkit.k.root_alignment import align_fitted_root
        translated=changed.copy();translated[:,labels.index('pelvis_tx')]+=.3
        shifted,_=transport(biomech.MODEL,biomech.MODEL,labels,q,translated)
        source,names=positions(biomech.MODEL,np.arange(4)/100,labels,q)
        pre,_=positions(biomech.MODEL,np.arange(4)/100,labels,output)
        shifted_pre,_=positions(biomech.MODEL,np.arange(4)/100,labels,shifted)
        np.testing.assert_allclose(shifted_pre-pre,np.broadcast_to([.3,0.,0.],pre.shape),atol=1e-6)
        _,a,_,_,_=align_fitted_root(biomech.MODEL,np.arange(4)/100,labels,output,source,names)
        _,b,_,_,_=align_fitted_root(biomech.MODEL,np.arange(4)/100,labels,shifted,source,names)
        np.testing.assert_allclose(a,b,atol=1e-6)


class DispatchTests(unittest.TestCase):
    def test_explicit_domain_config_uses_new_path(self):
        from unittest.mock import patch
        from motiogen.evalkit.k.domain_fitting import fit_kinematic_domain
        cfg={'output_kind':'kinematic_only','strategy':'trustworthy_fixed_geometry'}
        with patch('motiogen.evalkit.k.selection.fit_kinematic',return_value={'force_eligible':False}) as call:
            result=fit_kinematic_domain('source','work',61.,.02,cfg,'kimodo')
            call.assert_called_once_with('source','work',61.,.02,cfg,'kimodo')
            self.assertFalse(result['force_eligible'])

if __name__=='__main__':unittest.main()
