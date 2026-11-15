import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.arms import geometry_quality, policy, configure_model
from motiogen.model_resources import model_root, model_path


class ArmFitTests(unittest.TestCase):
    def test_candidate_failure_keeps_successful_model_and_reason(self):
        from types import SimpleNamespace
        from motiogen.evalkit.k import fitting
        def candidate(run,out,mass,floor,settings):
            if out.name=='expanded_range':raise RuntimeError('diagnostic solver failure')
            (out/'model.osim').write_text(out.name)
            return {'calibration':{'qualified':False}}
        q={'valid':np.ones(2,bool),'position_error_m':np.zeros((2,3)),'elbow_bias_deg':np.zeros(2)}
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(fitting,'load',return_value=SimpleNamespace(method='motionlcm',legacy=lambda:(np.zeros((2,3,3)),['a','b','c'],20))), \
             patch.object(fitting,'_prepare_candidate',side_effect=candidate), \
             patch.object(fitting.biomech,'write_coordinates',return_value=(np.arange(2)/20,[],np.empty((2,0)))), \
             patch('motiogen.evalkit.k.arms.assess',return_value={'l':q,'r':q}):
            out=Path(temp);receipt=fitting.prepare('unused',out,75.,0.)
            self.assertEqual((out/'model.osim').read_text(),'original_range')
            self.assertEqual(receipt['arm_selection']['selected'],'original_range')
            self.assertIn('expanded_range',receipt['arm_selection']['failures'])
            self.assertFalse(receipt['calibration']['qualified'])

    def test_expanded_range_t_pose_warm_start_preserves_delivered_model(self):
        import opensim as osim
        import biomech
        from motiogen.evalkit.k.arms import inverse_kinematics
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);model=osim.Model(str(biomech.MODEL))
            for name,(joint,body,_) in biomech.MARKERS.items():
                location=model.getJointSet().get(joint).getChildFrame().findTransformInBaseFrame().p()
                model.addMarker(osim.Marker(name,model.getBodySet().get(body),location))
            model.finalizeConnections();model.printToXML(str(root/'model.osim'))
            configure_model(root/'model.osim',policy('motionlcm'))
            original=(root/'model.osim').read_bytes()
            model=osim.Model(str(root/'model.osim'));state=model.initSystem()
            for side in 'lr':
                model.getCoordinateSet().get('arm_add_'+side).setValue(state,-np.pi/2,False)
                model.getCoordinateSet().get('elbow_flex_'+side).setValue(state,.05,False)
            model.realizePosition(state)
            target=np.array([biomech._child_origin(model,state,joint) for joint,_,_ in biomech.MARKERS.values()])
            table=osim.TimeSeriesTableVec3();table.setColumnLabels(list(biomech.MARKERS))
            for t in [2.89999999999999,2.94999999999999,2.99999999999999]:
                row=osim.RowVectorVec3(len(target))
                for i,p in enumerate(target):row[i]=osim.Vec3(*p)
                table.appendRow(t,row)
            table.addTableMetaDataString('DataRate','20');table.addTableMetaDataString('Units','m')
            osim.TRCFileAdapter.write(table,str(root/'markers.trc'))
            (root/'ik').mkdir()
            inverse_kinematics(root/'model.osim',root/'markers.trc',root/'ik/ik.mot',2.9,3.,
                               {n:1. for n in biomech.MARKERS},policy('motionlcm'))
            self.assertEqual((root/'model.osim').read_bytes(),original)
            _,labels,rows=biomech.read_storage(root/'ik/ik.mot')
            for row in rows:
                biomech._set(model,state,labels,row)
                fitted=np.array([biomech._child_origin(model,state,joint) for joint,_,_ in biomech.MARKERS.values()])
                self.assertLess(np.max(np.linalg.norm(fitted-target,axis=-1)),.005)

    def test_legacy_upper_body_import_uses_shared_resources(self):
        import subprocess
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-c',
                'import upper_body; from model_resources import model_path; assert upper_body.FBLS == model_path("fbls"); assert upper_body.FBLS.is_file()'],
                cwd=folder,env={**os.environ,'PYTHONPATH':str(ROOT/'utils/motiogen')},capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)

    def test_calibration_outlier_does_not_hide_other_eligible_frames(self):
        from motiogen.evalkit.k.fitting import upright_frames
        names=['left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle','left_shoulder','right_shoulder']
        rest=np.array([[-.1,1,0],[.1,1,0],[-.1,.5,0],[.1,.5,0],[-.1,0,0],[.1,0,0],[-.2,1.5,0],[.2,1.5,0]])
        p=np.broadcast_to(rest,(40,8,3)).copy()
        p[1:,6:,2]=.1  # About 11 degrees: all eligible, only one near the best frame.
        indices,receipt=upright_frames(p,names,20)
        self.assertTrue(receipt['qualified'])
        self.assertGreaterEqual(len(indices),10)
        self.assertEqual(receipt['eligible_frames'],40)
        p[5:,6:,1:]=[1.,.5]  # Only five eligible frames, still insufficient.
        self.assertFalse(upright_frames(p,names,20)[1]['qualified'])

    def test_unknown_arm_widens_ergo_score_bounds(self):
        from motiogen.evalkit.ergo import score_union
        for method in ('RULA','REBA'):
            unknown=score_union({},method)
            measured=score_union({'upper_arm_flexion_deg':30,'elbow_flexion_deg':90,'arm_abducted_or_rotated':0},method)
            self.assertLessEqual(unknown[0],measured[0])
            self.assertGreaterEqual(unknown[1],measured[1])

    def test_angle_and_endpoint_validity_are_independent(self):
        p=np.array([[[0.,0,0],[0,-1,0],[1,-1,0]]])
        q=p+np.array([.2,0,0])
        result=geometry_quality(p,q,10,.1)
        self.assertTrue(result['angle_valid'][0])
        self.assertFalse(result['position_valid'][0])
        self.assertFalse(result['valid'][0])
        self.assertAlmostEqual(result['elbow_bias_deg'][0],0)

    def test_missing_and_collapsed_bones_have_no_angle_support(self):
        p=np.array([[[0.,0,0],[0,-1,0],[1,-1,0]]]*3)
        p[1,0]=np.nan;p[2,1]=p[2,0]
        result=geometry_quality(p,p.copy(),10,.1)
        np.testing.assert_array_equal(result['valid'],[True,False,False])
        self.assertTrue(np.isnan(result['elbow_bias_deg'][1:]).all())

    def test_resource_relocation_and_absolute_models(self):
        config={'data_root':'/default/data','model_root':'models','models':{'fbls':'fbls/model.osim','other':'/explicit/other.osim'}}
        with patch.dict(os.environ,{'MORIONGEN_DATA_ROOT':'/relocated/data'}):
            self.assertEqual(model_root(config),Path('/relocated/data/models'))
            self.assertEqual(model_path('fbls',config),Path('/relocated/data/models/fbls/model.osim'))
            self.assertEqual(model_path('other',config),Path('/explicit/other.osim'))

    def test_model_policy_changes_only_requested_coordinate_ranges(self):
        import opensim as osim
        import biomech
        self.assertFalse(policy('ardy'))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.osim';model=osim.Model(str(biomech.MODEL));model.printToXML(str(path))
            before={c.getName():(c.getRangeMin(),c.getRangeMax()) for c in model.getCoordinateSet()}
            configure_model(path,policy('motionlcm'));after=osim.Model(str(path))
            for c in after.getCoordinateSet():
                if c.getName() in ('arm_flex_l','arm_flex_r'):
                    np.testing.assert_allclose([c.getRangeMin(),c.getRangeMax()],[-np.pi,np.pi])
                else:self.assertEqual((c.getRangeMin(),c.getRangeMax()),before[c.getName()])


if __name__=='__main__':unittest.main()
