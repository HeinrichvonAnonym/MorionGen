import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.fixed_geometry import build
from motiogen.evalkit.k.native_frames import read


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.run_data=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
        cls.input_config=ROOT/'config/evalkit_k4_20261117/input.json'
        cls.scratch=ROOT/'develop_logs/test_runs/20261117_k4'

    def test_fixed_geometry_uses_declared_rig_not_motion_phase(self):
        m=load(self.run_data/'inputs/074/kimodo/candidate','kimodo',config=self.input_config)
        other=copy.deepcopy(m);other.positions[:]=0
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            a=build(m,Path(temp)/'a.osim',75,True);b=build(other,Path(temp)/'b.osim',75,True)
        self.assertEqual(a['after'],b['after'])
        for key in ('shoulder_width','rest_hip_to_shoulder_midpoint'):
            self.assertAlmostEqual(a['after'][key],a['targets']['lengths_m'][key],places=5)
        self.assertFalse(a['force_eligible'])
        self.assertTrue(a['joint_domain']['subtalar_angle_l']['locked'])

    def test_native_head_transport_is_not_anatomical_or_model_fit(self):
        m=load(self.run_data/'inputs/000/closd/candidate','closd',config=self.input_config)
        native=read(m)
        self.assertTrue(native['head_relative_valid'].all())
        self.assertIn('no anatomical',native['scope'])
        p=load(self.run_data/'inputs/000/motionlcm/candidate','motionlcm',config=self.input_config)
        self.assertEqual(read(p)['status'],'unavailable')

    def test_effective_rigid_shoulder_width_is_fixed_and_input_only(self):
        from motiogen.evalkit.k.input_rig import geometry_targets
        m=load(self.run_data/'inputs/017/chois/candidate','chois',config=self.input_config)
        old=geometry_targets(m);new=geometry_targets(m,'observed_rigid_median')
        indices=[m.names.index(n) for n in ('left_shoulder','right_shoulder')]
        expected=np.median(np.linalg.norm(m.positions[:,indices[0]]-m.positions[:,indices[1]],axis=-1))
        self.assertAlmostEqual(new['lengths_m']['shoulder_width'],expected)
        self.assertGreater(abs(old['lengths_m']['shoulder_width']-expected),.04)
        self.assertEqual(old['lengths_m']['rest_hip_to_shoulder_midpoint'],new['lengths_m']['rest_hip_to_shoulder_midpoint'])
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            path=Path(temp)/'model.osim'
            result=build(m,path,75,True,'owned_joint_frame',shoulder_width_source='observed_rigid_median')
            self.assertAlmostEqual(result['after']['shoulder_width'],expected,places=8)
            restored=osim.Model(str(path));state=restored.initSystem()
            from motiogen.evalkit.k.fixed_geometry import measure
            self.assertAlmostEqual(measure(restored,state)['shoulder_width'],expected,places=8)
        # Changing the root or reference metadata cannot change clip-fixed width.
        moved=copy.deepcopy(m);moved.positions+=np.array([1.,2.,3.]);moved.provenance['reference']='UNREADABLE'
        self.assertAlmostEqual(geometry_targets(moved,'observed_rigid_median')['lengths_m']['shoulder_width'],expected)

    def test_owned_joint_markers_match_fk_after_serialization(self):
        import biomech
        from motiogen.evalkit.k.k4_fitting import model_fk
        from motiogen.evalkit.k.initialization import weighted_marker_loss
        m=load(self.run_data/'inputs/022/kimodo/candidate','kimodo',config=self.input_config)
        with tempfile.TemporaryDirectory(dir=self.scratch) as temp:
            path=Path(temp)/'owned.osim';receipt=build(m,path,75,True,'owned_joint_frame')
            self.assertLess(max(receipt['marker_joint_frame_offset_m'].values()),1e-9)
            model=osim.Model(str(path));state=model.initSystem();cs=model.getCoordinateSet()
            labels=[cs.get(i).getName() for i in range(cs.getSize())]
            row=np.array([cs.get(n).getDefaultValue()*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
            q=np.tile(row,(2,1))
            for s in ('l','r'):q[1,labels.index('knee_angle_'+s)]=110
            points,names,q=model_fk(path,labels,q,direct_couplers=True,return_coordinates=True)
            loss=weighted_marker_loss(path,labels,q,points,names,{n:1. for n in biomech.MARKERS})
            self.assertLess(loss['mean_weighted_squared_marker_error_m2'],1e-18)
            legacy=Path(temp)/'legacy.osim';old=build(m,legacy,75,True,'legacy')
            self.assertEqual(old['after'],receipt['after'])
            old_points,_=model_fk(legacy,labels,q,direct_couplers=True)
            np.testing.assert_allclose(old_points,points,atol=1e-12)
            self.assertGreater(old['marker_joint_frame_offset_m']['knee_r'],.001)


if __name__=='__main__':unittest.main()

class DeclaredCentresTests(unittest.TestCase):
    def test_declared_feet_keep_ankles_axes_and_markers_after_serialization(self):
        import biomech
        from motiogen.evalkit.k.rig_geometry import transport
        run=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
        motion=load(run/'inputs/002/kimodo/candidate','kimodo',config=ROOT/'config/evalkit_k4_20261117/input.json')
        rig=motion.provenance['input_rig'];rest=np.asarray(rig['neutral_joints']);idx={n:i for i,n in enumerate(rig['names'])}
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261117_k4') as temp:
            a=Path(temp)/'asset.osim';b=Path(temp)/'declared.osim'
            old=build(motion,a,75,True,'owned_joint_frame')
            new=build(motion,b,75,True,'owned_joint_frame',foot_geometry='declared_rig')
            self.assertEqual(old['joint_domain'],new['joint_domain'])
            self.assertLess(max(new['marker_joint_frame_offset_m'].values()),1e-9)
            model_a=osim.Model(str(a));state_a=model_a.initSystem()
            model_b=osim.Model(str(b));state_b=model_b.initSystem()
            self.assertEqual(model_a.getConstraintSet().getSize(),model_b.getConstraintSet().getSize())
            for side,s in [('left','l'),('right','r')]:
                ankle=biomech._child_origin(model_b,state_b,'ankle_'+s)
                np.testing.assert_allclose(ankle,biomech._child_origin(model_a,state_a,'ankle_'+s),atol=1e-12)
                vector=biomech._child_origin(model_b,state_b,'mtp_'+s)-ankle
                expected=transport(rig)@(rest[idx[side+'_foot']]-rest[idx[side+'_ankle']])
                np.testing.assert_allclose(vector,expected,atol=1e-10)
                for prefix in ('ankle_','subtalar_','mtp_'):
                    ja=osim.CustomJoint.safeDownCast(model_a.getJointSet().get(prefix+s))
                    jb=osim.CustomJoint.safeDownCast(model_b.getJointSet().get(prefix+s))
                    if ja and jb:
                        for k in range(6):
                            xa=ja.getSpatialTransform().getTransformAxis(k);xb=jb.getSpatialTransform().getTransformAxis(k)
                            self.assertEqual([xa.getAxis().get(i) for i in range(3)],[xb.getAxis().get(i) for i in range(3)])
            # Declared geometry is independent of sampled motion and reference metadata.
            altered=copy.deepcopy(motion);altered.positions[:]=0;altered.provenance['reference']='UNREADABLE'
            again=build(altered,Path(temp)/'again.osim',75,True,'owned_joint_frame',foot_geometry='declared_rig')
            self.assertEqual(new['foot_geometry'],again['foot_geometry'])
            positional=load(run/'inputs/008/motionlcm/candidate','motionlcm',config=ROOT/'config/evalkit_k4_20261117/input.json')
            fallback=build(positional,Path(temp)/'fallback.osim',75,True,'owned_joint_frame',foot_geometry='declared_rig')
            self.assertEqual(fallback['foot_geometry']['status'],'unavailable')

    def test_initialization_search_subset_preserves_final_model(self):
        import biomech,hashlib
        from motiogen.evalkit.k.initialization import search_model
        original=hashlib.sha256(Path(biomech.MODEL).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261117_k4') as temp:
            path,receipt=search_model(biomech.MODEL,temp,{'hip_rotation_l':[-20,20]})
            model=osim.Model(str(path));c=model.getCoordinateSet().get('hip_rotation_l')
            self.assertAlmostEqual(np.rad2deg(c.getRangeMin()),-20)
            self.assertEqual(set(receipt['changes']),{'hip_rotation_l'})
            with self.assertRaisesRegex(ValueError,'subset'):
                search_model(biomech.MODEL,temp,{'hip_rotation_l':[-90,90]})
        self.assertEqual(hashlib.sha256(Path(biomech.MODEL).read_bytes()).hexdigest(),original)

    def test_declared_centres_survive_save_and_keep_model_constraints(self):
        import biomech
        from motiogen.evalkit.k.rig_geometry import transport
        run=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
        m=load(run/'inputs/022/kimodo/candidate','kimodo',config=ROOT/'config/evalkit_k4_20261117/input.json')
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261117_k4') as temp:
            a=build(m,Path(temp)/'a.osim',75,True,'owned_joint_frame')
            b=build(m,Path(temp)/'b.osim',75,True,'owned_joint_frame','declared_rig')
            self.assertEqual(a['joint_domain'],b['joint_domain'])
            model=osim.Model(str(Path(temp)/'b.osim'));state=model.initSystem()
            for joint,record in b['joint_centres']['changes'].items():
                np.testing.assert_allclose(biomech._child_origin(model,state,joint),record['target_neutral_world_m'],atol=1e-10)
            self.assertLess(max(b['marker_joint_frame_offset_m'].values()),1e-9)
            # The geometry depends on the declared rig, not sampled poses.
            other=copy.deepcopy(m);other.positions[:]=0
            c=build(other,Path(temp)/'c.osim',75,True,'owned_joint_frame','declared_rig')
            self.assertEqual(b['joint_centres'],c['joint_centres'])
            effective=build(m,Path(temp)/'effective.osim',75,True,'owned_joint_frame','declared_rig','observed_rigid_median')
            self.assertAlmostEqual(effective['after']['shoulder_width'],effective['targets']['lengths_m']['shoulder_width'],places=9)
            self.assertLess(max(effective['marker_joint_frame_offset_m'].values()),1e-9)

    def test_expanded_domain_is_explicit_and_leaves_other_constraints(self):
        from motiogen.evalkit.k.diagnostic_domain import apply
        import biomech
        with tempfile.TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261117_k4') as temp:
            path=Path(temp)/'model.osim';model=osim.Model(str(biomech.MODEL));model.printToXML(str(path))
            policy={'base':'release_subtalar_original_range','scope':'counterfactual_kinematic_only','coordinate_ranges_deg':{'arm_flex_r':[-90,160]}}
            result=apply(path,policy);restored=osim.Model(str(path));cs=restored.getCoordinateSet()
            self.assertAlmostEqual(np.rad2deg(cs.get('arm_flex_r').getRangeMax()),160)
            self.assertEqual(cs.get('arm_flex_l').getRangeMax(),model.getCoordinateSet().get('arm_flex_l').getRangeMax())
            self.assertEqual(set(result['changes']),{'subtalar_angle_l','subtalar_angle_r','arm_flex_r'})
            self.assertIn('no anatomical',result['scope'])
            with self.assertRaises(ValueError):apply(path,{**policy,'scope':'promoted'})
