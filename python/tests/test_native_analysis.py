import unittest
import numpy as np
from motiogen.native_analysis import clustered_interval,foot_sensitivity,physics_contact_sensitivity,boundary_kinematics,task_sensitivity,common_history_consistency

class NativeAnalysisTests(unittest.TestCase):
    def test_common_history_excludes_extra_old_frames_without_changing_denominator(self):
        # Two joints, horizon 2. Long history adds old errors of 100, but its
        # common last two frames exactly match short history at every call.
        short=np.ones(3*2*2)
        long=np.concatenate([np.ones(4),np.r_[np.full(4,100.),np.ones(4)],np.r_[np.full(8,100.),np.ones(4)]])
        a=common_history_consistency(short,2,2,2,4,2)
        b=common_history_consistency(long,6,2,2,4,2)
        self.assertEqual(a,b)
        self.assertEqual(a['mean_m'],1.)
        self.assertEqual(a['samples'],12)
    def test_common_history_keeps_startup_and_rejects_missing_receipts(self):
        r=common_history_consistency(np.ones((2+4)*3),4,2,3,3,4)
        self.assertEqual(r['history_frames_per_call'],[2,4])
        for errors in [np.ones(17),np.r_[np.ones(17),np.nan],np.r_[np.ones(17),-1.]]:
            with self.assertRaises(ValueError):common_history_consistency(errors,4,2,3,3,4)
    def test_all_failures_do_not_claim_zero_population_uncertainty(self):
        result=clustered_interval([0.,0.,0.,0.],['a','a','b','b'])
        self.assertEqual(result['ci95'],[0.,0.])
        self.assertTrue(result['interval_degenerate'])
        self.assertIn('does not establish zero',result['interval_caveat'])
        self.assertFalse(clustered_interval([0.,1.],['a','b'])['interval_degenerate'])
    def test_omission_removes_shared_parent_and_preserves_task_weights(self):
        result=task_sensitivity([-2.,0.,3.,10.],['short','short','long','other'],['same','same','same','other'])
        values={r['omitted_parent']:r['equal_task_mean_delta'] for r in result['leave_one_parent_out']}
        self.assertEqual(values,{'same':10.,'other':1.})
        self.assertEqual(result['task_means_negative'],1)
        self.assertEqual(result['task_means_positive'],2)
        self.assertEqual(task_sensitivity([1.],['one'])['leave_one_parent_out'],[])
    def test_group_weight_not_frame_or_pair_weight(self):
        result=clustered_interval([0.,0.,9.],['a','a','b'])
        self.assertEqual(result['mean_equal_group_weight'],4.5)
        self.assertEqual(result['independent_groups'],2)
        self.assertEqual(result['ci95'],[0.,9.])

    def test_one_task_has_no_population_interval(self):
        self.assertIsNone(clustered_interval([1.,3.],['same','same'])['ci95'])
        with self.assertRaises(ValueError): clustered_interval([float('nan')],['a'])

    def test_shared_warm_rng_is_one_resampling_parent(self):
        result=clustered_interval([0.,2.,10.],['short','long','other'],resample_groups=['same_rng','same_rng','other_rng'])
        self.assertEqual(result['independent_groups'],2)
        self.assertEqual(result['task_groups'],3)
        self.assertEqual(result['mean_equal_group_weight'],4.)
        self.assertEqual(result['ci95'],[1.,10.])

    def test_continuous_motion_boundary_step_not_discontinuity(self):
        joints=np.zeros((30,2,3));joints[:,:,0]=np.arange(30)[:,None]/30
        result=boundary_kinematics(joints,30,[14,28])
        self.assertAlmostEqual(result['boundary_position_jump'],1/30)
        self.assertAlmostEqual(result['boundary_velocity_jump'],0.)
        self.assertAlmostEqual(result['boundary_jerk_rms'],0.)

    def test_foot_proxy_retains_empty_contact_denominator(self):
        motion={'joint_names':['left_foot','right_foot'],'joints':np.ones((3,2,3)),'fps':30}
        result=foot_sensitivity(motion,{'foot_height_thresholds_m':[.05],
            'foot_speed_thresholds_m_s':[.1],'floor_penetration_threshold_m':.02,
            'foot_contact_basis':'height proxy, not force'})
        self.assertIsNone(result['rows'][0]['sliding_fraction_given_candidate'])
        self.assertEqual(result['rows'][0]['candidate_contact_samples'],0)

    def test_physical_contact_denominators_and_body_motion(self):
        names=['L_Ankle','L_Toe','R_Ankle','R_Toe','Pelvis']
        joints=np.zeros((3,5,3));forces=np.zeros_like(joints)
        joints[:,0,0]=[0,.02,.04];forces[:,0,2]=10;forces[1,4,0]=30
        result=physics_contact_sensitivity(joints,forces,names,30,{
            'physics_contact_force_thresholds_n':[5.,20.], 'physics_contact_basis':'force, not contact-point velocity'})
        low,high=result['rows']
        self.assertEqual(low['contact_body_intervals'],2)
        self.assertEqual(low['all_body_intervals'],8)
        self.assertEqual(low['foot_body_translation_over_10cms_given_contact'],1.)
        self.assertAlmostEqual(low['nonfoot_contact_frame_fraction'],1/3)
        self.assertIsNone(high['foot_body_translation_over_10cms_given_contact'])
        with self.assertRaises(ValueError):physics_contact_sensitivity(joints,forces[:2],names,30,{})

class ProcessIncidentTests(unittest.TestCase):
    def test_recovered_failure_is_preserved(self):
        from motiogen.native_analysis import process_incidents
        failed={'status':'timeout','run_dir':'old','reason':'expired'}
        passed={'status':'success','run_dir':'new'}
        rows=process_incidents({**passed,'attempts':[failed,passed]})
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['reason'],'expired')

    def test_final_failed_row_does_not_duplicate_attempt(self):
        from motiogen.native_analysis import process_incidents
        failed={'status':'failed','run_dir':'old','reason':'exit 1'}
        self.assertEqual(len(process_incidents({**failed,'attempts':[failed]})),1)
        self.assertEqual(process_incidents({'status':'pending'}),[])

if __name__=='__main__': unittest.main()
