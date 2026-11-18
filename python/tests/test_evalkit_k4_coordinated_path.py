from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.resampling.coordinated import anchored_targets,refine_path


def test_anchor_support_and_gap_propagation():
    t=np.array([0.,1.,2.]);out=np.arange(0,2.01,.5)
    p=np.arange(18,dtype=float).reshape(3,2,3);support=np.ones((3,2),bool);support[1,0]=False
    target,fixed=anchored_targets(t,p,out,support)
    np.testing.assert_array_equal(fixed,[True,False,True,False,True])
    assert np.isnan(target[1:4,0]).all()
    np.testing.assert_array_equal(target[[0,4]],p[[0,2]])


def test_coordinated_fk_restores_inserted_pose_and_keeps_native_coordinates():
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.k4_fitting import model_fk
    from motiogen.evalkit.k.motion import Motion
    osim.Logger.setLevelString('error')
    model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
    row[labels.index('knee_angle_l')]=60
    native=np.tile(row,(2,1));points,names,native=model_fk(biomech.MODEL,labels,native,direct_couplers=True,return_coordinates=True)
    motion=Motion(points,names,np.array([10.,10.1]),'known_fk',provenance={'reference':'UNREADABLE'})
    initial=np.tile(native[0],(3,1));initial[1,labels.index('knee_angle_l')]=10
    cfg={'version':'fixed_native_anchors_v1','refinement':{'max_nfev':30,'continuity_weight':0,
         'active_coordinates':['knee_angle_l'],'joint_angle_weight':1.}}
    q,receipt=refine_path(biomech.MODEL,labels,native,np.array([0.,.1]),initial,np.array([0.,.05,.1]),motion,cfg)
    np.testing.assert_array_equal(q[[0,2]],native)
    actual,_=model_fk(biomech.MODEL,labels,q,direct_couplers=True)
    np.testing.assert_allclose(actual[1],points[0],atol=1e-6)
    assert receipt['inserted_frames']==1 and receipt['native_knots']==2
    np.testing.assert_equal(initial[1,labels.index('knee_angle_l')],10)


def test_prior_wrists_do_not_disable_observed_leg_path():
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.k4_fitting import model_fk
    from motiogen.evalkit.k.motion import Motion
    model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
    row[labels.index('knee_angle_l')]=60
    native=np.tile(row,(2,1));points,names,native=model_fk(biomech.MODEL,labels,native,direct_couplers=True,return_coordinates=True)
    motion=Motion(points,names,np.array([10.,10.1]),'known_fk',provenance={'degradation':{'prior_joints':['left_wrist','right_wrist']}})
    initial=np.tile(native[0],(3,1));initial[1,labels.index('knee_angle_l')]=10
    cfg={'version':'fixed_native_anchors_v1','refinement':{'max_nfev':30,'continuity_weight':0,
         'active_coordinates':['knee_angle_l'],'joint_angle_weight':1.}}
    old,old_receipt=refine_path(biomech.MODEL,labels,native,np.array([0.,.1]),initial,np.array([0.,.05,.1]),motion,cfg)
    np.testing.assert_array_equal(old,initial)
    assert old_receipt['frames'][1]['status']=='input_missing'
    cfg['refinement']['observation_policy']='partial_body_v1'
    q,receipt=refine_path(biomech.MODEL,labels,native,np.array([0.,.1]),initial,np.array([0.,.05,.1]),motion,cfg)
    np.testing.assert_array_equal(q[[0,2]],native)
    actual,_=model_fk(biomech.MODEL,labels,q,direct_couplers=True)
    np.testing.assert_allclose(actual[1],points[0],atol=1e-6)
    assert receipt['frames'][1]['supported_position_targets']==len(names)-2
    assert receipt['frames'][1]['supported_angle_targets']==2


def test_hand_tip_prior_keeps_complete_observed_body_path():
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.k4_fitting import model_fk
    from motiogen.evalkit.k.motion import Motion
    model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
    row[labels.index('knee_angle_l')]=60
    native=np.tile(row,(2,1));points,names,native=model_fk(biomech.MODEL,labels,native,direct_couplers=True,return_coordinates=True)
    raw_names=[n+'_body' if n.endswith('_wrist') else n for n in names]+['left_wrist','right_wrist']
    raw_points=np.concatenate([points,np.ones((2,2,3))*10],axis=1)
    motion=Motion(raw_points,raw_names,np.array([10.,10.1]),'known_fk',provenance={'degradation':{'prior_joints':['left_wrist','right_wrist']}})
    initial=np.tile(native[0],(3,1));initial[1,labels.index('knee_angle_l')]=10
    cfg={'version':'fixed_native_anchors_v1','refinement':{'max_nfev':30,'continuity_weight':0,
         'active_coordinates':['knee_angle_l'],'joint_angle_weight':1.}}
    q,receipt=refine_path(biomech.MODEL,labels,native,np.array([0.,.1]),initial,np.array([0.,.05,.1]),motion,cfg)
    np.testing.assert_array_equal(q[[0,2]],native)
    actual,_=model_fk(biomech.MODEL,labels,q,direct_couplers=True)
    np.testing.assert_allclose(actual[1],points[0],atol=1e-6)
    assert receipt['inserted_computed_frames']==receipt['inserted_supported_frames']==1
