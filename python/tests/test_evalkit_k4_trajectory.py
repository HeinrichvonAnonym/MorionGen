import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.trajectory_objective import TrajectoryObjective


def fixture(target=None):
    times=np.array([0.,.02,.08,.1]);x=np.array([.1,.3,-.1,.2])
    feature=lambda i,q:np.array([q[0],np.sin(q[0]),np.cos(q[0])])
    target=np.array([feature(i,[q]) for i,q in enumerate(x)]) if target is None else target
    return TrajectoryObjective(feature,target,times,[.05,.1,.1],[1,2],.2,.5,[-2.],[2.]),x


def test_exact_fast_motion_has_zero_residual_on_nonuniform_clock():
    obj,x=fixture();np.testing.assert_array_equal(obj.residual(x),np.zeros(18))


def test_sparse_jacobian_matches_independent_whole_vector_difference():
    obj,x=fixture();x=x+np.array([.01,-.01,.03,.02]);columns=[]
    for i in range(len(x)):
        d=np.eye(len(x))[i]*1e-6;columns.append((obj.residual(x+d)-obj.residual(x-d))/(2e-6))
    np.testing.assert_allclose(obj.jacobian(x).toarray(),np.stack(columns,1),rtol=2e-6,atol=2e-7)


def test_missing_target_does_not_create_temporal_bridge():
    obj,x=fixture();target=obj.target.copy();target[1]=np.nan
    obj,_=fixture(target);x[1]=1.7
    np.testing.assert_allclose(obj.residual(x),0.,atol=1e-12)
    np.testing.assert_allclose(obj.jacobian(x).toarray()[:,1],0.,atol=0)


def test_joint_objective_tracks_observed_motion_from_perturbed_seed():
    from scipy.optimize import least_squares
    obj,truth=fixture();initial=truth+np.array([.08,-.1,.12,-.04])
    result=least_squares(obj.residual,initial,jac=obj.jacobian,bounds=(-2,2),gtol=1e-10,ftol=1e-10,xtol=1e-10)
    np.testing.assert_allclose(result.x,truth,atol=1e-7)


def test_fixed_model_known_fk_and_reference_free_interface():
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.k4_fitting import model_fk
    from motiogen.evalkit.k.trajectory import refine
    osim.Logger.setLevelString('error');model=osim.Model(str(biomech.MODEL));s=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    q=np.tile([cs.get(n).getValue(s)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels],(3,1))
    q[:,labels.index('knee_angle_l')]=[20,50,80];q[:,labels.index('arm_flex_r')]=[10,30,60]
    p,names,q=model_fk(biomech.MODEL,labels,q,direct_couplers=True,return_coordinates=True)
    result,candidate,receipt=refine(biomech.MODEL,labels,q,p,names,np.arange(3)/30,{'version':'joint_trajectory_v1','max_nfev':2})
    actual,_=model_fk(biomech.MODEL,labels,result,direct_couplers=True)
    np.testing.assert_allclose(actual,p,atol=1e-8)
    assert receipt['initial_squared_residual']<1e-12
    assert receipt['candidate_squared_residual']<=receipt['initial_squared_residual']+1e-8


def test_post_seed_domain_preserves_known_fk_and_changes_only_search_bounds(tmp_path):
    import shutil
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.k4_fitting import model_fk
    from motiogen.evalkit.k.trajectory import problem
    from motiogen.evalkit.k.diagnostic_domain import apply
    path=tmp_path/'model.osim';shutil.copyfile(biomech.MODEL,path)
    model=osim.Model(str(path));state=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    q=np.tile([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels],(3,1))
    q[:,labels.index('arm_rot_r')]=[80,89,90]
    p,names,q=model_fk(path,labels,q,direct_couplers=True,return_coordinates=True)
    settings={'version':'joint_trajectory_v1'}
    before,x0,_,_,active=problem(path,labels,q,p,names,np.arange(3)/30,settings)
    policy={'base':'preserve_existing','scope':'independent_kinematic','coordinate_ranges_deg':{'arm_rot_r':[-180,180]}}
    receipt=apply(path,policy)
    actual,_=model_fk(path,labels,q,direct_couplers=True)
    np.testing.assert_allclose(actual,p,atol=1e-9,rtol=0)
    after,x1,_,_,active1=problem(path,labels,q,p,names,np.arange(3)/30,settings)
    assert active==active1 and set(receipt['changes'])=={'arm_rot_r'}
    i=active.index('arm_rot_r')
    assert before.upper[i]<after.upper[i]
    assert after.upper[i]==np.pi and after.lower[i]==-np.pi
    np.testing.assert_allclose(x0,x1,atol=1.1e-9,rtol=0)
    np.testing.assert_allclose(before.residual(x0),after.residual(x1),atol=1e-6,rtol=0)


def test_bounded_joint_solver_recovers_fk_features_with_monotone_steps():
    from motiogen.evalkit.k.trajectory_solver import solve
    obj,truth=fixture();initial=truth+np.array([.08,-.1,.12,-.04])
    result=solve(obj,initial,{'max_nfev':20})
    np.testing.assert_allclose(result.x,truth,atol=1e-6)
    assert result.nfev<=20 and result.success
    assert all(row['cost_after']<=row['cost_before'] for row in result.history)


def test_bounded_joint_solver_respects_domain_and_evaluation_limit():
    from motiogen.evalkit.k.trajectory_solver import solve
    obj,truth=fixture();obj.upper[:]=.15
    result=solve(obj,np.zeros_like(truth),{'max_nfev':3})
    assert result.nfev<=3 and (result.x<=.15).all() and (result.x>=obj.lower[0]).all()
    assert np.linalg.norm(result.fun)<=np.linalg.norm(obj.residual(np.zeros_like(truth)))


def test_projected_fallback_rejects_non_descent_linear_solver_step(monkeypatch):
    from scipy.optimize import OptimizeResult
    from motiogen.evalkit.k.trajectory_solver import solve
    obj,truth=fixture();initial=truth+.1
    monkeypatch.setattr('motiogen.evalkit.k.trajectory_solver.lsq_linear',lambda *a,**kw:OptimizeResult(x=np.ones(len(initial)),status=0,nit=50))
    result=solve(obj,initial,{'max_nfev':4,'projected_gradient_fallback':True})
    assert result.history[0]['step_source']=='projected_diagonal_gradient'
    assert np.linalg.norm(result.fun)<np.linalg.norm(obj.residual(initial))


def test_target_group_violation_jacobian_and_missing_support():
    original,truth=fixture();target=original.target.copy();target[1,1:]=np.nan
    obj=TrajectoryObjective(original.feature,target,original.times,original.scales,original.indices,.2,.5,[-2],[2],
        violation_groups=[(0,1),(1,3)],violation_weight=4,violation_margin=.8)
    x=truth+.2;columns=[]
    for i in range(len(x)):
        delta=np.eye(len(x))[i]*1e-6;columns.append((obj.residual(x+delta)-obj.residual(x-delta))/(2e-6))
    np.testing.assert_allclose(obj.jacobian(x).toarray(),np.stack(columns,1),atol=1e-6,rtol=2e-6)
    assert obj.residual(x)[-8:].reshape(4,2)[1,1]==0
