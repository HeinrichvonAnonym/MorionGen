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
