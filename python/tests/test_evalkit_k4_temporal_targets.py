import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.temporal_targets import direction_error_rate,rate_excess_residual


def test_physical_rate_and_per_direction_excess():
    rate=direction_error_rate(np.array([[.01,0,0],[0,.02,0]]),np.zeros((2,3)),.1)
    np.testing.assert_allclose(rate,[[.1,0,0],[0,.2,0]])
    np.testing.assert_allclose(rate_excess_residual(rate,.1,.8,4),[.8,4.8])
    for dt in (0,-1,float('nan')):
        with pytest.raises(ValueError):direction_error_rate(np.zeros((2,3)),np.zeros((2,3)),dt)


def test_observed_fast_rotation_is_not_smoothed():
    # Exact following of a 90-degree observed turn has zero added velocity.
    observed=np.array([[1.,0,0],[0,1.,0]])
    fitted=observed.copy();errors=fitted-observed
    rate=direction_error_rate(errors[1:],errors[:1],1/30)
    np.testing.assert_array_equal(rate_excess_residual(rate,np.deg2rad(10),.8,4),[0.])


def test_rate_only_solver_requires_clock_and_preserves_known_fk():
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.refinement import refine
    from motiogen.evalkit.k.k4_fitting import model_fk
    osim.Logger.setLevelString('error')
    model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
    q=np.tile(row,(3,1));q[:,labels.index('hip_flexion_l')]=[10,25,40];q[:,labels.index('knee_angle_l')]=[30,50,70]
    points,names,q=model_fk(biomech.MODEL,labels,q,direct_couplers=True,return_coordinates=True)
    settings=dict(max_nfev=2,continuity_weight=0.,direction_continuity_weight=0.,direction_rate_violation_weight=4.)
    with pytest.raises(ValueError,match='clock'):refine(biomech.MODEL,labels,q,points,names,settings)
    output,_=refine(biomech.MODEL,labels,q,points,names,settings,times=np.arange(3)/30)
    actual,_=model_fk(biomech.MODEL,labels,output,direct_couplers=True)
    np.testing.assert_allclose(actual,points,atol=1e-9)
