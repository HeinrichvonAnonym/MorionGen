import sys
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.k.trajectory_solver import solve


class LinearProblem:
    n=2
    lower=np.array([-1.,-1.]);upper=np.array([1.,1.])
    def __init__(self):
        self.matrix=sparse.csr_matrix([[1.,.999,0,0],[0,0,1.,.999],[1e-4,0,0,0],[0,0,1e-4,0]])
        self.target=np.array([.7,-.7,0,0])
    def residual(self,x):return self.matrix@x-self.target
    def jacobian(self,x):return self.matrix


def test_explicit_zero_options_keep_control_numerics():
    p=LinearProblem();x=np.zeros(4);settings=dict(max_nfev=10,projected_gradient_fallback=True)
    a=solve(p,x,settings);b=solve(p,x,dict(settings,linear_damping=0.,max_coordinate_step_rad=None))
    np.testing.assert_array_equal(a.x,b.x)
    assert a.history==b.history


def test_damped_step_limits_and_actual_objective_acceptance():
    p=LinearProblem();x=np.zeros(4)
    r=solve(p,x,dict(max_nfev=20,linear_damping=.01,max_coordinate_step_rad=.1,projected_gradient_fallback=True))
    assert np.linalg.norm(r.fun)<.01
    for row in r.history:
        assert row['cost_after']<=row['cost_before']
        if 'accepted_alpha' in row:assert row['max_coordinate_step_rad']<=.1+1e-12
    assert (r.x>=np.tile(p.lower,2)).all() and (r.x<=np.tile(p.upper,2)).all()


def test_invalid_numerical_controls_rejected():
    import pytest
    for options in [dict(linear_damping=-1),dict(linear_damping=float('nan')),dict(max_coordinate_step_rad=0),dict(max_coordinate_step_rad=float('inf'))]:
        with pytest.raises(ValueError):solve(LinearProblem(),np.zeros(4),options)
