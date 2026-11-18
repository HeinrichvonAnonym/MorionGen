from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.k.foot_domain import polar_range


def test_known_cone_has_correct_global_polar_interval():
    beta=np.deg2rad(40)
    d=polar_range(np.zeros(3),np.array([1.,0,0]),np.array([0,np.cos(beta),-np.sin(beta)]),np.array([0.,0,1]))
    np.testing.assert_allclose(np.rad2deg(d['polar_interval_rad']),[50,130],atol=1e-8)


def test_translated_circle_extrema_agree_with_independent_dense_curve():
    c=np.array([.03,.01,.02]);u=np.array([.13,0,0]);v=np.array([0,.1,np.sqrt(.13**2-.1**2)]);axis=np.array([0.,0,1])
    d=polar_range(c,u,v,axis);q=np.linspace(-np.pi,np.pi,200001)
    w=c+np.cos(q)[:,None]*u+np.sin(q)[:,None]*v
    theta=np.arccos(np.clip((w@axis)/np.linalg.norm(w,axis=-1),-1,1))
    np.testing.assert_allclose(d['polar_interval_rad'],[theta.min(),theta.max()],atol=1e-8)
