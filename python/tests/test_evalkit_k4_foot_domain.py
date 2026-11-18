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


def test_endpoint_search_retains_feasible_geometry_and_finds_fixed_alternative():
    from scipy.spatial.transform import Rotation
    from motiogen.evalkit.k.position_foot_geometry import search
    axis=np.array([0.,0.,1.]);subaxis=np.array([0.,np.sin(.6),np.cos(.6)])
    rotations=[Rotation.from_rotvec(subaxis*q).as_matrix() for q in (0,np.pi,np.pi/2)]
    neutral=np.array([.15,0,0]);vectors=[r@neutral for r in rotations]
    c=(vectors[0]+vectors[1])/2;u=(vectors[0]-vectors[1])/2;v=vectors[2]-c
    curve=dict(center=c,cosine=u,sine=v,axis=axis)
    delta,receipt=search(curve,rotations,np.deg2rad([70,100]),.15,.02)
    np.testing.assert_array_equal(delta,np.zeros(3));assert receipt['status']=='existing_geometry_covers_input'
    delta,receipt=search(curve,rotations,np.deg2rad([120,150]),.15,.02)
    assert delta is not None and receipt['status']=='estimated_effective_endpoint'
    # Independent dense FK, not the optimizer's polar-polynomial implementation.
    q=np.linspace(-np.pi,np.pi,30001);w=Rotation.from_rotvec(q[:,None]*subaxis).apply(neutral+delta)
    polar=np.arccos(np.clip((w@axis)/np.linalg.norm(w,axis=-1),-1,1))
    assert polar.min()<=np.deg2rad(120)+1e-7 and polar.max()>=np.deg2rad(150)-1e-7
    assert np.linalg.norm(w,axis=-1).min()>=.13-1e-7 and np.linalg.norm(w,axis=-1).max()<=.17+1e-7
