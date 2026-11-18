"""Foot-direction envelope for two audited pin joints on a fixed model.

For a complete ankle revolution, the polar angle about its axis is invariant.
The subtalar circle supplies the attainable polar interval. Polynomial extrema
avoid interpreting a coarse two-angle grid as a global reachability proof.
"""
import numpy as np
from numpy.polynomial import Polynomial as P


def polar_range(center,cosine,sine,axis):
    c,u,v,a=[np.asarray(x,float) for x in (center,cosine,sine,axis)]
    if any(x.shape!=(3,) or not np.isfinite(x).all() for x in (c,u,v,a)):raise ValueError('Finite3D vectors required')
    if abs(np.linalg.norm(a)-1)>1e-8:raise ValueError('Unit ankle axis required')
    if abs(u@v)>1e-8 or abs(u@u-v@v)>1e-8:raise ValueError('Audited circular foot curve required')
    g=c@c+u@u;h=2*c@u;k=2*c@v
    minimum_length_squared=g-np.hypot(h,k)
    if minimum_length_squared<=1e-12:raise ValueError('Foot direction degenerates on this curve')
    denominator=P([1,0,1]);co=P([1,0,-1]);si=P([0,2])
    xyz=[c[j]*denominator+u[j]*co+v[j]*si for j in range(3)]
    numerator=sum((a[j]*xyz[j] for j in range(3)),P([0.]))
    norm=sum((x*x for x in xyz),P([0.]))
    derivative=2*numerator.deriv()*norm-numerator*norm.deriv()
    scale=float(abs(derivative.coef).max()) or 1.;roots=derivative.trim(tol=1e-12*scale).roots()
    angles=np.r_[0.,np.pi,2*np.arctan([r.real for r in roots if abs(r.imag)<1e-7])]
    w=c+np.cos(angles)[:,None]*u+np.sin(angles)[:,None]*v
    values=(w@a)/np.linalg.norm(w,axis=-1);lo,hi=int(values.argmin()),int(values.argmax())
    interval=np.arccos(np.clip([values[hi],values[lo]],-1,1))
    return dict(polar_interval_rad=interval.tolist(),cosine_interval=[float(values[lo]),float(values[hi])],
                subtalar_extrema_rad=[float(angles[hi]),float(angles[lo])],minimum_foot_length_m=float(np.sqrt(minimum_length_squared)))


def model_curve(model_path,labels,row,side):
    import opensim as osim
    import biomech
    from scipy.spatial.transform import Rotation
    from .constraints import couplers,synchronize
    from .resampling.model import _matrix
    s='l' if side=='left' else 'r';model=osim.Model(str(model_path));state=model.initSystem();bindings=couplers(model,state)
    ankle=model.getJointSet().get('ankle_'+s);sub=model.getJointSet().get('subtalar_'+s)
    if ankle.getConcreteClassName()!='PinJoint' or sub.getConcreteClassName()!='PinJoint':raise ValueError('Two pin-joint foot geometry required')
    coords=model.getCoordinateSet();ca=coords.get('ankle_angle_'+s);cb=coords.get('subtalar_angle_'+s)
    def vector(qa,qb):
        biomech._set(model,state,labels,row);ca.setValue(state,float(qa),False);cb.setValue(state,float(qb),False)
        synchronize(state,bindings);model.realizePosition(state)
        return biomech._child_origin(model,state,'mtp_'+s)-biomech._child_origin(model,state,'ankle_'+s)
    zero=vector(0,0);axis=_matrix(ankle.getParentFrame(),state)[:,2]
    a,b=vector(0,np.pi),vector(0,np.pi/2);center=(zero+a)/2;cosine=(zero-a)/2;sine=b-center
    error=max(float(np.linalg.norm(vector(0,q)-(center+cosine*np.cos(q)+sine*np.sin(q)))) for q in np.linspace(-np.pi,np.pi,37))
    rotation_error=float(np.linalg.norm(vector(np.pi/2,.37)-Rotation.from_rotvec(axis*np.pi/2).apply(vector(0,.37))))
    if max(error,rotation_error)>1e-8:raise ValueError('Foot curve/ankle-axis formula disagrees with model FK')
    domain=polar_range(center,cosine,sine,axis)
    return dict(center=center.tolist(),cosine=cosine.tolist(),sine=sine.tolist(),axis=axis.tolist(),
                curve_FK_max_error_m=error,ankle_rotation_FK_error_m=rotation_error,**domain)
