"""Equivalent interior Euler representatives for audited SO(3) model joints.

This changes coordinate charts only, preserving every sampled physical pose.
It supplies an optional seed for feature-space trajectory optimization; no
Euler smoothness objective or anatomical interpretation is introduced.
"""
import warnings
import numpy as np
from scipy.spatial.transform import Rotation
from .resampling.model import audit_model,_lift


def interior(angles,order,signs,lower,upper):
    angles=np.asarray(angles,float);signs=np.asarray(signs,float)
    lower,upper=np.asarray(lower,float),np.asarray(upper,float)
    if angles.ndim!=2 or angles.shape[1]!=3 or np.any(lower>=upper):raise ValueError('Three-dimensional nondegenerate chart required')
    rot=Rotation.from_euler(order,angles*signs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning);canonical=rot.as_euler(order)
    alternative=canonical.copy();alternative[:,0]+=np.pi;alternative[:,1]=np.pi-alternative[:,1];alternative[:,2]+=np.pi
    candidates=[angles.copy()];valid=[np.all((angles>=lower-1e-7)&(angles<=upper+1e-7),axis=1)]
    center=np.broadcast_to((lower+upper)/2,angles.shape)
    for raw in (canonical,alternative):
        q,ok=_lift(raw/signs,center,lower,upper)
        # At gimbal poles the existing representative remains a safe fallback.
        ok &= (rot.inv()*Rotation.from_euler(order,q*signs)).magnitude()<1e-8
        candidates.append(q);valid.append(ok)
    def margin(q):return np.minimum(q-lower,upper-q).min(axis=1) # absolute radians, same units across axes
    margins=np.stack([np.where(ok,margin(q),-np.inf) for q,ok in zip(candidates,valid)])
    # Original representative wins ties; avoid unnecessary changes.
    margins[0]+=1e-10
    chosen=np.argmax(margins,axis=0)
    result=np.stack(candidates)[chosen,np.arange(len(angles))]
    error=(rot.inv()*Rotation.from_euler(order,result*signs)).magnitude()
    if np.any(error>1e-8):raise ValueError('Equivalent chart changed physical rotation')
    return result,dict(changed_frames=np.flatnonzero(np.max(abs(result-angles),axis=1)>1e-7).tolist(),
                       max_rotation_error_rad=float(error.max()),minimum_margin_before_rad=float(margin(angles).min()),
                       minimum_margin_after_rad=float(margin(result).min()))


def rechart(model_path,labels,coordinates):
    from .k4_fitting import model_fk
    q=np.array(coordinates,float,copy=True);audit=audit_model(model_path);records={}
    for name,joint in audit['joints'].items():
        if joint['classification']!='three_axis_so3':continue
        names=joint['rotation_coordinates'];cols=[list(labels).index(n) for n in names]
        lo,hi=np.array([joint['coordinates'][n]['range_rad'] for n in names]).T
        updated,receipt=interior(np.deg2rad(q[:,cols]),joint['order'],joint['signs'],lo,hi)
        q[:,cols]=np.rad2deg(updated);records[name]=receipt
    before,_=model_fk(model_path,labels,coordinates,direct_couplers=True)
    after,_=model_fk(model_path,labels,q,direct_couplers=True)
    difference=float(np.max(abs(before-after)))
    if difference>1e-8:raise ValueError('Equivalent joint chart changed body FK')
    return q,dict(policy='max_margin_v1',joints=records,max_body_FK_error_m=difference,
                  scope='Same native physical poses and original coordinate domains; only audited three-axis independent SO3 charts. D-only optimizer seed, no R.')
