"""Per-landmark direction continuity on the complete physical input clock."""
import numpy as np
from motiogen.evalkit.k.bones import NAMES,PARENTS
from motiogen.evalkit.k.geometry import unit
from motiogen.evalkit.k.scoring import angles,stats


def vectors(points):
    out={NAMES[j]:unit(points[:,j]-points[:,p]) for j,p in enumerate(PARENTS) if p>=0}
    out['shoulder_line']=unit(points[:,13]-points[:,10])
    out['pelvis_line']=unit(points[:,6]-points[:,2])
    return out


def angular_increment(v):
    cross=np.cross(v[:-1],v[1:]);norm=np.linalg.norm(cross,axis=-1)
    theta=np.arctan2(norm,np.clip((v[:-1]*v[1:]).sum(-1),-1,1))
    rotation=cross*np.divide(theta,norm,out=np.zeros_like(theta),where=norm>1e-10)[:,None]
    # Antipodal steps have no identifiable axis. Keep the 180-degree jump,
    # but propagate unknown angular-velocity direction instead of inventing it.
    rotation[(norm<=1e-10)&(theta>1.)]=np.nan
    return np.rad2deg(theta),np.rad2deg(rotation)


def temporal(source,fitted,times):
    dt=np.diff(times)
    if len(dt)==0 or not np.all(dt>0):raise ValueError('strictly increasing clock with at least two frames required')
    result={}
    for name,v in vectors(source).items():
        step,rotation=angular_increment(vectors(fitted)[name]);_,wanted=angular_increment(v)
        result['direction:'+name]={'step':stats(step),'velocity_error':stats(np.linalg.norm((rotation-wanted)/dt[:,None],axis=1))}
    observed=angles(source)
    for name,a in angles(fitted).items():
        result['angle:'+name]={'step':stats(np.diff(a)),'velocity_error':stats(np.diff(a-observed[name])/dt)}
    # Maxima use the same finite arrays as the distribution summaries.
    for name,v in vectors(fitted).items():
        step,_=angular_increment(v)
        result['direction:'+name]['step_max_deg']=float(np.nanmax(step)) if np.isfinite(step).any() else None
    for name,a in angles(fitted).items():
        step=abs(np.diff(a));result['angle:'+name]['step_max_deg']=float(np.nanmax(step)) if np.isfinite(step).any() else None
    return result
