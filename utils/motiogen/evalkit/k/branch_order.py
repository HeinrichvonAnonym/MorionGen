"""Offline continuation order from observable input geometry, without references."""
import numpy as np
from .geometry import unit


def anchored_order(target,min_plane_angle_deg=10.):
    """A knee plane is best conditioned away from both 0 and 180 degrees.

    Choose the largest normalized cross product on the supplied fitting clock,
    then traverse adjacent frames backward and forward from the same anchor.
    """
    if len(target)==0:raise ValueError('empty leg trajectory')
    u=unit(target[:,1]-target[:,0]);v=unit(target[:,2]-target[:,1])
    strength=np.linalg.norm(np.cross(u,v),axis=-1);valid=np.isfinite(strength)
    anchor=int(np.argmax(np.where(valid,strength,-np.inf))) if valid.any() else 0
    qualified=bool(valid[anchor] and strength[anchor]>=np.sin(np.deg2rad(min_plane_angle_deg)))
    if not qualified:anchor=0
    order=np.r_[np.arange(anchor,-1,-1),np.arange(anchor+1,len(target))].astype(int)
    return order,anchor,{'anchor_index':anchor,'plane_strength':float(strength[anchor]) if valid[anchor] else None,'qualified':qualified,'origin':'input-only normalized knee-plane cross product','latency':'offline complete-window continuation'}
