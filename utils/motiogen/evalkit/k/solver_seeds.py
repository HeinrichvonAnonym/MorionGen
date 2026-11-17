"""Deterministic observed-geometry seeds; never read evaluation references."""
import numpy as np


def knee_seed(initial, coordinate_names, target, lower, upper):
    """Interior flexion seed breaks the straight-knee stationary branch.

    The joint-centre angle is only an initialization proxy for the model's knee
    coordinate. Model FK and the unchanged input objective decide acceptance.
    """
    indices=[i for i,n in enumerate(coordinate_names) if n.startswith('knee_angle_')]
    if len(indices)!=1: return None
    u=target[1]-target[0];v=target[2]-target[1]
    if min(np.linalg.norm(u),np.linalg.norm(v))<1e-6: return None
    angle=np.arccos(np.clip(u@v/(np.linalg.norm(u)*np.linalg.norm(v)),-1.,1.))
    i=indices[0]; seed=initial.copy()
    seed[i]=np.clip(angle,lower[i]+1e-4,upper[i]-1e-4)
    return seed
