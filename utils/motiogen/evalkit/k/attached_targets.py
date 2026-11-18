"""Fixed head-chain positions as targets of the existing body coordinates.

For the attached chain, every distal point has the form o + R b + c(D).
The constant b and the input-dependent world offsets are compiled from the
existing chain FK. This neither adds a joint nor independently translates a
head point. Final outputs still use the full chain FK.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class AttachedTargets:
    names: tuple
    times: np.ndarray
    body_offset: np.ndarray
    world_offsets: np.ndarray
    observed: np.ndarray
    receipt: dict

    def positions(self, frame, origin, rotation):
        return np.asarray(origin)+np.asarray(rotation)@self.body_offset+self.world_offsets[frame]


def compile_targets(motion, model_path, labels, coordinates, times, definition):
    """Compile the already-fixed chain at native D times; no reference input."""
    from .head_neck import attach,body_frames
    times=np.asarray(times,float)
    if times.shape!=motion.times.shape or not np.allclose(times,motion.times,atol=1e-9,rtol=0):
        raise ValueError('Attached body targets require the complete native input clock')
    if definition is None or not definition.get('geometry_fixed_for_clip'):
        raise ValueError('An existing fixed head-chain definition is required')
    origin,rotation,_=body_frames(model_path,labels,coordinates)
    _,chain=attach(motion,model_path,labels,coordinates,times,fixed_definition=definition)
    # The first child follows the fitted body. Subsequent link directions
    # follow their existing D-derived joint targets, not a new observation.
    first=chain['base_origin']+np.einsum('tij,j->ti',chain['base_rotation'],chain['offsets'][0])
    local=np.einsum('tij,tj->ti',rotation.swapaxes(-1,-2),first-origin)
    if not np.isfinite(local).all() or np.max(abs(local-local[0]))>1e-8:
        raise ValueError('Head-chain attachment is not a fixed body-local offset')
    names=('neck','head');indices=[list(chain['names']).index(n) for n in names]
    world=chain['positions'][:,indices]-first[:,None,:]
    observed=np.stack([motion.positions[:,list(motion.names).index(n)] for n in names],1)
    prior=motion.legacy_prior_names()
    for i,name in enumerate(names):
        if name in prior:observed[:,i]=np.nan
    supported=np.isfinite(observed).all(-1)
    observed[~supported]=np.nan
    if not supported.any():raise ValueError('No observed attached point supports this objective')
    if np.any(supported & ~np.isfinite(world).all(-1)):
        raise ValueError('Existing chain FK lacks a supported input point; coverage cannot be reduced')
    reconstructed=origin[:,None,:]+np.einsum('tij,j->ti',rotation,local[0])[:,None,:]+world
    delta=reconstructed-chain['positions'][:,indices]
    finite=np.isfinite(delta)
    error=float(np.max(abs(delta[finite]))) if finite.any() else None
    if error is None or error>1e-8:raise ValueError('Compiled chain differs from existing chain FK')
    # Internal finite placeholders carry a NaN target and hence zero objective
    # support. Final full-chain FK continues to preserve the original missingness.
    world=np.where(supported[...,None],world,0.)
    receipt=dict(version='fixed_attached_node_targets_v1',names=list(names),
        definition_sha256=hashlib.sha256(json.dumps(definition,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        body_offset_m=local[0].tolist(),supported_frames={n:int(supported[:,i].sum()) for i,n in enumerate(names)},
        seed_fk_max_abs_difference_m=error,
        scope='Existing fixed head-chain FK targets of body coordinates. D-only; unchanged joints/geometry. No new rotation observability or force qualification.')
    arrays=[np.array(x,copy=True) for x in (times,local[0],world,observed)]
    for value in arrays:value.setflags(write=False)
    return AttachedTargets(names,*arrays,receipt)
