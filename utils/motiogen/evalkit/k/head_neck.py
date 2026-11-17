"""Attached, fixed-geometry head/neck FK. Observed targets are D only.

This is a kinematic chain, not a muscle model or anatomical calibration.
Source-native relative head rotation and model-body torso rotation remain named
separately. Position-only twist gauges never acquire rotation observability.
"""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .geometry import valid_rotation
from .native_frames import read as native_frames
from .timebase import supported


def interpolate_rotations(times,rotations,target):
    rotations=np.asarray(rotations);valid=valid_rotation(rotations)
    result=np.full((len(target),3,3),np.nan)
    inside=supported(target,times[0],times[-1]);clipped=np.clip(target,times[0],times[-1])
    hi=np.minimum(np.searchsorted(times,clipped,side='left'),len(times)-1);lo=np.maximum(hi-1,0)
    exact=np.isclose(clipped,times[hi],rtol=0,atol=1e-9);lo[exact]=hi[exact]
    mask=inside&valid[lo]&valid[hi]
    good=np.flatnonzero(valid)
    if len(good)>=2 and mask.any():
        result[mask]=Slerp(times[good],Rotation.from_matrix(rotations[good]))(clipped[mask]).as_matrix()
    elif len(good)==1:
        use=mask&np.isclose(clipped,times[good[0]],rtol=0,atol=1e-9);result[use]=rotations[good[0]]
    return result


def forward(origin,base_rotation,offsets,local_rotations):
    """Serial chain; local rotations belong to each child joint, never endpoints."""
    count=len(origin);n=len(offsets)
    p=np.full((count,n+1,3),np.nan);r=np.full((count,n+1,3,3),np.nan)
    p[:,0]=origin;r[:,0]=base_rotation
    for j,offset in enumerate(offsets):
        p[:,j+1]=p[:,j]+np.einsum('tij,j->ti',r[:,j],offset)
        r[:,j+1]=r[:,j]@local_rotations[:,j]
    return p,r


def swing(rotation,offset,direction):
    """Minimum swing; twist is a gauge when no independent rotation is observed."""
    a=rotation@offset;na=np.linalg.norm(a);nb=np.linalg.norm(direction)
    if not np.isfinite(a).all() or not np.isfinite(direction).all() or min(na,nb)<1e-9:
        return np.full((3,3),np.nan)
    a=a/na;b=direction/nb;cross=np.cross(a,b);norm=np.linalg.norm(cross);dot=np.clip(a@b,-1,1)
    if norm<1e-10:
        if dot>0:return rotation.copy()
        axis=np.cross(a,np.eye(3)[np.argmin(abs(a))]);axis/=np.linalg.norm(axis)
    else:axis=cross/norm
    return Rotation.from_rotvec(axis*np.arctan2(norm,dot)).as_matrix()@rotation


def body_frames(model_path,labels,coordinates):
    import opensim as osim
    import biomech
    from .constraints import couplers,synchronize
    model=osim.Model(str(model_path));state=model.initSystem();bindings=couplers(model,state)
    body=model.getBodySet().get('torso');origins=[];rotations=[];chests=[]
    marker=model.getMarkerSet().get('chest_c') if model.getMarkerSet().contains('chest_c') else None
    for row in coordinates:
        biomech._set(model,state,labels,row);synchronize(state,bindings);model.realizePosition(state)
        transform=body.getTransformInGround(state)
        origins.append([transform.p().get(j) for j in range(3)])
        rotations.append([[transform.R().get(j,k) for k in range(3)] for j in range(3)])
        if marker is not None:
            value=marker.getLocationInGround(state);chests.append([value.get(j) for j in range(3)])
    return np.asarray(origins),np.asarray(rotations),np.asarray(chests) if chests else None


def attach(motion,model_path,labels,coordinates,times,fixed_definition=None):
    body_origin,body_rotation,registered_chest=body_frames(model_path,labels,coordinates)
    native=native_frames(motion);rig=motion.provenance.get('input_rig');idx={n:i for i,n in enumerate(motion.names)}
    observed=np.stack([np.stack([np.interp(times,motion.times,motion.positions[:,j,k]) for k in range(3)],1)
                       for j in range(len(motion.names))],1)
    observed[~supported(times,motion.times[0],motion.times[-1])]=np.nan
    if native['status']=='declared_native_frames':
        names=rig['names'];chest=names.index('Spine3' if 'Spine3' in names else 'spine3');node=names.index('head');chain=[]
        while node!=chest:
            if node<0:raise ValueError('Declared head is not a descendant of the chest')
            chain.append(node);node=rig['parents'][node]
        chain=[chest,*reversed(chain)]
        if names.index('neck') not in chain:raise ValueError('Declared head chain must retain neck node')
        rest=np.asarray(rig['neutral_joints']);bases=np.asarray(rig['segment_frame_basis'])
        left=rest[names.index('left_hip')]-rest[names.index('right_hip')];left/=np.linalg.norm(left)
        up=np.array([0.,1,0]);up-=left*np.dot(left,up);up/=np.linalg.norm(up)
        source_basis=np.stack([left,up,np.cross(left,up)],1)
        transport=np.array([[0.,0,1],[0,1,0],[-1,0,0]])@source_basis.T
        if registered_chest is None:raise ValueError('Declared-rig chain requires a registered model chest anchor')
        origin=registered_chest;base_source=body_rotation@transport;base_rotation=base_source@bases[chest]
        offsets=[rest[j]-rest[i] for i,j in zip(chain[:-1],chain[1:])]
        source_rotations=[interpolate_rotations(motion.times,native['frames'][:,j],times) for j in chain]
        relative_target=source_rotations[0].swapaxes(-1,-2)@source_rotations[-1]
        desired=[base_rotation]
        # Internal nodes fit their observed world frame. The leaf fits the
        # observed head-relative-chest target; its world discrepancy is reported.
        desired.extend(source_rotations[k]@bases[j] for k,j in enumerate(chain[1:-1],1))
        desired.append(base_source@relative_target@bases[chain[-1]])
        local=np.stack([a.swapaxes(-1,-2)@b for a,b in zip(desired[:-1],desired[1:])],1)
        chain_names=[names[j] for j in chain]
        p,r=forward(origin,base_rotation,offsets,local)
        source_frames=np.stack([r[:,k]@bases[j].T for k,j in enumerate(chain)],1)
        relative=source_frames[:,0].swapaxes(-1,-2)@source_frames[:,-1]
        rotation_valid=valid_rotation(relative_target)&valid_rotation(relative)
        definition={'version':'attached-head-neck-1','names':chain_names,'offsets_m':np.asarray(offsets).tolist(),
            'source':'declared input rig; original hierarchy retained','attachment':'registered chest_c on fitted torso',
            'source_frame_bases':bases[chain].tolist(),'body_to_source_transport':transport.tolist(),
            'targets':'internal observed world frames; head relative fitted chest; no R access',
            'scope':'fixed kinematic chain; full head world rotation discrepancy remains a separate diagnostic'}
    else:
        chest_name='Spine3' if 'Spine3' in idx else 'spine3'
        chest_obs=observed[:,idx[chest_name]];neck_obs=observed[:,idx['neck']];head_obs=observed[:,idx['head']]
        inverse=body_rotation.swapaxes(-1,-2)
        attachment=np.nanmedian(np.einsum('tij,tj->ti',inverse,chest_obs-body_origin),axis=0)
        if fixed_definition is not None:attachment=np.asarray(fixed_definition['attachment_local_m'])
        origin=body_origin+np.einsum('tij,j->ti',body_rotation,attachment);base_rotation=body_rotation
        neck_local=np.nanmedian(np.einsum('tij,tj->ti',inverse,neck_obs-chest_obs),axis=0)
        neck_length=np.nanmedian(np.linalg.norm(neck_obs-chest_obs,axis=-1))
        head_length=np.nanmedian(np.linalg.norm(head_obs-neck_obs,axis=-1))
        if not np.isfinite(np.r_[attachment,neck_local,neck_length,head_length]).all() or min(np.linalg.norm(neck_local),neck_length,head_length)<1e-9:
            raise ValueError('Fixed head/neck geometry unidentifiable from input')
        neck_local=neck_local/np.linalg.norm(neck_local)*neck_length
        offsets=(np.asarray(fixed_definition['offsets_m']) if fixed_definition else [neck_local,np.array([0.,head_length,0.])]);local=np.full((len(times),2,3,3),np.nan)
        for i in range(len(times)):
            neck_rotation=swing(base_rotation[i],offsets[1],head_obs[i]-neck_obs[i])
            local[i,0]=base_rotation[i].T@neck_rotation;local[i,1]=np.eye(3)
        p,r=forward(origin,base_rotation,offsets,local);chain_names=['chest','neck','head']
        source_frames=r;relative=np.full((len(times),3,3),np.nan);rotation_valid=np.zeros(len(times),bool)
        definition={'version':'attached-head-neck-1','names':chain_names,'offsets_m':np.asarray(offsets).tolist(),
            'source':'input segment-length medians; estimated constant attachment from D and fitted torso',
            'attachment_local_m':attachment.tolist(),'attachment':'fixed on fitted torso',
            'targets':'observed head-neck direction; axial twist gauge unobserved',
            'scope':'position geometry only; no neutral anatomical posture or full head rotation inferred'}
    definition.update(geometry_fixed_for_clip=True,force_eligible=False,head_muscle_model=False)
    return definition,{'times':np.asarray(times),'names':np.asarray(chain_names),'positions':p,'world_rotations':source_frames,
        'base_origin':origin,'base_rotation':base_rotation,'local_rotations':local,'offsets':np.asarray(offsets),
        'head_relative_thorax':relative,'relative_rotation_valid':rotation_valid,
        'position_finite':np.isfinite(p).all(-1)}


def combine(body_positions,body_names,chain):
    """Append FK-generated neck/head; preserve body nodes and the full chain archive."""
    names=list(body_names);positions=[body_positions]
    for name in ('neck','head'):
        if name in names:raise ValueError('Body already owns a head/neck node')
        positions.append(chain['positions'][:,[list(chain['names']).index(name)]])
        names.append(name)
    return np.concatenate(positions,axis=1),names
