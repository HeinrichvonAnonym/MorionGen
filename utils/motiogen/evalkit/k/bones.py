"""Separate input-only fixed-length projection from reference-only scoring normalization."""
import numpy as np
from .geometry import unit

# Common endpoint tree. Virtual centres have explicit geometry, not anatomical labels.
NAMES=['pelvis_mid','shoulder_mid','left_hip','left_knee','left_ankle','left_foot',
       'right_hip','right_knee','right_ankle','right_foot','left_shoulder','left_elbow','left_wrist',
       'right_shoulder','right_elbow','right_wrist']
PARENTS=[-1,0,0,2,3,4,0,6,7,8,1,10,11,1,13,14]


def common(points,names):
    p={n:points[:,i] for i,n in enumerate(names)}
    for s in ('left','right'):
        if s+'_wrist_body' in p:p[s+'_wrist']=p[s+'_wrist_body']
    missing=np.full((len(points),3),np.nan)
    p['pelvis_mid']=.5*(p.get('left_hip',missing)+p.get('right_hip',missing))
    p['shoulder_mid']=.5*(p.get('left_shoulder',missing)+p.get('right_shoulder',missing))
    return np.stack([p.get(n,missing) for n in NAMES],1)


def lengths(points,parents=PARENTS):
    result=np.full(len(parents),np.nan)
    for j,p in enumerate(parents):
        if p<0:continue
        d=np.linalg.norm(points[:,j]-points[:,p],axis=-1)
        good=np.isfinite(d)&(d>1e-6)
        if good.any():result[j]=np.median(d[good])
    return result


def normalize(points,target_lengths,parents=PARENTS):
    """Keep each root and edge direction; missing ancestors propagate, never impute."""
    out=np.full_like(points,np.nan,dtype=float)
    for j,p in enumerate(parents):
        if p<0:out[:,j]=points[:,j]
        elif np.isfinite(target_lengths[j]) and target_lengths[j]>1e-6:
            out[:,j]=out[:,p]+unit(points[:,j]-points[:,p])*target_lengths[j]
    return out


def project_motion(motion):
    """Input-only projection using its native topology. Rotations remain original observations."""
    from .export import target_template
    from .motion import Motion
    names,_,parents=target_template(motion.method)
    idx={n:i for i,n in enumerate(motion.names)}
    # CHOIS carries two extra hand points separately from the body wrists.
    source_names=[n+'_body' if n.endswith('_wrist') and n+'_body' in idx else n for n in names]
    active=[j for j,n in enumerate(source_names) if n in idx]
    p=np.stack([motion.positions[:,idx[source_names[j]]] for j in active],1)
    remap={j:i for i,j in enumerate(active)}
    parent=[remap.get(parents[j],-1) for j in active]
    target=lengths(p,parent)
    projected=normalize(p,target,parent)
    if not np.isfinite(projected).all():raise ValueError('projection_unavailable_missing_or_degenerate_native_bones')
    out=motion.positions.copy()
    for k,j in enumerate(active):out[:,idx[source_names[j]]]=projected[:,k]
    for side in ('left','right'):
        if side+'_wrist_body' in idx and side+'_wrist' in idx:
            delta=out[:,idx[side+'_wrist_body']]-motion.positions[:,idx[side+'_wrist_body']]
            out[:,idx[side+'_wrist']]+=delta
    receipt={'strategy':'per_input_per_edge_median','source':'input_only','target_lengths_m':dict(zip([source_names[j] for j in active],[None if not np.isfinite(x) else float(x) for x in target])),
             'rotations':'original observations retained; projection only changes fitting positions',
             'max_displacement_m':float(np.max(np.linalg.norm(out-motion.positions,axis=-1)))}
    return Motion(out,motion.names.copy(),motion.times.copy(),motion.method,motion.rotations,motion.rotation_names.copy(),{**motion.provenance,'bone_projection':receipt}),receipt
