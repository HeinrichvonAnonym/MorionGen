"""Offline, input-construction candidate; not a runtime K solver.

Retain registered fixed geometry for a controlled ablation. Changing shape is
still an open conversion problem: this module never calls it inevitable loss.
"""
import numpy as np
from motiogen.evalkit.k.export import target_template, ALIASES
from motiogen.evalkit.k.motion import Motion
from .positions import SOURCE


def point_rotation(rotation, local, direction):
    """Minimum swing; retain existing axial component, handle opposite vectors."""
    from scipy.spatial.transform import Rotation
    x=np.einsum('tij,j->ti',rotation,local/np.linalg.norm(local))
    y=direction/np.maximum(np.linalg.norm(direction,axis=-1,keepdims=True),1e-12)
    cross=np.cross(x,y);dot=np.clip(np.sum(x*y,axis=-1),-1,1)
    norm=np.linalg.norm(cross,axis=-1);axis=cross/np.maximum(norm[:,None],1e-12)
    opposite=dot < -1+1e-10
    if opposite.any():
        fallback=np.cross(x[opposite],np.eye(3)[np.argmin(np.abs(x[opposite]),axis=1)])
        axis[opposite]=fallback/np.linalg.norm(fallback,axis=-1,keepdims=True)
    return Rotation.from_rotvec(axis*np.arctan2(norm,dot)[:,None]).as_matrix()@rotation


def aligned(reference, legacy, source_geometry=False):
    names, rest, parents = target_template(legacy.method)
    lookup = {n:i for i,n in enumerate(reference.names)}
    mapping = dict(SOURCE)
    if 'Spine3' not in lookup: mapping['spine3'] = 'Spine2'
    if 'Neck' not in lookup: mapping['neck'] = 'Neck1'
    aliases = {'Spine':'spine1','Spine3':'spine3','LeftShoulder':'left_collar','RightShoulder':'right_collar'}
    desired = legacy.positions.copy()
    observed = [];sources=[]
    for j,n in enumerate(names):
        source = mapping.get(aliases.get(n,n), n)
        # cskel27 has FOUR spine nodes, unlike HumanML22. Do not alias two
        # distinct nodes to Spine. Use its native asset names directly.
        if legacy.method == 'ardy' and n in ('Spine','spine1','spine2','Spine3'):
            source = {'Spine':'Spine','spine1':'Spine1','spine2':'Spine2','Spine3':'Spine3'}[n]
        if source in lookup:
            desired[:,j] = reference.positions[:,lookup[source]]
            observed.append(True)
        else:
            observed.append(False)
        sources.append(source)
    effective_links=[]
    if source_geometry:
        source_rest=np.asarray(reference.provenance['rest_positions'])
        rest=rest.copy()
        for j,observed_j in enumerate(observed):
            if observed_j:rest[j]=source_rest[lookup[sources[j]]]
        # Unobserved intermediate joints interpolate inside the observed chain;
        # leaves preserve an explicitly synthetic template offset.
        for j,observed_j in enumerate(observed):
            if observed_j:continue
            par=parents[j]
            children=[c for c,p in enumerate(parents) if p==j and observed[c]]
            if par>=0 and observed[par] and children:
                child=children[0]
                old=target_template(legacy.method)[1]
                fraction=np.linalg.norm(old[j]-old[par])/(np.linalg.norm(old[j]-old[par])+np.linalg.norm(old[child]-old[j]))
                rest[j]=rest[par]+fraction*(rest[child]-rest[par])
                desired[:,j]=desired[:,par]+fraction*(desired[:,child]-desired[:,par])
                observed[j]=True
                sources[j]=None
            elif par>=0:
                old=target_template(legacy.method)[1]
                rest[j]=rest[par]+old[j]-old[par]
        # A collapsed two-link shoulder has a varying endpoint distance. Its
        # best constant length for worst-case radial error is (min+max)/2.
        # This is an EFFECTIVE link, never an estimate of anatomical neutral.
        source_parents=reference.provenance.get('parents')
        if source_parents is not None:
            for name in ('left_shoulder','right_shoulder'):
                j=names.index(name);par=parents[j]
                if sources[j] not in lookup or sources[par] not in lookup:continue
                if source_parents[lookup[sources[j]]]==lookup[sources[par]]:continue
                lengths=np.linalg.norm(desired[:,j]-desired[:,par],axis=-1)
                lo,hi=float(lengths.min()),float(lengths.max());length=(lo+hi)/2
                offset=rest[j]-rest[par]
                delta=offset*(length/np.linalg.norm(offset)-1)
                descendants={j}
                for k,pj in enumerate(parents):
                    if pj in descendants:descendants.add(k)
                for k in descendants:rest[k]+=delta
                effective_links.append({'joint':name,'parent':names[par],
                    'source_distance_range_m':[lo,hi],'fixed_effective_length_m':length,
                    'minimax_radial_lower_bound_m':(hi-lo)/2,
                    'scope':'fixed parent anchor and single constant-length link; internal source articulation discarded'})
    r = legacy.rotations.copy()
    bases=np.broadcast_to(np.eye(3),(len(names),3,3)).copy()
    source_rest=np.asarray(reference.provenance['rest_positions'])
    from scipy.spatial.transform import Rotation
    for j in range(len(names)):
        if not observed[j] or sources[j] is None:continue
        children=[c for c,p in enumerate(parents) if p==j and observed[c] and sources[c] is not None]
        if children:
            a=np.stack([rest[c]-rest[j] for c in children]);a/=np.linalg.norm(a,axis=-1,keepdims=True)
            b=np.stack([source_rest[lookup[sources[c]]]-source_rest[lookup[sources[j]]] for c in children])
            b/=np.linalg.norm(b,axis=-1,keepdims=True)
            bases[j]=Rotation.align_vectors(b,a)[0].as_matrix()
        r[:,j]=reference.rotations[:,lookup[sources[j]]]@bases[j]
    for j in range(len(names)):
        children = [c for c,p in enumerate(parents) if p==j and observed[c] and observed[j]]
        if not children: continue
        # Minimal change to native orientation that aligns the outgoing bones.
        # Multi-child joints use a proper least-squares rotation (no reflection).
        a = np.stack([rest[c]-rest[j] for c in children])
        b = desired[:,children]-desired[:,j,None]
        a = a/np.linalg.norm(a,axis=-1,keepdims=True)
        b = b/np.linalg.norm(b,axis=-1,keepdims=True)
        if len(children)==1:
            x = np.einsum('tij,j->ti',r[:,j],a[0]); y=b[:,0]
            cross=np.cross(x,y); dot=np.clip(np.sum(x*y,axis=-1),-1,1)
            norm=np.linalg.norm(cross,axis=-1)
            axis=cross/np.maximum(norm[:,None],1e-12)
            opposite=dot < -1+1e-10
            if opposite.any():
                fallback=np.cross(x[opposite],np.eye(3)[np.argmin(np.abs(x[opposite]),axis=1)])
                axis[opposite]=fallback/np.linalg.norm(fallback,axis=-1,keepdims=True)
            swing=Rotation.from_rotvec(axis*np.arctan2(norm,dot)[:,None]).as_matrix()
            r[:,j]=swing@r[:,j]
        else:
            u,_,vh=np.linalg.svd(np.einsum('tci,cj->tij',b,a))
            d=np.broadcast_to(np.eye(3),u.shape).copy();d[:,2,2]=np.linalg.det(u@vh)
            r[:,j]=u@d@vh
    p=np.zeros_like(desired)
    for j,par in enumerate(parents):
        p[:,j]=desired[:,j] if par<0 else p[:,par]+np.einsum('tij,j->ti',r[:,par],rest[j]-rest[par])
    hips=[names.index(n) for n in ('left_hip','right_hip')]
    p += (desired[:,hips].mean(1)-p[:,hips].mean(1))[:,None]
    closure={}
    if source_geometry and legacy.method!='ardy':
        # Preserve chest endpoint using the two available lumbar links. The
        # source's omitted Spine2 changes the length of the naive shortcut;
        # redistributing bending across retained joints avoids that error.
        i,j,k=[names.index(n) for n in ('spine1','spine2','spine3')]
        assert parents[j]==i and parents[k]==j
        start=p[:,i];end=desired[:,k];v=end-start;distance=np.linalg.norm(v,axis=-1)
        u=v/np.maximum(distance[:,None],1e-12)
        l1=np.linalg.norm(rest[j]-rest[i]);l2=np.linalg.norm(rest[k]-rest[j])
        reachable=(distance>=abs(l1-l2)-1e-9)&(distance<=l1+l2+1e-9)&(distance>1e-9)
        d=np.clip(distance,max(abs(l1-l2),1e-9),l1+l2)
        along=(l1*l1-l2*l2+d*d)/(2*d)
        radius=np.sqrt(np.maximum(l1*l1-along*along,0))
        hint=desired[:,j]-start
        transverse=hint-u*np.sum(hint*u,axis=-1,keepdims=True)
        n=np.linalg.norm(transverse,axis=-1)
        bad=n<1e-9
        fallback=np.einsum('tij,j->ti',r[:,i],np.array([1.,0,0]))
        fallback-=u*np.sum(fallback*u,axis=-1,keepdims=True)
        transverse[bad]=fallback[bad]
        transverse/=np.maximum(np.linalg.norm(transverse,axis=-1,keepdims=True),1e-12)
        middle=start+u*along[:,None]+transverse*radius[:,None]
        ri=point_rotation(r[:,i],rest[j]-rest[i],middle-start)
        rj=point_rotation(r[:,j],rest[k]-rest[j],end-middle)
        r[reachable,i]=ri[reachable];r[reachable,j]=rj[reachable]
        root=p[:,0].copy()
        for node,par in enumerate(parents):
            p[:,node]=root if par<0 else p[:,par]+np.einsum('tij,j->ti',r[:,par],rest[node]-rest[par])
        closure={'chest_endpoint_two_link_reachable_frames':int(reachable.sum()),'frames':len(p),
                 'intermediate_joint':names[j],'source_internal_spine_pose':'redistributed; not independently preserved',
                 'chest_endpoint_max_error_m':float(np.max(np.linalg.norm(p[:,k]-end,axis=-1)))}
    provenance=dict(legacy.provenance)
    provenance.update({'candidate':'fixed_asset_direction_alignment_v1',
        'retarget_bases':bases.tolist(),
        'rotation_gauge':'static source OFFSET to target asset directions; single-bone twist minimal-rotation convention; leaf identity convention',
        'acceptance':'pending','geometry':'registered_fixed_template; shape mismatch unresolved',
        'root':'source_bilateral_hip_midpoint', 'source_position_observed':observed})
    if source_geometry:
        provenance.update({'candidate':'source_geometry_direction_alignment_v2',
            'geometry':'per_window_fixed_source_rest','neutral_joints':rest.tolist(),
            'parents':parents,'source_nodes':sources,
            'effective_compressed_links':effective_links,
            'chain_compression':closure,
            'source_position_observed':[bool(o and s is not None) for o,s in zip(observed,sources)]})
    return Motion(p,names,reference.times,legacy.method,r,names,provenance),parents
