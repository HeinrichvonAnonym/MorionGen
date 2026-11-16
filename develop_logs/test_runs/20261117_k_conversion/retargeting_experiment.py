"""Explicit source-chain correspondence for controlled representation experiments.

These mappings are project-derived geometric correspondences, not baseline claims.
"""
from itertools import combinations
import numpy as np

SPINE={'spine1','spine2','spine_mid','spine3'}

def spine_mapping(reference,names,rest,parents,source_for,source_names):
    target=[i for i,n in enumerate(source_for) if n in SPINE]
    source=[i for i,n in enumerate(source_names) if n in SPINE]
    if not target or not source:return source_for,{'status':'unavailable'}
    if len(target)>len(source):return source_for,{'status':'target_has_more_nodes; legacy prior retained'}
    # Both chains are serialized proximal to distal. Arc length distinguishes
    # nodes despite superficially similar naming in different skeletons.
    srcroot=source_names.index('pelvis')
    src_lengths=[np.median(np.linalg.norm(reference.positions[:,j]-reference.positions[:,p],axis=-1)) for p,j in zip([srcroot]+source[:-1],source)]
    dst_lengths=[np.linalg.norm(rest[j]-rest[parents[j]]) for j in target]
    a=np.cumsum(src_lengths);a=a/a[-1];b=np.cumsum(dst_lengths);b=b/b[-1]
    choices=[c+(len(source)-1,) for c in combinations(range(len(source)-1),len(target)-1)]
    selected=min(choices,key=lambda c:float(np.sum((a[list(c)]-b)**2)))
    result=list(source_for)
    for j,k in zip(target,selected):result[j]=source_names[source[k]]
    return result,{'status':'derived_from_chain_arc_lengths','source_fraction':a.tolist(),'target_fraction':b.tolist(),
                   'mapping':{names[j]:source_names[source[k]] for j,k in zip(target,selected)}}

