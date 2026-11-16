"""Versioned CoLLaboratE reference interpretation; independent of the fitter."""
import numpy as np
from .motion import Motion
from .geometry import frame,coordinate_precision_valid


def collaborate(bvh,shared_float32_contract=True):
    """Interpret animated local translations directly; preserve rotation channels.

    Units and world up follow the audited dataset convention. Local rest basis is
    a project geometric construction, independent of arbitrary world translation.
    """
    p,r=bvh.forward_kinematics(True,translation_mode='absolute_channels',dtype=np.float64)
    names=[j.name for j in bvh.joints]
    local=np.median(bvh.local_positions(True,'absolute_channels'),0);rest=np.zeros_like(local)
    for j,bone in enumerate(bvh.joints):
        if bone.parent>=0:rest[j]=rest[bone.parent]+local[j]
    lookup={n.lower():i for i,n in enumerate(names)}
    ix=lambda *keys:next(lookup[n.lower()] for n in keys if n.lower() in lookup)
    c=ix('Spine3','Chest','Spine2');neck=ix('Neck','Neck1')
    basis=frame(rest[neck]-rest[c],rest[ix('RightShoulder','RightArm')]-rest[ix('LeftShoulder','LeftArm')])
    metres=p*.01
    good=coordinate_precision_valid(metres.astype(np.float32)) if shared_float32_contract else np.ones(len(p),bool)
    metres[~good]=np.nan
    return Motion(metres,names,np.arange(len(p))/bvh.fps,'reference',r,names,
        {'segment_basis':basis.tolist(),'scope':'full_body_published_bvh_cm',
         'axes_evidence':'BVH channels plus audited geometry inference; anatomical correspondence unverified',
         'translation_contract':'animated local position channels replace OFFSET; rest-only axes retain OFFSET',
         'shared_export_precision_rejected_frames':int((~good).sum()),
         'reference_version':'collaborate-bvh-absolute-channels-2'})
