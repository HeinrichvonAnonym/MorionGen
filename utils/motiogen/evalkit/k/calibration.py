"""Reuse source-model rest frames; test bone-frame consistency on each actual output."""
import json
import numpy as np
from .config import ROOT,digest,CONFIG
from .geometry import frame,unit

CLOSD = ['Pelvis','L_Hip','R_Hip','Torso','L_Knee','R_Knee','Spine','L_Ankle','R_Ankle','Chest','L_Toe','R_Toe','Neck','L_Thorax','R_Thorax','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist','L_Hand','R_Hand']
UPRIGHT=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])


def model_basis(motion,source,config=None):
    from .config import resolve_config
    settings=resolve_config(config)
    path=ROOT/'config/evalkit_k_calibration'/f'{motion.method}.json'
    if motion.rotations is None or not path.exists():return None,{'status':'source_orientation_unavailable'}
    cal=json.loads(path.read_text())
    reconstructed=False
    if motion.method=='chois':
        from ..sources import chois_window
        try:cal=cal['windows'][chois_window(source)]
        except (KeyError,ValueError,OSError):
            from .motion import PARENTS
            # CHOIS stores body positions and independent global rotations.
            # Reconstruct its own rest offsets, then check across all frames.
            neutral=np.zeros((22,3))
            for j,parent in enumerate(PARENTS):
                if parent>=0:
                    offsets=np.einsum('tji,tj->ti',motion.rotations[:,parent],motion.positions[:,j]-motion.positions[:,parent])
                    neutral[j]=neutral[parent]+np.nanmedian(offsets,axis=0)
            cal={'neutral_joints':neutral.tolist(),'names':motion.rotation_names,'parents':PARENTS}
            reconstructed=True
    neutral=np.asarray(cal['neutral_joints']);names=cal['names'];idx={n:i for i,n in enumerate(names)}
    basis=frame(np.array([0.,1,0]),neutral[idx['right_hip']]-neutral[idx['left_hip']])
    right_basis=UPRIGHT if motion.method=='closd' else np.eye(3)
    posnames=[{'left_wrist_body':'left_wrist','right_wrist_body':'right_wrist'}.get(n,n) for n in motion.names]
    errors=[]
    # Upper-body bones establish the relevant parent-frame convention; leaf head twist is native information.
    for child in ['neck','head','left_shoulder','right_shoulder']:
        if child not in idx:continue
        j=idx[child];parent=cal['parents'][j]
        if parent<0:continue
        pn=names[parent]
        if child not in posnames or pn not in posnames:continue
        rn=CLOSD[parent] if motion.method=='closd' else pn
        if rn not in motion.rotation_names:continue
        r=motion.rotations[:,motion.rotation_names.index(rn)]@right_basis
        observed=motion.positions[:,posnames.index(child)]-motion.positions[:,posnames.index(pn)]
        local=np.einsum('tji,tj->ti',r,observed)
        cosine=(unit(local)*unit(neutral[j]-neutral[parent])).sum(-1)
        errors.append(np.degrees(np.arccos(np.clip(cosine,-1,1))))
    finite=np.concatenate(errors) if errors else np.array([]);finite=finite[np.isfinite(finite)]
    p95=float(np.percentile(finite,95)) if len(finite) else None
    tolerance=settings['calibration']['native_bone_tolerance_deg']
    frame_valid=np.all(np.asarray(errors)<=tolerance,axis=0) if errors else np.zeros(len(motion.times),bool)
    record={'status':'model_rest_convention_supported' if p95 is not None and p95<=tolerance else 'model_rest_convention_mismatch',
            'source':str(path.relative_to(ROOT)),'sha256':digest(path),'bone_direction_p95_deg':p95,
            'scope':'source-model neutral assumption; independent anatomical precision remains separately reported',
            'rest_geometry':'per_run_inverse_fk' if reconstructed else 'source_model_asset',
            'frame_valid':frame_valid.tolist()}
    if record['status']!='model_rest_convention_supported':return None,record
    return {'head':right_basis@basis,'thorax':right_basis@basis},record
