"""Input-only, time-constant kinematic shape calibration on a copied model."""
import json
from pathlib import Path
import numpy as np
from .config import ROOT,digest
from .export import target_template


def calibrate(source_model,destination,motion):
    import opensim as osim
    import biomech
    if Path(source_model).resolve()==Path(destination).resolve():
        raise ValueError('Shape calibration requires a separate model asset')
    points,names,_=motion.legacy();p={n:points[:,i] for i,n in enumerate(names)}
    model=osim.Model(str(source_model));state=model.initSystem()
    c=lambda n:biomech._child_origin(model,state,n)
    model.realizePosition(state)
    factors={};measurements={};prior=set(motion.legacy_prior_joints())
    def distance(a,b):
        if a in prior or b in prior:return None
        v=np.linalg.norm(p[a]-p[b],axis=1);v=v[np.isfinite(v)&(v>1e-6)]
        return float(np.median(v)) if len(v) else None
    for side,s in [('left','l'),('right','r')]:
        for key,a,b,ja,jb,bodies in [
            ('thigh','hip','knee','hip','walker_knee',['femur','patella']),
            ('shin','knee','ankle','walker_knee','ankle',['tibia']),
            ('foot','ankle','foot','ankle','mtp',['talus','calcn','toes']),
            ('upper_arm','shoulder','elbow','acromial','elbow',['humerus']),
            ('forearm','elbow','wrist','elbow','radius_hand',['ulna','radius','hand'])]:
            wanted=distance(side+'_'+a,side+'_'+b);base=float(np.linalg.norm(c(ja+'_'+s)-c(jb+'_'+s)))
            ratio=1. if wanted is None else wanted/base
            if not .25<=ratio<=4:raise ValueError('Unsupported input scale for '+side+'_'+key)
            for body in bodies:factors[body+'_'+s]=[ratio]*3
            measurements[side+'_'+key]={'input_median_m':wanted,'model_neutral_m':base,'ratio':ratio,'source':'input_segment_median' if wanted is not None else 'unchanged_prior'}
    tn,rest,_=target_template(motion.method);ix={n:i for i,n in enumerate(tn)}
    desired=np.linalg.norm(.5*(rest[ix['left_shoulder']]+rest[ix['right_shoulder']]-rest[ix['left_hip']]-rest[ix['right_hip']]))
    # Source asset is explicit; never infer neutral torso length from a bent clip.
    asset_method='primal' if motion.method in ('motionlcm','omnicontrol','chois') else motion.method
    asset=ROOT/'config/evalkit_k_calibration'/f'{asset_method}.json'
    hips=.5*(c('hip_l')+c('hip_r'));shoulders=.5*(c('acromial_l')+c('acromial_r'));lumbar=c('back')
    width=distance('left_shoulder','right_shoulder');base_width=np.linalg.norm(c('acromial_l')-c('acromial_r'))
    zscale=width/base_width if width is not None else 1.
    mobile=shoulders-lumbar;fixed=lumbar-hips
    horizontal=fixed[[0,2]]+mobile[[0,2]]*np.array([1.,zscale])
    radicand=desired**2-float(horizontal@horizontal)
    supported=radicand>0 and abs(mobile[1])>1e-6
    yscale=(np.sqrt(radicand)-fixed[1])/mobile[1] if supported else 1.
    if not .25<=yscale<=4:yscale=1.;supported=False
    if not .25<=zscale<=4:raise ValueError('Unsupported shoulder width')
    factors['torso']=[1.,float(yscale),float(zscale)]
    measurements['torso']={'neutral_span_m':float(desired),'source':'audited_format_asset_prior','asset':str(asset),'sha256':digest(asset),'length_identified_from_motion':False,'neutral_scale_applied':bool(supported),'shoulder_width_m':width,'axis_scale_xyz':factors['torso'],'scope':'constant source-rig geometry prior; no anatomical shape claim'}
    scales=osim.ScaleSet()
    for body,factor in factors.items():
        scale=osim.Scale();scale.setSegmentName(body);scale.setScaleFactors(osim.Vec3(*factor));scale.setApply(True);scales.cloneAndAppend(scale)
    model.scale(state,scales,True,model.getTotalMass(state));model.finalizeConnections();model.printToXML(str(destination))
    return {'source_model_sha256':digest(source_model),'model_sha256':digest(destination),'measurements':measurements,'scales':factors,'scope':'fixed per-input geometry; no evaluation reference; no per-frame shape changes'}
