"""Independent input-fidelity scoring when no paired original motion exists.

Only F-D is reported. Common lengths come from the actual input D and never
enter fitting. This module is for independent reporting, not candidate choice.
"""
import numpy as np
from motiogen.degradation.metrics import points,normalize,angle,REGIONS
from .candidate_quality import DIRECTIONS,exact_indices,assess
from .motion import Motion
from .scoring import summary,unit


def score(motion,fitted,head_rotation=None):
    t=motion.times;indices=exact_indices(t,fitted['times'])
    raw=points(motion);predicted=points(Motion(np.asarray(fitted['positions'])[indices],list(fitted['names']),t,'model_fk'))
    for name in motion.legacy_prior_names():
        if name in raw:raw[name]=np.full_like(raw[name],np.nan)
    raw['pelvis_mid']=(raw['left_hip']+raw['right_hip'])/2
    raw['shoulder_mid']=(raw['left_shoulder']+raw['right_shoulder'])/2
    for name in raw:
        observed=np.isfinite(raw[name]).all(-1);predicted[name]=predicted[name].copy();predicted[name][~observed]=np.nan
    nd,nf=normalize(raw,raw),normalize(raw,predicted)
    root=predicted['pelvis_mid']-raw['pelvis_mid'];positions={};directions={};angles={};temporal={}
    for name in dict.fromkeys(n for names in REGIONS.values() for n in names):
        delta=predicted[name]-raw[name];norm=nf[name]-nd[name]
        positions[name]=dict(world=summary(np.linalg.norm(delta,axis=-1)*100,[5]),
                            pelvis_relative=summary(np.linalg.norm(delta-root,axis=-1)*100,[5]),
                            common_length_world=summary(np.linalg.norm(norm,axis=-1)*100,[5]),
                            common_length_pelvis_relative=summary(np.linalg.norm(norm-root,axis=-1)*100,[5]),
                            signed_bias_xyz_cm=[summary(delta[:,j]*100,signed=delta[:,j]*100)['signed_bias'] for j in range(3)])
    for name,(start,end) in DIRECTIONS.items():
        d,f=raw[end]-raw[start],predicted[end]-predicted[start]
        directions[name]=summary(angle(d,f),[5,10,20])
        rate=np.diff(unit(f)-unit(d),axis=0)/np.diff(t)[:,None]
        temporal[name]=dict(direction_rate_vector_error_deg_s=summary(np.linalg.norm(rate,axis=-1)*180/np.pi),
                            input_step_deg=summary(angle(d[:-1],d[1:])),output_step_deg=summary(angle(f[:-1],f[1:])))
    for side in ('left','right'):
        for joint,parent,child in [('elbow','shoulder','wrist'),('knee','hip','ankle')]:
            def flex(p):return 180-angle(p[side+'_'+parent]-p[side+'_'+joint],p[side+'_'+child]-p[side+'_'+joint])
            delta=flex(predicted)-flex(raw);angles[side+'_'+joint]=summary(abs(delta),[5,10,20],delta)
    from .native_frames import read
    source=read(motion)
    return dict(comparison='F-D',clock='Actual input timestamps; full window, no reference jump strata',frames=len(t),
                positions_cm=positions,directions_deg=directions,joint_angles_deg=angles,temporal=temporal,
                input_quality=assess(motion,fitted,head_rotation=head_rotation,include_export_path=True,ranking_policy='observed_then_path_v1'),
                native_frame_support={k:source[k] for k in ('status','reason','scope') if k in source},
                normalization='Window-median actual D lengths, independent left/right, score-only. Raw input and fitted model remain unchanged.',
                scope='Fit fidelity to a generated baseline output. No paired original R, no end-to-end accuracy or anatomical qualification.')
