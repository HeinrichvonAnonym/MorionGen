"""Independent R/D/F geometry scoring. This module is never imported by fitting.

World positions and directions describe named kinematic points, not anatomical
accuracy. All three normalized geometries use the same R-only fixed lengths.
"""
import numpy as np
from motiogen.degradation.metrics import points, normalize, angle, stats, REGIONS


DIRECTIONS = {'trunk_axis': ('pelvis_mid', 'shoulder_mid'),
              'shoulder_line': ('left_shoulder', 'right_shoulder'),
              'head_neck_axis': ('neck', 'head')}
for _side in ('left', 'right'):
    for _part, _a, _b in [('upper_arm','shoulder','elbow'), ('forearm','elbow','wrist'),
                         ('thigh','hip','knee'), ('shank','knee','ankle'), ('foot_axis','ankle','foot')]:
        DIRECTIONS[_side+'_'+_part] = (_side+'_'+_a, _side+'_'+_b)


def summary(values, thresholds=(), signed=None):
    values=np.asarray(values); result=stats(values)
    result['valid_fraction']=result['n']/result['denominator'] if result['denominator'] else None
    result['within']={str(t):{'n':int(np.sum(values<=t)), 'denominator':values.size,
                            'valid_denominator':result['n']} for t in thresholds}
    if signed is not None:
        signed=np.asarray(signed); valid=np.isfinite(signed)
        result['signed_bias']=float(np.mean(signed[valid])) if valid.any() else None
    return result


def unit(v):
    den=np.linalg.norm(v,axis=-1,keepdims=True)
    return np.where(den>1e-9,v/np.maximum(den,1e-9),np.nan)


def score(reference, degraded, fitted, tolerance_mask=None, normalization_reference=None):
    if not (np.array_equal(reference.times,degraded.times) and np.array_equal(reference.times,fitted.times)):
        raise ValueError('R, D, F require identical physical timestamps')
    t=reference.times
    raw={'R':points(reference,True), 'D':points(degraded), 'F':points(fitted)}
    lengths_source=raw['R'] if normalization_reference is None else points(normalization_reference,True)
    norm={key:normalize(lengths_source,value) for key,value in raw.items()}
    if tolerance_mask is None:tolerance_mask=np.zeros(len(t),bool)
    if np.shape(tolerance_mask)!=(len(t),):raise ValueError('Invalid frozen temporal stratum')
    strata={'full':np.ones(len(t),bool),'continuous':~tolerance_mask,'source_jump_tolerance':tolerance_mask}
    result={}
    for label, x, y in [('D-R','R','D'),('F-D','D','F'),('F-R','R','F')]:
        a,b=raw[x],raw[y]; pair={}
        for stratum,mask in strata.items():
            metrics={'positions_cm':{},'directions_deg':{},'joint_angles_deg':{},'temporal':{}}
            root=b['pelvis_mid']-a['pelvis_mid']
            for region,names in REGIONS.items():
                region_metrics={}
                for name in names:
                    delta=b[name]-a[name]
                    nd=norm[y][name]-norm[x][name]
                    region_metrics[name]={'world':summary(np.linalg.norm(delta[mask],axis=-1)*100,[5]),
                        'pelvis_relative':summary(np.linalg.norm((delta-root)[mask],axis=-1)*100,[5]),
                        'common_length_world':summary(np.linalg.norm(nd[mask],axis=-1)*100,[5]),
                        'common_length_pelvis_relative':summary(np.linalg.norm((nd-root)[mask],axis=-1)*100,[5]),
                        'world_bias_xyz_cm':[summary(delta[mask,j]*100,signed=delta[mask,j]*100)['signed_bias'] for j in range(3)]}
                metrics['positions_cm'][region]=region_metrics
            for name,(start,end) in DIRECTIONS.items():
                va,vb=a[end]-a[start],b[end]-b[start]
                error=angle(va,vb)
                metrics['directions_deg'][name]=summary(error[mask],[5,10,20])
                steps=mask[1:]&mask[:-1]
                sa,sb=angle(va[:-1],va[1:]),angle(vb[:-1],vb[1:])
                rate=(np.diff(unit(vb),axis=0)-np.diff(unit(va),axis=0))/np.diff(t)[:,None]
                metrics['temporal'][name]={'reference_step_deg':summary(sa[steps]),
                    'output_step_deg':summary(sb[steps]),'signed_step_change_deg':summary((sb-sa)[steps],signed=(sb-sa)[steps]),
                    'direction_rate_vector_error_deg_s':summary(np.linalg.norm(rate[steps],axis=-1)*180/np.pi)}
            for side in ('left','right'):
                for joint,parent,child in [('knee','hip','ankle'),('elbow','shoulder','wrist')]:
                    def flex(p):return 180-angle(p[side+'_'+parent]-p[side+'_'+joint],p[side+'_'+child]-p[side+'_'+joint])
                    delta=flex(b)-flex(a)
                    metrics['joint_angles_deg'][side+'_'+joint]=summary(abs(delta[mask]),[5,10,20],delta[mask])
            for name in ('pelvis_mid','left_foot','right_foot','left_ankle','right_ankle'):
                dv=np.diff(b[name]-a[name],axis=0)/np.diff(t)[:,None]
                steps=mask[1:]&mask[:-1]
                metrics['temporal'][name+'_velocity_cm_s']=summary(np.linalg.norm(dv[steps],axis=-1)*100)
            pair[stratum]=metrics
        result[label]=pair
    return {'comparisons':result,'clock':{'frames':len(t),'start_s':float(t[0]),'end_s':float(t[-1]),
                    'source_jump_frames':int(tolerance_mask.sum())},
            'normalization':'Window-median R lengths applied to R,D,F independently; separate sides; root retained; scoring only',
            'scope':'Kinematic point directions and centres; head/neck model unavailable stays NaN; no anatomical accuracy claim',
            'temporal_definition':'direction rate uses derivative of unit direction in world coordinates, converted rad/s to deg/s; not anatomical joint angular velocity'}
