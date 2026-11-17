"""Separate observed input geometry from model-fit fidelity, per metric.

These masks compare the model to its input, never to an evaluation reference.
They describe geometric fidelity; anatomical accuracy needs separate evidence.
"""
import numpy as np
from .bones import common
from .scoring import angles,region


def dependencies(metric):
    torso={'left_hip','right_hip','left_shoulder','right_shoulder'}
    if metric=='trunk_incline':return torso
    if metric=='pelvis_line_incline':return {'left_hip','right_hip'}
    side='left' if metric.startswith('left_') else 'right'
    suffix=metric[len(side)+1:]
    nodes={'elbow_flexion':['shoulder','elbow','wrist'],
           'knee_flexion':['hip','knee','ankle'],
           'thigh_incline':['hip','knee'],
           'upper_arm_elevation':['shoulder','elbow'],
           'hip_trunk_angle':['hip','knee'],
           'ankle_chain_angle':['knee','ankle','foot'],
           'foot_incline':['ankle','foot']}[suffix]
    return {side+'_'+n for n in nodes} | (torso if suffix=='hip_trunk_angle' else set())


def metric_fidelity(source,fitted,names,limit,prior=()):
    a,b=angles(common(source,names)),angles(common(fitted,names));result={}
    for name,value in a.items():
        observed=np.isfinite(value)&(not bool(dependencies(name)&set(prior)))
        delta=b[name]-value;finite=np.isfinite(delta)
        valid=observed&finite&(abs(delta)<=limit)
        reason=np.where(~observed,1,np.where(~finite,2,np.where(valid,0,3))).astype(np.uint8)
        result[name]={'input_deg':value,'model_deg':b[name],'error_deg':delta,
                      'observed':observed,'model_fit_valid':valid,'reason':reason}
    return result


def apply_metric_gates(quality,metrics):
    """Preserve the direction-only mask and add joint-angle agreement."""
    for reg,q in quality.items():
        q['direction_valid']=q['angle_valid'].copy()
        masks=[m['model_fit_valid'] for n,m in metrics.items() if region(n)==reg]
        q['geometry_valid']=np.logical_and.reduce(masks)
        q['angle_valid']=q['direction_valid']&q['geometry_valid']
    return quality


REASONS={0:'input_geometry_and_model_agree_within_engineering_tolerance',
         1:'input_geometry_missing_degenerate_or_prior',
         2:'model_geometry_unavailable',
         3:'model_geometry_does_not_reliably_reproduce_input'}
