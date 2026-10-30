"""Transparent RULA/REBA scoring from explicit anatomical/observational inputs.

Unknown inputs are sets of admissible values, never neutral defaults. This
module performs no inverse dynamics and makes no claim of human validation.
Angles must be anatomically calibrated degrees, not arbitrary Euler channels.
"""
from functools import lru_cache
from itertools import product
import json
import math
from numbers import Real
from pathlib import Path

TABLES=Path(__file__).resolve().parents[2]/'config/ergonomic_tables.json'

def choices(value,domain):
    if value is None:return tuple(domain)
    values=tuple(value) if isinstance(value,(tuple,list,set)) else (value,)
    if not values or any(v not in domain for v in values):raise ValueError(f'Invalid scoring input {value}; expected {domain}')
    return tuple(sorted(set(values)))

def angle(value):
    if not isinstance(value,Real) or not math.isfinite(value):raise ValueError('Finite angle in degrees required')
    return float(value)

def upper_arm_base(flexion):
    a=angle(flexion)
    return 1 if -20<=a<=20 else 2 if a< -20 or a<=45 else 3 if a<=90 else 4

def lower_arm_base(flexion):
    a=angle(flexion)
    if not 0<=a<=180:raise ValueError('Elbow flexion must be in [0,180]')
    return 1 if 60<=a<=100 else 2

def trunk_base(flexion,method):
    a=angle(flexion)
    if method=='REBA':return 1 if a==0 else 2 if abs(a)<=20 else 3 if a<0 or a<=60 else 4
    if method=='RULA':
        if a<0:raise ValueError('Original RULA trunk extension has no explicit flexion bin; retain as missing')
        return 2 if a<=20 else 3 if a<=60 else 4
    raise ValueError(method)

def neck_base(flexion,method):
    a=angle(flexion)
    if method=='RULA':return 4 if a<0 else 1 if a<=10 else 2 if a<=20 else 3
    if method=='REBA':return 1 if 0<=a<=20 else 2
    raise ValueError(method)

def wrist_base(flexion,method):
    a=abs(angle(flexion))
    if method=='RULA':return 1 if a==0 else 2 if a<=15 else 3
    if method=='REBA':return 1 if a<=15 else 2
    raise ValueError(method)

def component_scores(inputs,method):
    """Return discrete score sets with explicit unknown adjustment variables.

    Caller supplies anatomical angles and categorical observations. For REBA,
    twist OR side bend contributes one point, not two separate additions.
    """
    if method not in ['RULA','REBA']:raise ValueError(method)
    missing=[]
    def field(key,domain):
        if inputs.get(key) is None:missing.append(key)
        return choices(inputs.get(key),domain)
    def base(key,fn,domain):
        if inputs.get(key) is None:missing.append(key);return tuple(domain)
        return (fn(inputs[key]),)
    upper=base('upper_arm_flexion_deg',upper_arm_base,range(1,5))
    arm=sorted({max(1,min(6,b+raised+abducted-supported)) for b,raised,abducted,supported in product(upper,field('shoulder_raised',[0,1]),field('arm_abducted_or_rotated',[0,1]),field('arm_supported',[0,1]))})
    lower=base('elbow_flexion_deg',lower_arm_base,[1,2])
    if method=='RULA':lower=sorted({b+x for b,x in product(lower,field('forearm_across_midline_or_outside',[0,1]))})
    wrist=base('wrist_flexion_deg',lambda x:wrist_base(x,method),range(1,4) if method=='RULA' else [1,2])
    wrist=sorted({b+x for b,x in product(wrist,field('wrist_deviated' if method=='RULA' else 'wrist_deviated_or_twisted',[0,1]))})
    neck=base('neck_flexion_deg',lambda x:neck_base(x,method),range(1,5) if method=='RULA' else [1,2])
    if method=='RULA' and inputs.get('trunk_flexion_deg') is not None and angle(inputs['trunk_flexion_deg'])<0:
        missing.append('trunk_extension_rule_not_specified');trunk=(2,3,4)
    else:trunk=base('trunk_flexion_deg',lambda x:trunk_base(x,method),range(1,5))
    result={'upper_arm':arm,'lower_arm':lower,'wrist':wrist}
    for part,bs in [('neck',neck),('trunk',trunk)]:
        twist=field(part+'_twisted',[0,1]);bend=field(part+'_side_bent',[0,1])
        result[part]=sorted({b+(t+s if method=='RULA' else int(bool(t or s))) for b,t,s in product(bs,twist,bend)})
    if method=='RULA':
        # Supported seated trunk is an explicit category, not inferred from pose.
        seated=field('trunk_supported_seated',[0,1])
        # Supported seating supplies BASE 1, while twist/side-bend still count.
        result['trunk']=sorted({(1 if s else b)+t+sb for s,b,t,sb in product(seated,trunk,choices(inputs.get('trunk_twisted'),[0,1]),choices(inputs.get('trunk_side_bent'),[0,1]))})
        result['legs']=field('legs_score',[1,2])
        result['wrist_twist']=field('wrist_twist_score',[1,2])
        for group in ['a','b']:
            result['muscle_'+group]=field('muscle_use_'+group,[0,1])
            result['load_'+group]=field('load_score_'+group,range(4))
    else:
        legbase=field('leg_support_score',[1,2]);seated=field('seated',[0,1])
        knee=base('knee_flexion_deg',lambda x:0 if angle(x)<=30 else 1 if angle(x)<=60 else 2,[0,1,2])
        result['legs']=sorted({b+(0 if s else k) for b,s,k in product(legbase,seated,knee)})
        result['load']=field('load_score',range(4));result['coupling']=field('coupling_score',range(4))
        result['activity']=sorted({sum(v) for v in product(field('static_over_minute',[0,1]),field('repeated_small_range_over4_per_minute',[0,1]),field('rapid_change_or_unstable',[0,1]))})
    return {k:tuple(v) for k,v in result.items()},sorted(set(missing))

@lru_cache(maxsize=1)
def tables():return json.loads(TABLES.read_text())

def action_level(score,method):
    if method=='RULA':return 1 if score<=2 else 2 if score<=4 else 3 if score<=6 else 4
    if method=='REBA':return 0 if score==1 else 1 if score<=3 else 2 if score<=7 else 3 if score<=10 else 4
    raise ValueError(method)

@lru_cache(maxsize=8192)
def _combine(method,signature):
    c=dict(signature);t=tables()[method]
    if method=='RULA':
        a={t['A'][(u-1)*3+l-1][(w-1)*2+tw-1] for u,l,w,tw in product(c['upper_arm'],c['lower_arm'],c['wrist'],c['wrist_twist'])}
        b={t['B'][n-1][(tr-1)*2+le-1] for n,tr,le in product(c['neck'],c['trunk'],c['legs'])}
        aa={min(8,x+m+l) for x,m,l in product(a,c['muscle_a'],c['load_a'])}
        bb={min(7,x+m+l) for x,m,l in product(b,c['muscle_b'],c['load_b'])}
        scores={t['C'][x-1][y-1] for x,y in product(aa,bb)}
    else:
        a={t['A'][tr-1][(n-1)*4+le-1] for tr,n,le in product(c['trunk'],c['neck'],c['legs'])}
        b={t['B'][u-1][(l-1)*3+w-1] for u,l,w in product(c['upper_arm'],c['lower_arm'],c['wrist'])}
        aa={min(12,x+l) for x,l in product(a,c['load'])};bb={min(12,x+k) for x,k in product(b,c['coupling'])}
        scores={t['C'][x-1][y-1]+act for x,y,act in product(aa,bb,c['activity'])}
    return tuple(sorted(scores)),tuple(sorted(a)),tuple(sorted(b))

def score(inputs,method):
    components,missing=component_scores(inputs,method)
    scores,a,b=_combine(method,tuple(sorted(components.items())))
    return {'method':method,'components':components,'posture_a':a,'posture_b':b,
            'possible_scores':scores,'score_range':[min(scores),max(scores)],
            'total':scores[0] if not missing and len(scores)==1 else None,
            'possible_action_levels':sorted({action_level(s,method) for s in scores}),
            'missing_inputs':missing,'input_status':'complete' if not missing else 'partial',
            'human_validation':'pending_no_raters','table_version':tables()['version'],
            'scope':'Rule-based posture screening; independent missing-field combinations give conservative outer bounds, not confidence intervals or force estimates.'}
