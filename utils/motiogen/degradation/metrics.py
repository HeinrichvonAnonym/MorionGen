"""D-R metrics on actual timestamps; all denominators and vector biases retained."""
import numpy as np
from .positions import SOURCE


NAMES = ['pelvis','neck','head']+[f'{s}_{n}' for s in ('left','right')
         for n in ('hip','shoulder','elbow','wrist','knee','ankle','foot')]
REGIONS = {'head_neck':['neck','head'], 'shoulders':['left_shoulder','right_shoulder'],
           'trunk':['pelvis_mid','shoulder_mid'], 'hands':['left_wrist','right_wrist'],
           'knees':['left_knee','right_knee'],
           'feet':['left_ankle','right_ankle','left_foot','right_foot']}

EDGES=[('pelvis_mid','shoulder_mid'),('shoulder_mid','neck'),('neck','head')]+[
    (parent,child) for side in ('left','right') for parent,child in [
        ('pelvis_mid',side+'_hip'),(side+'_hip',side+'_knee'),(side+'_knee',side+'_ankle'),
        (side+'_ankle',side+'_foot'),('shoulder_mid',side+'_shoulder'),
        (side+'_shoulder',side+'_elbow'),(side+'_elbow',side+'_wrist')]]


def normalize(reference, other):
    out={'pelvis_mid':other['pelvis_mid'].copy()}
    for parent,child in EDGES:
        length=np.nanmedian(np.linalg.norm(reference[child]-reference[parent],axis=-1))
        v=other[child]-other[parent];den=np.linalg.norm(v,axis=-1,keepdims=True)
        out[child]=out[parent]+np.where(den>1e-9,v/np.maximum(den,1e-9)*length,np.nan)
    return out


def points(motion, source=False):
    index={n:i for i,n in enumerate(motion.names)}
    result={}
    for name in NAMES:
        key=SOURCE[name] if source else name
        if source and key=='Neck' and key not in index:key='Neck1'
        if not source and name.endswith('_wrist') and name+'_body' in index:key=name+'_body'
        result[name]=motion.positions[:,index[key]] if key in index else np.full((len(motion.times),3),np.nan)
    result['pelvis_mid']=(result['left_hip']+result['right_hip'])/2
    result['shoulder_mid']=(result['left_shoulder']+result['right_shoulder'])/2
    return result


def stats(values, total=None):
    a=np.asarray(values);a=a[np.isfinite(a)]
    return {'n':len(a),'denominator':int(total if total is not None else np.size(values)),
            'median':float(np.median(a)) if len(a) else None,
            'p95':float(np.percentile(a,95)) if len(a) else None,
            'max':float(np.max(a)) if len(a) else None}


def angle(a,b):
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    c=np.sum(a*b,axis=-1)/np.maximum(den,1e-15)
    return np.where(den>1e-12,np.degrees(np.arccos(np.clip(c,-1,1))),np.nan)


def compare(reference, decoded):
    if not np.allclose(reference.times,decoded.times,atol=1e-9):raise ValueError('Unmatched physical clock')
    a,b=points(reference,True),points(decoded)
    an,bn=normalize(a,a),normalize(a,b)
    metrics={}
    for region,names in REGIONS.items():
        d=np.stack([b[n]-a[n] for n in names],1)
        root_delta=b['pelvis_mid']-a['pelvis_mid']
        metrics[region]={'world_cm':stats(np.linalg.norm(d,axis=-1)*100),
            'pelvis_relative_cm':stats(np.linalg.norm(d-root_delta[:,None],axis=-1)*100),
            'common_length_world_cm':stats(np.stack([np.linalg.norm(bn[n]-an[n],axis=-1)*100 for n in names],1)),
            'endpoints_cm':{n:stats(np.linalg.norm(b[n]-a[n],axis=-1)*100) for n in names},
            'world_bias_xyz_cm':np.nanmean(d,axis=(0,1)).tolist()}
        metrics[region]['world_bias_xyz_cm']=(np.asarray(metrics[region]['world_bias_xyz_cm'])*100).tolist()
    pairs={'head_neck_axis':('neck','head'), 'trunk_axis':('pelvis_mid','shoulder_mid'),
           'shoulder_line':('left_shoulder','right_shoulder')}
    for side in ('left','right'):
        pairs[side+'_upper_arm']=(side+'_shoulder',side+'_elbow')
        pairs[side+'_forearm']=(side+'_elbow',side+'_wrist')
        pairs[side+'_thigh']=(side+'_hip',side+'_knee')
        pairs[side+'_shank']=(side+'_knee',side+'_ankle')
        pairs[side+'_foot_axis']=(side+'_ankle',side+'_foot')
    directions={}
    for key,(x,y) in pairs.items():
        e=angle(a[y]-a[x],b[y]-b[x]);directions[key]=stats(e)
        directions[key]['within_deg']={str(t):int(np.sum(e<=t)) for t in [5,10,20]}
    knees={}
    for side in ('left','right'):
        def flex(p):return 180-angle(p[side+'_hip']-p[side+'_knee'],p[side+'_ankle']-p[side+'_knee'])
        delta=flex(b)-flex(a);knees[side]={**stats(abs(delta)),'signed_bias_deg':float(np.nanmean(delta))}
    return {'positions':metrics,'directions_deg':directions,'knee_flexion_deg':knees,
            'normalization':'fixed window median R segment lengths, independent left/right; score-only common tree',
            'scope':'wrist joint centres; ankle and toe-base proxies; native BVH geometry, not anatomical joint accuracy'}
