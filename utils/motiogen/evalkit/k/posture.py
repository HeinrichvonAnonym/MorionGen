import numpy as np
from .config import CONFIG, VERSION
from .geometry import frame, valid_rotation
from .body import evaluate_body


def evaluate(motion, anatomical_bases=None):
    """Native segment pose is distinct from anatomically calibrated neck angles."""
    model_assumption = anatomical_bases is None and 'model_bases' in motion.provenance
    if model_assumption: anatomical_bases=motion.provenance['model_bases']
    n = len(motion.times); idx = {x:i for i,x in enumerate(motion.names)}
    profile = CONFIG['profiles'].get(motion.method,{})
    result = {'version':VERSION,'method':motion.method,'times':motion.times,'metrics':{},'body':evaluate_body(motion)}
    for region,key in [('head','head'),('thorax','thorax')]:
        name = profile.get(key)
        r = np.full((n,3,3),np.nan)
        origin = 'unavailable'
        if profile.get('orientation') != 'positions_only' and motion.rotations is not None and name in motion.rotation_names:
            r = motion.rotations[:,motion.rotation_names.index(name)].copy(); origin='native_segment_rotation'
        elif region=='thorax' and {'neck','left_shoulder','right_shoulder'} <= idx.keys():
            lower = next((x for x in ('spine3','Spine3','spine2','Spine') if x in idx),None)
            if lower:
                left = 'left_collar' if 'left_collar' in idx else 'left_shoulder'
                right = 'right_collar' if 'right_collar' in idx else 'right_shoulder'
                r = frame(motion.positions[:,idx['neck']]-motion.positions[:,idx[lower]],
                          motion.positions[:,idx[right]]-motion.positions[:,idx[left]])
                origin='position_derived_frame'
        calibrated = anatomical_bases is not None and region in anatomical_bases
        if calibrated:
            b = np.asarray(anatomical_bases[region],float)
            if b.shape!=(3,3) or not valid_rotation(b): raise ValueError('Invalid anatomical basis')
            r = r @ b
        valid = valid_rotation(r)
        prior = motion.provenance.get('degradation',{}).get('prior_joints',[])
        canonical_name = {'Head':'head','Chest':'spine3'}.get(name,name)
        if name in prior or canonical_name in prior:valid[:] = False
        result['metrics'][region] = dict(rotations=r,valid=valid,origin=origin,
            status='supported_native_frame' if valid.any() else 'unavailable',
            anatomical_calibration='provided' if calibrated else 'unverified',
            reason='' if valid.any() else 'independent_orientation_missing_or_invalid')
    head,thorax = (result['metrics'][k] for k in ('head','thorax'))
    valid = head['valid'] & thorax['valid']
    if model_assumption:
        valid &= np.asarray(motion.provenance['calibration'].get('frame_valid',np.ones(n,bool)))
    relative = thorax['rotations'].swapaxes(-1,-2) @ head['rotations']
    calibrated = anatomical_bases is not None and {'head','thorax'} <= anatomical_bases.keys()
    reason = ('head_orientation_missing_or_invalid' if not head['valid'].any() else
              'thorax_orientation_missing_or_invalid' if not thorax['valid'].any() else
              'model_calibration_frame_gate' if not valid.any() else
              'anatomical_calibration_unverified' if not calibrated else '')
    result['metrics']['neck'] = dict(rotations=relative,valid=valid,anatomical_valid=valid & calibrated,
        status='model_convention' if model_assumption and valid.any() else 'supported' if calibrated and valid.any() else 'native_frame_only' if valid.any() else 'unreliable',
        reason=reason,
        message='' if calibrated and valid.any() else '颈部指标不可可靠评价')
    result['metrics']['neck']['precision_scope']='source_model_convention_pending_external_precision' if model_assumption else 'explicit_calibration' if calibrated else 'unverified'
    return result


def summary(result):
    return {'version':result['version'],'method':result['method'], 'body':{
        k:{**{n:v for n,v in item.items() if n not in ('values_deg','valid')},
           'valid_frames':int(item['valid'].sum()),'total_frames':len(item['valid'])}
        for k,item in result['body'].items()}, 'metrics':{
        k:{**{n:v for n,v in item.items() if n not in ('rotations','valid','anatomical_valid')},
           'valid_frames':int(item['valid'].sum()),'total_frames':len(item['valid']),
           'coverage':float(item['valid'].mean())} for k,item in result['metrics'].items()}}
