"""Position-supported full-body angles, with metric-level masks and explicit semantics."""
import numpy as np
from .geometry import unit


def evaluate_body(motion):
    names = {n:i for i,n in enumerate(motion.names)}
    aliases = {'spine3':'Spine3','spine1':'Spine','left_wrist':'left_wrist_body','right_wrist':'right_wrist_body'}
    prior = set(motion.provenance.get('degradation', {}).get('prior_joints', []))
    def point(name):
        key = aliases.get(name) if aliases.get(name) in names else name
        return motion.positions[:,names[key]] if key in names else np.full((len(motion.times),3),np.nan)
    def measured(keys):
        return all((aliases.get(k) if aliases.get(k) in names else k) not in prior for k in keys)
    result = {}
    def add(name, a, b, keys, definition):
        angle = np.degrees(np.arccos(np.clip((unit(a)*unit(b)).sum(-1),-1,1)))
        valid = np.isfinite(angle) & measured(keys)
        result[name] = {'values_deg':angle, 'valid':valid, 'origin':'joint_position_geometry',
                        'definition':definition, 'reason':'' if valid.any() else 'missing_degenerate_or_template_points'}
    hips = .5*(point('left_hip')+point('right_hip'))
    shoulders = .5*(point('left_shoulder')+point('right_shoulder'))
    add('trunk_incline', shoulders-hips, np.broadcast_to([0.,1.,0.], hips.shape),
        ['left_hip','right_hip','left_shoulder','right_shoulder'], 'shoulder-to-hip midpoint axis versus world vertical')
    for side in ('left','right'):
        for joint, proximal, distal in [('knee','hip','ankle'),('elbow','shoulder','wrist')]:
            a,b,c = [f'{side}_{x}' for x in (proximal,joint,distal)]
            add(f'{side}_{joint}_flexion',point(b)-point(a),point(c)-point(b),[a,b,c],
                'angle between successive bone directions; straight limb is zero')
        a,b=f'{side}_hip',f'{side}_knee'
        add(f'{side}_thigh_incline',point(a)-point(b),np.broadcast_to([0.,1.,0.],hips.shape),[a,b],
            'knee-to-hip direction versus world vertical')
    # This direction survives position-only formats but does not define a head frame.
    add('neck_head_axis_incline',point('head')-point('neck'),np.broadcast_to([0.,1.,0.],hips.shape),['neck','head'],
        'neck-to-head joint direction versus vertical; excludes independent head axial rotation')
    return result
