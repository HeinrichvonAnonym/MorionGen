"""Clip-fixed effective joint centres from positional geometric frames.

These frames define a reduced-model prior, not observed segment rotations or
neutral/anatomical geometry. The strategy never changes observation support.
"""
import numpy as np
from .geometry import unit, valid_rotation


def estimate(motion):
    receipt={'policy':'observed_position_centres_v1','fixed_for_clip':True,
             'neutral_geometry_identified':False,'total_frames':len(motion.times),
             'scope':'Effective fixed-centre prior; no observed-rotation, anatomical or force qualification',
             'frame_assumption':'Pelvis: hip line and hip-midpoint-to-spine1; torso: shoulder line and spine1-to-spine3. Left line fixed, superior projected perpendicular; model x=forward,y=superior,z=right.',
             'estimator':'Componentwise median of geometric-frame centre vectors; all finite nondegenerate input frames. Variation retained as uncertainty, not converted into observed rotation.'}
    if motion.provenance.get('input_rig'):
        return None,dict(receipt,status='unavailable',reason='declared_rig_requires_declared_centres')
    index={n:i for i,n in enumerate(motion.names)}
    required=['left_hip','right_hip','left_shoulder','right_shoulder','spine1','spine3']
    if any(n not in index for n in required):
        return None,dict(receipt,status='unavailable',reason='required_nodes_missing')
    prior=set(motion.provenance.get('degradation',{}).get('prior_joints',[]))
    if prior.intersection(required):
        return None,dict(receipt,status='unavailable',reason='required_nodes_are_prior')
    p=lambda n:motion.positions[:,index[n]]
    hip=(p('left_hip')+p('right_hip'))/2
    shoulder=(p('left_shoulder')+p('right_shoulder'))/2
    def basis(left,up):
        left=unit(left);up=unit(up-left*np.sum(left*up,axis=-1,keepdims=True))
        return np.stack([np.cross(left,up),up,-left],axis=-1)
    pelvis=basis(p('left_hip')-p('right_hip'),p('spine1')-hip)
    torso=basis(p('left_shoulder')-p('right_shoulder'),p('spine3')-p('spine1'))
    local={'proximal':np.einsum('tji,tj->ti',pelvis,p('spine1')-hip),
           'distal':np.einsum('tji,tj->ti',torso,shoulder-p('spine1'))}
    valid=valid_rotation(pelvis)&valid_rotation(torso)
    for v in local.values():valid&=np.isfinite(v).all(1)
    receipt['observed_frames']=int(valid.sum())
    if valid.sum()<2:
        return None,dict(receipt,status='unavailable',reason='insufficient_nondegenerate_frames')
    vectors={k:np.median(v[valid],axis=0) for k,v in local.items()}
    if any(np.linalg.norm(v)<1e-6 for v in vectors.values()):
        return None,dict(receipt,status='unavailable',reason='degenerate_fixed_vector')
    receipt.update(status='estimated_effective_centres',local_vectors_m={k:v.tolist() for k,v in vectors.items()},
                   local_vector_quantiles_m={k:np.percentile(v[valid],[5,50,95],axis=0).tolist() for k,v in local.items()})
    return vectors,receipt


def register(model,motion):
    import opensim as osim
    import biomech
    vectors,receipt=estimate(motion)
    if vectors is None:return receipt
    def transform(frame,state):
        tf=frame.getTransformInGround(state)
        return (np.array([[tf.R().get(i,j) for j in range(3)] for i in range(3)]),
                np.array([tf.p().get(i) for i in range(3)]))
    state=model.initSystem()
    hip=(biomech._child_origin(model,state,'hip_l')+biomech._child_origin(model,state,'hip_r'))/2
    width=np.linalg.norm(biomech._child_origin(model,state,'acromial_l')-biomech._child_origin(model,state,'acromial_r'))
    rp,_=transform(model.getBodySet().get('pelvis'),state)
    rt,_=transform(model.getBodySet().get('torso'),state)
    back=hip+rp@vectors['proximal'];mid=back+rt@vectors['distal']
    targets={'back':back,'acromial_l':mid+rt@np.array([0,0,-width/2]),
             'acromial_r':mid+rt@np.array([0,0,width/2])}
    changes={}
    for joint,target in targets.items():
        state=model.initSystem()
        frame=osim.PhysicalOffsetFrame.safeDownCast(model.getJointSet().get(joint).getParentFrame())
        if frame is None:raise ValueError('Effective centre registration requires existing parent offset frame')
        r,o=transform(frame.getParentFrame(),state);local=r.T@(target-o)
        before=[frame.get_translation().get(i) for i in range(3)]
        frame.set_translation(osim.Vec3(*local));model.finalizeConnections();state=model.initSystem()
        actual=biomech._child_origin(model,state,joint)
        if np.linalg.norm(actual-target)>1e-8:raise ValueError('Effective centre registration FK mismatch')
        changes[joint]={'parent_translation_before_m':before,'parent_translation_after_m':local.tolist(),
                        'target_neutral_world_m':target.tolist(),'fk_neutral_world_m':actual.tolist()}
    return dict(receipt,changes=changes,effective_shoulder_width_m=float(width),
                retained='Axes, DOFs, child frames, coupling, limits and shoulder width unchanged; only three parent translations modified')
