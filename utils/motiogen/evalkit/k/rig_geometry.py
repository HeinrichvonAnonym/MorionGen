"""Declared rest-rig joint centres for clip-fixed kinematic geometry only."""
import numpy as np


def transport(rig):
    rest=np.asarray(rig['neutral_joints']);index={n:i for i,n in enumerate(rig['names'])}
    left=rest[index['left_hip']]-rest[index['right_hip']];left/=np.linalg.norm(left)
    up=np.array([0.,1,0]);up-=left*np.dot(left,up);up/=np.linalg.norm(up)
    return np.array([[0.,0,1],[0,1,0],[-1,0,0]])@np.stack([left,up,np.cross(left,up)],1).T


def register_centres(model,rig):
    """Move existing parent offset frames once; keep axes, child frames and DOFs."""
    import opensim as osim
    import biomech
    if rig is None:return {'status':'unavailable','reason':'Position profile has no declared neutral rig; existing model prior retained'}
    rest=np.asarray(rig['neutral_joints']);index={n:i for i,n in enumerate(rig['names'])}
    hip=(rest[index['left_hip']]+rest[index['right_hip']])/2;rotation=transport(rig)
    state=model.initSystem();center=(biomech._child_origin(model,state,'hip_l')+biomech._child_origin(model,state,'hip_r'))/2
    spine='Spine' if 'Spine' in index else 'spine1';records={}
    for joint,node in [('back',spine),('acromial_l','left_shoulder'),('acromial_r','right_shoulder')]:
        if node not in index:raise ValueError('Declared rig lacks joint-centre correspondence: '+node)
        state=model.initSystem();frame=model.updJointSet().get(joint).getParentFrame()
        offset=osim.PhysicalOffsetFrame.safeDownCast(frame)
        if offset is None:raise ValueError('Joint parent must be an explicit offset frame')
        parent=offset.getParentFrame();tf=parent.getTransformInGround(state)
        r=np.array([[tf.R().get(i,j) for j in range(3)] for i in range(3)]);o=np.array([tf.p().get(i) for i in range(3)])
        target=center+rotation@(rest[index[node]]-hip);local=r.T@(target-o)
        before=[offset.get_translation().get(i) for i in range(3)]
        offset.set_translation(osim.Vec3(*local));model.finalizeConnections()
        state=model.initSystem();actual=biomech._child_origin(model,state,joint)
        if np.linalg.norm(actual-target)>1e-8:raise ValueError('Declared joint-centre registration FK mismatch')
        records[joint]={'source_node':node,'parent_translation_before_m':before,'parent_translation_after_m':local.tolist(),
                        'target_neutral_world_m':target.tolist(),'fk_neutral_world_m':actual.tolist()}
    return {'status':'declared_rig_centres','changes':records,'fixed_for_clip':True,
            'scope':'Input-rig neutral joint centres; existing joint axes, DOFs, ranges and child frames unchanged; kinematic only'}


def register_foot_vectors(model,rig):
    """Match declared neutral ankle-to-toe-base vectors, preserving ankle centres."""
    import opensim as osim
    import biomech
    if rig is None:return {'status':'unavailable','reason':'No declared neutral foot vector; scaled asset retained'}
    rest=np.asarray(rig['neutral_joints']);index={n:i for i,n in enumerate(rig['names'])};basis=transport(rig);records={}
    for side,s in [('left','l'),('right','r')]:
        state=model.initSystem();ankle=biomech._child_origin(model,state,'ankle_'+s)
        vector=basis@(rest[index[side+'_foot']]-rest[index[side+'_ankle']])
        if np.linalg.norm(vector)<1e-6:raise ValueError('Degenerate declared foot vector')
        frame=osim.PhysicalOffsetFrame.safeDownCast(model.getJointSet().get('mtp_'+s).getParentFrame())
        if frame is None:raise ValueError('MTP requires an existing parent offset frame')
        tf=frame.getParentFrame().getTransformInGround(state)
        r=np.array([[tf.R().get(i,j) for j in range(3)] for i in range(3)]);o=np.array([tf.p().get(i) for i in range(3)])
        before=biomech._child_origin(model,state,'mtp_'+s)-ankle
        local=r.T@(ankle+vector-o);frame.set_translation(osim.Vec3(*local));model.finalizeConnections();state=model.initSystem()
        actual=biomech._child_origin(model,state,'mtp_'+s)-biomech._child_origin(model,state,'ankle_'+s)
        np.testing.assert_allclose(actual,vector,atol=1e-9,rtol=0)
        records[side]={'asset_neutral_vector_m':before.tolist(),'declared_neutral_vector_m':vector.tolist(),'fk_neutral_vector_m':actual.tolist(),
                       'parent_translation_m':local.tolist()}
    return {'status':'declared_rig_foot_vectors','changes':records,'fixed_for_clip':True,
            'scope':'Ankle-to-toe-base point geometry; axes, DOFs, ranges and ankle centres retained; no force/contact qualification'}
