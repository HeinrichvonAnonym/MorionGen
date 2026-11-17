"""Input-derived, clip-fixed OpenSim geometry for kinematic-only K4 fits."""
import numpy as np
from scipy.optimize import least_squares
from .input_rig import geometry_targets


def measure(model,state):
    import biomech
    p=lambda name:biomech._child_origin(model,state,name)
    result={}
    for side,s in [('left','l'),('right','r')]:
        for part,a,b in [('thigh','hip','walker_knee'),('shank','walker_knee','ankle'),
                         ('foot','ankle','mtp'),('upper_arm','acromial','elbow'),('forearm','elbow','radius_hand')]:
            result[side+'_'+part]=float(np.linalg.norm(p(a+'_'+s)-p(b+'_'+s)))
    result['hip_width']=float(np.linalg.norm(p('hip_l')-p('hip_r')))
    result['shoulder_width']=float(np.linalg.norm(p('acromial_l')-p('acromial_r')))
    result['rest_hip_to_shoulder_midpoint']=float(np.linalg.norm((p('acromial_l')+p('acromial_r')-p('hip_l')-p('hip_r'))/2))
    return result


def scale_bodies(model, factors, mass):
    import opensim as osim
    scales=osim.ScaleSet()
    for body,factor in factors.items():
        entry=osim.Scale();entry.setSegmentName(body)
        entry.setScaleFactors(osim.Vec3(*np.broadcast_to(factor,(3,))))
        entry.setApply(True);scales.cloneAndAppend(entry)
    state=model.initSystem();model.scale(state,scales,True,mass)


def build(motion,out,mass,independent_geometry=False,marker_registration='legacy',joint_centres='scaled_asset',shoulder_width_source='declared_or_observed',position_trunk_strategy='model_prior',foot_geometry='scaled_asset'):
    import opensim as osim
    import biomech
    targets=geometry_targets(motion,shoulder_width_source);lengths=targets['lengths_m']
    if position_trunk_strategy not in ('model_prior','observed_chain_proxy_v1'):
        raise ValueError('Unknown position-input trunk geometry strategy')
    if position_trunk_strategy!='model_prior' and not motion.provenance.get('input_rig'):
        if not independent_geometry:raise ValueError('Chain proxy requires independent fixed geometry')
        from .position_geometry import trunk_chain_proxy
        estimate,description=trunk_chain_proxy(motion)
        targets['effective_trunk_estimate']=description
        if estimate is not None:
            lengths['rest_hip_to_shoulder_midpoint']=estimate
            targets['neutral_trunk_length']='unidentified; effective chain proxy used as explicit fixed model approximation'
    if shoulder_width_source!='declared_or_observed' and (not independent_geometry or joint_centres!='scaled_asset'):
        raise ValueError('Observed rigid shoulder width requires independent geometry and scaled-asset centres')
    if any(not np.isfinite(v) or v<=1e-6 for v in lengths.values()):
        raise ValueError('Fixed geometry unavailable: missing or degenerate input segments')
    model=osim.Model(str(biomech.MODEL));state=model.initSystem();base=measure(model,state)
    factors={'pelvis':lengths['hip_width']/base['hip_width']}
    groups={'thigh':['femur','patella'],'shank':['tibia'],'foot':['talus','calcn','toes'],
            'upper_arm':['humerus'],'forearm':['ulna','radius','hand']}
    for part,bodies in groups.items():
        ratios={side:lengths[side+'_'+part]/base[side+'_'+part] for side in ('left','right')}
        for side,s in [('left','l'),('right','r')]:
            factor=ratios[side] if independent_geometry else np.mean(list(ratios.values()))
            for body in bodies:factors[body+'_'+s]=factor
    neutral=lengths.get('rest_hip_to_shoulder_midpoint')
    if not independent_geometry:
        factors['torso']=neutral/base['rest_hip_to_shoulder_midpoint'] if neutral else 1.
    scale_bodies(model,factors,mass)
    torso_receipt={'policy':'uniform torso; existing left/right averaging', 'neutral_length_source':targets['source'] if neutral else 'model_prior'}
    if independent_geometry:
        # Rajagopal neutral torso local Y is superior and Z is mediolateral.
        # Fit two fixed scale factors against independently measured model FK.
        template=osim.Model(model)
        desired=np.array([neutral if neutral else measure(model,model.initSystem())['rest_hip_to_shoulder_midpoint'],lengths['shoulder_width']])
        def residual(log_scale):
            candidate=osim.Model(template);y,z=np.exp(log_scale)
            scale_bodies(candidate,{'torso':[1.,y,z]},mass)
            measured=measure(candidate,candidate.initSystem())
            return (np.array([measured['rest_hip_to_shoulder_midpoint'],measured['shoulder_width']])-desired)/desired
        fit=least_squares(residual,[0.,0.],bounds=(np.log(.25),np.log(4.)),diff_step=1e-3,max_nfev=30)
        y,z=np.exp(fit.x);scale_bodies(model,{'torso':[1.,y,z]},mass)
        torso_receipt={'policy':'independent fixed torso height and shoulder width','scale_yz':[float(y),float(z)],
                       'relative_residual':residual(fit.x).tolist(),'neutral_length_source':targets['source'] if neutral else 'model_prior',
                       'bounds':'0.25..4 engineering guard; boundary hits remain diagnostic'}
        if targets.get('effective_trunk_estimate',{}).get('status')=='estimated_effective_length':
            torso_receipt['neutral_length_source']='observed_chain_proxy; neutral length remains unidentified'
    if joint_centres not in ('scaled_asset','declared_rig'):raise ValueError('Unknown fixed joint-centre policy')
    centre_receipt={'status':'scaled_asset'}
    if joint_centres=='declared_rig':
        from .rig_geometry import register_centres
        centre_receipt=register_centres(model,motion.provenance.get('input_rig'))
    if foot_geometry not in ('scaled_asset','declared_rig'):
        raise ValueError('Unknown fixed foot geometry policy')
    foot_receipt={'status':'scaled_asset'}
    if foot_geometry=='declared_rig':
        from .rig_geometry import register_foot_vectors
        foot_receipt=register_foot_vectors(model,motion.provenance.get('input_rig'))
    # Retain anatomical joint structure. Only global position/heading are free.
    for name in ('pelvis_tx','pelvis_ty','pelvis_tz','pelvis_rotation'):
        coordinate=model.updCoordinateSet().get(name);coordinate.setRangeMin(-1e3);coordinate.setRangeMax(1e3);coordinate.setDefaultClamped(False)
    state=model.initSystem()
    if marker_registration not in ('legacy','owned_joint_frame'):
        raise ValueError('Unsupported marker registration policy')
    for name,(joint,body,_) in biomech.MARKERS.items():
        if marker_registration=='owned_joint_frame':
            transform=model.getJointSet().get(joint).getChildFrame().findTransformInBaseFrame()
            location=osim.Vec3(*[transform.p().get(j) for j in range(3)])
        else:
            # Audited frozen common baseline: all 9,184 limb-marker offsets
            # across 656 saved models are zero. Reproduce that control
            # deterministically without retaining a borrowed temporary Vec3.
            location=osim.Vec3(0,0,0)
        model.addMarker(osim.Marker(name,model.getBodySet().get(body),location))
    rig=motion.provenance.get('input_rig')
    registered=[];axial=[]
    if rig:
        state=model.initSystem()
        rest=np.asarray(rig['neutral_joints']);idx={n:i for i,n in enumerate(rig['names'])}
        hip=(rest[idx['left_hip']]+rest[idx['right_hip']])/2
        # Rest-rig to model-neutral orientation uses only the declared geometry.
        left=rest[idx['left_hip']]-rest[idx['right_hip']];left/=np.linalg.norm(left)
        up=np.array([0.,1,0]);up-=left*np.dot(up,left);up/=np.linalg.norm(up)
        source_basis=np.stack([left,up,np.cross(left,up)],1)
        target_basis=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
        transform=target_basis@source_basis.T
        center=(biomech._child_origin(model,state,'hip_l')+biomech._child_origin(model,state,'hip_r'))/2
        for name,(body,node) in biomech.AXIAL.items():
            source=node if node in idx else {'Spine':'spine1','Spine3':'spine3'}.get(node,node)
            if source not in idx:continue
            world=center+transform@(rest[idx[source]]-hip)
            frame=model.getBodySet().get(body).getTransformInGround(state)
            rot=np.array([[frame.R().get(i,j) for j in range(3)]for i in range(3)])
            origin=np.array([frame.p().get(i) for i in range(3)])
            local=rot.T@(world-origin)
            axial.append((name,body,local))
            registered.append(name)
    for name,body,local in axial:
        model.addMarker(osim.Marker(name,model.getBodySet().get(body),osim.Vec3(*local)))
    model.finalizeConnections();state=model.initSystem();final=measure(model,state);model.printToXML(str(out))
    coordinates=model.getCoordinateSet()
    domain={coordinates.get(i).getName():{
        'range_native':[coordinates.get(i).getRangeMin(),coordinates.get(i).getRangeMax()],
        'locked':coordinates.get(i).getDefaultLocked(),'clamped':coordinates.get(i).getDefaultClamped(),
        'dependent':coordinates.get(i).isDependent(state),'motion_type':int(coordinates.get(i).getMotionType())}
        for i in range(coordinates.getSize())}
    marker_offsets={}
    for name,(joint,_,_) in biomech.MARKERS.items():
        marker=model.getMarkerSet().get(name).getLocationInGround(state)
        center=model.getJointSet().get(joint).getChildFrame().getPositionInGround(state)
        marker_offsets[name]=float(np.linalg.norm([marker.get(j)-center.get(j) for j in range(3)]))
    if marker_registration=='owned_joint_frame' and max(marker_offsets.values())>1e-9:
        raise ValueError('Registered IK marker does not coincide with its declared joint frame')
    return {'targets':targets,'before':base,'after':final,'body_scales':factors,'torso':torso_receipt,
            'joint_centres':centre_receipt,'foot_geometry':foot_receipt,'marker_registration':marker_registration,'marker_joint_frame_offset_m':marker_offsets,
            'registered_axial_markers':registered,'kinematic_only':True,'force_eligible':False,
            'geometry_fixed_for_clip':True,'head_model_support':'no independent head joint',
            'joint_domain':domain,'joint_domain_source':'Original asset except explicitly freed global translations and heading',
            'axial_registration_basis':'Declared rig rest hip line and Y-up mapped to model neutral left=-Z, up=Y; fixed project coordinate transport'}
