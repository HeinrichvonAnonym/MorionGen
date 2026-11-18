"""Experimental input-only pose refinement on a fixed OpenSim model.

Directions use a 5-degree chord scale and positions use 5 cm. Both are explicit
engineering weights. The supplied model's declared ranges/locks are respected.
"""
import numpy as np
from scipy.optimize import least_squares


def bounded_jacobian(function,x,lower,upper,step):
    """Absolute radians avoid vanishing relative steps at a coordinate bound."""
    if not np.isfinite(step) or step<=0:raise ValueError('Positive finite angular difference step required')
    columns=[]
    for j in range(len(x)):
        left=x.copy();right=x.copy();left[j]=max(lower[j],x[j]-step);right[j]=min(upper[j],x[j]+step)
        if right[j]<=left[j]:raise ValueError('Degenerate active coordinate interval')
        columns.append((function(right)-function(left))/(right[j]-left[j]))
    return np.stack(columns,axis=1)


def refine(model_path, labels, coordinates, points, names, settings, orientation_targets=None,times=None,fixed_frames=None):
    if fixed_frames is None:fixed_frames=np.zeros(len(coordinates),dtype=bool)
    fixed_frames=np.asarray(fixed_frames)
    if fixed_frames.dtype!=np.bool_ or fixed_frames.shape!=(len(coordinates),):
        raise ValueError('Fixed-frame mask must be boolean and match the coordinate clock')
    traversal=settings.get('traversal','forward')
    if traversal not in ('forward','reverse'):
        raise ValueError('Unknown refinement traversal')
    if traversal=='reverse':
        # Offline whole-clip candidate: use later observed geometry to seed
        # earlier ambiguous poses. Negating the reversed physical clock keeps
        # positive intervals; output samples remain on their original clock.
        inner={**settings,'traversal':'forward'}
        reversed_targets={body:value[::-1] for body,value in (orientation_targets or {}).items()}
        reversed_times=None if times is None else -np.asarray(times)[::-1]
        result,receipt=refine(model_path,labels,np.asarray(coordinates)[::-1],
                              np.asarray(points)[::-1],names,inner,reversed_targets,reversed_times,fixed_frames[::-1])
        receipt['frames']=receipt['frames'][::-1]
        receipt['settings']=dict(settings)
        receipt['traversal']={'direction':'reverse','output_order':'original input order',
                              'scope':'Offline candidate using future D samples; no reference or window-specific selection'}
        return result[::-1].copy(),receipt
    import opensim as osim
    import biomech
    model=osim.Model(str(model_path));state=model.initSystem();cs=model.getCoordinateSet()
    from .constraints import couplers,synchronize
    bindings=couplers(model,state)
    model_names=[source for _,_,source in biomech.MARKERS.values()]
    joints=[model.getJointSet().get(joint).getChildFrame() for joint,_,_ in biomech.MARKERS.values()]
    index={n:j for j,n in enumerate(names)}
    target=points[:,[index[n] for n in model_names]]
    ni={n:j for j,n in enumerate(model_names)}
    edges=[(ni[side+'_'+a],ni[side+'_'+b]) for side in ('left','right')
           for a,b in [('hip','knee'),('knee','ankle'),('ankle','foot'),('shoulder','elbow'),('elbow','wrist')]]
    hip=[ni['left_hip'],ni['right_hip']];shoulder=[ni['left_shoulder'],ni['right_shoulder']]
    def vectors(p):
        values=[p[b]-p[a] for a,b in edges]+[p[shoulder].mean(0)-p[hip].mean(0),p[shoulder[1]]-p[shoulder[0]]]
        if settings.get('include_pelvis_line',False):values.append(p[hip[1]]-p[hip[0]])
        return np.vstack(values)
    def unit(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)
    triples=[(ni[side+'_'+a],ni[side+'_'+b],ni[side+'_'+c]) for side in ('left','right')
             for a,b,c in [('hip','knee','ankle'),('shoulder','elbow','wrist')]]
    def joint_angles(p):
        values=[]
        for a,b,c in triples:
            u,v=p[a]-p[b],p[c]-p[b]
            values.append(np.pi-np.arctan2(np.linalg.norm(np.cross(u,v)),np.dot(u,v)))
        return np.asarray(values)
    label_index={n:i for i,n in enumerate(labels)}
    active=[n for n in labels if cs.contains(n) and not cs.get(n).getDefaultLocked()
            and not cs.get(n).isDependent(state) and cs.get(n).getMotionType()==1]
    requested=settings.get('active_coordinates')
    if requested is not None:
        if not requested or len(set(requested))!=len(requested) or not set(requested)<=set(active):
            raise ValueError('Requested refinement coordinates must be distinct active angular coordinates')
        active=[n for n in active if n in requested]
    # Coordinate limits come from this model, including its explicitly free heading.
    lo=np.array([cs.get(n).getRangeMin() for n in active]);hi=np.array([cs.get(n).getRangeMax() for n in active])
    cols=[label_index[n] for n in active]
    handles=[cs.get(n) for n in active]
    neutral=np.array([h.getDefaultValue() for h in handles])
    extra_start=settings.get('extra_start','none')
    if extra_start not in ('none','neutral_knee_if_target_failed'):raise ValueError('Unknown refinement restart policy')
    position_scale=float(settings.get('position_scale_m',.05))
    direction_scale=2*np.sin(np.deg2rad(float(settings.get('direction_scale_deg',5)))/2)
    jw=float(settings.get('joint_angle_weight',0.))
    vw=float(settings.get('target_violation_weight',0.));margin=float(settings.get('target_margin_fraction',1.))
    pw=float(settings.get('position_weight',.25));cw=float(settings.get('continuity_weight',.02))
    tw=float(settings.get('direction_continuity_weight',0.))
    rate_scale=np.deg2rad(float(settings.get('direction_rate_scale_deg_s',10.)))
    rate_weight=float(settings.get('direction_rate_violation_weight',0.))
    rate_margin=float(settings.get('direction_rate_margin_fraction',.8))
    if rate_weight:
        from .temporal_targets import direction_error_rate,rate_excess_residual
        rate_excess_residual(np.zeros((1,3)),rate_scale,rate_margin,rate_weight)
    transport_settings=settings.get('leg_axial_transport')
    transport_weight=float(transport_settings.get('weight',0.)) if transport_settings else 0.
    transport_rate=np.deg2rad(float(transport_settings.get('rate_scale_deg_s',90.))) if transport_settings else 1.
    transport_threshold=float(transport_settings.get('weak_knee_flexion_deg',30.)) if transport_settings else 30.
    if transport_settings:
        from .branch_continuity import transport_direction,weak_flexion_weight
        weak_flexion_weight(0.,transport_threshold)
        if transport_weight<0 or not np.isfinite(transport_weight) or transport_rate<=0 or not np.isfinite(transport_rate):
            raise ValueError('Finite nonnegative transport weight and positive rate scale required')
    temporal_active=bool(tw or rate_weight or transport_weight)
    if temporal_active:
        times=np.asarray(times)
        if times.shape!=(len(coordinates),) or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or rate_scale<=0:
            raise ValueError('Physical input clock required for direction continuity')
    pelvis_weight=float(settings.get('pelvis_line_weight',1.))
    native_weight=float(settings.get('native_frame_weight',0.))
    orientation_targets=orientation_targets or {}
    for body,target_rotation in orientation_targets.items():
        if target_rotation.shape!=(len(coordinates),3,3):raise ValueError('Native body target clock/layout mismatch')
        if body not in ('pelvis','torso'):raise ValueError('Undeclared body target')
    result=np.array(coordinates,copy=True);receipts=[];previous=None;knee_seed_tables=None;previous_direction_error=None
    previous_body_rotations=None;previous_input_directions=None
    def body_rotations():
        result=[]
        for side in ('l','r'):
            rot=model.getBodySet().get('femur_'+side).getTransformInGround(state).R()
            result.append(np.array([[rot.get(i,j) for j in range(3)] for i in range(3)]))
        return result
    for frame,(row,goal) in enumerate(zip(coordinates,target)):
        if not np.isfinite(goal).all():
            receipts.append({'status':'input_missing','selected':'nominal'});previous=None;previous_direction_error=None
            previous_body_rotations=None;previous_input_directions=None;continue
        biomech._set(model,state,labels,row)
        x0=np.clip(np.deg2rad(row[cols]),lo+1e-9,hi-1e-9)
        gu=unit(vectors(goal));gp=goal-goal[hip].mean(0);ga=joint_angles(goal)
        transported=[]
        if transport_weight and previous_body_rotations is not None:
            for side,edge,joint in [(0,0,0),(1,5,2)]:
                rotation=transport_direction(previous_input_directions[edge],gu[edge])
                if rotation is not None:
                    transported.append((side,rotation@previous_body_rotations[side],
                                        weak_flexion_weight(ga[joint],transport_threshold)))
        def evaluate(x):
            biomech._set(model,state,labels,row)
            for h,v in zip(handles,x):h.setValue(state,float(v),False)
            synchronize(state,bindings);model.realizePosition(state)
            return np.array([[joint.getPositionInGround(state).get(k) for k in range(3)] for joint in joints])
        def residual(x,continuity=True):
            p=evaluate(x);u=unit(vectors(p));local=p-p[hip].mean(0)
            directional=(u-gu)/direction_scale;positional=(local-gp)/position_scale
            weighted=directional.copy()
            if settings.get('include_pelvis_line',False):weighted[-1]*=pelvis_weight
            terms=[weighted.ravel(),(pw*positional).ravel()]
            if jw:
                angular=(joint_angles(p)-ga)/np.deg2rad(float(settings.get('direction_scale_deg',5)))
                terms.append(jw*angular)
            if vw:
                # Emphasize each declared input target; score/reference never enter.
                violation=[np.maximum(np.linalg.norm(directional,axis=-1)-margin,0),
                           np.maximum(np.linalg.norm(positional,axis=-1)-margin,0)]
                if jw:violation.append(np.maximum(abs(angular)-margin,0))
                terms.append(vw*np.concatenate(violation))
            if native_weight:
                for body,rotations in orientation_targets.items():
                    goal_rotation=rotations[frame]
                    if not np.isfinite(goal_rotation).all():continue
                    tf=model.getBodySet().get(body).getTransformInGround(state)
                    current_rotation=np.array([[tf.R().get(i,j) for j in range(3)] for i in range(3)])
                    terms.append((native_weight*(current_rotation-goal_rotation)/(np.sqrt(2)*direction_scale)).ravel())
            if continuity and previous is not None:terms.append(cw*(x-previous)/direction_scale)
            if continuity and transported:
                rotations=body_rotations()
                dt=times[frame]-times[frame-1]
                for side,target_rotation,weight in transported:
                    # D transports the observed direction; only the weakly
                    # observed axial gauge is regularized by a temporal prior.
                    terms.append((transport_weight*np.sqrt(weight)*(rotations[side]-target_rotation)
                                  /(np.sqrt(2)*transport_rate*dt)).ravel())
            if continuity and temporal_active and previous_direction_error is not None:
                # Penalize added direction velocity, not the input motion itself.
                if tw:terms.append((tw*((u-gu)-previous_direction_error)/(rate_scale*(times[frame]-times[frame-1]))).ravel())
                if rate_weight:
                    error_rate=direction_error_rate(u-gu,previous_direction_error,times[frame]-times[frame-1])
                    terms.append(rate_excess_residual(error_rate,rate_scale,rate_margin,rate_weight))
            return np.concatenate(terms)
        if fixed_frames[frame]:
            chosen=np.deg2rad(row[cols])
            if np.any(chosen<lo-1e-8) or np.any(chosen>hi+1e-8):
                raise ValueError('Fixed native anchor outside declared coordinate domain')
            final_points=evaluate(chosen)
            previous=chosen.copy();previous_direction_error=unit(vectors(final_points))-gu
            if transport_weight:
                previous_body_rotations=body_rotations();previous_input_directions=gu.copy()
            receipts.append({'status':'fixed_native_anchor','selected':'unchanged_native_coordinates'})
            continue
        initial_loss=float(np.dot(residual(x0,False),residual(x0,False)))
        def selection_loss(x,pose_loss):
            value=residual(x,True) if temporal_active else None
            return float(np.dot(value,value)) if temporal_active else pose_loss
        candidates=[('nominal',x0,initial_loss,selection_loss(x0,initial_loss))]
        starts=[('nominal_start',x0)]
        if previous is not None:starts.append(('continuity_start',np.clip(previous,lo+1e-9,hi-1e-9)))
        attempts=[]
        def attempt(name,start):
            jacobian=settings.get('jacobian','relative_forward')
            if jacobian not in ('relative_forward','absolute_central'):raise ValueError('Unknown Jacobian policy')
            options=({'jac':lambda x:bounded_jacobian(residual,x,lo,hi,float(settings.get('jacobian_step_rad',1e-5)))}
                     if jacobian=='absolute_central' else {'diff_step':1e-4})
            opt=least_squares(residual,start,bounds=(lo,hi),max_nfev=settings.get('max_nfev',12),
                              ftol=1e-5,xtol=1e-5,gtol=1e-5,**options)
            err=residual(opt.x,False);loss=float(np.dot(err,err))
            attempts.append({'start':name,'nfev':int(opt.nfev),'status':int(opt.status),
                             'optimality':float(opt.optimality),'cost_with_continuity':float(opt.cost),
                             'input_loss':loss})
            if np.isfinite(loss):candidates.append((name,opt.x,loss,selection_loss(opt.x,loss)))
        for name,start in starts:attempt(name,start)
        restart_triggered=False
        if extra_start=='neutral_knee_if_target_failed':
            best=min(candidates,key=lambda c:c[3]);p=evaluate(best[1])
            direction_error=np.linalg.norm(unit(vectors(p))-gu,axis=-1)/direction_scale
            position_error=np.linalg.norm((p-p[hip].mean(0))-gp,axis=-1)/position_scale
            angle_error=abs(joint_angles(p)-ga)/np.deg2rad(float(settings.get('direction_scale_deg',5)))
            restart_triggered=bool(max(direction_error.max(),position_error.max(),angle_error.max())>1.)
            if restart_triggered:
                start=neutral.copy()
                if knee_seed_tables is None:
                    from .initialization import seed_definitions
                    _,seed_info=seed_definitions(model_path,points,names)
                    knee_seed_tables=seed_info['knee_mapping']
                for side,long_side,angle_index in [('l','left',0),('r','right',2)]:
                    name='knee_angle_'+side
                    if name in active:
                        table=knee_seed_tables.get(long_side)
                        if table:
                            index=int(np.argmin(abs(np.asarray(table['geometric_flexion_deg'])-np.rad2deg(ga[angle_index]))))
                            start[active.index(name)]=np.deg2rad(table['coordinate_deg'][index])
                        else:start[active.index(name)]=x0[active.index(name)]
                attempt('neutral_knee_start',np.clip(start,lo+1e-9,hi-1e-9))
        name,chosen,loss,selected_loss=min(candidates,key=lambda c:c[3])
        final_points=evaluate(chosen)
        for j,label in enumerate(labels):
            c=cs.get(label);result[frame,j]=c.getValue(state)*(180/np.pi if c.getMotionType()==1 else 1)
        previous=np.deg2rad(result[frame,cols])
        previous_direction_error=unit(vectors(final_points))-gu
        if transport_weight:
            previous_body_rotations=body_rotations();previous_input_directions=gu.copy()
        receipts.append({'status':'computed','selected':name,'before_loss':initial_loss,'after_loss':loss,'attempts':attempts,
                         'selection_loss':selected_loss,
                         'leg_axial_transport_weights':[float(x[2]) for x in transported],
                         'extra_start_triggered':restart_triggered})
    return result,{'frames':receipts,'active_coordinates':active,'settings':settings,
                   'neutral_restart_knee_mapping':knee_seed_tables,
                   'native_target_bodies':list(orientation_targets),'scope':'experimental local direction/position refinement; full-window regression gates required',
                   'candidate_selection':('input pose plus physical direction-error continuity/target excess; nominal retained' if temporal_active else 'input-only objective; nominal retained as a per-frame candidate')}
