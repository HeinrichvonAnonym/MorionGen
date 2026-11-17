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


def refine(model_path, labels, coordinates, points, names, settings, orientation_targets=None):
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
    native_weight=float(settings.get('native_frame_weight',0.))
    orientation_targets=orientation_targets or {}
    for body,target_rotation in orientation_targets.items():
        if target_rotation.shape!=(len(coordinates),3,3):raise ValueError('Native body target clock/layout mismatch')
        if body not in ('pelvis','torso'):raise ValueError('Undeclared body target')
    result=np.array(coordinates,copy=True);receipts=[];previous=None;knee_seed_tables=None
    for frame,(row,goal) in enumerate(zip(coordinates,target)):
        if not np.isfinite(goal).all():
            receipts.append({'status':'input_missing','selected':'nominal'});previous=None;continue
        biomech._set(model,state,labels,row)
        x0=np.clip(np.deg2rad(row[cols]),lo+1e-9,hi-1e-9)
        gu=unit(vectors(goal));gp=goal-goal[hip].mean(0);ga=joint_angles(goal)
        def evaluate(x):
            biomech._set(model,state,labels,row)
            for h,v in zip(handles,x):h.setValue(state,float(v),False)
            synchronize(state,bindings);model.realizePosition(state)
            return np.array([[joint.getPositionInGround(state).get(k) for k in range(3)] for joint in joints])
        def residual(x,continuity=True):
            p=evaluate(x);u=unit(vectors(p));local=p-p[hip].mean(0)
            directional=(u-gu)/direction_scale;positional=(local-gp)/position_scale
            terms=[directional.ravel(),(pw*positional).ravel()]
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
            return np.concatenate(terms)
        initial_loss=float(np.dot(residual(x0,False),residual(x0,False)))
        candidates=[('nominal',x0,initial_loss)]
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
            if np.isfinite(loss):candidates.append((name,opt.x,loss))
        for name,start in starts:attempt(name,start)
        restart_triggered=False
        if extra_start=='neutral_knee_if_target_failed':
            best=min(candidates,key=lambda c:c[2]);p=evaluate(best[1])
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
        name,chosen,loss=min(candidates,key=lambda c:c[2])
        evaluate(chosen)
        for j,label in enumerate(labels):
            c=cs.get(label);result[frame,j]=c.getValue(state)*(180/np.pi if c.getMotionType()==1 else 1)
        previous=np.deg2rad(result[frame,cols])
        receipts.append({'status':'computed','selected':name,'before_loss':initial_loss,'after_loss':loss,'attempts':attempts,
                         'extra_start_triggered':restart_triggered})
    return result,{'frames':receipts,'active_coordinates':active,'settings':settings,
                   'neutral_restart_knee_mapping':knee_seed_tables,
                   'native_target_bodies':list(orientation_targets),'scope':'experimental local direction/position refinement; full-window regression gates required',
                   'candidate_selection':'input-only objective; nominal retained as a per-frame candidate'}
