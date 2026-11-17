"""Input-supported direction refinement of an existing IK solution; no reference access."""
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from .geometry import unit


def refine(model_path,motion,times,labels,data,settings):
    if settings.get('rescue_after_baseline',False):
        baseline=dict(settings,rescue_after_baseline=False,interior_rescue=False,
                      max_nfev=settings['baseline_max_nfev'],
                      continue_arm_branch=False,accept_continued_arm=False)
        initial,first=refine(model_path,motion,times,labels,data,baseline)
        rescue=dict(settings,rescue_after_baseline=False)
        result,second=refine(model_path,motion,times,labels,initial,rescue)
        second['baseline_pass']=first
        second['strategy']='baseline_then_input_only_interior_rescue'
        return result,second
    import opensim as osim
    import biomech
    model=osim.Model(str(model_path));state=model.initSystem();cs=model.getCoordinateSet()
    # These are coordinate chart ranges, not a claim about muscle validity.
    for side in 'lr':
        c=cs.get('arm_flex_'+side);c.setRangeMin(-np.pi);c.setRangeMax(np.pi)
    model.finalizeConnections();model.printToXML(str(model_path))
    state=model.initSystem()
    points,names,_=motion.legacy();p={n:points[:,i] for i,n in enumerate(names)}
    source_times=motion.times-motion.times[0]
    index=np.unique(np.r_[np.arange(0,len(times),max(1,int(round(100/settings['refinement_hz'])))),len(times)-1]).astype(int)
    out=data[index].copy();query=times[index]
    point=lambda n:np.stack([np.interp(query,source_times,p[n][:,j]) for j in range(3)],1)
    targets={n:point(n) for n in names}
    groups=[('pelvis',['pelvis_tilt','pelvis_list','pelvis_rotation'],['hip_l','hip_r'],['left_hip','right_hip']),
            ('torso',['lumbar_extension','lumbar_bending','lumbar_rotation'],['hip_l','hip_r','acromial_l','acromial_r'],['left_hip','right_hip','left_shoulder','right_shoulder'])]
    for side,s in [('left','l'),('right','r')]:
        groups.extend([(side+'_arm',[n+'_'+s for n in ['arm_flex','arm_add','arm_rot','elbow_flex']],
                        [n+'_'+s for n in ['acromial','elbow','radius_hand']],[side+'_'+n for n in ['shoulder','elbow','wrist']]),
                       (side+'_leg',[n+'_'+s for n in ['hip_flexion','hip_adduction','hip_rotation','knee_angle','ankle_angle','subtalar_angle']],
                        [n+'_'+s for n in ['hip','walker_knee','ankle','mtp']],[side+'_'+n for n in ['hip','knee','ankle','foot']])])
    from .constraints import coordinate_contract
    constraint_receipt=coordinate_contract(model,state,[n for _,coords,_,_ in groups for n in coords])
    diagnostics={};previous={}
    def directions(a,group):
        if group=='torso':return unit(np.stack([.5*(a[2]+a[3]-a[0]-a[1]),a[3]-a[2]]))
        return unit(np.diff(a,axis=0))
    for group,coord_names,joints,source_names in groups:
        if settings.get('independent_coordinates_only',False):
            coord_names=[n for n in coord_names if not cs.get(n).getLocked(state) and not cs.get(n).isConstrained(state)]
        coords=[cs.get(n) for n in coord_names];columns=[labels.index(n) for n in coord_names]
        lo=np.array([c.getRangeMin() for c in coords])+1e-7;hi=np.array([c.getRangeMax() for c in coords])-1e-7
        errors=[];failures=0
        for i,t in enumerate(query):
            biomech._set(model,state,labels,out[i])
            target=np.stack([targets[n][i] for n in source_names]);wanted=directions(target,group)
            if not np.isfinite(wanted).all():failures+=1;continue
            x0=np.clip(np.deg2rad(out[i,columns]),lo,hi);prior=previous.get(group,x0)
            def residual(x):
                for c,v in zip(coords,x):c.setValue(state,float(v),False)
                model.realizePosition(state)
                actual=np.array([biomech._child_origin(model,state,j) for j in joints]);v=directions(actual,group)
                return np.r_[settings['direction_weight']*(v-wanted).ravel(),
                             settings['position_weight']*(actual-target).ravel(),
                             settings['continuity_weight']*(x-prior),.001*(x-x0)]
            initial=residual(x0)
            candidates=[(float(initial@initial),x0)]
            seeds=[x0]
            if group in previous:seeds.append(np.clip(previous[group],lo,hi))
            # Opt-in development rescue: interior starts avoid a boundary seed's
            # tiny trust region. All ranking uses the observed input only.
            rescue=settings.get('interior_rescue',False) and group.endswith(('arm','leg'))
            if rescue and float(initial@initial)>.01:
                rng=np.random.default_rng(1729)
                seeds.extend(lo+(hi-lo)*f for f in rng.uniform(.15,.85,(6,len(x0))))
            if settings.get('geometry_knee_seed',False) and group.endswith('leg'):
                from .solver_seeds import knee_seed
                seed=knee_seed(x0,coord_names,target,lo,hi)
                if seed is not None:seeds.append(seed)
            # Prefer the better observed-input seed; retry the other only for a large residual.
            seeds.sort(key=lambda x:float(residual(x)@residual(x)))
            # Continue the preceding coordinate chart before considering a new
            # branch. A small input residual accepts it; large residuals still
            # trigger the ordinary multistart rescue. No reference is consulted.
            if settings.get('continue_arm_branch',False) and group.endswith('arm') and group in previous:
                seeds.insert(0,np.clip(previous[group],lo,hi))
            for seed_index,seed in enumerate(seeds):
                if seed_index and min(v[0] for v in candidates)<.01 and not (settings.get('geometry_knee_seed',False) and group.endswith('leg')):break
                sol=least_squares(residual,seed,bounds=(lo,hi),max_nfev=settings['max_nfev'],ftol=1e-5,xtol=1e-5,gtol=1e-5)
                candidates.append((float(sol.fun@sol.fun),sol.x))
                if (seed_index==0 and settings.get('continue_arm_branch',False)
                        and group.endswith('arm') and group in previous
                        and settings.get('accept_continued_arm',False)
                        and float(sol.fun@sol.fun)<.01):
                    # Accept the already adequate continuation rather than a
                    # marginally closer seed on a distant coordinate branch.
                    candidates=[candidates[-1]]
                    break
            best=min(candidates,key=lambda x:x[0])[1]
            previous[group]=best;out[i,columns]=np.rad2deg(best)
            errors.append(float(np.linalg.norm(residual(best)[:wanted.size])))
        diagnostics[group]={'optimized_frames':len(errors),'missing_frames':failures,'direction_residual_p95':float(np.percentile(errors,95)) if errors else None}
    # Interpolate the optimized coordinates continuously; no post-hoc angle copying.
    refined=PchipInterpolator(query,out,axis=0)(times)
    coupling={'status':'legacy_preserved; dependent coordinates may retain pre-refinement values'}
    if settings.get('enforce_coordinate_couplers',False):
        from .constraints import synchronize_couplers
        refined,coupling=synchronize_couplers(model_path,model,labels,refined)
    return refined,{'strategy':'input_bone_directions_with_weak_position_and_continuity','sample_frames':len(index),
                    'groups':diagnostics,'settings':settings,'model_constraint_contract':constraint_receipt,
                    'coordinate_coupling':coupling,
                    'anatomical_scope':'joint-centre directions; unobserved axial twist retains seed prior'}
