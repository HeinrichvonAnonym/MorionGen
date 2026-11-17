"""Input-only staged leg fitting with whole-window acceptance guards.

The model, ranges, locks, input landmarks and clock are unchanged. A proximal
solve initializes the joint solve; foot weighting is explicit. Candidate and
baseline are checked on the complete clock before accepting a whole leg, so
framewise switches cannot manufacture temporal discontinuities.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import PchipInterpolator
from .geometry import unit


def errors(actual,target):
    a=unit(np.diff(actual,axis=1));b=unit(np.diff(target,axis=1))
    direction=np.degrees(np.arccos(np.clip((a*b).sum(-1),-1,1)))
    knee=lambda p:np.degrees(np.arccos(np.clip((unit(p[:,1]-p[:,0])*unit(p[:,2]-p[:,1])).sum(-1),-1,1)))
    return direction,knee(actual)-knee(target),knee(actual)


def accept_window(baseline,candidate,target,times,settings):
    bd,bk,ba=errors(baseline,target);cd,ck,ca=errors(candidate,target)
    if not all(np.isfinite(x).all() for x in [bd,cd,bk,ck]):return False,{'reason':'nonfinite_geometry'}
    p95=lambda x:float(np.percentile(x,95))
    score=lambda d:p95(np.max(d[:,:2],axis=1))
    measures={'proximal_before':score(bd),'proximal_after':score(cd),
        'knee_before':p95(abs(bk)),'knee_after':p95(abs(ck)),
        'foot_direction_increase':p95(cd[:,2])-p95(bd[:,2]),
        'foot_position_increase_m':p95(np.linalg.norm(candidate[:,3]-target[:,3],axis=1))-p95(np.linalg.norm(baseline[:,3]-target[:,3],axis=1)),
        'maximum_step_increase_deg':float(abs(np.diff(ca)).max()-abs(np.diff(ba)).max()),
        'velocity_error_increase_deg_s':p95(abs(np.gradient(ck,times)))-p95(abs(np.gradient(bk,times)))}
    bdirections=unit(np.diff(baseline,axis=1));cdirections=unit(np.diff(candidate,axis=1));tdirections=unit(np.diff(target,axis=1))
    step=lambda d:float(np.max(np.degrees(np.arccos(np.clip((d[1:]*d[:-1]).sum(-1),-1,1)))))
    velocity=lambda d:p95(np.linalg.norm(np.gradient(d,times,axis=0)-np.gradient(tdirections,times,axis=0),axis=-1)*180/np.pi)
    measures['bone_step_increase_deg']=step(cdirections)-step(bdirections)
    measures['bone_velocity_error_increase_deg_s']=velocity(cdirections)-velocity(bdirections)
    valid=lambda d,k:float(np.mean((d<=settings['quality_angle_deg']).all(1)&(abs(k)<=settings['quality_angle_deg'])))
    measures['coverage_loss']=valid(bd,bk)-valid(cd,ck)
    tests={'proximal_improvement':measures['proximal_after']<=measures['proximal_before']-settings['minimum_improvement_deg'],
        'knee_nonregression':measures['knee_after']<=measures['knee_before']+settings['knee_tolerance_deg'],
        'foot_direction':measures['foot_direction_increase']<=settings['foot_tolerance_deg'],
        'foot_position':measures['foot_position_increase_m']<=settings['foot_position_tolerance_m'],
        'step':measures['maximum_step_increase_deg']<=settings['step_tolerance_deg'],
        'velocity':measures['velocity_error_increase_deg_s']<=settings['velocity_tolerance_deg_s'],
        'bone_step':measures['bone_step_increase_deg']<=settings['step_tolerance_deg'],
        'bone_velocity':measures['bone_velocity_error_increase_deg_s']<=settings['velocity_tolerance_deg_s'],
        'coverage':measures['coverage_loss']<=settings['coverage_tolerance']}
    return all(tests.values()),{'measures':measures,'checks':tests,'reason':'accepted' if all(tests.values()) else 'candidate_guard_failed'}


def refine_legs(model_path,times,labels,data,source,names,settings):
    import opensim as osim
    import biomech
    model=osim.Model(str(model_path));state=model.initSystem();cs=model.getCoordinateSet()
    out=data.copy();raw=data.copy();diagnostics={}
    step=max(1,int(round(1/(np.median(np.diff(times))*settings['hz']))))
    indices=np.unique(np.r_[np.arange(0,len(times),step),len(times)-1]).astype(int)
    for side,s in [('left','l'),('right','r')]:
        coord_names=[n+'_'+s for n in ['hip_flexion','hip_adduction','hip_rotation','knee_angle','ankle_angle','subtalar_angle']]
        coord_names=[n for n in coord_names if not cs.get(n).getLocked(state) and not cs.get(n).isConstrained(state)]
        coords=[cs.get(n) for n in coord_names];cols=[labels.index(n) for n in coord_names]
        lo=np.array([c.getRangeMin() for c in coords])+1e-7;hi=np.array([c.getRangeMax() for c in coords])-1e-7
        joints=[n+'_'+s for n in ['hip','walker_knee','ankle','mtp']]
        target=source[:,[names.index(side+'_'+n) for n in ['hip','knee','ankle','foot']]]
        optimized=np.empty((len(indices),len(coords)));previous=None;anchor_solution=None
        order=np.arange(len(indices));anchor=0;anchor_receipt={'enabled':False}
        if settings.get('anchor_bent_frame',False):
            from .branch_order import anchored_order
            order,anchor,anchor_receipt=anchored_order(target[indices],settings.get('anchor_min_plane_angle_deg',10.))
            anchor_receipt['time']=float(times[indices[anchor]])
        for k in order:
            i=indices[k]
            if k==anchor+1 and anchor_solution is not None:previous=anchor_solution.copy()
            biomech._set(model,state,labels,data[i]);wanted=unit(np.diff(target[i],axis=0));seed=np.clip(np.deg2rad(data[i,cols]),lo,hi)
            if not np.isfinite(wanted).all():optimized[k]=seed;continue
            def residual(x,foot_weight):
                for c,v in zip(coords,x):c.setValue(state,float(v),False)
                model.realizePosition(state);actual=np.array([biomech._child_origin(model,state,j) for j in joints])
                direction=(unit(np.diff(actual,axis=0))-wanted)*np.array([1.,1.,foot_weight])[:,None]
                position_delta=actual-target[i]
                if settings.get('root_relative_positions',False):
                    model_root=.5*(biomech._child_origin(model,state,'hip_l')+biomech._child_origin(model,state,'hip_r'))
                    input_root=.5*(source[i,names.index('left_hip')]+source[i,names.index('right_hip')])
                    position_delta=position_delta-(model_root-input_root)
                continuity=x-(previous if previous is not None else seed)
                return np.r_[settings.get('direction_weight',1.)*direction.ravel(),settings['position_weight']*position_delta.ravel(),settings['continuity_weight']*continuity]
            # Fixed deterministic interior starts are bounded by the same asset.
            seeds=[seed,lo+.5*(hi-lo)]
            if previous is not None:seeds.append(np.clip(previous,lo,hi))
            solutions=[least_squares(lambda x:residual(x,0.),x,bounds=(lo,hi),max_nfev=settings['max_nfev']) for x in seeds]
            proximal=min(solutions,key=lambda s:float(s.fun@s.fun)).x
            joint_seeds=[seed,proximal]
            if settings.get('continue_joint_branch',False) and previous is not None:
                joint_seeds.insert(0,np.clip(previous,lo,hi))
            joint=[least_squares(lambda x:residual(x,settings['foot_weight']),x,bounds=(lo,hi),max_nfev=settings['max_nfev']) for x in joint_seeds]
            best=min(joint,key=lambda s:float(s.fun@s.fun)).x
            optimized[k]=best;previous=best
            if k==anchor:anchor_solution=best.copy()
        values=PchipInterpolator(times[indices],np.asarray(optimized),axis=0)(times)
        candidate=data.copy();candidate[:,cols]=np.rad2deg(values);raw[:,cols]=candidate[:,cols]
        def points(q):
            result=[]
            for row in q:
                biomech._set(model,state,labels,row);result.append([biomech._child_origin(model,state,j) for j in joints])
            return np.asarray(result)
        accepted,receipt=accept_window(points(data),points(candidate),target,times,settings)
        if accepted:out[:,cols]=candidate[:,cols]
        diagnostics[side]={'accepted':accepted,'coordinates':coord_names,'anchor':anchor_receipt,**receipt}
    from .constraints import synchronize_couplers
    out,coupling=synchronize_couplers(model_path,model,labels,out)
    raw,_=synchronize_couplers(model_path,model,labels,raw)
    return out,raw,{'legs':diagnostics,'coupling':coupling,'input_contract':'existing joint positions; no reference, new fields or range changes','settings':settings}
