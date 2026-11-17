"""Experimental input-only pose refinement on a fixed OpenSim model.

Directions use a 5-degree chord scale and positions use 5 cm. Both are explicit
engineering weights. The original model's ranges/locks and structure are kept.
"""
import numpy as np
from scipy.optimize import least_squares


def refine(model_path, labels, coordinates, points, names, settings):
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
    label_index={n:i for i,n in enumerate(labels)}
    active=[n for n in labels if cs.contains(n) and not cs.get(n).getDefaultLocked()
            and not cs.get(n).isDependent(state) and cs.get(n).getMotionType()==1]
    # Coordinate limits come from this model, including its explicitly free heading.
    lo=np.array([cs.get(n).getRangeMin() for n in active]);hi=np.array([cs.get(n).getRangeMax() for n in active])
    cols=[label_index[n] for n in active]
    handles=[cs.get(n) for n in active]
    position_scale=float(settings.get('position_scale_m',.05))
    direction_scale=2*np.sin(np.deg2rad(float(settings.get('direction_scale_deg',5)))/2)
    pw=float(settings.get('position_weight',.25));cw=float(settings.get('continuity_weight',.02))
    result=np.array(coordinates,copy=True);receipts=[];previous=None
    for frame,(row,goal) in enumerate(zip(coordinates,target)):
        if not np.isfinite(goal).all():
            receipts.append({'status':'input_missing','selected':'nominal'});previous=None;continue
        biomech._set(model,state,labels,row)
        x0=np.clip(np.deg2rad(row[cols]),lo+1e-9,hi-1e-9)
        gu=unit(vectors(goal));gp=goal-goal[hip].mean(0)
        def evaluate(x):
            biomech._set(model,state,labels,row)
            for h,v in zip(handles,x):h.setValue(state,float(v),False)
            synchronize(state,bindings);model.realizePosition(state)
            return np.array([[joint.getPositionInGround(state).get(k) for k in range(3)] for joint in joints])
        def residual(x,continuity=True):
            p=evaluate(x);u=unit(vectors(p));local=p-p[hip].mean(0)
            terms=[((u-gu)/direction_scale).ravel(),(pw*(local-gp)/position_scale).ravel()]
            if continuity and previous is not None:terms.append(cw*(x-previous)/direction_scale)
            return np.concatenate(terms)
        initial_loss=float(np.dot(residual(x0,False),residual(x0,False)))
        candidates=[('nominal',x0,initial_loss)]
        starts=[('nominal_start',x0)]
        if previous is not None:starts.append(('continuity_start',np.clip(previous,lo+1e-9,hi-1e-9)))
        for name,start in starts:
            opt=least_squares(residual,start,bounds=(lo,hi),max_nfev=settings.get('max_nfev',12),
                              diff_step=1e-4,ftol=1e-5,xtol=1e-5,gtol=1e-5)
            err=residual(opt.x,False);loss=float(np.dot(err,err))
            if np.isfinite(loss):candidates.append((name,opt.x,loss))
        name,chosen,loss=min(candidates,key=lambda c:c[2])
        evaluate(chosen)
        for j,label in enumerate(labels):
            c=cs.get(label);result[frame,j]=c.getValue(state)*(180/np.pi if c.getMotionType()==1 else 1)
        previous=np.deg2rad(result[frame,cols])
        receipts.append({'status':'computed','selected':name,'before_loss':initial_loss,'after_loss':loss})
    return result,{'frames':receipts,'active_coordinates':active,'settings':settings,
                   'scope':'experimental local direction/position refinement; full-window regression gates required',
                   'candidate_selection':'input-only objective; nominal retained as a per-frame candidate'}
