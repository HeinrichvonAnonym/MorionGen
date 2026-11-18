"""Optional D-only joint optimization of native poses on a fixed body model."""
import numpy as np
from scipy.optimize import least_squares
from .trajectory_objective import TrajectoryObjective


def problem(model_path, labels, coordinates, observed, names, times, settings, *, attached_targets=None):
    if settings.get('version')!='joint_trajectory_v1':raise ValueError('Unknown trajectory policy')
    import opensim as osim
    import biomech
    from .constraints import couplers,synchronize
    from .candidate_quality import DIRECTIONS
    model=osim.Model(str(model_path));state=model.initSystem();cs=model.getCoordinateSet();bindings=couplers(model,state)
    body_names=[n for _,_,n in biomech.MARKERS.values()]
    joints=[j for j,_,_ in biomech.MARKERS.values()]
    index={n:i for i,n in enumerate(names)}
    target=np.asarray(observed)[:,[index[n] for n in body_names]]
    active=[n for n in labels if cs.contains(n) and cs.get(n).getMotionType()==1 and not cs.get(n).getDefaultLocked() and not cs.get(n).isDependent(state)]
    cols=[list(labels).index(n) for n in active];handles=[cs.get(n) for n in active]
    lower=np.array([h.getRangeMin() for h in handles]);upper=np.array([h.getRangeMax() for h in handles])
    ni={n:i for i,n in enumerate(body_names)}
    edges=[(a,b) for name,(a,b) in DIRECTIONS.items() if name!='head_neck_axis']
    def features(p):
        d={n:p[i] for n,i in ni.items()}
        d['pelvis_mid']=(d['left_hip']+d['right_hip'])/2
        d['shoulder_mid']=(d['left_shoulder']+d['right_shoulder'])/2
        vectors=np.array([d[b]-d[a] for a,b in edges]);length=np.linalg.norm(vectors,axis=-1,keepdims=True)
        directions=np.where(length>1e-9,vectors/np.maximum(length,1e-9),np.nan)
        angles=[]
        for side in ('left','right'):
            for a,b,c in [('hip','knee','ankle'),('shoulder','elbow','wrist')]:
                u=d[side+'_'+a]-d[side+'_'+b];v=d[side+'_'+c]-d[side+'_'+b]
                angles.append(np.pi-np.arctan2(np.linalg.norm(np.cross(u,v)),np.dot(u,v)) if min(np.linalg.norm(u),np.linalg.norm(v))>1e-9 else np.nan)
        return np.r_[(p-d['pelvis_mid']).ravel(),directions.ravel(),angles]
    goals=np.array([features(p) for p in target]);np_=len(body_names)*3;nd=len(edges)*3
    scale_position=float(settings.get('position_scale_m',.05));scale_angle=np.deg2rad(float(settings.get('angle_scale_deg',5)))
    scales=np.r_[np.full(np_,scale_position),np.full(nd,2*np.sin(scale_angle/2)),np.full(4,scale_angle)]
    groups=[(i,i+3) for i in range(0,np_+nd,3)]+[(i,i+1) for i in range(np_+nd,np_+nd+4)]
    body=None
    if attached_targets is not None:
        if not np.array_equal(attached_targets.times,np.asarray(times)):
            raise ValueError('Attached targets and body objective clocks differ')
        if attached_targets.names!=('neck','head'):
            raise ValueError('Declare existing neck/head point semantics')
        pelvis=(target[:,ni['left_hip']]+target[:,ni['right_hip']])/2
        extra=(attached_targets.observed-pelvis[:,None,:]).reshape(len(times),6)
        start=goals.shape[1];goals=np.c_[goals,extra]
        scales=np.r_[scales,np.full(6,scale_position)]
        groups.extend([(start,start+3),(start+3,start+6)])
        body=model.getBodySet().get('torso')
    seed=np.asarray(coordinates,float).copy()
    def feature(frame,x):
        biomech._set(model,state,labels,seed[frame])
        for h,value in zip(handles,x):h.setValue(state,float(value),False)
        synchronize(state,bindings);model.realizePosition(state)
        points=np.array([biomech._child_origin(model,state,j) for j in joints])
        values=features(points)
        if attached_targets is not None:
            transform=body.getTransformInGround(state)
            origin=np.array([transform.p().get(j) for j in range(3)])
            rotation=np.array([[transform.R().get(j,k) for k in range(3)] for j in range(3)])
            pelvis=(points[ni['left_hip']]+points[ni['right_hip']])/2
            values=np.r_[values,(attached_targets.positions(frame,origin,rotation)-pelvis).ravel()]
        return values
    objective=TrajectoryObjective(feature,goals,times,scales,np.arange(np_,np_+nd),
        np.deg2rad(float(settings.get('direction_rate_scale_deg_s',10))),float(settings.get('direction_rate_weight',1)),lower,upper,
        violation_groups=groups,
        violation_weight=float(settings.get('target_violation_weight',0)),
        violation_margin=float(settings.get('target_margin_fraction',.8)))
    x0=np.clip(np.deg2rad(seed[:,cols]),lower+1e-9,upper-1e-9).ravel()
    return objective,x0,seed,cols,active


def refine(model_path, labels, coordinates, observed, names, times, settings, *, attached_targets=None):
    chart=None
    if settings.get('equivalent_seed_chart'):
        if settings['equivalent_seed_chart']!='max_margin_v1':raise ValueError('Unknown equivalent seed chart')
        from .rotation_chart import rechart
        coordinates,chart=rechart(model_path,labels,coordinates)
    objective,x0,seed,cols,active=problem(model_path,labels,coordinates,observed,names,times,settings,attached_targets=attached_targets)
    lower,upper=objective.lower,objective.upper
    r0=objective.residual(x0);initial=float(np.dot(r0,r0))
    solver=settings.get('solver','trf_lsmr')
    if solver=='trf_lsmr':
        fit=least_squares(objective.residual,x0,jac=objective.jacobian,bounds=(np.tile(lower,len(seed)),np.tile(upper,len(seed))),
                          method='trf',tr_solver='lsmr',x_scale=.1,max_nfev=int(settings.get('max_nfev',20)))
    elif solver=='bounded_gauss_newton':
        from .trajectory_solver import solve
        fit=solve(objective,x0,settings)
    else:raise ValueError('Unknown native trajectory solver')
    candidate=seed.copy();candidate[:,cols]=np.rad2deg(fit.x.reshape(len(seed),len(cols)))
    cost=float(np.dot(fit.fun,fit.fun));accepted=np.isfinite(cost) and cost<=initial+1e-8
    receipt=dict(settings=settings,active_coordinates=active,initial_squared_residual=initial,candidate_squared_residual=cost,
                 selected='joint_trajectory' if accepted else 'input_seed',optimizer_success=bool(fit.success),optimizer_status=int(fit.status),
                 message=str(fit.message),nfev=int(fit.nfev),njev=int(fit.njev),optimality=float(fit.optimality),
                 solver=solver,iterations=fit.get('history',[]),seed_chart=chart,
                 supported_features_per_frame=objective.mask.sum(1).tolist(),
                 scope='Fixed model; all native frames optimized jointly using D positions, directions and geometric angles. Rate is derivative of fitted-minus-input unit directions, never raw motion speed or Euler differences. R is absent. Cost acceptance is not accuracy qualification.')
    if attached_targets is not None:receipt['attached_node_targets']=attached_targets.receipt
    return candidate if accepted else seed, candidate, receipt
