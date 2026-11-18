"""Input-derived fixed foot endpoint candidate, preserving joint structure.

This estimates an effective endpoint geometry, not anatomical neutral axes.
The polar envelope is a necessary reachability condition; full fitting and
independent scoring must still qualify the result.
"""
import numpy as np
from scipy.optimize import minimize
from .foot_domain import model_curve,polar_range


def search(curve,rotations,polar_angles,length,tolerance):
    c,u,v,a=[np.asarray(curve[n]) for n in ('center','cosine','sine','axis')]
    r0,rpi,rhalf=rotations;C=(r0+rpi)/2;U=(r0-rpi)/2;V=rhalf-C
    lo,hi=float(min(polar_angles)),float(max(polar_angles))
    def describe(delta):
        cc,uu,vv=c+C@delta,u+U@delta,v+V@delta
        domain=polar_range(cc,uu,vv,a);g=cc@cc+uu@uu;amp=np.hypot(2*cc@uu,2*cc@vv)
        return domain,float(np.sqrt(max(0,g-amp))),float(np.sqrt(g+amp))
    def constraints(delta):
        try:domain,mn,mx=describe(delta)
        except ValueError:return np.full(4,-10.)
        lower,upper=domain['polar_interval_rad']
        return np.array([lo-lower,upper-hi,mn-(length-tolerance),(length+tolerance)-mx])
    zero=np.zeros(3)
    if min(constraints(zero))>=-1e-7:return zero,dict(status='existing_geometry_covers_input',attempts=[],domain=describe(zero))
    attempts=[]
    for start in [zero]+[sign*length/3*np.eye(3)[i] for i in (1,2) for sign in (-1,1)]:
        fit=minimize(lambda x:float(x@x)/length**2,start,method='SLSQP',bounds=[(-length,length)]*3,
                     constraints=[dict(type='ineq',fun=constraints)],options=dict(maxiter=200,ftol=1e-12))
        attempts.append(dict(delta_m=fit.x.tolist(),optimizer_success=bool(fit.success),message=str(fit.message),
                             objective=float(fit.fun),constraints=constraints(fit.x).tolist()))
    valid=[x for x in attempts if min(x['constraints'])>=-1e-7]
    if not valid:return None,dict(status='no_feasible_fixed_endpoint_found',attempts=attempts)
    chosen=min(valid,key=lambda x:x['objective']);delta=np.array(chosen['delta_m'])
    return delta,dict(status='estimated_effective_endpoint',attempts=attempts,selected=chosen,domain=describe(delta))


def apply(model_path,motion,policy):
    import opensim as osim
    import biomech
    from .leg_seed import LegDirectionSeeder
    from .constraints import synchronize
    from .resampling.model import _matrix
    from .fixed_geometry import measure
    from .config import digest
    if policy.get('version')!='input_polar_envelope_v1':raise ValueError('Explicit foot geometry policy required')
    tolerance=float(policy['length_tolerance_m'])
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Positive fixed length tolerance required')
    if motion.provenance.get('input_rig'):
        return dict(status='declared_rig_preserved',changes={},model_sha256=digest(model_path))
    observed,names,_=motion.legacy();names=list(names)
    model=osim.Model(str(model_path));state=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    nominal=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
    records={};changes={}
    for side,suffix in [('left','l'),('right','r')]:
        required=[side+'_'+n for n in ('hip','knee','ankle','foot')]
        if any(n not in names or n in motion.legacy_prior_names() for n in required):
            records[side]=dict(status='insufficient_independent_positions');continue
        target=observed[:,[names.index(n) for n in required]]
        seeder=LegDirectionSeeder(model_path,labels,side);m,st=seeder.model,seeder.state
        ankle=m.getJointSet().get('ankle_'+suffix);frame=osim.PhysicalOffsetFrame.safeDownCast(m.getJointSet().get('mtp_'+suffix).getParentFrame())
        if frame is None:raise ValueError('Existing MTP offset frame required')
        angles=[];rows=[];excluded=[]
        for i,p in enumerate(target):
            if not np.isfinite(p).all() or np.linalg.norm(p[3]-p[2])<1e-8:
                excluded.append(dict(frame=i,reason='missing_foot_vector'));continue
            row,receipt=seeder.seed(nominal,p[:3])
            if row is None or max(receipt['direction_error_deg'])>1e-3:
                excluded.append(dict(frame=i,reason=receipt['status']));continue
            axis=_matrix(ankle.getParentFrame(),st)[:,2];foot=p[3]-p[2]
            angles.append(float(np.arccos(np.clip(axis@foot/np.linalg.norm(foot),-1,1))));rows.append(row)
        if len(rows)<3:
            records[side]=dict(status='insufficient_identifiable_knee_planes',supported=len(rows),excluded=excluded);continue
        row=rows[len(rows)//2];curve=model_curve(model_path,labels,row,side);rot=[]
        for angle in (0,np.pi,np.pi/2):
            biomech._set(m,st,labels,row);seeder.cs.get('ankle_angle_'+suffix).setValue(st,0,False)
            seeder.cs.get('subtalar_angle_'+suffix).setValue(st,float(angle),False)
            synchronize(st,seeder.bindings);m.realizePosition(st);rot.append(_matrix(frame.getParentFrame(),st))
        length=float(np.nanmedian(np.linalg.norm(target[:,3]-target[:,2],axis=-1)))
        if length<=tolerance:raise ValueError('Foot length must exceed estimation tolerance')
        delta,receipt=search(curve,rot,angles,length,tolerance)
        receipt.update(supported=len(rows),total=len(target),excluded=excluded,input_polar_range_deg=np.rad2deg([min(angles),max(angles)]).tolist(),input_length_m=length)
        records[side]=receipt
        if delta is not None and np.linalg.norm(delta)>1e-10:changes[side]=(delta,row,receipt['domain'][0])
    before=digest(model_path)
    for side,(delta,row,domain) in changes.items():
        suffix='l' if side=='left' else 'r';frame=osim.PhysicalOffsetFrame.safeDownCast(model.updJointSet().get('mtp_'+suffix).getParentFrame())
        old=np.array([frame.get_translation().get(i) for i in range(3)])
        frame.set_translation(osim.Vec3(*(old+delta)));records[side]['parent_translation_before_m']=old.tolist();records[side]['parent_translation_after_m']=(old+delta).tolist()
    model.finalizeConnections();state=model.initSystem();model.printToXML(str(model_path))
    for side,(_,row,domain) in changes.items():
        actual=model_curve(model_path,labels,row,side)
        np.testing.assert_allclose(actual['polar_interval_rad'],domain['polar_interval_rad'],atol=1e-8,rtol=0)
    offsets={}
    for name,(joint,_,_) in biomech.MARKERS.items():
        p=model.getMarkerSet().get(name).getLocationInGround(state);q=model.getJointSet().get(joint).getChildFrame().getPositionInGround(state)
        offsets[name]=float(np.linalg.norm([p.get(i)-q.get(i) for i in range(3)]))
    if max(offsets.values())>1e-9:raise ValueError('Fixed endpoint changed marker/FK registration')
    return dict(status='evaluated_input_geometry',changes=records,fixed_for_clip=True,model_sha256_before=before,
                model_sha256_after=digest(model_path),measurements_after=measure(model,state),marker_offsets_m=offsets,
                scope='Effective fixed MTP translation estimated from D only; unchanged joint axes, structure, couplings and ankle centres. Polar envelope assumes complete rotations; finite-domain fit still needs independent acceptance. No neutral anatomical or force qualification.')
