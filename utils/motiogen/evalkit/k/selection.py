"""Whole-window K3.9 candidate selection; receives no scoring reference."""
import json,shutil
from pathlib import Path
import numpy as np
from motiogen.evalkit.k.config import resolve_config
from motiogen.evalkit.k.trustworthy import fit_variant,guard,fidelity
from motiogen.evalkit.k.limb_transport import transport
from motiogen.evalkit.k.root_alignment import align_fitted_root
from motiogen.evalkit.k.fitting3 import positions,write_coordinates
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.temporal import temporal


def selection_guard(source,baseline,candidate,times):
    accepted,receipt=guard(source,baseline,candidate,times)
    a=temporal(source,baseline,times);b=temporal(source,candidate,times)
    for key in a:
        for field,limit in [('step_max_deg',5.),('velocity_error',10.)]:
            av,bv=a[key][field],b[key][field]
            if isinstance(av,dict):av,bv=av['p95_abs'],bv['p95_abs']
            if av is not None and bv is not None and bv>av+limit:receipt['failures'].append('temporal:'+key+':'+field)
    return accepted and not receipt['failures'],receipt


def fit(run,work,config,mass=75.,floor=0.):
    """Retain baseline, raw candidate and transported candidates with their models.

    This is an opt-in independent kinematic path; no force qualification is added.
    The configuration selects a geometry rule, never a reference-specific frame.
    """
    work=Path(work);work.mkdir(parents=True,exist_ok=True);cfg=resolve_config(config)
    baseline=work/'baseline';raw=work/'candidate'
    fit_variant(run,baseline,cfg,nominal_work=work/'nominal',mass=mass,floor=floor)
    z=np.load(baseline/'fit.npz');names=list(z['names']);labels=list(z['labels']);times=z['times']
    source=common(z['source'],names);base=common(z['fitted'],names)
    # Runtime method is supplied by the canonical reader, not inferred from R.
    from motiogen.evalkit.k.motion import load
    motion=load(run,config=cfg['baseline_config']);rule=cfg.get('per_method',{}).get(motion.method,{'geometry':True})
    fit_variant(run,raw,cfg,geometry=rule['geometry'],optimized=rule.get('optimized',True),nominal_work=work/'nominal',mass=mass,floor=floor)
    c=np.load(raw/'fit.npz');checks={};eligible=[('baseline',baseline,base)]
    ok,receipt=selection_guard(source,base,common(c['fitted'],names),times);checks['candidate']=dict(receipt,accepted=bool(ok))
    require_transport=rule.get('require_limb_transport',False)
    checks['candidate']['eligible_by_format_rule']=not require_transport
    if ok and not require_transport:eligible.append(('candidate',raw,common(c['fitted'],names)))
    # Parent-frame transport is only needed for raw candidates that would regress.
    if not ok or require_transport:
        for limbs in [('legs',),('legs','arms')]:
            name='transport_'+'_'.join(limbs);path=work/name;path.mkdir(exist_ok=True)
            q,transport_receipt=transport(baseline/'kinematic_model.osim',raw/'kinematic_model.osim',labels,z['coordinates'],c['pre_root_coordinates'],limbs)
            pre=q.copy();pre_points,_=positions(raw/'kinematic_model.osim',times,labels,pre)
            q,F,valid,shift,root=align_fitted_root(raw/'kinematic_model.osim',times,labels,q,c['source'],names,motion.legacy_prior_joints())
            shutil.copy2(raw/'kinematic_model.osim',path/'kinematic_model.osim')
            np.savez_compressed(path/'fit.npz',times=times,names=names,labels=labels,source=c['source'],fitted=F,coordinates=q,pre_root_fitted=pre_points,pre_root_coordinates=pre,root_valid=valid,root_shift=shift)
            ok,receipt=selection_guard(source,base,common(F,names),times);checks[name]=dict(receipt,accepted=bool(ok),transport=transport_receipt,root=root)
            if ok:eligible.append((name,path,common(F,names)))
    def objective(item):
        metrics=fidelity(source,item[2],times)
        # Equal angular terms in engineering units; position breaks small ties.
        return sum(v['p95_abs']/5 for v in metrics['angles'].values() if v['p95_abs'] is not None)+sum(v['p95_abs']/.05 for v in metrics['positions'].values() if v['p95_abs'] is not None)
    chosen=min(eligible,key=objective);selected=work/'selected';selected.mkdir(exist_ok=True)
    for filename in ('fit.npz','kinematic_model.osim'):shutil.copy2(chosen[1]/filename,selected/filename)
    arrays=np.load(selected/'fit.npz');write_coordinates(selected/'kinematic_coordinates.mot',times,labels,arrays['coordinates'])
    from motiogen.evalkit.k.scoring import angles
    metric_masks={}
    for key,value in angles(source).items():
        delta=angles(chosen[2])[key]-value
        metric_masks['scorable_'+key]=np.isfinite(delta)
        metric_masks['input_fidelity_'+key]=np.isfinite(delta)&(abs(delta)<=5.)
    np.savez_compressed(selected/'metric_masks.npz',**metric_masks)
    receipt={'selected':chosen[0],'checks':checks,'source':'input-only complete-window selection','reference_access':False,
             'output_kind':'kinematic_only','force_eligible':False,'baseline':str(baseline),'selected_path':str(selected),'model_and_coordinates':'selected together; fixed model geometry within window',
             'fidelity':fidelity(source,chosen[2],times),'temporal':temporal(source,chosen[2],times),
             'metric_masks':str(selected/'metric_masks.npz'),
             'prior_joints':motion.legacy_prior_joints(),
             'qualification':'geometric input fidelity only; source-prior dependencies retain prior status; anatomical spinal coordinates and force qualification unavailable'}
    (work/'selection.json').write_text(json.dumps(receipt,allow_nan=False));return receipt


def fit_kinematic(run,work,mass=75.,floor=0.,config=None,method=None):
    """Compatible opt-in domain API, keeping kinematic and force paths separate."""
    from .motion import load
    from .quality import metric_fidelity,apply_metric_gates
    from .fitting3 import support
    cfg=resolve_config(config)
    motion=load(run,method,config=cfg['baseline_config'])
    receipt=fit(motion,work,cfg,mass,floor)
    selected=Path(receipt['selected_path']);z=np.load(selected/'fit.npz')
    times=z['times'];names=list(z['names']);prior=motion.legacy_prior_joints()
    metrics=metric_fidelity(z['source'],z['fitted'],names,cfg['quality_angle_deg'],prior)
    quality=apply_metric_gates(support(z['source'],z['fitted'],names,cfg['quality_angle_deg'],cfg['quality_position_m'],prior),metrics)
    np.savez_compressed(selected/'kinematic_metric_quality.npz',times=times,**{n+'_'+k:v for n,q in metrics.items() for k,v in q.items()})
    return {'times':times,'labels':list(z['labels']),'coords':z['coordinates'],'names':names,'source':z['source'],'fitted':z['fitted'],
        'receipt':receipt,'region_angle_valid':{n:q['angle_valid'] for n,q in quality.items()},
        'region_position_valid':{n:q['position_valid'] for n,q in quality.items()},'metric_quality':metrics,
        'output_kind':'kinematic_only','force_eligible':False,'model_path':str(selected/'kinematic_model.osim')}
