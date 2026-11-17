"""Explicit kinematic-only fitting API using existing motion inputs.

Outputs intentionally use distinct filenames from the force/muscle pipeline.
No expanded model replaces the nominal musculoskeletal asset or default fit.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from .config import resolve_config,digest,implementation_key
from .fitting import fit as nominal_fit
from .motion import load
from .model_domain import prepare
from .legs import refine_legs
from .fitting3 import positions,write_coordinates,support
from .quality import metric_fidelity,apply_metric_gates


def fit_kinematic_domain(run,work,mass=75.,floor=0.,config=None,method=None):
    cfg=resolve_config(config)
    if cfg.get('output_kind')!='kinematic_only':raise ValueError('an explicit kinematic_only domain configuration is required')
    if cfg.get('strategy')=='trustworthy_fixed_geometry':
        from .selection import fit_kinematic
        return fit_kinematic(run,work,mass,floor,cfg,method)
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    motion=load(run,method,config=cfg['baseline_config'])
    nominal_fit(motion,work/'nominal',mass,floor,config=cfg['baseline_config'])
    source_model=work/'nominal/model.osim';source_arrays=work/'nominal/kinematics.npz'
    evidence={'baseline_arrays':digest(source_arrays),'baseline_model':digest(source_model),'config':cfg,'implementation':implementation_key()}
    key=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest();receipt_path=work/'kinematic_domain_receipt.json';arrays_path=work/'kinematic_fit.npz';model=work/'kinematic_model.osim'
    if receipt_path.exists() and arrays_path.exists() and model.exists() and json.loads(receipt_path.read_text()).get('key')==key:
        receipt=json.loads(receipt_path.read_text())
    else:
        with np.load(source_arrays) as z:times=z['times'];labels=list(z['labels']);names=list(z['names']);source=z['source'];nominal=z['coordinates']
        initial,domain=prepare(source_model,model,labels,nominal,cfg['profile'])
        selected,candidate,optimization=refine_legs(model,times,labels,initial,source,names,cfg['leg_refinement'])
        fitted,fit_names=positions(model,times,labels,selected)
        if fit_names!=names:raise ValueError('kinematic domain output node order changed')
        pre_root=selected.copy();pre_root_fitted=fitted.copy();pre_root_candidate=candidate.copy()
        root_valid=np.zeros(len(times),bool);root_shift=np.zeros((len(times),3))
        root_receipt={'status':'disabled'}
        root_policy=cfg.get('root_alignment','none')
        if root_policy=='observed_hip_midpoint':
            from .root_alignment import align_fitted_root
            selected,fitted,root_valid,root_shift,root_receipt=align_fitted_root(
                model,times,labels,selected,source,names,motion.legacy_prior_joints())
            candidate,_,_,_,_=align_fitted_root(
                model,times,labels,candidate,source,names,motion.legacy_prior_joints())
        elif root_policy!='none':
            raise ValueError('Unknown root alignment: '+root_policy)
        write_coordinates(work/'kinematic_coordinates.mot',times,labels,selected)
        np.savez_compressed(arrays_path,times=times,labels=labels,names=names,source=source,fitted=fitted,coordinates=selected,baseline_coordinates=initial,candidate_coordinates=candidate,
                            pre_root_coordinates=pre_root,pre_root_fitted=pre_root_fitted,pre_root_candidate_coordinates=pre_root_candidate,
                            root_alignment_valid=root_valid,root_shift_m=root_shift)
        receipt={'key':key,'evidence':evidence,'domain':domain,'optimization':optimization,'output_kind':'kinematic_only','force_eligible':False,
                 'root_alignment':root_receipt,'leg_guard_scope':'whole-window candidates before independent root alignment',
                 'scope':'geometric pose reproduction; joint coordinates and expanded model are not validated for muscle/force inference'}
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    with np.load(arrays_path) as z:times=z['times'];labels=list(z['labels']);names=list(z['names']);source=z['source'];fitted=z['fitted'];coordinates=z['coordinates']
    metrics=metric_fidelity(source,fitted,names,cfg['quality_angle_deg'],motion.legacy_prior_joints())
    quality=apply_metric_gates(support(source,fitted,names,cfg['quality_angle_deg'],cfg['quality_position_m'],motion.legacy_prior_joints()),metrics)
    np.savez_compressed(work/'kinematic_metric_quality.npz',times=times,**{n+'_'+k:v for n,q in metrics.items() for k,v in q.items()})
    return {'times':times,'labels':labels,'coords':coordinates,'names':names,'source':source,'fitted':fitted,'receipt':receipt,
        'region_angle_valid':{n:q['angle_valid'] for n,q in quality.items()},'region_position_valid':{n:q['position_valid'] for n,q in quality.items()},
        'metric_quality':metrics,'output_kind':'kinematic_only','force_eligible':False,'model_path':str(model)}
