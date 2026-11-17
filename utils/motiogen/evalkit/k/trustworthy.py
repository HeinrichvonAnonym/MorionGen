"""Versioned fixed-shape and root-relative pose fitting; kinematic outputs only."""
import json,shutil
from pathlib import Path
import numpy as np
from .config import resolve_config,digest,implementation_key
from .motion import load
from .fitting import fit as nominal_fit
from .fitting3 import positions,write_coordinates,support
from .root_alignment import align_fitted_root
from .model_domain import prepare
from .geometric_calibration import calibrate
from .refinement import refine
from .legs import refine_legs
from .bones import common,NAMES
from .scoring import angles,stats
from .geometry import unit


def directions(points):
    return {'trunk_axis':unit(points[:,1]-points[:,0]),
            'shoulder_line':unit(points[:,13]-points[:,10])}


def fidelity(source,fitted,times):
    observed=angles(source);actual=angles(fitted);result={}
    for key in observed:result[key]=stats(actual[key]-observed[key])
    for key,value in directions(source).items():
        result[key]=stats(np.degrees(np.arccos(np.clip((value*directions(fitted)[key]).sum(-1),-1,1))))
    positions_error={name:stats(np.linalg.norm(fitted[:,i]-source[:,i],axis=1),False) for i,name in enumerate(NAMES)}
    step=max(float(np.nanmax(abs(np.diff(v)))) for v in actual.values())
    velocity=max(float(np.nanpercentile(abs(np.gradient(actual[k]-v,times)),95)) for k,v in observed.items())
    return {'angles':result,'positions':positions_error,'maximum_angle_step_deg':step,'maximum_velocity_error_p95_deg_s':velocity}


def guard(source,baseline,candidate,times):
    a=fidelity(source,baseline,times);b=fidelity(source,candidate,times)
    bad=[]
    for key in a['angles']:
        if b['angles'][key]['p95_abs']>a['angles'][key]['p95_abs']+1:bad.append('angle:'+key)
        if b['angles'][key]['within_5']<a['angles'][key]['within_5']-.01:bad.append('coverage:'+key)
    for key in ['left_shoulder','right_shoulder','left_knee','right_knee','left_foot','right_foot','left_wrist','right_wrist']:
        if b['positions'][key]['p95_abs']>a['positions'][key]['p95_abs']+.01:bad.append('position:'+key)
    if b['maximum_angle_step_deg']>a['maximum_angle_step_deg']+5:bad.append('step')
    if b['maximum_velocity_error_p95_deg_s']>a['maximum_velocity_error_p95_deg_s']+10:bad.append('velocity')
    improvement=(sum(v['p95_abs'] for v in b['angles'].values())<sum(v['p95_abs'] for v in a['angles'].values())-1e-5 or
                 sum(v['p95_abs'] for v in b['positions'].values())<sum(v['p95_abs'] for v in a['positions'].values())-1e-5)
    return not bad and improvement,{'failures':bad,'improvement':bool(improvement),'scope':'whole-window input-only guard; complete pose/model retained on rejection'}


def fit_variant(run,work,config,geometry=False,optimized=False,nominal_work=None):
    cfg=resolve_config(config);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    motion=load(run,config=cfg['baseline_config']);nominal=Path(nominal_work or work/'nominal')
    nominal_fit(motion,nominal,75.,0.,config=cfg['baseline_config'])
    evidence={'inputs':motion.content_key(),'nominal_model':digest(nominal/'model.osim'),'nominal_arrays':digest(nominal/'kinematics.npz'),'config':cfg,'geometry':geometry,'optimized':optimized,'implementation':implementation_key()}
    receipt_path=work/'receipt.json';arrays_path=work/'fit.npz'
    if receipt_path.exists() and arrays_path.exists() and json.loads(receipt_path.read_text()).get('evidence')==evidence:return json.loads(receipt_path.read_text())
    z=np.load(nominal/'kinematics.npz');t=z['times'];labels=list(z['labels']);names=list(z['names']);source=z['source'];initial=z['coordinates']
    domain_model=work/'domain.osim';initial,domain=prepare(nominal/'model.osim',domain_model,labels,initial,cfg['profile'])
    model=work/'kinematic_model.osim';shape={'policy':'current_geometry'}
    if geometry:shape=calibrate(domain_model,model,motion)
    else:shutil.copy2(domain_model,model)
    settings=dict(resolve_config(cfg['baseline_config'])['fitting']);settings.update(settings['per_method'][motion.method]);settings['rescue_after_baseline']=False
    if optimized:settings.update(cfg['pose_optimizer'])
    # Both geometries and solvers receive the same nominal initialization.
    q,pose=refine(model,motion,t,labels,initial,settings)
    leg_settings=dict(cfg['leg_refinement'])
    if optimized:leg_settings.update(cfg['leg_optimizer'])
    q,leg_candidate,legs=refine_legs(model,t,labels,q,source,names,leg_settings)
    before=q.copy();before_points,_=positions(model,t,labels,q)
    q,fitted,root_valid,shift,root=align_fitted_root(model,t,labels,q,source,names,motion.legacy_prior_joints())
    write_coordinates(work/'kinematic_coordinates.mot',t,labels,q)
    np.savez_compressed(arrays_path,times=t,labels=labels,names=names,source=source,fitted=fitted,coordinates=q,pre_root_coordinates=before,pre_root_fitted=before_points,root_valid=root_valid,root_shift=shift,leg_candidate_coordinates=leg_candidate)
    result={'evidence':evidence,'shape':shape,'domain':domain,'pose':pose,'legs':legs,'root':root,'work':str(work),'output_kind':'kinematic_only','force_eligible':False,
            'fidelity':fidelity(common(source,names),common(fitted,names),t)}
    receipt_path.write_text(json.dumps(result,allow_nan=False));return result
