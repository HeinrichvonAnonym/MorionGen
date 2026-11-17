"""Explicit K4 fitting entry. Only D and declared model/configuration are read."""
import json
from pathlib import Path
import numpy as np
from .config import ROOT,digest
from .motion import load
from .fixed_geometry import build


def model_fk(model_path,labels,coordinates,assemble=True,return_coordinates=False,direct_couplers=False):
    import opensim as osim
    import biomech
    model=osim.Model(str(model_path));state=model.initSystem();points=[];consistent=[]
    if direct_couplers:
        from .constraints import couplers,synchronize
        bindings=couplers(model,state)
    for row in coordinates:
        biomech._set(model,state,labels,row)
        if direct_couplers:synchronize(state,bindings)
        elif assemble:model.assemble(state)
        model.realizePosition(state)
        points.append([biomech._child_origin(model,state,joint) for joint,_,_ in biomech.MARKERS.values()])
        consistent.append([model.getCoordinateSet().get(name).getValue(state)*(180/np.pi if model.getCoordinateSet().get(name).getMotionType()==1 else 1) for name in labels])
    result=(np.asarray(points),[source for _,_,source in biomech.MARKERS.values()])
    return (*result,np.asarray(consistent)) if return_coordinates else result


def write_coordinates(path,times,labels,coordinates):
    import opensim as osim
    table=osim.TimeSeriesTable();table.setColumnLabels(list(labels))
    for t,row in zip(times,coordinates):table.appendRow(float(t),osim.RowVector(row.tolist()))
    table.addTableMetaDataString('inDegrees','yes');osim.STOFileAdapter.write(table,str(path))


def resample_native(times,labels,coordinates,rate=100.):
    """Preserve native samples as targets, unwrap only the free heading chart.

    Use a uniform grid including both boundaries. Actual spacing is recorded;
    no endpoint extrapolation or time shift is introduced.
    """
    data=np.array(coordinates,copy=True)
    if 'pelvis_rotation' in labels:
        j=list(labels).index('pelvis_rotation');data[:,j]=np.rad2deg(np.unwrap(np.deg2rad(data[:,j])))
    n=max(1,int(round((times[-1]-times[0])*rate)))
    new=np.linspace(times[0],times[-1],n+1)
    return new,np.stack([np.interp(new,times,c) for c in data.T],1)


def fit(run,work,mass,floor,config):
    import biomech
    settings=json.loads(Path(config).read_text()) if not isinstance(config,dict) else dict(config)
    if settings.get('version')!='k4-fit-1':raise ValueError('Unsupported K4 fit configuration')
    if settings.get('solver') not in ('nominal_model_marker_IK','direction_position_refinement'):
        raise ValueError('Unsupported K4 solver')
    if settings.get('root_policy') not in ('IK','input_hip_midpoint'):
        raise ValueError('Unsupported K4 root policy')
    if settings.get('hip_rotation_resampling') and settings.get('postprocess')!='native_preserving':
        raise ValueError('Hip joint interpolation requires native-preserving postprocessing')
    if settings.get('rotation_resampling'):
        if settings.get('hip_rotation_resampling') or settings.get('postprocess')!='native_preserving' or settings['rotation_resampling']!='whole_body_so3':
            raise ValueError('Select one explicit native-preserving rotation resampling policy')
    motion=load(run,settings.get('method'),config=ROOT/'config/evalkit_k4_20261117/input.json')
    out=Path(work);out.mkdir(parents=True,exist_ok=True)
    key={'inputs':motion.provenance['inputs'],'config':settings,'mass_kg':mass,'floor_m':floor,
         'model_sha256':digest(biomech.MODEL),'implementation':{str(p.relative_to(Path(__file__).parent)):digest(p) for p in sorted(Path(__file__).parent.rglob('*.py'))},
         'input_config_sha256':digest(ROOT/'config/evalkit_k4_20261117/input.json'),
         'biomech_sha256':digest(biomech.__file__)}
    receipt_path=out/'receipt.json'
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt['key']!=key:raise ValueError('K4 cache identity changed; use a new output directory')
        with np.load(out/'fit.npz') as z:return {**{k:z[k] for k in z.files},'receipt':receipt}
    geometry=build(motion,out/'model.osim',mass,settings['independent_geometry'],settings.get('marker_registration','legacy'),settings.get('joint_centres','scaled_asset'),settings.get('shoulder_width_source','declared_or_observed'))
    if settings.get('diagnostic_domain') and settings.get('kinematic_domain'):raise ValueError('Select one explicit model domain')
    model_domain=settings.get('kinematic_domain',settings.get('diagnostic_domain'))
    if model_domain:
        from .diagnostic_domain import apply
        domain=apply(out/'model.osim',model_domain)
        geometry['diagnostic_domain']=domain
        for name,change in domain['changes'].items():
            geometry['joint_domain'][name].update(locked=change['locked_after'],range_native=change['range_native_after'])
        geometry['joint_domain_source']='Original asset plus recorded diagnostic lock/range interventions'
    if settings.get('kinematic_coupling'):
        if not settings.get('kinematic_domain'):raise ValueError('Coupling extension requires independent model domain')
        from .coupling_extension import apply as extend_coupling
        geometry['coupling_extension']=extend_coupling(out/'model.osim',settings['kinematic_coupling'])
    points,names,fps=motion.legacy()
    biomech.write_trc(points,names,fps,out/'markers.trc')
    weights={n:w for n,w in biomech.WEIGHTS.items() if n in biomech.MARKERS or n in geometry['registered_axial_markers']}
    (out/'ik').mkdir(exist_ok=True)
    initialization=None
    if settings.get('initialization'):
        import shutil
        from .initialization import run as initialize
        selected,initialization=initialize(out/'model.osim',out/'markers.trc',points,names,motion.times,
                                           weights,out/'initialization',settings['initialization'])
        for path in selected.iterdir():
            if path.is_file():shutil.copyfile(path,out/'ik'/path.name)
    else:
        biomech.inverse_kinematics(out/'model.osim',out/'markers.trc',out/'ik/ik.mot',0.,(len(points)-1)/fps,weights)
    nt,labels,native=biomech.read_storage(out/'ik/ik.mot')
    from .timebase import native_clock
    nt,clock_receipt=native_clock(nt,motion.times)
    refinement=None
    if settings['solver']=='direction_position_refinement':
        from .refinement import refine
        if len(nt)!=len(points) or not np.allclose(nt,np.arange(len(points))/fps,atol=1e-6):
            raise ValueError('Native IK/input clocks differ')
        body_targets={};body_target_receipt={'status':'not_requested'}
        if settings['refinement'].get('native_frame_weight',0):
            from .body_frames import targets
            body_targets,body_target_receipt=targets(motion)
        native,refinement=refine(out/'model.osim',labels,native,points,names,settings['refinement'],body_targets,times=motion.times)
        refinement['body_target_source']=body_target_receipt
        write_coordinates(out/'coordinates_refined_native.mot',nt,labels,native)
        (out/'refinement.json').write_text(json.dumps(refinement,indent=2)+'\n')
    rotation_resampling=None
    if settings.get('postprocess','legacy_lowpass')=='native_preserving':
        times,coordinates=resample_native(nt,labels,native)
        if settings.get('hip_rotation_resampling'):
            if settings['hip_rotation_resampling']!='joint_slerp':raise ValueError('Unknown hip resampling policy')
            from .rotation_resampling import hip_slerp
            coordinates,rotation_resampling=hip_slerp(out/'model.osim',labels,native,nt,times,coordinates)
        elif settings.get('rotation_resampling')=='whole_body_so3':
            from .resampling import interpolate_rotations
            coordinates,rotation_resampling=interpolate_rotations(out/'model.osim',labels,native,nt,times,coordinates)
    else:
        if refinement is not None:raise ValueError('Refined coordinates require explicit native-preserving resampling')
        times,labels,coordinates=biomech.write_coordinates(out/'ik/ik.mot',out/'coordinates.mot')
    direct=settings.get('constraint_policy','assembly')=='direct_couplers'
    physical_times=(np.linspace(motion.times[0],motion.times[-1],len(times))
                    if settings.get('postprocess')=='native_preserving' else times+motion.times[0])
    fitted,fitted_names,consistent=model_fk(out/'model.osim',labels,coordinates,return_coordinates=True,direct_couplers=direct)
    pre_root=fitted.copy();pre_root_coordinates=consistent.copy();root_shift=np.zeros((len(times),3));root_valid=np.zeros(len(times),bool)
    if settings['root_policy']=='input_hip_midpoint':
        from .root import compensate
        consistent,root=compensate(out/'model.osim',physical_times,labels,consistent,motion)
        fitted=root['after'];root_shift=root['shift_m'];root_valid=root['valid']
    write_coordinates(out/'coordinates_filtered_unassembled.mot',times,labels,coordinates)
    write_coordinates(out/'coordinates.mot',times,labels,consistent)
    values={'times':physical_times,'labels':np.asarray(labels),'coords':consistent,
            'coordinates_filtered_unassembled':coordinates,
            'positions':fitted,'positions_pre_root':pre_root,'coordinates_pre_root':pre_root_coordinates,
            'root_shift_m':root_shift,'root_compensation_valid':root_valid,
            'names':np.asarray(fitted_names),'position_finite':np.isfinite(fitted).all(-1)}
    head_definition=None
    if settings.get('head_neck'):
        if settings['head_neck']!='attached_head_neck_v1':raise ValueError('Unknown head/neck policy')
        from .head_neck import attach,combine
        head_definition,head=attach(motion,out/'model.osim',labels,consistent,physical_times,
            attachment_policy=settings.get('head_neck_attachment','registered_chest'))
        _,head_before=attach(motion,out/'model.osim',labels,pre_root_coordinates,physical_times,fixed_definition=head_definition)
        # Fixed attachment/lengths must be identical across root-only compensation.
        if not np.allclose(head['offsets'],head_before['offsets'],equal_nan=True):
            raise ValueError('Root compensation changed head/neck geometry')
        np.savez_compressed(out/'head_neck.npz',**head)
        np.savez_compressed(out/'head_neck_pre_root.npz',**head_before)
        (out/'head_neck_model.json').write_text(json.dumps(head_definition,indent=2)+'\n')
        values['body_positions']=fitted.copy();values['body_names']=np.asarray(fitted_names)
        fitted,fitted_names=combine(fitted,fitted_names,head)
        pre_root,_=combine(pre_root,list(values['body_names']),head_before)
        values.update(positions=fitted,positions_pre_root=pre_root,names=np.asarray(fitted_names),
                      position_finite=np.isfinite(fitted).all(-1))
    np.savez_compressed(out/'fit.npz',**values)
    from .quality import assess
    quality=assess(motion,values['times'],fitted,fitted_names)
    np.savez_compressed(out/'input_fidelity.npz',times=values['times'],**quality)
    from .native_frames import read as native_frames
    native_frames_result=native_frames(motion)
    if native_frames_result['status']=='declared_native_frames':
        np.savez_compressed(out/'native_head_thorax.npz',times=motion.times,
                            head_relative_thorax=native_frames_result['head_relative_thorax'],
                            valid=native_frames_result['head_relative_valid'])
    from .original_domain import assess as assess_original_domain
    original_domain=assess_original_domain(biomech.MODEL,labels,consistent)
    (out/'original_domain_diagnostics.json').write_text(json.dumps(original_domain,indent=2)+'\n')
    receipt={'original_domain_diagnostics':original_domain,'key':key,'version':settings['version'],'geometry':geometry,
             'scope':'constraint-consistent body FK and optional attached head/neck FK; native rotations reported separately',
             'head_neck_model':head_definition,
             'kinematic_only':True,'force_eligible':False,'clock_origin_s':float(motion.times[0]),
             'input_rig_status':motion.provenance['input_geometry'],
             'coordinate_contract':'coordinates.mot satisfies declared model constraints; pre-root and pre-constraint coordinates preserved separately',
             'constraint_policy':settings.get('constraint_policy','assembly'),
             'resampling':settings.get('postprocess','legacy_lowpass'),
             'rotation_resampling':rotation_resampling,
             'root_policy':settings['root_policy'],'refinement_ran':refinement is not None,
             'initialization':initialization,
             'output_dt_s':float(np.median(np.diff(times))),
             'clock_boundary_roundoff_s':1e-9,
             'native_clock':clock_receipt,
             'input_fidelity_scope':'Per-input-point engineering fidelity; separate from independent end-to-end accuracy',
             'native_head_thorax_scope':native_frames_result.get('scope',native_frames_result.get('reason'))}
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return {**values,'receipt':receipt}
