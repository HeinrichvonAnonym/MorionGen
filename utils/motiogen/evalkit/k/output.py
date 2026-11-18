"""Shared model-FK output stage for native fits and fixed-anchor export variants.

D supplies root localization. A supplied head definition is held fixed across
export variants. No original reference or scoring is read.
"""
import json
import numpy as np


def write_output(motion,out,labels,coordinates,times,settings,*,fixed_head=None):
    from .k4_fitting import model_fk,write_coordinates
    direct=settings.get('constraint_policy','assembly')=='direct_couplers'
    physical_times=((times-times[0])+motion.times[0]
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
            fixed_definition=fixed_head,attachment_policy=settings.get('head_neck_attachment','registered_chest'))
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
    return values,head_definition,native_frames_result
