"""Native ARDY streaming inputs and optional official motion correction.

The task supplies root waypoints; it is not a project feedback controller.
Correction is applied to the generated horizon and fed back as in the official GUI.
"""


def install(adapter, condition, evidence):
    import numpy as np
    import torch
    import time
    from .runtime import event,json_write
    from ardy.constraints import Root2DConstraintSet
    from ardy.postprocess import post_process_motion
    model = adapter.model
    keep = int(condition['history_frames'])
    if keep <= 0 or keep % model.num_frames_per_token:
        raise ValueError('History must contain a positive integer number of native tokens')
    correction = bool(condition['correction'])
    # Optional 20261030 protocol switches; absent keys keep the 20261024 behaviour.
    waypoints=condition.get('waypoints','sparse')
    target_mode=condition.get('target_mode','path')
    if waypoints not in ('sparse','dense') or target_mode not in ('path','relative_offset'):
        raise ValueError('Unknown ARDY waypoint density or target mode')
    if target_mode=='relative_offset' and waypoints!='sparse':
        raise ValueError('Relative offsets are defined for one end-of-window waypoint')
    offset=float(condition.get('offset_m',0.))
    # Optional 20261031 M1/V3 arms; absent keys keep the native settings.
    from .ardy_boundary import TREATMENTS,apply as apply_boundary,renoise
    margin=float(condition.get('root_margin_m',.04))
    cfg_constraint=float(condition.get('constraint_cfg',2.))
    boundary=condition.get('boundary','none');writeback=bool(condition.get('boundary_writeback',True))
    renoise_fraction=float(condition.get('renoise_fraction',0.))
    if boundary not in TREATMENTS or not 0.<=renoise_fraction<=1.:
        raise ValueError('Unknown ARDY boundary treatment or renoise fraction')
    if (boundary!='none' or renoise_fraction) and not correction:
        raise ValueError('Boundary treatments and renoising act on corrected windows')
    # 20261101 V4: corrected output, next window conditioned on the uncorrected generation.
    correction_writeback=bool(condition.get('correction_writeback',True))
    if not correction_writeback and (not correction or (boundary!='none' and writeback) or renoise_fraction):
        raise ValueError('Output-only correction applies to MotionCorrection with output-only boundary treatment and no renoising')
    # 20261101 M2: after a boundary treatment, a second MotionCorrection pass constrained to the
    # treated root path (every window frame) re-pins the feet the treatment moved.
    repin=bool(condition.get('boundary_repin',False))
    repin_margin=float(condition.get('boundary_repin_margin_m',.01))
    if repin and boundary=='none':
        raise ValueError('Foot re-pinning follows a boundary treatment')
    # 20261101 M2: boundary term inside MotionCorrection. The last k output frames of the
    # previous window are prepended as full-body constraints, so the solver's smoothness
    # and foot-contact terms act across the window boundary.
    prefix_frames=int(condition.get('boundary_prefix_frames',0))
    if prefix_frames<0 or prefix_frames>model.gen_horizon_len or (prefix_frames and (not correction or boundary!='none' or renoise_fraction)):
        raise ValueError('The MotionCorrection boundary prefix needs correction without boundary treatment or renoising')
    if prefix_frames:
        from ardy.constraints import FullBodyConstraintSet
        from ardy.skeleton import fk
    # 20261102 P0: which window frames carry the relative-offset root constraint.
    # end: last frame only; end_token: every frame of the last native token;
    # ramp: every frame, reference root path plus offset ramped linearly to the full offset at the end.
    constraint_frames=condition.get('constraint_frames','end')
    if constraint_frames not in ('end','end_token','ramp') or (constraint_frames!='end' and (target_mode!='relative_offset' or correction)):
        raise ValueError('Constraint frame layouts other than end apply to uncorrected relative-offset runs')
    receipts='waypoints' in condition or 'target_mode' in condition
    path=adapter.scene.get('native_root_path',{})
    direction=np.asarray(path.get('direction_xz',[0.,1.]),float)
    if direction.shape!=(2,) or not np.isclose(np.linalg.norm(direction),1.):
        raise ValueError('Root path direction must be a unit native XZ vector')
    lateral=np.asarray(path.get('lateral_xz',[direction[1],-direction[0]]),float)
    if target_mode=='relative_offset' and (lateral.shape!=(2,) or not np.isclose(np.linalg.norm(lateral),1.)):
        raise ValueError('Relative offset requires a unit native XZ lateral vector')
    before=float(path.get('speed_before_m_s',.4));after=float(path.get('speed_after_m_s',-.2))
    switch=float(path.get('switch_seconds',5.))
    windows=[];reference_pass=[False]
    def count(key):
        evidence['calls'][key] = evidence['calls'].get(key,0)+1
    def verify_guidance(_module,inputs):
        if reference_pass[0]:
            count('denoiser_reference_pass_without_constraint')
            return
        actual=[float(inputs[0].item()),float(inputs[1].item())]
        if actual != [2.,cfg_constraint]:
            raise RuntimeError(f'ARDY denoiser received wrong text/constraint guidance: {actual}')
        count('denoiser_with_text_and_spatial_guidance')
        evidence['observed_denoiser_guidance']=actual
    model.denoiser.register_forward_pre_hook(verify_guidance)
    def path_targets(frames):
        seconds=np.asarray(frames,float)/adapter.fps
        distance=before*np.minimum(seconds,switch)+after*np.maximum(seconds-switch,0.)
        return distance[:,None]*direction[None]
    def step(state,prompt):
        history = state['history']
        n = 0 if history is None else history.shape[1]
        horizon = model.gen_horizon_len
        seed = adapter.seed*100000+state['frames']
        reference_end=None
        if target_mode=='relative_offset':
            # Same per-step noise without the constraint: the offset is then
            # the correction this window needs, not an accumulated path drift.
            with adapter.timed('reference_pass'),torch.no_grad():
                reference_pass[0]=True
                try:
                    torch.manual_seed(seed)
                    free=model.autoregressive_step(num_frames=n+horizon,init_history_sequence=history,
                        texts=[prompt],num_denoising_steps=int(model.diffusion.num_base_steps),cfg_weight=(2.,2.),
                        motion_mask=None,observed_motion=None)
                finally:
                    reference_pass[0]=False
                count('native_reference_pass')
                reference_root=model.motion_rep.inverse(free,is_normalized=True)['root_positions'][0,n:][:, [0,2]].double().cpu().numpy()
            reference_end=reference_root[-1]
            if constraint_frames=='end':
                local_frames=[horizon-1]
                targets=(reference_end+offset*lateral)[None]
            elif constraint_frames=='end_token':
                local_frames=list(range(horizon-model.num_frames_per_token,horizon))
                targets=reference_root[local_frames]+offset*lateral[None]
            else:
                local_frames=list(range(horizon))
                targets=reference_root+offset*lateral[None]*((np.arange(horizon)+1.)/horizon)[:,None]
        elif waypoints=='dense':
            local_frames=list(range(horizon))
            targets=path_targets([state['frames']+k for k in local_frames])
        else:
            # Fixed task path, same absolute frame-indexed targets in every condition.
            local_frames=[horizon-1]
            targets=path_targets([state['frames']+horizon-1])
        target = torch.tensor(targets,dtype=torch.float32,device='cuda')
        constraint = Root2DConstraintSet(model.skeleton,torch.tensor([n+k for k in local_frames]),target)
        observed,mask=model.motion_rep.create_conditions_from_constraints_batched(
            [constraint],torch.tensor([n+horizon],device='cuda'),to_normalize=True,device='cuda')
        with adapter.timed('replan'),torch.no_grad():
            # Explicit per-step RNG makes differing history lengths share sampling noise.
            torch.manual_seed(seed)
            motion=model.autoregressive_step(num_frames=n+horizon,init_history_sequence=history,
                texts=[prompt],num_denoising_steps=int(model.diffusion.num_base_steps),cfg_weight=(2.,cfg_constraint),
                motion_mask=mask,observed_motion=observed)
            count('native_autoregressive_step')
            decoded=model.motion_rep.inverse(motion,is_normalized=True)
            generated_root=decoded['root_positions'][0,n:][:, [0,2]].double().cpu().numpy()
            if history is not None:
                old=model.motion_rep.inverse(history,is_normalized=True)['posed_joints']
                state['overlap_errors_m']=torch.linalg.vector_norm(decoded['posed_joints'][:,:n]-old,dim=-1).cpu().numpy()
            if correction:
                prefix=state.get('correction_prefix') if prefix_frames else None
                if prefix is None:
                    local_constraint=Root2DConstraintSet(model.skeleton,torch.tensor(local_frames),target)
                    corrected=post_process_motion(decoded['local_rot_mats'][:,n:],
                        decoded['root_positions'][:,n:],decoded['foot_contacts'][:,n:],
                        model.skeleton,constraint_lst=[local_constraint],contact_threshold=.5,root_margin=margin)
                else:
                    p_rot,p_root,p_contacts=prefix;k=p_root.shape[1]
                    p_global,p_joints,_=fk(p_rot,p_root,model.skeleton)
                    pin=FullBodyConstraintSet(model.skeleton,torch.arange(k),p_joints[0],p_global[0])
                    local_constraint=Root2DConstraintSet(model.skeleton,torch.tensor(local_frames)+k,target)
                    joined=post_process_motion(torch.cat([p_rot,decoded['local_rot_mats'][:,n:]],1),
                        torch.cat([p_root,decoded['root_positions'][:,n:]],1),
                        torch.cat([p_contacts,decoded['foot_contacts'][:,n:]],1),
                        model.skeleton,constraint_lst=[pin,local_constraint],contact_threshold=.5,root_margin=margin)
                    drift=float(torch.linalg.vector_norm(joined['root_positions'][0,:k].cpu()-p_root[0].cpu(),dim=-1).max())
                    summary=evidence.setdefault('boundary_prefix',{'frames':prefix_frames,'windows':0,'prefix_root_drift_max_m':0.})
                    summary['windows']+=1;summary['prefix_root_drift_max_m']=max(summary['prefix_root_drift_max_m'],drift)
                    corrected={key:joined[key][:,k:] for key in ('local_rot_mats','root_positions')}
                    count('native_motion_correction_with_boundary_prefix')
                count('native_motion_correction')
                normalized=model.motion_rep(local_joint_rots=corrected['local_rot_mats'].to(motion.device),
                    root_positions=corrected['root_positions'].to(motion.device),to_normalize=True)
                if not correction_writeback:
                    uncorrected=motion.clone()
                motion[:,n:]=normalized
                decoded=model.motion_rep.inverse(motion,is_normalized=True)
            if renoise_fraction:
                with adapter.timed('renoise'):
                    text_feat,text_pad_mask=model._encode_text([prompt])
                    motion,info=renoise(model,motion,history,n,renoise_fraction,text_feat,text_pad_mask,
                        mask,observed,(2.,cfg_constraint),seed+7)
                count('native_renoise_pass')
                summary=evidence.setdefault('renoise',{'steps':info['steps'],'signal_scale':info['signal_scale'],'roundtrip_root_max_m':0.})
                summary['roundtrip_root_max_m']=max(summary['roundtrip_root_max_m'],info['roundtrip_root_max_m'])
                decoded=model.motion_rep.inverse(motion,is_normalized=True)
            if boundary!='none' and state.get('last_output') is not None:
                with adapter.timed('boundary'):
                    previous_root,previous_rot=state['last_output']
                    root,rot=apply_boundary(boundary,previous_root,previous_rot,decoded['root_positions'][0,n:],
                        decoded['local_rot_mats'][0,n:],adapter.fps)
                    if repin:
                        path_constraint=Root2DConstraintSet(model.skeleton,torch.arange(root.shape[0]),root[:,[0,2]].float())
                        fixed=post_process_motion(rot[None],root[None],decoded['foot_contacts'][:,n:],model.skeleton,
                            constraint_lst=[path_constraint],contact_threshold=.5,root_margin=repin_margin)
                        root,rot=fixed['root_positions'][0].to(root.dtype),fixed['local_rot_mats'][0].to(rot.dtype)
                        count('native_boundary_repin')
                    smoothed=motion.clone()
                    smoothed[:,n:]=model.motion_rep(local_joint_rots=rot[None].to(motion.device),
                        root_positions=root[None].to(motion.device),to_normalize=True)
                count('native_boundary_'+boundary)
                decoded=model.motion_rep.inverse(smoothed,is_normalized=True)
                if writeback:
                    motion=smoothed
            if prefix_frames:
                state['correction_prefix']=tuple(decoded[key][:,-prefix_frames:].detach()
                                                 for key in ('local_rot_mats','root_positions','foot_contacts'))
            if boundary!='none':
                state['last_output']=(decoded['root_positions'][0,-4:].detach(),decoded['local_rot_mats'][0,-4:].detach())
            if receipts:
                final_root=decoded['root_positions'][0,n:][:, [0,2]].double().cpu().numpy()
                windows.append({'frame':int(state['frames']),'local_frames':local_frames,'targets_xz':targets.tolist(),
                    'reference_end_xz':None if reference_end is None else reference_end.tolist(),
                    'generated_at_targets_xz':generated_root[local_frames].tolist(),
                    'final_at_targets_xz':final_root[local_frames].tolist(),
                    'correction_root_displacement_max_m':float(np.linalg.norm(final_root-generated_root,axis=-1).max())})
            state['history']=(motion if correction_writeback else uncorrected)[:,-keep:].detach()
            state['frames']+=horizon
        return {key:value[:,-horizon:] if torch.is_tensor(value) and value.ndim>=2 and value.shape[1]==motion.shape[1]
                else value for key,value in decoded.items()}
    adapter.step=step
    if adapter.scene.get('clock','paused')=='realtime':
        def streaming():
            state=adapter.initialize_state();joints=[];rotations=[];boundaries=[];errors=[];updates=set();trace=[]
            prompt=adapter.scene['prompt'];total=round(adapter.scene['duration_seconds']*adapter.fps)
            started=time.perf_counter()
            while state['frames']<total:
                frame=state['frames'];scheduled=frame/adapter.fps
                time.sleep(max(0.,started+scheduled-time.perf_counter()))
                wall=time.perf_counter()-started
                for i,update in enumerate(adapter.scene.get('updates',[])):
                    if wall>=update['time_seconds'] and i not in updates:
                        prompt=update['prompt'];updates.add(i)
                        event(adapter.events,'prompt_update',requested_seconds=update['time_seconds'],
                              applied_seconds=scheduled,applied_wall_seconds=wall,prompt=prompt)
                tick=time.perf_counter();output=step(state,prompt);count_frames=min(state['frames']-frame,total-frame)
                if frame:boundaries.append(frame)
                joints.append(output['posed_joints'][0,:count_frames].detach().cpu().numpy())
                rotations.append(output['global_rot_mats'][0,:count_frames].detach().cpu().numpy())
                if 'overlap_errors_m' in state:errors.extend(state['overlap_errors_m'].ravel().tolist())
                trace.append({'frame':frame,'request_wall_seconds':wall,'seconds':time.perf_counter()-tick,
                              'schedule_lag_seconds':max(0.,wall-scheduled),'prompt':prompt})
                event(adapter.events,'chunk_committed',frame=frame,frames=count_frames,external_stitch=False)
            json_write(adapter.path/'clock_trace.json',{'clock':'realtime','steps':trace,
                'scope':'Paced native autoregressive calls; model-generated history, not a dynamics execution sensor',
                'root_path':'Frame-indexed future waypoints fixed in advance; the text update is delivered at wall time 5s'})
            np.save(adapter.path/'raw/streaming_joints.npy',np.concatenate(joints))
            return adapter.save(np.concatenate(joints),adapter.names,np.concatenate(rotations),chunk_boundaries=boundaries,overlap_errors_m=errors)
        adapter.streaming=streaming
    evidence['native_source']='scripts/interactive_demo/generation.py:313; ardy/model/ardy_model.py:706'
    evidence['guidance_weights']={'text':2.,'constraint':cfg_constraint,'source':'scripts/interactive_demo/gui/model.py:188-203'}
    evidence['task']={'root_direction_xz':direction.tolist(),'speed_before_m_s':before,'speed_after_m_s':after,
        'waypoints':'Frame-indexed future constraints fixed in advance; text update is a separate event at 5s'}
    if receipts:
        evidence['task'].update(root_margin_m=margin,constraint_cfg=cfg_constraint,boundary=boundary,
            boundary_writeback=writeback,renoise_fraction=renoise_fraction,correction_writeback=correction_writeback,
            boundary_prefix_frames=prefix_frames,boundary_repin=repin,boundary_repin_margin_m=repin_margin,
            constraint_frames=constraint_frames)
        evidence['task'].update(waypoint_density=waypoints,target_mode=target_mode,offset_m=offset,
            lateral_xz=lateral.tolist(),switch_seconds=switch,windows=windows,
            target_scope='path: absolute frame-indexed native XZ path; relative_offset: end of the same-noise unconstrained '
                         'prediction of this window plus offset_m along lateral_xz')
    evidence['history_is']='model-generated explicit motion, corrected when correction is enabled'
