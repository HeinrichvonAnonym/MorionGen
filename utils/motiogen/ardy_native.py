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
        if actual != [2.,2.]:
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
                reference_end=model.motion_rep.inverse(free,is_normalized=True)['root_positions'][0,-1,[0,2]].double().cpu().numpy()
            local_frames=[horizon-1]
            targets=(reference_end+offset*lateral)[None]
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
                texts=[prompt],num_denoising_steps=int(model.diffusion.num_base_steps),cfg_weight=(2.,2.),
                motion_mask=mask,observed_motion=observed)
            count('native_autoregressive_step')
            decoded=model.motion_rep.inverse(motion,is_normalized=True)
            generated_root=decoded['root_positions'][0,n:][:, [0,2]].double().cpu().numpy()
            if history is not None:
                old=model.motion_rep.inverse(history,is_normalized=True)['posed_joints']
                state['overlap_errors_m']=torch.linalg.vector_norm(decoded['posed_joints'][:,:n]-old,dim=-1).cpu().numpy()
            if correction:
                local_constraint=Root2DConstraintSet(model.skeleton,torch.tensor(local_frames),target)
                corrected=post_process_motion(decoded['local_rot_mats'][:,n:],
                    decoded['root_positions'][:,n:],decoded['foot_contacts'][:,n:],
                    model.skeleton,constraint_lst=[local_constraint],contact_threshold=.5,root_margin=.04)
                count('native_motion_correction')
                normalized=model.motion_rep(local_joint_rots=corrected['local_rot_mats'].to(motion.device),
                    root_positions=corrected['root_positions'].to(motion.device),to_normalize=True)
                motion[:,n:]=normalized
                decoded=model.motion_rep.inverse(motion,is_normalized=True)
            if receipts:
                final_root=decoded['root_positions'][0,n:][:, [0,2]].double().cpu().numpy()
                windows.append({'frame':int(state['frames']),'local_frames':local_frames,'targets_xz':targets.tolist(),
                    'reference_end_xz':None if reference_end is None else reference_end.tolist(),
                    'generated_at_targets_xz':generated_root[local_frames].tolist(),
                    'final_at_targets_xz':final_root[local_frames].tolist(),
                    'correction_root_displacement_max_m':float(np.linalg.norm(final_root-generated_root,axis=-1).max())})
            state['history']=motion[:,-keep:].detach()
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
    evidence['guidance_weights']={'text':2.,'constraint':2.,'source':'scripts/interactive_demo/gui/model.py:188-203'}
    evidence['task']={'root_direction_xz':direction.tolist(),'speed_before_m_s':before,'speed_after_m_s':after,
        'waypoints':'Frame-indexed future constraints fixed in advance; text update is a separate event at 5s'}
    if receipts:
        evidence['task'].update(waypoint_density=waypoints,target_mode=target_mode,offset_m=offset,
            lateral_xz=lateral.tolist(),switch_seconds=switch,windows=windows,
            target_scope='path: absolute frame-indexed native XZ path; relative_offset: end of the same-noise unconstrained '
                         'prediction of this window plus offset_m along lateral_xz')
    evidence['history_is']='model-generated explicit motion, corrected when correction is enabled'
