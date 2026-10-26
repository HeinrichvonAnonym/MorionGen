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
    path=adapter.scene.get('native_root_path',{})
    direction=np.asarray(path.get('direction_xz',[0.,1.]),float)
    if direction.shape!=(2,) or not np.isclose(np.linalg.norm(direction),1.):
        raise ValueError('Root path direction must be a unit native XZ vector')
    before=float(path.get('speed_before_m_s',.4));after=float(path.get('speed_after_m_s',-.2))
    def count(key):
        evidence['calls'][key] = evidence['calls'].get(key,0)+1
    def verify_guidance(_module,inputs):
        actual=[float(inputs[0].item()),float(inputs[1].item())]
        if actual != [2.,2.]:
            raise RuntimeError(f'ARDY denoiser received wrong text/constraint guidance: {actual}')
        count('denoiser_with_text_and_spatial_guidance')
        evidence['observed_denoiser_guidance']=actual
    model.denoiser.register_forward_pre_hook(verify_guidance)
    def step(state,prompt):
        history = state['history']
        n = 0 if history is None else history.shape[1]
        horizon = model.gen_horizon_len
        # Fixed task path, same absolute frame-indexed targets in every condition.
        end_seconds = (state['frames']+horizon-1)/adapter.fps
        distance = before*min(end_seconds,5.) + after*max(end_seconds-5.,0.)
        target = torch.tensor([direction*distance],dtype=torch.float32,device='cuda')
        constraint = Root2DConstraintSet(model.skeleton,torch.tensor([n+horizon-1]),target)
        observed,mask=model.motion_rep.create_conditions_from_constraints_batched(
            [constraint],torch.tensor([n+horizon],device='cuda'),to_normalize=True,device='cuda')
        with adapter.timed('replan'),torch.no_grad():
            # Explicit per-step RNG makes differing history lengths share sampling noise.
            torch.manual_seed(adapter.seed*100000+state['frames'])
            motion=model.autoregressive_step(num_frames=n+horizon,init_history_sequence=history,
                texts=[prompt],num_denoising_steps=int(model.diffusion.num_base_steps),cfg_weight=(2.,2.),
                motion_mask=mask,observed_motion=observed)
            count('native_autoregressive_step')
            decoded=model.motion_rep.inverse(motion,is_normalized=True)
            if history is not None:
                old=model.motion_rep.inverse(history,is_normalized=True)['posed_joints']
                state['overlap_errors_m']=torch.linalg.vector_norm(decoded['posed_joints'][:,:n]-old,dim=-1).cpu().numpy()
            if correction:
                local_constraint=Root2DConstraintSet(model.skeleton,torch.tensor([horizon-1]),target)
                corrected=post_process_motion(decoded['local_rot_mats'][:,n:],
                    decoded['root_positions'][:,n:],decoded['foot_contacts'][:,n:],
                    model.skeleton,constraint_lst=[local_constraint],contact_threshold=.5,root_margin=.04)
                count('native_motion_correction')
                normalized=model.motion_rep(local_joint_rots=corrected['local_rot_mats'].to(motion.device),
                    root_positions=corrected['root_positions'].to(motion.device),to_normalize=True)
                motion[:,n:]=normalized
                decoded=model.motion_rep.inverse(motion,is_normalized=True)
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
    evidence['history_is']='model-generated explicit motion, corrected when correction is enabled'
