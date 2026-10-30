"""Observe native DiP/PHC/PhysX execution with runtime-only goal messages."""
import hashlib
import os
import sys
import time


def run(args, upstream, root, data):
    import numpy as np
    import torch
    from isaacgym import gymtorch
    from closd.env.tasks.closd import CLoSD
    from closd.env.tasks.closd_multitask import CLoSDMultiTask
    from closd.utils.closd_util import STATES
    import closd.diffusion_planner.model.mdm as mdm_module
    import closd.diffusion_planner.data_loaders.get_data as loader_module
    from .runtime import json_write, checksum
    from .runtime_feedback import EventReceiver, stamp
    from .motion import save_motion, HUMANML_NAMES

    out = args.run_dir
    assets = data / 'outputs/assets/closd-official'
    start = stamp()
    receiver = EventReceiver(args.socket, out / 'events.jsonl')
    state = {'version': 0, 'committed_version': 0, 'action_version': 0,
             'frame': 0, 'start_ns': None, 'goal': [2., 0., 0.]}
    events, planners, steps, actions, poses, rotations, contacts, dofs = [], [], [], [], [], [], [], []
    upstream.hf_handler.get_dependencies = lambda: str(assets)
    bert = data / 'outputs/assets/closd-distilbert'
    original_bert = mdm_module.load_bert
    mdm_module.load_bert = lambda _path: original_bert(str(bert))
    original_loader = loader_module.DataLoader
    def serial_loader(*a, **kw):
        kw['num_workers'] = 0
        return original_loader(*a, **kw)
    loader_module.DataLoader = serial_loader
    checkpoint = assets / 'checkpoints/closd/CLoSD_multitask_finetune/Humanoid.pth'
    dip = assets / 'checkpoints/dip/DiP_multi-target_10steps_context20_predict40/model000300000.pt'
    weights = out / 'weights'
    weights.mkdir(exist_ok=True)
    (weights / 'Humanoid.pth').symlink_to(checkpoint)
    os.chdir(data / 'baselines/CLoSD')
    original_target = CLoSDMultiTask.set_random_target_loc
    def target(task, ids, native_state):
        original_target(task, ids, native_state)
        if native_state != STATES.REACH:
            raise ValueError('Only native reach supported in this event pilot')
        task._target_states[ids, :3] = torch.tensor(state['goal'], device=task.device)
    CLoSDMultiTask.set_random_target_loc = target

    original_context = CLoSD.build_completion_input
    def context(task, context_switch_vec=None):
        result = original_context(task, context_switch_vec)
        condition = result[0]['target_cond'].detach().cpu().numpy()
        planners.append({'condition_version': state['version'], 'condition_consumed_ns': stamp(),
                         'native_frame': int(task.frame_idx), 'frame': state['frame'],
                         'target_world': task._tar_pos[0].detach().cpu().tolist(),
                         'target_condition': condition.tolist(),
                         'target_condition_sha256': hashlib.sha256(condition.tobytes()).hexdigest()})
        return result
    CLoSD.build_completion_input = context
    original_horizon = CLoSD.get_mdm_next_planning_horizon
    def horizon(task):
        # Fixed random stream by planner invocation, shared across update/ignore.
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(args.seed * 100000 + len(planners))
            result = original_horizon(task)
        torch.cuda.synchronize()
        planners[-1]['compute_end_ns'] = stamp()
        state['committed_version'] = planners[-1]['condition_version']
        planners[-1]['reference_commit_ns'] = stamp()
        return result
    CLoSD.get_mdm_next_planning_horizon = horizon
    player = upstream.im_amp_players.IMAMPPlayerContinuous
    original_action = player.get_action
    def action(self, *a, **kw):
        version = state['committed_version']
        begin = stamp()
        result = original_action(self, *a, **kw)
        torch.cuda.synchronize()
        state['action_version'] = version
        actions.append({'frame': state['frame'], 'condition_version': version,
                        'compute_start_ns': begin, 'compute_end_ns': stamp()})
        return result
    player.get_action = action
    original_step = player.env_step
    class Finished(BaseException):
        pass

    def step(self, env, action_value):
        task = env.task
        frame = state['frame']
        dt = float(task.dt)
        if abs(dt - 1 / 30) > 1e-6:
            raise ValueError(f'Unexpected control dt: {dt}')
        if state['start_ns'] is None:
            state['start_ns'] = stamp()
            json_write(out / 'ready.json', {'start_ns': state['start_ns'], 'worker_pid': os.getpid(),
                       'cold_seconds': (state['start_ns'] - start) / 1e9,
                       'cache': 'Models warmed; no future goal payload in worker arguments or cache.'})
        if args.clock == 'realtime':
            time.sleep(max(0., (state['start_ns'] + round(frame * dt * 1e9) - stamp()) / 1e9))
        elif frame >= 150 and not events:
            while receiver.pending.empty():
                if stamp() - state['start_ns'] > 15e9:
                    raise TimeoutError('Independent event did not arrive')
                time.sleep(.005)
        for event in receiver.poll():
            event['ignored'] = args.ignore
            events.append(event)
            if not args.ignore:
                goal = event['payload']['goal']
                if len(goal) != 2 or not np.isfinite(goal).all():
                    raise ValueError('Expected finite native 2D reach goal')
                state['goal'] = [float(goal[0]), float(goal[1]), 0.]
                state['version'] = int(event['version'])
                ids = torch.arange(task.num_envs, device=task.device, dtype=torch.long)
                task._reset_reach_target(ids)
                actor_ids = task._tar_actor_ids[ids]
                task.gym.set_actor_root_state_tensor_indexed(task.sim, gymtorch.unwrap_tensor(task._root_states),
                    gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
                event['goal_applied_ns'] = stamp()
                event['goal_applied_frame'] = frame
        # This action was computed before this step. A newly received goal cannot
        # label it affected until DiP commits and PHC consumes that reference.
        version = state['action_version']
        submit = stamp()
        result = original_step(self, env, action_value)
        torch.cuda.synchronize()
        complete = stamp()
        poses.append(task._rigid_body_pos[0].detach().cpu().numpy().copy())
        rotations.append(task._rigid_body_rot[0].detach().cpu().numpy().copy())
        contacts.append(task._contact_forces[0].detach().cpu().numpy().copy())
        dofs.append(task._dof_pos[0].detach().cpu().numpy().copy())
        steps.append({'frame': frame, 'condition_version': version, 'action_submit_ns': submit,
                      'simulation_step_end_ns': complete, 'native_step_seconds': (complete-submit)/1e9,
                      'deadline_lateness_seconds': max(0., (submit-state['start_ns'])/1e9-frame*dt) if args.clock=='realtime' else None,
                      'done': bool(result[2].reshape(-1)[0]), 'task_done': bool(task.is_done[0]),
                      'goal': task._tar_pos[0].detach().cpu().tolist()})
        state['frame'] += 1
        if frame % 30 == 0:
            print(f'Native PhysX step {frame}; goal version {state["version"]}; action version {version}', flush=True)
        if state['frame'] < 600:
            return result
        names = task._body_names
        mapping = ['Pelvis','L_Hip','R_Hip','Torso','L_Knee','R_Knee','Spine','L_Ankle','R_Ankle','Chest',
                   'L_Toe','R_Toe','Neck','L_Thorax','R_Thorax','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist']
        xyz = np.stack(poses)[:, [names.index(n) for n in mapping]]
        save_motion(out/'motion.npz', xyz, HUMANML_NAMES, 30, {'units':'m','axes':'right-handed Z-up',
                    'model':'CLoSD','protocol':'runtime_reveal_v1','seed':args.seed})
        np.savez_compressed(out/'native_physics.npz', joints=np.stack(poses), rotations_xyzw=np.stack(rotations),
                            dof_positions=np.stack(dofs), body_names=names, contact_forces=np.stack(contacts))
        json_write(out/'runtime_trace.json', {'protocol':'runtime_reveal_v1','clock':args.clock,
            'execution':'native PHC action + PhysX simulation, no kinematic playback',
            'start_ns':state['start_ns'],'events':events,'planner':planners,'actions':actions,'steps':steps,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'tracker_sha256':checksum(checkpoint),'dip_sha256':checksum(dip),
            'scope':'Native simulation delays retained; runtime goal assigned on receipt, reference consumed on next native planning cycle.'})
        json_write(out/'resolved_config.json', upstream.cfg)
        raise Finished()
    player.env_step = step
    overrides = ['learning=im_big','robot=smpl_humanoid','epoch=-1','test=True','no_log=True','no_virtual_display=True',
        'headless=True','env.num_envs=1','env=closd_multitask','env.task_filter=reach','env.episode_length=1000000',
        'exp_name=CLoSD_multitask_finetune',f'seed={args.seed}',f'output_path={weights}',
        f'env.dip.model_path={dip}','env.dip.context_switch_prob=0.0',f'hydra.run.dir={out}/hydra','hydra.job.chdir=False']
    from hydra import compose, initialize_config_dir
    try:
        with initialize_config_dir(config_dir=str(data/'baselines/CLoSD/closd/data/cfg'), version_base='1.2'):
            native_config = compose(config_name='config', overrides=overrides)
        upstream.main.__wrapped__(native_config)
    except Finished:
        return
    finally:
        receiver.close()
    raise RuntimeError('Native player ended before the full observation window')
