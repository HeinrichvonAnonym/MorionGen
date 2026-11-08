"""Observe the official PHC player loop; intervene only on DiP prefix source."""
import json
import hashlib
import os
import sys
import time
from pathlib import Path

def run(args,upstream,root,data):
    import numpy as np
    import torch
    from isaacgym import gymtorch
    from closd.env.tasks.closd import CLoSD
    from closd.env.tasks.closd_multitask import CLoSDMultiTask
    from closd.utils.closd_util import STATES
    import closd.diffusion_planner.model.mdm as mdm_module
    import closd.diffusion_planner.data_loaders.get_data as loader_module
    from .runtime import json_write,checksum
    from .motion import save_motion,HUMANML_NAMES
    from .metrics import evaluate
    out=args.run_dir;assets=data/'outputs/assets/closd-official'
    bert=data/'outputs/assets/closd-distilbert'
    if not (bert/'source_receipt.json').is_file(): raise FileNotFoundError('Pinned DistilBERT staging is not complete')
    if (out/'motion.npz').exists(): raise ValueError('Refusing to overwrite CLoSD motion')
    from .config import load_config
    from .weight_identity import bind,verify_unchanged
    weight_binding=bind(load_config(),'closd',out)
    upstream.hf_handler.get_dependencies=lambda:str(assets)
    original_bert=mdm_module.load_bert
    mdm_module.load_bert=lambda _path:original_bert(str(bert))
    # One evaluation environment needs no eight-process data-loader pool.
    # Retain the same dataset/collation/shuffle; avoid fork RSS duplication and
    # needless worker processes on this 32 GiB host.
    original_loader=loader_module.DataLoader
    def serial_loader(*a,**kw):
        kw['num_workers']=0
        return original_loader(*a,**kw)
    loader_module.DataLoader=serial_loader
    checkpoint=assets/'checkpoints/closd/CLoSD_multitask_finetune/Humanoid.pth'
    dip=assets/'checkpoints/dip/DiP_multi-target_10steps_context20_predict40/model000300000.pt'
    weights=out/'weights';weights.mkdir(exist_ok=True)
    if not (weights/'Humanoid.pth').exists(): (weights/'Humanoid.pth').symlink_to(checkpoint)
    os.chdir(data/'baselines/CLoSD')
    history=[];poses=[];rotations=[];dofs=[];contacts=[];planner=[];planner_samples=[];phase={'changed':False,'frame':0};start=time.perf_counter()
    original_target=CLoSDMultiTask.set_random_target_loc
    def fixed_target(task,env_ids,state):
        original_target(task,env_ids,state)
        if state!=STATES.REACH: raise ValueError('This paired protocol supports the native reach task only')
        # Task inputs, not a tracking controller. Identical absolute world goals
        # remain fixed after native resets and are independent of divergent poses.
        goal=[*args.goal_next,0.] if phase['changed'] else [*args.goal_start,0.]
        task._target_states[env_ids,:3]=torch.tensor(goal,device=task.device)
    CLoSDMultiTask.set_random_target_loc=fixed_target
    original_context=CLoSD.build_completion_input
    def context(task,context_switch_vec=None):
        expected=args.context_source=='prediction' and task.frame_idx>0
        actual=context_switch_vec is not None and bool(torch.all(context_switch_vec==1).item())
        if actual!=expected: raise RuntimeError('Native DiP context switch did not match intervention')
        if not hasattr(task,'_paired_sample_calls'):
            task._paired_sample_calls=0;original_sample=task.sample_fn
            def sample(*a,**kw):
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(args.seed*100000+task._paired_sample_calls)
                    task._paired_sample_calls+=1
                    torch.cuda.synchronize();tick=time.perf_counter()
                    value=original_sample(*a,**kw)
                    torch.cuda.synchronize()
                    planner_samples.append({'frame':phase['frame'],'seconds':time.perf_counter()-tick,
                        'during_execution':execution_start[0] is not None})
                    return value
            task.sample_fn=sample
        result=original_context(task,context_switch_vec)
        prefix=result[0].get('prefix')
        planner.append({'frame':phase['frame'],'native_frame_idx':task.frame_idx,
            'context_wall_seconds':time.perf_counter()-execution_start[0] if execution_start[0] is not None else None,
            'source':'prediction' if actual else 'simulation','prefix_shape':list(prefix.shape),
            'prefix_sha256':hashlib.sha256(prefix.detach().cpu().numpy().tobytes()).hexdigest(),
            'prefix_l2':float(prefix.norm().item()),'target':task._tar_pos[0].detach().cpu().tolist() if hasattr(task,'_tar_pos') else None})
        return result
    CLoSD.build_completion_input=context
    player=upstream.im_amp_players.IMAMPPlayerContinuous
    original_action=player.get_action
    def action(self,*a,**kw):
        torch.cuda.synchronize();tick=time.perf_counter()
        value=original_action(self,*a,**kw)
        torch.cuda.synchronize();phase['last_action_seconds']=time.perf_counter()-tick
        return value
    player.get_action=action
    original_step=player.env_step
    class Finished(BaseException): pass
    execution_start=[None];cold=[None]
    def step(self,env,action):
        task=env.task;frame=phase['frame'];dt=float(task.dt)
        if abs(dt-1/30)>1e-6: raise ValueError(f'Unexpected native control timestep {dt}')
        if execution_start[0] is None:
            torch.cuda.synchronize();cold[0]=time.perf_counter()-start;execution_start[0]=time.perf_counter()
        if args.clock=='realtime': time.sleep(max(0.,execution_start[0]+frame*dt-time.perf_counter()))
        wall=time.perf_counter()-execution_start[0]
        if not phase['changed'] and (wall>=5. if args.clock=='realtime' else frame>=150):
            phase['changed']=True
            ids=torch.arange(task.num_envs,device=task.device,dtype=torch.long)
            task._reset_reach_target(ids)
            actor_ids=task._tar_actor_ids[ids]
            task.gym.set_actor_root_state_tensor_indexed(task.sim,gymtorch.unwrap_tensor(task._root_states),gymtorch.unwrap_tensor(actor_ids),len(actor_ids))
        torch.cuda.synchronize();tick=time.perf_counter()
        result=original_step(self,env,action)
        torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        poses.append(task._rigid_body_pos[0].detach().cpu().numpy().copy())
        contacts.append(task._contact_forces[0].detach().cpu().numpy().copy())
        rotations.append(task._rigid_body_rot[0].detach().cpu().numpy().copy())
        dofs.append(task._dof_pos[0].detach().cpu().numpy().copy())
        history.append({'frame':frame,'seconds':elapsed,'request_wall_seconds':wall,
            'phc_action_seconds':phase.get('last_action_seconds'),
            'timing_scope':'seconds covers native env_step including triggered DiP and PhysX; PHC action timing is separate',
            'goal':task._tar_pos[0].detach().cpu().tolist(),'done':bool(result[2].reshape(-1)[0]),
            'reward':float(result[1].reshape(-1)[0]),'task_done':bool(task.is_done[0]),
            'schedule_lag_seconds':max(0.,wall-frame*dt) if args.clock=='realtime' else None})
        phase['frame']+=1
        if frame%30==0: print(json.dumps({'stage':'native_physics','frame':frame,'planner_calls':len(planner),'goal':history[-1]['goal']}),flush=True)
        if phase['frame']>=args.frames:
            verify_unchanged(weight_binding)
            names=task._body_names
            mapping={'pelvis':'Pelvis','left_hip':'L_Hip','right_hip':'R_Hip','spine1':'Torso','spine2':'Spine','spine3':'Chest',
                'left_knee':'L_Knee','right_knee':'R_Knee','left_ankle':'L_Ankle','right_ankle':'R_Ankle','left_foot':'L_Toe','right_foot':'R_Toe',
                'neck':'Neck','head':'Head','left_collar':'L_Thorax','right_collar':'R_Thorax',
                'left_shoulder':'L_Shoulder','right_shoulder':'R_Shoulder','left_elbow':'L_Elbow','right_elbow':'R_Elbow','left_wrist':'L_Wrist','right_wrist':'R_Wrist'}
            xyz=np.stack(poses)[:,[names.index(mapping[n]) for n in HUMANML_NAMES]]
            save_motion(out/'motion.npz',xyz,HUMANML_NAMES,30,{'units':'m','axes':'right-handed Z-up','model':'CLoSD',
                'variant':'native DiP + PHC + IsaacGym human','context_source':args.context_source,'seed':args.seed},contact_forces=np.stack(contacts))
            np.savez_compressed(out/'native_physics.npz',joints=np.stack(poses),rotations_xyzw=np.stack(rotations),dof_positions=np.stack(dofs),
                body_names=names,contact_forces=np.stack(contacts))
            json_write(out/'metrics.json',evaluate({'joints':xyz,'joint_names':HUMANML_NAMES,'fps':30},{'name':'native_root_goal_change'}))
            json_write(out/'trace.json',{'steps':history,'planner':planner,'planner_sampling':planner_samples,'cold_start_seconds':cold[0],'clock':args.clock,
                'execution_wall_seconds':time.perf_counter()-execution_start[0],
                'data_loader_workers':0,
                'requested_goals':[{'frame':0,'position':[*args.goal_start,0.]},{'frame':150,'position':[*args.goal_next,0.]}],
                'goal_request_seconds':5,'goal_applied_frame':next((r['frame'] for r in history if np.allclose(r['goal'],[*args.goal_next,0.])),None),
                'gpu_peak_bytes':torch.cuda.max_memory_allocated(),'tracker_sha256':checksum(checkpoint),'dip_sha256':checksum(dip),
                'scope':'Native fixed pretrained human simulation; explicit absolute root goals; native failure resets retained'})
            json_write(out/'intervention.json',{'condition':{'context_source':args.context_source},
                'calls':{'native_physics_steps':len(history),'planner_context':len(planner),'prediction_context':sum(r['source']=='prediction' for r in planner)},
                'unchanged':'PHC receives simulated execution state in both conditions'})
            json_write(out/'resolved_config.json',upstream.cfg)
            raise Finished()
        return result
    player.env_step=step
    sys.argv=[sys.argv[0],'learning=im_big','robot=smpl_humanoid','epoch=-1','test=True','no_log=True','no_virtual_display=True',
        'headless=True','env.num_envs=1','env=closd_multitask','env.task_filter=reach','env.episode_length=1000000',
        'exp_name=CLoSD_multitask_finetune',f'seed={args.seed}',f'output_path={weights}',
        f'env.dip.model_path={dip}',f'env.dip.context_switch_prob={1. if args.context_source=="prediction" else 0.}',
        f'hydra.run.dir={out}/hydra','hydra.job.chdir=False']
    # Imported upstream entry points make Hydra interpret its relative config
    # path as a Python package. The official config tree is filesystem-backed
    # (no package __init__), so compose that same tree explicitly.
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(data/'baselines/CLoSD/closd/data/cfg'),version_base='1.2'):
        native_config=compose(config_name='config',overrides=sys.argv[1:])
    try: upstream.main.__wrapped__(native_config)
    except Finished: return
    raise RuntimeError('Native player ended before requested observation duration')
