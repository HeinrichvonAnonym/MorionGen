"""CLoSD's pretrained PHC tracker on one shared UHC source reference."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--index',type=int,default=0);a=p.parse_args()
a.run_dir=a.run_dir.resolve();a.references=a.references.resolve();a.run_dir.mkdir(parents=True,exist_ok=True)
if not os.environ.get('MOTIONGEN_CLOSD_ENV_READY'):
    env=dict(os.environ);prefix=DATA/'envs/motiogen-closd'
    env.update(MOTIONGEN_CLOSD_ENV_READY='1',LD_LIBRARY_PATH=str(prefix/'lib')+':'+env.get('LD_LIBRARY_PATH',''),
        PATH=str(prefix/'bin')+':'+env['PATH'],MAX_JOBS='2',TORCH_EXTENSIONS_DIR=str(DATA/'staged_dependencies/torch_extensions'))
    os.execve(sys.executable,[sys.executable,*sys.argv],env)
repo=DATA/'baselines/CLoSD';sys.path[:0]=[str(repo),str(repo/'closd'),str(DATA/'staged_dependencies/pytorch3d'),str(ROOT/'utils')]
from isaacgym import gymapi
import closd.run as upstream
import joblib
import numpy as np
import torch
from motiogen.runtime import json_write,checksum
from closd.utils.motion_lib_smpl import MotionLibSMPL

source=joblib.load(a.references);key=sorted(source)[a.index];motion=source[key]
reference=a.run_dir/'reference.pkl';joblib.dump({key:motion},reference)
assets=DATA/'outputs/assets/closd-official';checkpoint=assets/'checkpoints/closd/CLoSD_multitask_finetune/Humanoid.pth'
weights=a.run_dir/'weights';weights.mkdir();(weights/'Humanoid.pth').symlink_to(checkpoint)
upstream.hf_handler.get_dependencies=lambda:str(assets)
os.chdir(repo);started=time.monotonic();height=[];rows=[];poses=[];references=[];contacts=[];rotations=[];action_values=[];policy_times=[]
original_height=MotionLibSMPL.fix_trans_height
def height_receipt(pose,trans,shape,parsers,fix_height_mode):
    value=original_height(pose,trans,shape,parsers,fix_height_mode)
    height.append({'mode':str(fix_height_mode),'offset_m':float(value[1]),'shape':shape.detach().cpu().tolist()})
    return value
MotionLibSMPL.fix_trans_height=staticmethod(height_receipt)
player=upstream.im_amp_players.IMAMPPlayerContinuous
original_step=player.env_step
original_action=player.get_action
def timed_action(self,*args,**kwargs):
    torch.cuda.synchronize();begin=time.monotonic()
    result=original_action(self,*args,**kwargs)
    torch.cuda.synchronize();policy_times.append(time.monotonic()-begin)
    return result
player.get_action=timed_action
class Finished(BaseException):pass
def step(self,env,action):
    task=env.task
    if not rows:
        json_write(a.run_dir/'load_receipt.json',{'checkpoint_sha256':checksum(checkpoint),'source_reference':key,
            'reference_sha256':checksum(a.references),'observation_size':int(task.num_obs),'action_size':int(task.num_actions),
            'cold_seconds':time.monotonic()-started,'native_task':type(task).__name__,
            'native_mesh_parsers_available':task._motion_lib.mesh_parsers is not None,
            'scope':'PHC tracking layer only, not full DiP+PHC CLoSD generation.'})
    tick=time.monotonic();result=original_step(self,env,action);torch.cuda.synchronize()
    t=task.progress_buf*task.dt+task._motion_start_times+task._motion_start_times_offset
    ref=task._motion_lib.get_motion_state(task._sampled_motion_ids,t,offset=task._global_offset)
    current=task._rigid_body_pos[0].detach().cpu().numpy().copy()
    target=ref['rg_pos'][0].detach().cpu().numpy().copy()
    poses.append(current);references.append(target);contacts.append(task._contact_forces[0].detach().cpu().numpy().copy())
    rotations.append(task._rigid_body_rot[0].detach().cpu().numpy().copy());action_values.append(action.detach().cpu().numpy().copy())
    done=bool(result[2].reshape(-1)[0]);terminated=bool(task._terminate_buf[0])
    rows.append({'step':len(rows),'motion_time_seconds':float(t[0]),'seconds':time.monotonic()-tick,
        'policy_seconds':policy_times[-1],
        'mean_joint_tracking_error_m':float(np.linalg.norm(current-target,axis=-1).mean()),
        'done':done,'native_early_termination':terminated,'root_height_m':float(current[0,2])})
    if len(rows)%30==0:print(json.dumps(rows[-1]),flush=True)
    if not done and len(rows)<len(motion['pose_aa'])-1:return result
    assert len(policy_times)==len(rows),'Policy-call and execution-step counts differ'
    np.savez_compressed(a.run_dir/'execution_trace.npz',joints=np.stack(poses),reference_joints=np.stack(references),
        rotations_xyzw=np.stack(rotations),contact_forces=np.stack(contacts),actions=np.stack(action_values),
        body_names=task._body_names,fps=1/task.dt)
    json_write(a.run_dir/'tracking.json',{'source':key,'steps':rows,'reference_frames':len(motion['pose_aa']),
        'completion_fraction':min(1.,len(rows)/(len(motion['pose_aa'])-1)),'native_height_corrections':height,
        'native_termination_distance_m':float(task._termination_distances.max()),
        'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'height_correction_applied':bool(height),
        'native_mesh_parsers_available':task._motion_lib.mesh_parsers is not None,
        'policy_call_seconds':policy_times,
        'timing_scope':'Synchronized policy call and native env_step measured separately; stored per-step policy_seconds + seconds covers these two stages, not process cold start or final export.',
        'scope':'Subject source motion retargeted to PHC neutral human. Height correction is only reported as applied when a callback receipt exists; mesh parser absence skips it in the native loader. Native tracking-distance termination is not automatically classified as a fall. No training or policy adaptation.'})
    json_write(a.run_dir/'resolved_config.json',upstream.cfg)
    raise Finished()
player.env_step=step
overrides=['learning=im_big','robot=smpl_humanoid','epoch=-1','test=True','no_log=True','no_virtual_display=True',
    'headless=True','env.num_envs=1','env=im_single_prim','env.episode_length=1000000','env.stateInit=Start',
    f'env.motion_file={reference}','exp_name=CLoSD_multitask_finetune','seed=0',f'output_path={weights}',
    f'hydra.run.dir={a.run_dir}/hydra','hydra.job.chdir=False']
from hydra import compose,initialize_config_dir
try:
    with initialize_config_dir(config_dir=str(repo/'closd/data/cfg'),version_base='1.2'):
        config=compose(config_name='config',overrides=overrides)
    upstream.main.__wrapped__(config)
except Finished:
    pass
except Exception:
    import traceback
    json_write(a.run_dir/'failure.json',{'traceback':traceback.format_exc()});raise
