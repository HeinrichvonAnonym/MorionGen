"""Bounded human-SMPL infrastructure checks; no generated-motion claim."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import load_config,data_root
from motiogen.runtime import json_write
root=data_root(load_config());repo=root/'baselines/SMPLSim'
sys.path.insert(0,str(repo));os.chdir(repo)
import numpy as np
# Legacy SMPL pickles import Chumpy's historical NumPy aliases. These are
# scoped to this worker, without changing the body arrays or upstream source.
for name, value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'unicode':str,'str':str}.items():
    if name not in np.__dict__:setattr(np,name,value)
import mujoco
from omegaconf import OmegaConf
from smpl_sim.envs.humanoid_env import HumanoidEnv

cfg=OmegaConf.load(repo/'smpl_sim/data/cfg/config.yaml')
cfg.env=OmegaConf.load(repo/'smpl_sim/data/cfg/env/base_env.yaml')
cfg.robot=OmegaConf.load(repo/'smpl_sim/data/cfg/robot/smpl_humanoid.yaml')
cfg.robot.smpl_data_dir=str(root/'outputs/assets/closd-official/data/smpl')
cfg.env.render_mode=None;cfg.headless=True
cfg.env.control_mode='torque'
cfg.env.enable_early_termination=False
args.run_dir.mkdir(parents=True,exist_ok=True)
json_write(args.run_dir/'resolved_config.json',OmegaConf.to_container(cfg,resolve=True))
start=time.monotonic();env=HumanoidEnv(cfg);obs,info=env.reset(seed=0)
model=env.mj_model;data=env.mj_data
rows=[];qpos=[];qvel=[]
for i in range(300):
    obs,reward,terminated,truncated,info=env.step(np.zeros(env.action_space.shape,dtype=np.float32))
    finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(obs).all())
    qpos.append(data.qpos.copy());qvel.append(data.qvel.copy())
    rows.append({'step':i,'simulation_time':float(data.time),'contacts':int(data.ncon),'finite':finite,
                 'terminated':bool(terminated),'truncated':bool(truncated),'root_height':float(data.qpos[2])})
    if not finite:break
    if i%30==0:print(json.dumps(rows[-1]),flush=True)
np.savez_compressed(args.run_dir/'trajectory.npz',qpos=qpos,qvel=qvel)
axes=np.linalg.norm(model.jnt_axis,axis=1)
result={'steps':len(rows),'elapsed_seconds':time.monotonic()-start,'rows':rows,
        'nq':model.nq,'nv':model.nv,'nu':model.nu,'body_names':[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) for i in range(model.nbody)],
        'joint_axes':model.jnt_axis.tolist(),'joint_ranges':model.jnt_range.tolist(),
        'joint_limited':model.jnt_limited.tolist(),'masses':model.body_mass.tolist(),
        'timestep':float(model.opt.timestep),'control_dt':env.dt,
        'checks':{'finite_states':all(r['finite'] for r in rows),'positive_body_masses':bool((model.body_mass[1:]>0).all()),
                  'unit_hinge_axes':bool(np.allclose(axes[model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE],1)),
                  'ordered_joint_limits':bool((model.jnt_range[:,0]<=model.jnt_range[:,1]).all()),
                  'contact_observed':any(r['contacts']>0 for r in rows)},
        'scope':'Zero torque human SMPL free dynamics; falling is expected and not a numerical-instability failure. Coordinate roundtrip and shape-variation tests remain separate.'}
json_write(args.run_dir/'infrastructure.json',result)
env.close()
assert all(result['checks'].values()),result['checks']
