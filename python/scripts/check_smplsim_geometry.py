"""CPU-only native human model geometry and timestep checks, no controller claim."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
repo=DATA/'baselines/SMPLSim'
sys.path.insert(0,str(repo));sys.path.insert(0,str(ROOT/'utils'))
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'unicode':str,'str':str}.items():
    if name not in np.__dict__:setattr(np,name,value)
import torch
import mujoco
from omegaconf import OmegaConf
from smpl_sim.envs.humanoid_env import HumanoidEnv
from motiogen.runtime import json_write

OUT=ROOT/'develop_logs/test_runs/20261028_supplement/smplsim_geometry'
OUT.mkdir(exist_ok=True)
cfg=OmegaConf.load(repo/'smpl_sim/data/cfg/config.yaml')
cfg.env=OmegaConf.load(repo/'smpl_sim/data/cfg/env/base_env.yaml')
cfg.robot=OmegaConf.load(repo/'smpl_sim/data/cfg/robot/smpl_humanoid.yaml')
cfg.robot.smpl_data_dir=str(DATA/'outputs/assets/closd-official/data/smpl')
cfg.env.render_mode=None;cfg.headless=True;cfg.env.control_mode='torque';cfg.env.enable_early_termination=False
torch.set_num_threads(2)
env=HumanoidEnv(cfg)
assert env.robot is not None, 'Native body assets required; fallback XML is not accepted'
rows=[];started=time.monotonic()
for gender in [0,1,2]:
    for shape in [0.,1.]:
        beta=torch.zeros((1,10));beta[0,0]=shape
        env.robot.load_from_skeleton(betas=beta,gender=[gender])
        xml=env.robot.export_xml_string()
        xmlpath=OUT/f'gender{gender}_beta0_{shape:g}.xml';xmlpath.write_bytes(xml)
        for scale in [1.,.5]:
            model=mujoco.MjModel.from_xml_string(xml.decode())
            model.opt.timestep=float(env.mj_model.opt.timestep)*scale
            data=mujoco.MjData(model)
            # Raise the free human clear of the ground, then let it fall.
            data.qpos[2]=1.1
            mujoco.mj_forward(model,data)
            quat_errors=[]
            for q in data.xquat:
                matrix=np.empty(9);recovered=np.empty(4)
                mujoco.mju_quat2Mat(matrix,q);mujoco.mju_mat2Quat(recovered,matrix)
                quat_errors.append(min(np.linalg.norm(q-recovered),np.linalg.norm(q+recovered)))
            finite=True;contact_steps=0;max_speed=0.;max_penetration=0.
            for _ in range(round(2./model.opt.timestep)):
                mujoco.mj_step(model,data)
                finite=finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
                contact_steps+=data.ncon>0
                max_speed=max(max_speed,float(np.abs(data.qvel).max()))
                if data.ncon:max_penetration=max(max_penetration,float(max(0.,-min(c.dist for c in data.contact))))
            hinge=model.jnt_type==mujoco.mjtJoint.mjJNT_HINGE
            checks=dict(finite_states=finite,positive_masses=bool((model.body_mass[1:]>0).all()),
                unit_hinge_axes=bool(np.allclose(np.linalg.norm(model.jnt_axis[hinge],axis=1),1)),
                ordered_limits=bool(np.all(model.jnt_range[:,0]<=model.jnt_range[:,1])),
                quaternion_matrix_roundtrip=bool(max(quat_errors)<1e-10),contact_observed=bool(contact_steps>0),
                no_mujoco_warnings=all(w.number==0 for w in data.warning))
            row=dict(gender=gender,beta0=shape,timestep=float(model.opt.timestep),simulated_seconds=float(data.time),
                nq=model.nq,nv=model.nv,nu=model.nu,total_mass=float(model.body_mass.sum()),
                contact_steps=int(contact_steps),max_abs_generalized_velocity=max_speed,
                max_penetration_m=max_penetration,quaternion_roundtrip_error=max(quat_errors),
                warnings=[int(w.number) for w in data.warning],checks=checks,
                xml=str(xmlpath),xml_sha256=hashlib.sha256(xml).hexdigest())
            rows.append(row);print(json.dumps(row),flush=True)
            json_write(OUT/'checks.json',dict(rows=rows,elapsed_seconds=time.monotonic()-started,
                protocol='Three native SMPL genders x neutral/+1 first shape coefficient x native/half timestep, two-second zero-actuation free dynamics.',
                scope='Infrastructure only. Falling is expected. Finite trajectories do not establish controller tracking, task success, or biomechanical safety. Shape perturbation is a diagnostic, not population coverage.'))
env.close()
assert len(rows)==12 and all(all(r['checks'].values()) for r in rows)
