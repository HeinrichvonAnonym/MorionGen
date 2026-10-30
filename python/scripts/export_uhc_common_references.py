"""Export native UHC reference FK for auditable PHC retargeting, no policy run."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'utils'))
from motiogen.config import data_root,load_config
from motiogen.runtime import json_write,checksum
data=data_root(load_config());repo=data/'baselines/UHC';assets=data/'outputs/assets'
sys.path.insert(0,str(repo));os.chdir(repo)
import joblib
import numpy as np
import torch
from uhc.utils.config_utils.copycat_config import Config
from uhc.data_loaders.dataset_amass_single import DatasetAMASSSingle
from uhc.envs.humanoid_im import HumanoidEnv

torch.set_num_threads(4);torch.set_default_dtype(torch.float64)
a.run_dir.mkdir(parents=True,exist_ok=True)
work=a.run_dir/'work';(work/'data').mkdir(parents=True)
(work/'data/smpl').symlink_to(assets/'closd-official/data/smpl',target_is_directory=True)
(work/'assets').symlink_to(repo/'assets',target_is_directory=True)
cfg=Config(cfg_id='uhc_implicit',base_dir=str(repo),create_dirs=False)
cfg.base_dir=str(work)
cfg.update(SimpleNamespace(test=True,num_threads=1,gpu_index=0,epoch=19000,show_noise=False,resume=None,no_log=True,debug=False,
    mode='stats',render_video=False,render_rfc=False,render=False,hide_expert=False,no_fail_safe=True,focus=False,
    output=str(work),shift_expert=False,smplx=False,hide_im=False))
source=repo/'sample_data/amass_copycat_take5_test_small.pkl'
cfg.data_specs['file_path']=str(source);cfg.data_specs.pop('test_file_path',None)
cfg.data_specs['neutral_path']=str(repo/'sample_data/standing_neutral.pkl')
loader=DatasetAMASSSingle(cfg.data_specs,data_mode='train')
keys=sorted(loader.pickle_data)[:8]
sample=loader.get_sample_from_key(take_key=keys[0],full_sample=True,fr_start=-1)
env=HumanoidEnv(cfg,init_expert=sample,data_specs=cfg.data_specs,mode='test',no_root=cfg.no_root)
rows=[]
for ordinal,key in enumerate(keys):
    sample=loader.get_sample_from_key(take_key=key,full_sample=True,fr_start=-1)
    env.load_expert(sample)
    folder=a.run_dir/f'reference_{ordinal:02d}';folder.mkdir()
    (folder/'humanoid.xml').write_bytes(env.smpl_robot.export_xml_string())
    qpos=np.asarray(env.expert['qpos'])
    positions=[];rotations=[]
    indices=[i for i,n in enumerate(env.model.body_names) if n!='world']
    for q in qpos:
        env.data.qpos[:]=q;env.data.qvel[:]=0;env.sim.forward()
        positions.append(env.data.body_xpos[indices].copy())
        rotations.append(env.data.body_xquat[indices][:,[1,2,3,0]].copy())
    names=[env.model.body_names[i] for i in indices]
    dt=float(env.dt)
    np.savez_compressed(folder/'reference.npz',qpos=qpos,joints=np.asarray(positions),
        rotations_xyzw=np.asarray(rotations),body_names=names,pose_aa=np.asarray(sample['pose_aa']),
        trans=np.asarray(sample['trans']),beta=np.asarray(sample['beta']),fps=1/dt)
    rows.append({'source':key,'frames':len(qpos),'native_dt':dt,'native_fps':1/dt,
        'path':str(folder/'reference.npz'),'sha256':checksum(folder/'reference.npz'),
        'body_names':names,'pose_shape':list(np.asarray(sample['pose_aa']).shape)})
    print(json.dumps(rows[-1]),flush=True)
json_write(a.run_dir/'reference_export.json',{'rows':rows,'source':str(source),'source_sha256':checksum(source),
    'scope':'Official UHC source preprocessing, body shape and MuJoCo FK. No new policy inference or training; PHC retargeting and reference errors must be measured separately.'})
