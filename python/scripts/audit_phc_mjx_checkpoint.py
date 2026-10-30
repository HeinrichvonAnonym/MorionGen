"""Inspect actual policy schema; never equate matching MLP widths with compatibility."""
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import yaml

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
repo=DATA/'baselines/PHC_MJX'
checkpoint=DATA/'outputs/assets/closd-official/checkpoints/closd/CLoSD_multitask_finetune/Humanoid.pth'
saved=torch.load(checkpoint,map_location='cpu')
network=saved['model']
mlp=yaml.safe_load((repo/'phc_mjx/cfg/learning/im_mlp.yaml').read_text())['mlp']
layers={k:list(v.shape) for k,v in network.items() if 'actor_mlp' in k and k.endswith('weight')}
missing=sorted(set(['policy','value','epoch','optimizer_value','optimizer_policy'])-set(saved))
files=[repo/'phc_mjx/cfg/learning/im_mlp.yaml',repo/'phc_mjx/cfg/env/env_im.yaml',
       repo/'phc_mjx/cfg/robot/smpl_humanoid.yaml',repo/'phc_mjx/envs/humanoid_im.py',
       DATA/'baselines/SMPLSim/smpl_sim/agents/agent_humanoid.py',
       DATA/'baselines/SMPLSim/smpl_sim/envs/humanoid_env.py']
result={'commit':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
        'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'actual_checkpoint_keys':list(saved),'required_loader_keys_missing':missing,
        'actual_actor_weight_shapes':layers,'candidate_actor_config':mlp,
        'hidden_widths_match':list(mlp['units'])==[v[0] for v in layers.values()],
        'native_loader_can_load_this_checkpoint':not missing,
        'default_observation_source_derivation':{
            'body_count':24,'articulated_joint_count':23,'trajectory_samples':1,
            'self_v1':1+23*3+24*6+3+3+23*3,
            'imitation_v1':24*(3+6+3+6)+(1+1+24-1)*3,
            'total_v1':796,'existing_checkpoint_input':574,
            'scope':'Static derivation for default SMPL configuration; not a measured environment rollout.'},
        'frequency':{'simulation_hz':450,'control_substeps':15,'control_hz':30},
        'policy_inference_started':False,'training_started':False,
        'blocker':'No released PHC_MJX policy in the recorded official repository/release audit. Existing CLoSD checkpoint cannot enter its native loader: required keys absent and default input width 796 differs from 574. Matching hidden MLP widths alone does not establish an interchangeable policy.',
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'scope':'No claim that independent policy conversion or retraining is impossible. Neither is silently substituted for native pretrained evaluation.'}
assert missing and result['hidden_widths_match']
output=ROOT/'develop_logs/test_runs/20261028_supplement/phc_mjx_interface_audit.json'
output.write_text(json.dumps(result,indent=2))
print(json.dumps({'native_loadable':not missing,'hidden_widths_match':result['hidden_widths_match'],'missing_keys':missing}))
