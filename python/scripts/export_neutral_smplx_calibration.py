"""Neutral SMPL-X calibration only; template fingers are not observations."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from human_body_prior.body_model.body_model import BodyModel
from smplx.joint_names import JOINT_NAMES

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.runtime import json_write,checksum
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2)
model=BodyModel(bm_fname=str(a.model),num_betas=16).eval().requires_grad_(False)
with torch.no_grad():
    body=model(root_orient=torch.zeros(1,3),pose_body=torch.zeros(1,63),pose_hand=torch.zeros(1,90),betas=torch.zeros(1,16),trans=torch.zeros(1,3))
neutral=body.Jtr[0].numpy();parents=model.kintree_table[0].numpy().astype(int);parents[0]=-1
aliases={f'{side}_{finger}{i}':side.title()+'Hand'+finger.title()+str(i) for side in ['left','right'] for finger in ['index','middle','ring','pinky','thumb'] for i in [1,2,3]}
json_write(a.output,{'names':[aliases.get(n,n) for n in JOINT_NAMES[:len(neutral)]],
    'neutral_joints':neutral.tolist(),'parents':parents.tolist(),'rotation_joint_count':22,
    'sha256':checksum(a.model),'source_path':str(a.model.resolve()),'shape':[0.]*16,'gender':'neutral',
    'hand_scope':'Neutral anatomical reference axes only; no observed finger articulation is inferred.'})
print(a.output)
