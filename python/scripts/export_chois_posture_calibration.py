"""Recreate source-window SMPL-X neutral frames, with actual shape/gender."""
import json,sys
from pathlib import Path
import joblib,numpy as np,torch
from human_body_prior.body_model.body_model import BodyModel
from smplx.joint_names import JOINT_NAMES
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config,data_root
from motiogen.runtime import checksum,json_write
assets=data_root(load_config())/'outputs/assets/chois/processed_data'
source=assets/'cano_test_diffusion_manip_window_120_joints24.p'
windows=[v for v in joblib.load(source).values() if (assets/'omomo_text_anno_json_data'/(v['seq_name']+'.json')).exists() and len(v['motion'])>=120 and v['start_t_idx']==0]
analysis=json.loads((ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json').read_text());models={};rows={}
torch.set_num_threads(2)
for row in analysis['rows']:
    if row['method']!='chois':continue
    idx=row['task_spec'].get('native_window_index',0)
    if str(idx) in rows:continue
    w=windows[idx];gender=str(np.asarray(w['gender']).item());path=assets/'smpl_all_models/smplx'/f'SMPLX_{gender.upper()}.npz'
    if gender not in models:models[gender]=BodyModel(bm_fname=str(path),num_betas=16).eval().requires_grad_(False)
    model=models[gender]
    with torch.no_grad():body=model(root_orient=torch.zeros(1,3),pose_body=torch.zeros(1,63),pose_hand=torch.zeros(1,90),betas=torch.as_tensor(w['betas'],dtype=torch.float32).reshape(1,-1),trans=torch.zeros(1,3))
    neutral=body.Jtr[0].numpy();parents=model.kintree_table[0].cpu().numpy().astype(int);parents[0]=-1
    names=JOINT_NAMES[:len(neutral)]
    aliases={f'{side}_{finger}{i}':side.title()+'Hand'+finger.title()+str(i) for side in ['left','right'] for finger in ['index','middle','ring','pinky','thumb'] for i in [1,2,3]}
    names=[aliases.get(n,n) for n in names]
    rows[str(idx)]={'names':names,'neutral_joints':neutral.tolist(),'parents':parents.tolist(),'rotation_joint_count':22,
        'sha256':checksum(path),'source_path':str(path),'shape':np.asarray(w['betas']).tolist(),'gender':gender,'sequence':w['seq_name'],
        'axes':'Native SMPL-X neutral Y-up; generated world rotations already Z-up',
        'hand_scope':'Native predicted wrist orientation; neutral hand geometry is calibration, not observed finger articulation. Upstream sets unpredicted finger pose to zero.'}
json_write(ROOT/'develop_logs/test_runs/20261028_supplement/chois_skeleton_calibration.json',{'source_sha256':checksum(source),'windows':rows});print(len(rows))
