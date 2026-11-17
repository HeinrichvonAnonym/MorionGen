"""Pinned HumanML features with an explicit, continuous yaw-velocity chart."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT=Path(__file__).resolve().parents[2]
data_root=Path(json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['local_roots']['andy']).parents[2]
sys.path.insert(0,str(data_root/'baselines/OmniControl'))
import torch
torch.set_num_threads(1)
if 'float' not in np.__dict__:np.float=float
from data_loaders.humanml.scripts.motion_process import extract_features,recover_from_ric
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain


def process(path):
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    y_to_z=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    p=data['joints']@y_to_z
    features=extract_features(p.copy(),.002,torch.from_numpy(t2m_raw_offsets),t2m_kinematic_chain,[2,1,17,16],[8,11],[7,10])
    # Mirror the upstream heading definition, using positions only. The source
    # arcsin(relative_quaternion_y) folds at quaternion sign changes. A continuous
    # half-yaw difference encodes the same rotations without that chart error.
    across=p[:,1]-p[:,2]+p[:,17]-p[:,16]
    across/=np.linalg.norm(across,axis=-1,keepdims=True)
    forward=np.cross(np.array([0.,1,0]),across)
    forward=gaussian_filter1d(forward,20,axis=0,mode='nearest')
    if np.any(np.linalg.norm(forward,axis=-1)<1e-9):raise ValueError('Heading is degenerate')
    yaw=-np.unwrap(np.arctan2(forward[:,0],forward[:,2]))/2
    yaw[0]=0.  # Upstream IK explicitly sets the first root quaternion to identity.
    features[:,0]=np.diff(yaw)
    decoded=recover_from_ric(torch.from_numpy(features).float(),22).numpy().astype(float)
    offset=p[0,0]*np.array([1,0,1])
    decoded+=offset
    raw=path.parent/'raw';raw.mkdir(exist_ok=True)
    np.save(raw/'features.npy',features[None,None])
    (path.parent/'codec_frame.json').write_text(json.dumps({
        'version':'continuous_half_yaw_1','world_offset_y_up_m':offset.tolist(),
        'orientation_information':'derived_only_from_retained_positions',
        'boundary':'last position frame supplies velocity, not a decoded output frame'})+'\n')
    data['joints']=decoded@y_to_z.T;data['times']=data['times'][:len(decoded)]
    np.savez_compressed(path,**data)


for line in sys.stdin:
    try:
        process(Path(json.loads(line)['motion']))
        print('K_CODEC '+json.dumps({'status':'ok'}),flush=True)
    except Exception as error:
        print('K_CODEC '+json.dumps({'status':'failed','error':str(error)}),flush=True)
