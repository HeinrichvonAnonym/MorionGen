"""Use the pinned OmniControl position-derived HumanML encoder/decoder on CPU."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--motion',type=Path);ap.add_argument('--server',action='store_true');a=ap.parse_args()
root=Path(json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['local_roots']['andy']).parents[2]
sys.path.insert(0,str(root/'baselines/OmniControl'))
import torch
torch.set_num_threads(1)
if 'float' not in np.__dict__:np.float=float
from data_loaders.humanml.scripts.motion_process import extract_features,recover_from_ric
from data_loaders.humanml.utils.paramUtil import t2m_raw_offsets,t2m_kinematic_chain

def process(motion):
    with np.load(motion,allow_pickle=False) as z:original={k:z[k] for k in z.files}
    y_to_z=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    p=original['joints']@y_to_z
    features=extract_features(p.copy(),.002,torch.from_numpy(t2m_raw_offsets),t2m_kinematic_chain,[2,1,17,16],[8,11],[7,10])
    decoded=recover_from_ric(torch.from_numpy(features).float(),22).numpy()
    # Restore the root's arbitrary initial horizontal pose using only retained positions.
    src=decoded[0][:,[0,2]];dst=p[0][:,[0,2]];x=src-src.mean(0);y=dst-dst.mean(0)
    u,_,vh=np.linalg.svd(x.T@y);d=np.eye(2);d[1,1]=np.linalg.det(u@vh);rot=u@d@vh
    decoded[:,:,[0,2]]=(decoded[:,:,[0,2]]-src.mean(0))@rot+dst.mean(0)
    error=np.linalg.norm(decoded-p[:len(decoded)],axis=-1)
    raw=motion.parent/'raw';raw.mkdir(exist_ok=True)
    np.save(raw/'features.npy',features[None,None])
    original['joints']=decoded@y_to_z.T
    if 'times' in original:original['times']=original['times'][:len(decoded)]
    np.savez_compressed(motion,**original)
    (motion.parent/'codec.json').write_text(json.dumps({'encoder':'OmniControl.extract_features','decoder':'recover_from_ric',
        'input_frames':len(p),'output_frames':len(decoded),'median_joint_error_m':float(np.median(error)),
        'p95_joint_error_m':float(np.percentile(error,95)),'initial_horizontal_pose':'restored from retained joint positions',
        'head_orientation_information':'position_derived_only'},indent=2)+'\n')

if a.server:
    for line in sys.stdin:
        try:
            process(Path(json.loads(line)['motion']))
            print('K_CODEC '+json.dumps({'status':'ok'}),flush=True)
        except Exception as error:
            print('K_CODEC '+json.dumps({'status':'failed','error':str(error)}),flush=True)
elif a.motion:
    process(a.motion)
else:
    ap.error('--motion or --server is required')
