"""Native text-only MotionLCM checkpoint on the frozen common text requests."""
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import load_config,repo_root
from motiogen.runtime import json_write,checksum
from motiogen.adapters import MotionLCM,require
from motiogen.motion import save_motion,canonicalize,HUMANML_NAMES
cfg=load_config();repo=repo_root(cfg,'motionlcm');sys.path.insert(0,str(repo));os.chdir(repo)
import numpy as np
import torch
torch.set_num_threads(4)
scene={'kind':'official','prompt':'','duration_seconds':9.8}
adapter=MotionLCM(cfg,'motionlcm',args.run_dir,scene,0)
start=time.monotonic();adapter.load();torch.cuda.synchronize();cold=time.monotonic()-start
assert not adapter.model.is_controlnet,'Native text-only checkpoint required'
original=adapter.model.feats2joints;captured=[]
def capture(features):
    captured.append(features.detach().cpu().numpy())
    return original(features)
adapter.model.feats2joints=capture
json_write(args.run_dir/'load_receipt.json',{'checkpoint_sha256':checksum(require(cfg,'motionlcm','motionlcm_humanml.ckpt')),
    'strict_load':True,'controlnet':False,'cold_seconds':cold,'native_entry':'MLD.forward; text-only native checkpoint',
    'length_policy':'Exact request sent to native forward; reject unsupported lengths, never clamp.'})
rows=[]
for i,request in enumerate(json.loads(args.inputs.read_text())['requests']):
    row=dict(request);folder=args.run_dir/f'case_{i:03d}';folder.mkdir();captured.clear();adapter.seed_all(request['seed'])
    try:
        n=int(request['requested_frames'])
        if not 40<=n<=196:raise ValueError('Requested length outside native pilot supported interval [40,196]')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
        with torch.no_grad():joints=adapter.model({'text':[request['text']],'length':[n]})[0][0].cpu().numpy()
        torch.cuda.synchronize();elapsed=time.monotonic()-t
        xyz,_=canonicalize(joints,'Y-up')
        save_motion(folder/'motion.npz',xyz,HUMANML_NAMES,20,{'units':'m','axes':'right-handed Z-up','method':'MotionLCM','native':True})
        features=captured[0][0]*adapter.std+adapter.mean
        np.savez_compressed(folder/'native.npz',normalized_features=captured[0][0],features=features)
        row.update(status='success',actual_frames=len(xyz),seconds=elapsed,rtf=elapsed/(len(xyz)/20),peak_allocated_bytes=torch.cuda.max_memory_allocated(),output=str(folder))
    except Exception as exc:row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
    rows.append(row);json_write(args.run_dir/'generation.json',{'cold_seconds':cold,'rows':rows});print(json.dumps(row),flush=True)
assert any(r['status']=='success' for r in rows)
