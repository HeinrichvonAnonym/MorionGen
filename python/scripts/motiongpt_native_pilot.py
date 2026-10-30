"""Official pretrained MLM -> VQVae decode, with untouched native sampling."""
import argparse,json,os,random,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);args=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import load_config,data_root
from motiogen.runtime import json_write,checksum
from motiogen.motion import save_motion,canonicalize,HUMANML_NAMES
root=data_root(load_config());repo=root/'baselines/MotionGPT';assets=root/'outputs/assets'
sys.path.insert(0,str(repo));os.chdir(repo)
import numpy as np
import torch
from mGPT.archs.mgpt_lm import MLM
from mGPT.archs.mgpt_vq import VQVae
from mGPT.data.humanml.scripts.motion_process import recover_from_ric
torch.set_num_threads(4)
start=time.monotonic();args.run_dir.mkdir(parents=True,exist_ok=True)
checkpoint=assets/'motiongpt-official/motiongpt_s3_h3d.tar'
state=torch.load(checkpoint,map_location='cpu')['state_dict']
lm=MLM(model_path=str(assets/'motiongpt-flan-t5-base'),stage='lm_instruct',motion_codebook_size=512)
vae=VQVae(nfeats=263,down_t=2,stride_t=2,norm=None)
loads={}
for name,model in [('lm',lm),('vae',vae)]:
    part={k[len(name)+1:]:v for k,v in state.items() if k.startswith(name+'.')}
    result=model.load_state_dict(part,strict=True)
    loads[name]={'tensors':len(part),'strict':True,'missing':result.missing_keys,'unexpected':result.unexpected_keys}
    model.eval().cuda()
del state
meta=assets/'motiongpt-official/t2m/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
mean=torch.tensor(np.load(meta/'mean.npy'),device='cuda',dtype=torch.float32)
std=torch.tensor(np.load(meta/'std.npy'),device='cuda',dtype=torch.float32)
torch.cuda.synchronize();cold=time.monotonic()-start
json_write(args.run_dir/'load_receipt.json',{'strict_loads':loads,'checkpoint_sha256':checksum(checkpoint),
    'normalization':{str(x):checksum(x) for x in [meta/'mean.npy',meta/'std.npy']},'cold_seconds':cold,
    'native_entry':'mGPT.models.mgpt.MotionGPT.forward t2m: lm.generate_direct(do_sample=True) then vae.decode',
    'length_policy':'Requested length recorded only; native model determines decoded length. No stretching/truncation.'})
rows=[]
for i,request in enumerate(json.loads(args.inputs.read_text())['requests']):
    random.seed(request['seed']);np.random.seed(request['seed']);torch.manual_seed(request['seed']);torch.cuda.manual_seed_all(request['seed'])
    row=dict(request);folder=args.run_dir/f'case_{i:03d}';folder.mkdir()
    try:
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.monotonic()
        with torch.no_grad():
            tokens,text=lm.generate_direct([request['prompt']],do_sample=True)
            motion=vae.decode(tokens[0]);features=motion*std+mean
            joints=recover_from_ric(features,22)[0]
        torch.cuda.synchronize();elapsed=time.monotonic()-t
        xyz,_=canonicalize(joints.cpu().numpy(),'Y-up')
        save_motion(folder/'motion.npz',xyz,HUMANML_NAMES,20,{'units':'m','axes':'right-handed Z-up','method':'MotionGPT','native':True})
        np.savez_compressed(folder/'native.npz',tokens=tokens[0].cpu().numpy(),normalized_features=motion[0].cpu().numpy(),features=features[0].cpu().numpy())
        row.update(status='success',actual_frames=len(xyz),seconds=elapsed,rtf=elapsed/(len(xyz)/20),peak_allocated_bytes=torch.cuda.max_memory_allocated(),cleaned_text=text,output=str(folder))
    except Exception as exc:row.update(status='failed',reason=f'{type(exc).__name__}: {exc}')
    rows.append(row);json_write(args.run_dir/'generation.json',{'cold_seconds':cold,'rows':rows})
    print(json.dumps(row),flush=True)
assert any(r['status']=='success' for r in rows),'No successful native generation'
