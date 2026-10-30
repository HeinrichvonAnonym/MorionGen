"""Live LLM2Vec encoding of runtime-revealed text; native human ARDY steps."""
import argparse,json,os,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--socket',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--ignore',action='store_true');p.add_argument('--clock',choices=['realtime','paused'],default='realtime');a=p.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'utils'))
    from motiogen.config import load_config,data_root,asset_root
    from motiogen.runtime import json_write
    from motiogen.runtime_feedback import EventReceiver,Playback,stamp
    from motiogen.adapters import Nvidia,joint_aliases
    from motiogen.local_llm import load_encoder
    from motiogen.motion import canonicalize,save_motion
    cfg=load_config();data=data_root(cfg);repo=data/'baselines/ardy';sys.path.insert(0,str(repo));os.chdir(repo)
    import torch,numpy as np
    from ardy.model.load_model import load_model
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    a.run_dir.mkdir(parents=True,exist_ok=True);initial_text='A person walks forward.'
    start=stamp();encoder=load_encoder(cfg,'ardy');cache={};encodes=[]
    class LiveEncoder:
        def get_device(self):return torch.device('cuda')
        def to(self,*args,**kwargs):return self
        def eval(self):return self
        def __call__(self,texts):
            scalar=isinstance(texts,str);requested=[texts] if scalar else list(texts)
            for text in requested:
                if text not in cache:
                    tick=stamp()
                    with torch.no_grad():values,lengths=encoder([text])
                    torch.cuda.synchronize();cache[text]=values.detach()
                    encodes.append({'prompt':text,'start_ns':tick,'end_ns':stamp(),'cache_hit':False})
                    json_write(a.run_dir/'text_encoding.json',{'rows':encodes,'resident_encoder':True,
                        'scope':'Initial prompt may be warmed; every previously unseen text is encoded only when requested during the runtime event loop.'})
            values=torch.cat([cache[text] for text in requested],dim=0)
            return (values[0],1) if scalar else (values,[1]*len(requested))
    live=LiveEncoder();live([initial_text])
    model=load_model(cfg['baselines']['ardy']['model'],device='cuda',text_encoder=live,checkpoints_dir=str(asset_root(cfg)/'ardy'))
    adapter=Nvidia(cfg,'ardy',a.run_dir,{'kind':'online','prompt':initial_text,'duration_seconds':20},a.seed);adapter.model=model
    adapter.fps=model.motion_rep.fps
    aliases=joint_aliases(model.skeleton.name);names=[aliases.get(x,x) for x,_ in model.skeleton.bone_order_names_with_parents]
    state=adapter.initialize_state();torch.manual_seed(a.seed*100000)
    first=adapter.step(state,initial_text)
    initial,initial_rot=canonicalize(first['posed_joints'][0].cpu().numpy(),'Y-up',first['global_rot_mats'][0].cpu().numpy())
    receiver=EventReceiver(a.socket,a.run_dir/'input_events.jsonl');torch.cuda.synchronize();cold=(stamp()-start)/1e9
    playback=Playback(a.run_dir,20,20,a.clock=='realtime',initial[0]);playback.submit(initial,0,-1)
    json_write(a.run_dir/'ready.json',{'start_ns':playback.start_ns,'cold_seconds':cold,'worker_pid':os.getpid(),
        'argv':sys.argv,'initial_condition':initial_text,'future_payload_in_initial_cache':False,'initial_cached_prompts':list(cache),
        'encoder_resident':True,'initial_model_warmup_frames':len(initial)})
    chunks=[initial];rotations=[initial_rot];traces=[];prompt=initial_text;version=0;seen=False
    try:
        while state['frames']<400:
            frame=state['frames']
            if a.clock=='realtime':
                while frame/20>(stamp()-playback.start_ns)/1e9+.5:time.sleep(.005)
                if stamp()-playback.start_ns>=20e9:break
            elif frame>=100 and not seen:
                while stamp()-playback.start_ns<5.05e9:time.sleep(.005)
            for event in receiver.poll():
                seen=True
                if not a.ignore:prompt=event['payload']['prompt'];version=event['version']
            torch.manual_seed(a.seed*100000+frame);torch.cuda.synchronize();consume=stamp()
            decoded=adapter.step(state,prompt);torch.cuda.synchronize();finish=stamp()
            xyz,rot=canonicalize(decoded['posed_joints'][0].cpu().numpy(),'Y-up',decoded['global_rot_mats'][0].cpu().numpy())
            commit=playback.submit(xyz,version,len(traces));chunks.append(xyz);rotations.append(rot)
            traces.append({'frame':frame,'condition_version':version,'condition_consumed_ns':consume,'compute_end_ns':finish,'commit_ns':commit,
                'seconds':(finish-consume)/1e9,'prompt':prompt,'history_frames':state['history'].shape[1]})
            if len(traces)%10==0:print(json.dumps(traces[-1]),flush=True)
        playback.finish();save_motion(a.run_dir/'motion.npz',np.concatenate(chunks)[:400],names,20,
            {'units':'m','axes':'right-handed Z-up','method':'ARDY','protocol':'runtime_reveal_v1'},rotations=np.concatenate(rotations)[:400])
        json_write(a.run_dir/'runtime_trace.json',{'steps':traces,'cold_seconds':cold,'ignored':a.ignore,'clock':a.clock,'event_received':seen,
            'state_transition':'Official autoregressive_step; native prediction history <=160 frames, no spatial wrapper',
            'execution':'Independent kinematic playback; not physical simulation'})
    finally:receiver.close()

if __name__=='__main__':main()
