"""Runtime goal input through a socket, official DART state transition intact."""
import argparse,json,os,random,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--socket',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--ignore',action='store_true');p.add_argument('--clock',choices=['realtime','paused'],default='realtime');p.add_argument('--goal-start',type=float,nargs=2,default=[1.,0.]);a=p.parse_args()
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'utils'))
    from motiogen.config import load_config,data_root
    from motiogen.runtime import json_write
    from motiogen.runtime_feedback import EventReceiver,Playback,stamp
    from motiogen.motion import save_motion,HUMANML_NAMES
    data=data_root(load_config());repo=data/'baselines/DART'
    os.chdir(data/'outputs/runtime/dart');sys.path.insert(0,str(data/'staged_dependencies/pytorch3d'));sys.path.insert(0,str(repo))
    import numpy as np
    if 'float' not in np.__dict__:np.float=float
    import torch,clip,tyro,yaml
    original_clip=clip.load
    def local_clip(name,*args,**kwargs):
        kwargs['download_root']=str(data/'outputs/assets/omnicontrol');return original_clip(name,*args,**kwargs)
    clip.load=local_clip
    from control.train_reach_location_mld import ReachLocationArgs,load_motion_model
    from control.env.env_reach_location_mld import EnvReachLocationMLD
    from control.policy.policy import PolicyReachLocationMLP,PolicyReachLocationTransformer
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
    start=stamp();directory=data/'outputs/assets/dart-official/policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100'
    cfg=tyro.extras.from_yaml(ReachLocationArgs,yaml.safe_load((directory/'args.yaml').read_text()))
    cfg.num_envs=cfg.batch_size=cfg.env_args.num_envs=1;cfg.num_steps=cfg.env_args.num_steps=1000
    cfg.env_args.enable_export=False;cfg.device=torch.device('cuda');cfg.seed=a.seed;cfg.init_data_path='./data/stand.pkl'
    da,dm,va,vm,diffusion,dataset=load_motion_model(cfg);env=EnvReachLocationMLD(cfg,da,dm,va,vm,diffusion,dataset)
    cfg.policy_args.observation_structure=env.observation_structure;cfg.policy_args.action_structure=env.action_structure
    cls=PolicyReachLocationMLP if cfg.policy_args.architecture=='mlp' else PolicyReachLocationTransformer
    policy=cls(cfg.policy_args).cuda().eval();policy.load_state_dict(torch.load(directory/'iter_5000.pth',map_location='cpu'));policy.requires_grad_(False)
    active=torch.tensor([[*a.goal_start,0.]],device='cuda');obs,_=env.reset(goal_location=active,goal_texts=['walk'])
    initial=env.get_global_joints()[0].detach().cpu().numpy();captured=[]
    transition=env.get_new_state_human
    def capture(action):
        result=transition(action);captured.append({k:v.detach().cpu().numpy() for k,v in result[2].items() if torch.is_tensor(v)});return result
    env.get_new_state_human=capture
    a.run_dir.mkdir(parents=True,exist_ok=True);receiver=EventReceiver(a.socket,a.run_dir/'input_events.jsonl')
    torch.cuda.synchronize();cold=(stamp()-start)/1e9
    playback=Playback(a.run_dir,30,20,a.clock=='realtime',initial[0]);playback.submit(initial,0,-1)
    json_write(a.run_dir/'ready.json',{'start_ns':playback.start_ns,'cold_seconds':cold,'worker_pid':os.getpid(),
        'argv':sys.argv,'initial_condition':a.goal_start,'future_payload_in_initial_cache':False})
    chunks=[initial];traces=[];frame=len(initial);version=0;event_seen=False
    try:
        while frame<600:
            if a.clock=='realtime':
                # Bound look-ahead to one half second. Pre-generating the whole
                # trajectory before the unknown event would invalidate the test.
                while frame/30>(stamp()-playback.start_ns)/1e9+.5:time.sleep(.005)
                if stamp()-playback.start_ns>=20e9:break
            elif frame>=150 and not event_seen:
                while stamp()-playback.start_ns<5.05e9:time.sleep(.005)
            for event in receiver.poll():
                event_seen=True
                if not a.ignore:
                    active=torch.tensor([[*event['payload']['goal_xy'],0.]],device='cuda')
                    env.reset_goal(None,goal_location=active,goal_texts=['walk']);obs=env.get_observation();version=event['version']
            torch.manual_seed(a.seed*100000+len(traces));torch.cuda.synchronize();consume=stamp()
            native_goal_before=env.state_goal['goal_location'][0].detach().cpu().tolist()
            with torch.no_grad():
                action=policy.get_action_and_value(obs)[0]
                obs,reward,success,terminated,truncated,info=env.step(action,next_goal_location=active,next_goal_texts=np.array(['walk']))
            torch.cuda.synchronize();finish=stamp();native=captured.pop();future=native['joints'][0]
            idx=len(traces);np.savez_compressed(a.run_dir/f'primitive_{idx:03d}.npz',**native)
            commit=playback.submit(future,version,idx);chunks.append(future)
            traces.append({'frame':frame,'condition_version':version,'condition_consumed_ns':consume,'compute_end_ns':finish,'commit_ns':commit,
                'seconds':(finish-consume)/1e9,'goal':active[0].cpu().tolist(),'native_goal_before':native_goal_before,
                'native_goal_after':env.state_goal['goal_location'][0].detach().cpu().tolist(),
                'terminated':bool(terminated.item()),'truncated':bool(truncated.item()),'success':bool(success.item())})
            frame+=len(future)
            if idx%15==0:print(json.dumps(traces[-1]),flush=True)
        playback.finish();xyz=np.concatenate(chunks)[:600]
        save_motion(a.run_dir/'motion.npz',xyz,HUMANML_NAMES,30,{'units':'m','axes':'right-handed Z-up','method':'DART','protocol':'runtime_reveal_v1'})
        json_write(a.run_dir/'runtime_trace.json',{'steps':traces,'cold_seconds':cold,'ignored':a.ignore,'clock':a.clock,'event_received':event_seen,
            'state_transition':'Official EnvReachLocationMLD.step with original termination/reset semantics',
            'execution':'Independent kinematic playback; not physical simulation'})
    finally:receiver.close()

if __name__=='__main__':main()
