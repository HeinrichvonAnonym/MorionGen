"""Official UHC fixed-policy tracking with per-step force/contact receipts."""
import argparse,copy,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--count',type=int,default=1);p.add_argument('--skip',type=int,default=0);args=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import data_root,load_config
from motiogen.runtime import json_write,checksum
root=data_root(load_config());repo=root/'baselines/UHC';assets=root/'outputs/assets'
sys.path.insert(0,str(repo));os.chdir(repo)
import numpy as np
import torch
import joblib
from uhc.utils.config_utils.copycat_config import Config
from uhc.agents.agent_copycat import AgentCopycat
from uhc.utils.flags import flags

args.run_dir.mkdir(parents=True,exist_ok=True)
work=args.run_dir/'work';(work/'data').mkdir(parents=True)
(work/'data/smpl').symlink_to(assets/'closd-official/data/smpl',target_is_directory=True)
(work/'assets').symlink_to(repo/'assets',target_is_directory=True)
cfg=Config(cfg_id='uhc_implicit',base_dir=str(repo),create_dirs=False)
cfg.base_dir=str(work);cfg.model_dir=str(work/'models');Path(cfg.model_dir).mkdir()
(Path(cfg.model_dir)/'iter_19000.p').symlink_to(assets/'uhc-official/uhc_implicit.p')
cfg.output_dir=cfg.result_dir=str(work/'results');Path(cfg.result_dir).mkdir()
cfg.log_dir=str(work/'log');Path(cfg.log_dir).mkdir()
cfg.update(SimpleNamespace(test=True,num_threads=1,gpu_index=0,epoch=19000,show_noise=False,resume=None,no_log=True,debug=False,
    mode='stats',render_video=False,render_rfc=False,render=False,hide_expert=False,no_fail_safe=True,focus=False,output=str(work),shift_expert=False,smplx=False,hide_im=False))
cfg.fail_safe=False;flags.debug=False
source=repo/'sample_data/amass_copycat_take5_test_small.pkl'
all_data=joblib.load(source);keys=sorted(all_data)[args.skip:args.skip+args.count]
sample=args.run_dir/'references.pkl';joblib.dump({k:all_data[k] for k in keys},sample)
cfg.data_specs['file_path']=str(sample);cfg.data_specs.pop('test_file_path',None)
cfg.data_specs['neutral_path']=str(repo/'sample_data/standing_neutral.pkl')
torch.set_num_threads(4);torch.set_default_dtype(torch.float64);np.random.seed(1);torch.manual_seed(1)
t=time.monotonic();agent=AgentCopycat(cfg,torch.float64,torch.device('cpu'),training=True,checkpoint_epoch=19000)
agent.policy_net.eval();agent.value_net.eval();cold=time.monotonic()-t
json_write(args.run_dir/'load_receipt.json',{'checkpoint_sha256':checksum(assets/'uhc-official/uhc_implicit.p'),
    'reference_sha256':checksum(source),'selected_keys':keys,'cold_seconds':cold,'state_dim':agent.state_dim,
    'action_dim':agent.action_dim,'native_evaluator':'AgentCopycat.eval_seq','optimizer_steps':0,
    'inference_device':'cpu (official eval_uhc.py choice)','fail_safe':False,'residual_force_mode':cfg.residual_force_mode})
original_step=agent.env.step;trace=[]
def instrumented_step(action):
    started=time.monotonic();result=original_step(action)
    trace.append({'seconds':time.monotonic()-started,'qfrc_applied':agent.env.data.qfrc_applied.copy(),
        'contacts':agent.env.data.ncon,'qpos':agent.env.data.qpos.copy(),'action':np.asarray(action).copy()})
    return result
agent.env.step=instrumented_step
rows=[];normalizer=copy.deepcopy(agent.running_state)
for ordinal,(key,rfc) in enumerate((key,rfc) for key in keys for rfc in [1,0]):
    agent.running_state=copy.deepcopy(normalizer)
    np.random.seed(1);torch.manual_seed(1);agent.env.seed(1)
    agent.env.rfc_rate=float(rfc)
    trace.clear();t=time.monotonic()
    result=agent.eval_seq(key,agent.data_loader);elapsed=time.monotonic()-t
    folder=args.run_dir/f'reference_{ordinal:02d}';folder.mkdir()
    joblib.dump(result,folder/'native_result.pkl')
    np.savez_compressed(folder/'execution_trace.npz',**{k:np.asarray([r[k] for r in trace]) for k in trace[0]})
    row={'source':key,'rfc_enabled':rfc,'intervention':'Fixed-policy inference: residual-force multiplier only; action dimensions and weights unchanged',
         'seconds':elapsed,'steps':len(trace),'reference_frames':len(all_data[key]['pose_aa']),
         'completion':float(result['percent']),'fail_safe':bool(result['fail_safe']),
         'qfrc_root_force_rms':float(np.sqrt(np.mean(np.asarray([r['qfrc_applied'][:3] for r in trace])**2))),
         'output':str(folder),'metrics':{k:float(v) for k,v in result.items() if isinstance(v,(float,np.floating,int))}}
    rows.append(row);json_write(args.run_dir/'tracking.json',{'rows':rows,'cold_seconds':cold,
        'scope':'Fixed official pretrained implicit-RFC policy; original motion retargeting and PHC common-reference comparison require separate receipts.'})
    print(json.dumps(row),flush=True)
