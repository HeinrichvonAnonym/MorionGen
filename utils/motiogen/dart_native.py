"""Fixed pretrained DART policy, official step/state update, no pose extrapolation."""
import json
import random
import time
from pathlib import Path


def run(args, data, root):
    import numpy as np
    import torch
    import tyro
    import yaml
    from control.train_reach_location_mld import ReachLocationArgs, load_motion_model
    from control.env.env_reach_location_mld import EnvReachLocationMLD
    from control.policy.policy import PolicyReachLocationMLP, PolicyReachLocationTransformer
    from motiogen.motion import HUMANML_NAMES, save_motion
    from motiogen.metrics import evaluate
    from motiogen.runtime import json_write, checksum

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'motion.npz').exists():
        raise ValueError('Refusing to overwrite experiment evidence')
    from .config import load_config
    from .weight_identity import bind,verify_unchanged
    weight_binding=bind(load_config(),'dart',out)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    policy_dir = data/'outputs/assets/dart-official/policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100'
    cfg = tyro.extras.from_yaml(ReachLocationArgs, yaml.safe_load((policy_dir/'args.yaml').read_text()))
    cfg.num_envs = cfg.batch_size = cfg.env_args.num_envs = 1
    cfg.num_steps = cfg.env_args.num_steps = 1000
    cfg.env_args.enable_export = False
    cfg.device = torch.device('cuda')
    cfg.seed = args.seed
    cfg.init_data_path = './data/stand.pkl'
    da, dm, va, vm, diffusion, dataset = load_motion_model(cfg)
    env = EnvReachLocationMLD(cfg, da, dm, va, vm, diffusion, dataset)
    cfg.policy_args.observation_structure = env.observation_structure
    cfg.policy_args.action_structure = env.action_structure
    agent_class = PolicyReachLocationMLP if cfg.policy_args.architecture == 'mlp' else PolicyReachLocationTransformer
    agent = agent_class(cfg.policy_args).cuda().eval()
    agent.load_state_dict(torch.load(policy_dir/'iter_5000.pth', map_location='cpu'))
    agent.requires_grad_(False)
    goal0 = torch.tensor([[*args.goal_start,0.]],device='cuda')
    goal1 = torch.tensor([[*args.goal_next,0.]],device='cuda')
    observation, _ = env.reset(goal_location=goal0, goal_texts=['walk'])
    initial = env.get_global_joints()[0].detach().cpu().numpy()
    captured = []
    native_transition = env.get_new_state_human
    def capture(action):
        result = native_transition(action)
        captured.append(result[2]['joints'][0].detach().cpu().numpy())
        return result
    env.get_new_state_human = capture
    torch.cuda.synchronize()
    cold = time.perf_counter()-start
    trace, boundaries = [], []
    chunks = [initial]
    frame = len(initial)
    changed = False
    active = goal0
    execution_start=time.perf_counter()
    while frame < args.frames:
        scheduled=frame/30
        if args.clock=='realtime':
            time.sleep(max(0.,execution_start+scheduled-time.perf_counter()))
        wall_elapsed=time.perf_counter()-execution_start
        if (wall_elapsed>=5. if args.clock=='realtime' else frame>=150) and not changed:
            changed = True
            if args.goal_feedback:
                active = goal1
                env.reset_goal(None, goal_location=active, goal_texts=['walk'])
                observation = env.get_observation()
        # Matched action noise at the same primitive index in both conditions.
        torch.manual_seed(args.seed*100000 + len(trace))
        torch.cuda.synchronize()
        tick = time.perf_counter()
        with torch.no_grad():
            action = agent.get_action_and_value(observation)[0]
            observation, reward, success, terminated, truncated, info = env.step(
                action, next_goal_location=active, next_goal_texts=np.array(['walk']))
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-tick
        boundaries.append(frame)
        future = captured.pop()
        chunks.append(future)
        trace.append({'frame':frame,'seconds':elapsed,'success':bool(success.item()),
                      'request_wall_seconds':wall_elapsed,'scheduled_motion_seconds':scheduled,
                      'completion_wall_seconds':time.perf_counter()-execution_start,
                      'schedule_lag_seconds':max(0.,wall_elapsed-scheduled) if args.clock=='realtime' else None,
                      'terminated':bool(terminated.item()),'truncated':bool(truncated.item()),
                      'goal':active[0].cpu().tolist()})
        frame += len(future)
        print(json.dumps(trace[-1]),flush=True)
    xyz = np.concatenate(chunks)[:args.frames]
    verify_unchanged(weight_binding)
    if not np.isfinite(xyz).all():
        raise ValueError('Non-finite native output')
    save_motion(out/'motion.npz',xyz,HUMANML_NAMES,30,
        {'units':'m','axes':'right-handed Z-up','model':'DART','seed':args.seed,
         'variant':'official env.step, same target retained after success, original termination reset'})
    motion={'joints':xyz,'joint_names':HUMANML_NAMES,'fps':30,'chunk_boundaries':np.array(boundaries)}
    json_write(out/'metrics.json',evaluate(motion,{'name':'native_goal_change'}))
    requested=np.repeat(goal0.cpu().numpy(),len(xyz),axis=0)
    requested[150:]=goal1.cpu().numpy()
    errors=np.linalg.norm(xyz[:,0,:2]-requested[:,:2],axis=-1)
    json_write(out/'trace.json',{'steps':trace,'cold_start_seconds':cold,'clock':args.clock,
        'requested_goals':[{'frame':0,'position':goal0[0].cpu().tolist()},
                           {'frame':150,'position':goal1[0].cpu().tolist()}],
        'execution_clock_definition':'Native env.step paced at primitive boundaries; no pose extrapolation or external dynamics controller',
        'requested_update_frame':150,'applied_update_frame':next((r['frame'] for r in trace if r['goal']==goal1[0].cpu().tolist()),None),
        'goal_error_xy_m':errors.tolist(),'policy_sha256':checksum(policy_dir/'iter_5000.pth'),
        'state_transition':'EnvReachLocationMLD.step -> get_new_state_human -> get_blended_feature',
        'goal_update':bool(args.goal_feedback),'gpu_peak_bytes':torch.cuda.max_memory_allocated()})
    json_write(out/'intervention.json',{'condition':{'goal_update':args.goal_feedback},
        'calls':{'native_step':len(trace),'native_state_transition':len(trace)},
        'scope':'Goal-update intervention only; native human-state feedback retained in both conditions'})
