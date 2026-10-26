"""DART pretrained goal policy connected to observed human poses, without physics."""
import json
import pickle
from pathlib import Path


def run(args, data, root):
    import numpy as np
    import torch
    import tyro
    import yaml
    from pytorch3d import transforms
    from control.train_reach_location_mld import ReachLocationArgs, load_motion_model
    from control.env.env_reach_location_mld import EnvReachLocationMLD
    from control.policy.policy import PolicyReachLocationMLP, PolicyReachLocationTransformer
    from motiogen.feedback import Observation, run_episode, score_root_goals
    from motiogen.motion import HUMANML_NAMES, save_motion
    from motiogen.metrics import evaluate

    policy_dir = data / 'outputs/assets/dart-official/policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100'
    cfg = tyro.extras.from_yaml(ReachLocationArgs, yaml.safe_load((policy_dir / 'args.yaml').read_text()))
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
    agent.load_state_dict(torch.load(policy_dir / 'iter_5000.pth', map_location='cpu'))
    agent.requires_grad_(False)
    with open(data / 'outputs/assets/dart-official/data/stand.pkl', 'rb') as handle:
        initial = pickle.load(handle)
    pose = np.concatenate([initial['transl'][:2], initial['global_orient'][:2], initial['body_pose'][:2]], axis=-1)
    utility = env.primitive_utility

    def body_joints(poses):
        tensor = torch.as_tensor(poses, dtype=torch.float32, device='cuda')
        result = utility.bm_male(betas=torch.zeros(len(tensor), 10, device='cuda'),
            transl=tensor[:, :3], global_orient=transforms.axis_angle_to_matrix(tensor[:, 3:6]),
            body_pose=transforms.axis_angle_to_matrix(tensor[:, 6:].reshape(-1, 21, 3)))
        return result.joints[:, :22]

    initial_root = body_joints(pose)[-1, 0].detach().cpu().numpy()
    goal0 = initial_root + np.array([1., 0., 0.])
    goals = [{'frame': 0, 'position': goal0.tolist()},
             {'frame': args.frames // 2, 'position': (initial_root + [-1., .5, 0.]).tolist()}]

    def generate(observation):
        torch.manual_seed(args.seed * 100000 + observation.frame)
        tensor = torch.as_tensor(observation.human_history, dtype=torch.float32, device='cuda')
        rotation = transforms.axis_angle_to_matrix(tensor[:, 3:].reshape(2, 22, 3))
        # Native features use forward differences. Estimate the next pose from
        # observed constant velocity; no future ground truth enters the adapter.
        next_rotation = rotation[1] @ rotation[0].transpose(-1, -2) @ rotation[1]
        rotation = torch.cat([rotation, next_rotation[None]], dim=0)
        translation = torch.cat([tensor[:, :3], (2 * tensor[1:2, :3] - tensor[:1, :3])])
        primitive = {'gender': 'male', 'betas': torch.zeros(1, 3, 10, device='cuda'),
            'transl': translation[None], 'global_orient': rotation[None, :, 0],
            'body_pose': rotation[None, :, 1:], 'transf_rotmat': torch.eye(3, device='cuda')[None],
            'transf_transl': torch.zeros(1, 1, 3, device='cuda')}
        primitive['pelvis_delta'] = utility.calc_calibrate_offset({
            'gender': 'male', 'betas': primitive['betas'][:, 0], 'transl': primitive['transl'][:, 0],
            'global_orient': primitive['global_orient'][:, 0], 'body_pose': primitive['body_pose'][:, 0]})
        _, _, primitive = utility.canonicalize(primitive)
        features = utility.calc_features(primitive)
        state = {key: value[:, :2] for key, value in features.items()}
        state.update({key: primitive[key] for key in ['gender', 'transf_rotmat', 'transf_transl', 'pelvis_delta']})
        state['betas'] = primitive['betas'][:, :2]
        env.state_human = state
        env.state_goal['goal_location'] = torch.as_tensor(observation.goal, dtype=torch.float32, device='cuda')[None]
        env.state_goal['goal_texts'] = ['walk']
        with torch.no_grad():
            action = agent.get_action_and_value(env.get_observation())[0]
            _, _, future = env.get_new_state_human(action)
        result = torch.cat([future['transl'], transforms.matrix_to_axis_angle(future['global_orient']),
                            transforms.matrix_to_axis_angle(future['body_pose']).flatten(-2)], dim=-1)
        return result[0].detach().cpu().numpy()

    if getattr(args, 'suite', False):
        from motiogen.feedback_benchmark import run_suite
        return run_suite(generate, pose, body_joints, 'dart', args, root)

    poses, trace = run_episode(generate, pose, goals, fps=30, seconds=args.frames / 30,
        mode=args.clock, feedback=bool(args.goal_feedback),
        handoff=args.handoff,
        warmup=lambda: generate(Observation(0, 30, pose, goal0)))
    xyz = body_joints(poses).detach().cpu().numpy()
    out = root / f'develop_logs/test_runs/20261024_dart_online_v2_{args.clock}_feedback{args.goal_feedback}_seed{args.seed}'
    if args.handoff == 'committed':
        out = out.with_name(out.name.replace('_v2_', '_v3_'))
    out.mkdir(parents=True, exist_ok=True)
    save_motion(out / 'motion.npz', xyz, HUMANML_NAMES, 30,
        {'units': 'm', 'axes': 'right-handed Z-up', 'model': 'DART', 'seed': args.seed,
         'variant': 'project_observed_pose_wrapper'})
    assessment = evaluate({'joints': xyz, 'joint_names': HUMANML_NAMES, 'fps': 30}, {'name': 'root_goal_change'})
    task = score_root_goals(xyz, goals)
    trace.update({'task': task, 'goal_events': goals, 'seed': args.seed,
        'wrapper': 'pretrained goal policy; observed pose history; constant-velocity extrapolation for native forward-difference features',
        'gpu_peak_bytes': torch.cuda.max_memory_allocated()})
    (out / 'trace.json').write_text(json.dumps(trace, indent=2))
    (out / 'metrics.json').write_text(json.dumps(assessment, indent=2))
    print({'output': str(out), 'task': task, 'hold_fraction': trace['hold_fraction']}, flush=True)
