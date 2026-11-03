#!/usr/bin/env python
"""Experiment 3 runner: one process per method loads the model once and loops over jobs.

generate (method environment, GPU): C0 and the generated arms.
post (Kimodo environment, CPU): output-layer arms through native MotionCorrection.
Registration: config/x3_cross_method_layers_20261104.json.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
DATA = Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])

import numpy as np

from motiogen import x3_tasks as xt
from motiogen.motion import HUMANML_NAMES, load_motion, save_motion
from motiogen.runtime import json_write


def done(path):
    return (path/'receipt.json').exists()


def write(path, joints, names, fps, receipt, **optional):
    path.mkdir(parents=True, exist_ok=True)
    save_motion(path/'motion.npz', joints, names, fps,
                {'units': 'm', 'axes': 'right-handed Z-up', 'protocol': 'x3', 'arm': receipt['arm'],
                 'method': receipt['method'], 'seed': receipt['seed']}, **optional)
    json_write(path/'receipt.json', receipt)


def receipt(method, task, seed, arm, offset, goal_rows, joints, names, **fields):
    return {'method': method, 'task': task['id'], 'seed': seed, 'arm': arm, 'offset_m': offset,
            'goals': xt.goal_receipt(joints, names, goal_rows) if goal_rows else None, **fields}


def goal_file(method, task, seed):
    return xt.base_dir(method, task['id'], seed)/'goals.json'


def save_goals(method, task, seed, joints, names, fps):
    rows = {f'{d:g}': xt.goals(joints, names, fps, task, d) for d in xt.prereg()['offsets_m']}
    json_write(goal_file(method, task, seed), rows)
    return rows


# ---------------------------------------------------------------- DART

def dart_setup(batch):
    if 'float' not in np.__dict__:
        np.float = float
    runtime = DATA/'outputs/runtime/dart'
    os.chdir(runtime)
    sys.path.insert(0, str(DATA/'staged_dependencies/pytorch3d'))
    sys.path.insert(0, str(DATA/'baselines/DART'))
    import torch
    import clip
    import tyro
    import yaml
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    original = clip.load
    def local_clip(name, *positional, **kwargs):
        kwargs['download_root'] = str(DATA/'outputs/assets/omnicontrol')
        return original(name, *positional, **kwargs)
    clip.load = local_clip
    from control.train_reach_location_mld import ReachLocationArgs, load_motion_model
    from control.env.env_reach_location_mld import EnvReachLocationMLD
    from control.policy.policy import PolicyReachLocationMLP, PolicyReachLocationTransformer
    policy_dir = DATA/'outputs/assets/dart-official/policy_train/reach_location_mld/fixtext_repeat_floor100_hop10_skate100'
    cfg = tyro.extras.from_yaml(ReachLocationArgs, yaml.safe_load((policy_dir/'args.yaml').read_text()))
    cfg.num_envs = cfg.batch_size = cfg.env_args.num_envs = batch
    cfg.num_steps = cfg.env_args.num_steps = 1000
    cfg.env_args.enable_export = False
    cfg.device = torch.device('cuda')
    cfg.init_data_path = './data/stand.pkl'
    da, dm, va, vm, diffusion, dataset = load_motion_model(cfg)
    env = EnvReachLocationMLD(cfg, da, dm, va, vm, diffusion, dataset)
    cfg.policy_args.observation_structure = env.observation_structure
    cfg.policy_args.action_structure = env.action_structure
    agent_class = PolicyReachLocationMLP if cfg.policy_args.architecture == 'mlp' else PolicyReachLocationTransformer
    agent = agent_class(cfg.policy_args).cuda().eval()
    agent.load_state_dict(torch.load(policy_dir/'iter_5000.pth', map_location='cpu'))
    agent.requires_grad_(False)
    return env, agent, policy_dir


def dart_reset(env, texts):
    """Official reset, then one shared stand.pkl initial state for every batch element."""
    import random
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    B = env.args.num_envs
    env.reset(goal_location=torch.zeros(B, 3, device='cuda'), goal_texts=texts)
    for key, value in env.state_human.items():
        if torch.is_tensor(value) and value.shape[0] == B:
            env.state_human[key] = value[:1].expand_as(value).clone()
    return env.get_observation()


def dart_noise(task, seed, k, shape):
    import torch
    g = torch.Generator().manual_seed(100000*(1000*task+seed)+k)
    return torch.randn(shape, generator=g), torch.randn(shape, generator=g)


def dart_policy_action(agent, observation, eps):
    """Native policy, its Normal sample drawn from the run's own generator (batch-independent)."""
    from torch.distributions import Normal
    original = Normal.sample
    Normal.sample = lambda self, sample_shape=None: self.loc+self.scale*eps.view_as(self.loc).to(self.loc)
    try:
        return agent.get_action_and_value(observation)[0]
    finally:
        Normal.sample = original


def dart_rollout(env, agent, jobs, frames):
    """jobs: dicts with task, seed, text, lookahead (None for C0) and goals (frame, target_xy).
    Returns joints [B, frames, 22, 3], boundaries, per-job policy-call receipts."""
    import torch
    B = env.args.num_envs
    rows = jobs+[jobs[-1]]*(B-len(jobs))
    shape = (int(np.prod(env.action_structure['shape'])),)
    zero = torch.zeros(B, 3, device='cuda')
    observation = dart_reset(env, [r['text'] for r in rows])
    chunks = [env.get_global_joints().detach().cpu().numpy()]
    frame = chunks[0].shape[1]
    boundaries, calls = [], [[] for _ in rows]
    active = [None]*B
    k = 0
    while frame < frames:
        noise, eps = zip(*[dart_noise(r['task'], r['seed'], k, shape) for r in rows])
        noise, eps = torch.stack(noise).cuda(), torch.stack(eps).cuda()
        use = torch.zeros(B, dtype=torch.bool, device='cuda')
        goal = torch.zeros(B, 3, device='cuda')
        for i, r in enumerate(rows):
            if r['lookahead'] is None:
                continue
            for g in r['goals']:
                if frame <= g['frame'] and frame+8 > g['frame']-r['lookahead']:
                    use[i] = True; goal[i, :2] = torch.tensor(g['target_xy'], device='cuda')
                    break
        changed = [i for i in range(B) if use[i] and (active[i] is None or not torch.equal(active[i], goal[i]))]
        if changed:
            idx = torch.tensor(changed, device='cuda')
            env.reset_goal(idx, goal_location=goal[idx], goal_texts=[rows[i]['text'] for i in changed])
            for i in changed:
                active[i] = goal[i].clone()
            observation = env.get_observation()
        with torch.no_grad():
            policy = dart_policy_action(agent, observation, eps) if use.any() else noise
            action = torch.where(use[:, None], policy.view(B, -1), noise)
            captured = {}
            native = env.get_new_state_human
            def capture(a):
                out = native(a); captured['joints'] = out[2]['joints']; return out
            env.get_new_state_human = capture
            try:
                observation, *_ = env.step(action, next_goal_location=torch.stack([a if a is not None else zero[0] for a in active]),
                                           next_goal_texts=np.array([r['text'] for r in rows]))
            finally:
                env.get_new_state_human = native
        for i in range(B):
            if use[i]:
                calls[i].append(frame)
        boundaries.append(frame)
        chunks.append(captured['joints'].detach().cpu().numpy())
        frame += chunks[-1].shape[1]
        k += 1
    joints = np.concatenate(chunks, axis=1)[:, :frames]
    return joints[:len(jobs)], boundaries, calls[:len(jobs)]


def dart_diff_rollout(env, jobs, noises, frames, autocast):
    """Differentiable re-execution of env.get_new_state_human from the reset state."""
    import torch
    from torch.cuda import amp
    zero = torch.zeros(env.args.num_envs, 3, device='cuda')
    B = env.args.num_envs
    rows = jobs+[jobs[-1]]*(B-len(jobs))
    if noises.shape[1] < B:
        noises = torch.cat([noises, noises[:, -1:].expand(-1, B-noises.shape[1], -1)], 1)
    dart_reset(env, [r['text'] for r in rows])
    state = {k: v.clone() if torch.is_tensor(v) else v for k, v in env.state_human.items()}
    text_embedding = env.get_text_embedding([r['text'] for r in rows])
    out = [env.get_global_joints(history_feature_dict=state)]
    frame = out[0].shape[1]
    pu, dataset, diffusion = env.primitive_utility, env.init_dataset, env.diffusion

    def primitive(action, state):
        history = dataset.normalize(pu.dict_to_tensor(state))
        guidance = torch.ones(B, *env.denoiser_args.model_args.noise_shape, device='cuda')*env.input_args.guidance_param
        y = {'text_embedding': text_embedding, 'history_motion_normalized': history, 'scale': guidance}
        with amp.autocast(enabled=autocast, dtype=torch.float16):
            x = diffusion.ddim_sample_loop_full_chain(env.denoiser_model, (B, *env.denoiser_args.model_args.noise_shape),
                                                      clip_denoised=False, model_kwargs={'y': y}, skip_timesteps=0,
                                                      init_image=None, progress=False, noise=action)
            future = env.vae_model.decode(x.permute(1, 0, 2), history, nfuture=env.future_length,
                                          scale_latent=env.denoiser_args.rescale_latent)
        future = dataset.denormalize(future.float())
        fd = pu.tensor_to_dict(future)
        fd.update({'transf_rotmat': state['transf_rotmat'], 'transf_transl': state['transf_transl'], 'gender': state['gender'],
                   'betas': state['betas'][:, [0], :].repeat(1, env.future_length, 1), 'pelvis_delta': state['pelvis_delta']})
        world = pu.transform_primitive_to_world(pu.feature_dict_to_smpl_dict(fd))
        hd = pu.tensor_to_dict(future[:, -env.history_length:])
        hd.update({'transf_rotmat': state['transf_rotmat'], 'transf_transl': state['transf_transl'], 'gender': state['gender'],
                   'betas': state['betas'], 'pelvis_delta': state['pelvis_delta']})
        _, blended = pu.get_blended_feature(hd, use_predicted_joints=env.args.use_predicted_joints)
        return world['joints'], blended

    from torch.utils.checkpoint import checkpoint
    for k in range(len(noises)):
        if frame >= frames:
            break
        action = noises[k].view(B, *env.action_structure['shape'])
        if torch.is_grad_enabled() and action.requires_grad:
            joints, state = checkpoint(primitive, action, state, use_reentrant=False)
        else:
            joints, state = primitive(action, state)
        out.append(joints)
        frame += env.future_length
    return torch.cat(out, 1)[:, :frames]


def dart_optimize(env, jobs, c0_noise, frames, autocast, steps=300, lr=.01):
    """Official optim_mld settings: Adam, unit-norm gradient, annealed lr, Huber goal loss, no jerk/floor terms."""
    import torch
    B = env.args.num_envs
    last = max(g['frame'] for j in jobs for g in j['goals'])
    n = int(np.ceil((last+1)/env.future_length))
    noise = torch.stack([c0_noise[i][:n] for i in range(len(jobs))]+[c0_noise[len(jobs)-1][:n]]*(B-len(jobs)), 1).cuda()
    noise = noise.clone().requires_grad_(True)
    frames_idx = torch.tensor([[g['frame'] for g in j['goals']] for j in jobs]+[[g['frame'] for g in jobs[-1]['goals']]]*(B-len(jobs)), device='cuda')
    target = torch.tensor([[g['target_xy'] for g in j['goals']] for j in jobs]+[[g['target_xy'] for g in jobs[-1]['goals']]]*(B-len(jobs)), device='cuda')
    optimizer = torch.optim.Adam([noise], lr=lr)
    criterion = torch.nn.HuberLoss(reduction='none')
    history = []
    for step in range(steps):
        optimizer.param_groups[0]['lr'] = lr*(1.-step/steps)
        optimizer.zero_grad()
        joints = dart_diff_rollout(env, jobs, noise, last+1, autocast)
        pelvis = joints[torch.arange(B, device='cuda')[:, None], frames_idx, 0, :2]
        per = criterion(pelvis, target).mean((1, 2))
        per.sum().backward()
        noise.grad.data /= noise.grad.norm(p=2, dim=2, keepdim=True).clamp(min=1e-6)
        optimizer.step()
        if step in (0, steps-1) or step % 50 == 0:
            history.append({'step': step, 'loss': per[:len(jobs)].detach().cpu().tolist()})
            print(json.dumps({'stage': 'dart_S', 'step': step, 'mean_loss': float(per[:len(jobs)].mean()),
                              'at': time.strftime('%H:%M:%S')}), flush=True)
    final = noise.detach()
    tail = torch.stack([c0_noise[i][n:] for i in range(len(jobs))]+[c0_noise[len(jobs)-1][n:]]*(B-len(jobs)), 1).cuda()
    return torch.cat([final, tail], 0), history


def dart_generate(args, stage):
    import torch
    from motiogen.config import load_config
    from motiogen.weight_identity import bind, verify_unchanged
    reg = xt.prereg()['methods']['dart']
    fps, frames = 30, int(reg['clip_seconds']*30)
    cells = xt.new_cells('dart', stage)
    offsets = xt.prereg()['offsets_m']
    B = len(cells)*len(offsets)
    env, agent, policy_dir = dart_setup(B)
    out = xt.OUT/'dart'; out.mkdir(parents=True, exist_ok=True)
    binding = bind(load_config(), 'dart', out)
    timing = {}
    # C0
    c0_jobs = [{'task': t['id'], 'seed': s, 'text': t['text'], 'lookahead': None, 'goals': []} for t, s in cells]
    start = time.perf_counter()
    joints, boundaries, _ = dart_rollout(env, agent, c0_jobs, frames)
    timing['C0'] = (time.perf_counter()-start)/len(cells)
    goal_rows = {}
    for (t, s), j in zip(cells, joints):
        goal_rows[(t['id'], s)] = save_goals('dart', t, s, j, HUMANML_NAMES, fps)
        path = xt.run_dir('dart', t['id'], s, 'C0')
        if not done(path):
            write(path, j, HUMANML_NAMES, fps, receipt('dart', t, s, 'C0', None, None, j, HUMANML_NAMES,
                  policy_calls=[], wall_seconds=timing['C0']), chunk_boundaries=np.array(boundaries))
    shape = (int(np.prod(env.action_structure['shape'])),)
    # replay check of the differentiable rollout on the C0 noise
    n = int(np.ceil(frames/env.future_length))+1
    c0_noise = [torch.stack([dart_noise(t['id'], s, k, shape)[0] for k in range(n)]) for t, s in cells]
    autocast = False
    with torch.no_grad():
        replay = dart_diff_rollout(env, c0_jobs, torch.stack(c0_noise, 1).cuda(), frames, autocast).cpu().numpy()
    replay_error = float(np.linalg.norm(replay[:len(cells), :, 0, :2]-joints[:, :, 0, :2], axis=-1).max())
    if replay_error > .01:
        autocast = True
        with torch.no_grad():
            replay = dart_diff_rollout(env, c0_jobs, torch.stack(c0_noise, 1).cuda(), frames, autocast).cpu().numpy()
        replay_error_autocast = float(np.linalg.norm(replay[:len(cells), :, 0, :2]-joints[:, :, 0, :2], axis=-1).max())
    else:
        replay_error_autocast = None
    json_write(out/f'{stage}_replay.json', {'max_root_xy_error_m_fp32': replay_error, 'max_root_xy_error_m_autocast': replay_error_autocast,
                                             'autocast_used': autocast, 'gate_m': .01})
    jobs = []
    for t, s in cells:
        for d in offsets:
            jobs.append({'task': t['id'], 'seed': s, 'text': t['text'], 'offset': d,
                         'goals': goal_rows[(t['id'], s)][f'{d:g}'], 'task_row': t})
    for arm, lookahead in (('G0', 0), ('G4', 4*fps)):
        start = time.perf_counter()
        joints, boundaries, calls = dart_rollout(env, agent, [{**j, 'lookahead': lookahead} for j in jobs], frames)
        timing[arm] = (time.perf_counter()-start)/len(jobs)
        for j, m, c in zip(jobs, joints, calls):
            path = xt.run_dir('dart', j['task'], j['seed'], arm, j['offset'])
            write(path, m, HUMANML_NAMES, fps, receipt('dart', j['task_row'], j['seed'], arm, j['offset'], j['goals'], m, HUMANML_NAMES,
                  lookahead_frames=lookahead, policy_calls=c, wall_seconds=timing[arm]), chunk_boundaries=np.array(boundaries))
    if replay_error_autocast is None or replay_error_autocast <= .01:
        start = time.perf_counter()
        noise_by_job = [c0_noise[cells.index((j['task_row'], j['seed']))] for j in jobs]
        optimized, history = dart_optimize(env, jobs, noise_by_job, frames, autocast)
        with torch.no_grad():
            joints = dart_diff_rollout(env, jobs, optimized, frames, autocast).cpu().numpy()[:len(jobs)]
        timing['S'] = (time.perf_counter()-start)/len(jobs)
        for i, (j, m) in enumerate(zip(jobs, joints)):
            path = xt.run_dir('dart', j['task'], j['seed'], 'S', j['offset'])
            write(path, m, HUMANML_NAMES, fps, receipt('dart', j['task_row'], j['seed'], 'S', j['offset'], j['goals'], m, HUMANML_NAMES,
                  optimizer={'steps': 300, 'lr': .01, 'autocast': autocast, 'loss_first': history[0]['loss'][i],
                             'loss_last': history[-1]['loss'][i], 'trace': [h['loss'][i] for h in history]},
                  wall_seconds=timing['S'], batch=len(jobs)), chunk_boundaries=np.array(boundaries))
    verify_unchanged(binding)
    json_write(out/f'{stage}_timing.json', {'seconds_per_run': timing, 'batch': B})


def dart_equivariance(stage):
    """W1 premise: shifting the DART world transform shifts every later C0 frame by the same vector."""
    import torch
    reg = xt.prereg()['methods']['dart']
    frames = int(reg['clip_seconds']*30)
    cells = [(t, s) for t in xt.job_tasks('dart', stage) for s in xt.seeds(stage)][:2]
    env, agent, _ = dart_setup(len(cells))
    jobs = [{'task': t['id'], 'seed': s, 'text': t['text'], 'lookahead': None, 'goals': []} for t, s in cells]
    base, boundaries, _ = dart_rollout(env, agent, jobs, frames)
    shift_at = boundaries[20]
    native = env.step
    shift = torch.tensor([.7, -.4, 0.], device='cuda')
    state = {'k': 0}
    def shifted(action, **kw):
        if state['k'] == 20:
            env.state_human['transf_transl'] = env.state_human['transf_transl']+shift.view(1, 1, 3)
        state['k'] += 1
        return native(action, **kw)
    env.step = shifted
    moved, _, _ = dart_rollout(env, agent, jobs, frames)
    env.step = native
    after = moved[:, shift_at:]-base[:, shift_at:]
    error = float(np.abs(after-shift.cpu().numpy()).max())
    json_write(xt.OUT/'dart'/f'{stage}_w1_equivariance.json', {'shift_m': shift.tolist(), 'from_frame': int(shift_at),
               'max_abs_error_m': error, 'before_identical_m': float(np.abs(moved[:, :shift_at]-base[:, :shift_at]).max()),
               'gate_m': .001, 'pass': error <= .001})


# ---------------------------------------------------------------- Kimodo

def kimodo_generate(args, stage):
    import torch
    from motiogen.adapters import create_adapter
    from motiogen.config import load_config, repo_root
    from motiogen.motion import canonicalize, native_positions
    from motiogen.weight_identity import bind, verify_unchanged
    cfg = load_config()
    repo = repo_root(cfg, 'kimodo'); os.chdir(repo); sys.path.insert(0, str(repo))
    reg = xt.prereg()
    out = xt.OUT/'kimodo'; out.mkdir(parents=True, exist_ok=True)
    prompts = reg['prompts']
    scene = {'name': 'x3', 'kind': 'position', 'duration_seconds': 10, 'prompt': prompts[0],
             'updates': [{'time_seconds': 0, 'prompt': p} for p in prompts[1:]], 'constraints': []}
    adapter = create_adapter(cfg, 'kimodo', out, scene, 0)
    adapter.preflight(); binding = bind(cfg, 'kimodo', out); adapter.load()
    model, axes = adapter.model, adapter.spec['axes']
    fps = model.fps; frames = int(10*fps)
    from kimodo.constraints import Root2DConstraintSet
    import kimodo.model.kimodo_model as km
    native_pp = km.post_process_motion
    stash = {}
    def patched(local_rot_mats, root_positions, contacts, skeleton, constraint_lst=None, **kw):
        stash['raw_root'] = root_positions.detach().cpu().numpy().copy()
        stash['received'] = len(constraint_lst or [])
        constraint_lst = stash.get('substitute', constraint_lst)
        stash['applied'] = len(constraint_lst or [])
        return native_pp(local_rot_mats, root_positions, contacts, skeleton, constraint_lst, **kw)
    km.post_process_motion = patched
    def root2d(goal_rows):
        frame_idx = torch.tensor([g['frame'] for g in goal_rows])
        xz = torch.tensor([[native_positions([*g['target_xy'], 0.], axes)[i] for i in (0, 2)] for g in goal_rows], dtype=torch.float32)
        constraint = Root2DConstraintSet(model.skeleton, frame_idx, xz)
        return constraint.to('cuda') if hasattr(constraint, 'to') else constraint
    def call(prompt, seed, constraints, weight, post, substitute=None):
        adapter.seed_all(seed)
        stash.clear()
        if substitute is not None:
            stash['substitute'] = substitute
        start = time.perf_counter()
        with torch.no_grad():
            o = model([prompt], [frames], constraint_lst=constraints, num_denoising_steps=100, num_samples=1,
                      post_processing=post, cfg_weight=[2.0, weight], return_numpy=True)
        seconds = time.perf_counter()-start
        joints, _ = canonicalize(o['posed_joints'][0], axes, None, units=adapter.spec.get('position_units', 'm'))
        return joints, seconds, o, dict(stash)
    names = adapter.output_names
    timing = {}
    for t, s in xt.new_cells('kimodo', stage):
        if all(done(xt.run_dir('kimodo', t['id'], s, a, d)) for a in ('T', 'T+', 'O', 'TO') for d in reg['offsets_m']):
            continue
        c0, sec, o_c0, _ = call(t['prompt'], s, [], 2.0, False)
        timing.setdefault('C0', []).append(sec)
        rows = save_goals('kimodo', t, s, c0, names, fps)
        write(xt.run_dir('kimodo', t['id'], s, 'C0'), c0, names, fps,
              receipt('kimodo', t, s, 'C0', None, None, c0, names, constraints=0, cfg_weight=[2.0, 2.0], post_processing=False, wall_seconds=sec),
              posed_joints=o_c0['posed_joints'][0])
        for d in reg['offsets_m']:
            g = rows[f'{d:g}']
            for arm, constraints, weight, post, sub in (
                    ('T', [root2d(g)], 2.0, False, None), ('T+', [root2d(g)], 4.0, False, None),
                    ('O', [], 2.0, True, [root2d(g)]), ('TO', [root2d(g)], 2.0, True, None)):
                joints, sec, o, info = call(t['prompt'], s, constraints, weight, post, sub)
                timing.setdefault(arm, []).append(sec)
                extra = {}
                if arm in ('O', 'TO'):
                    base = o_c0 if arm == 'O' else t_raw
                    extra = {'postprocess_constraints_received': info.get('received'), 'postprocess_constraints_applied': info.get('applied'),
                             'raw_root_matches_base_m': float(np.abs(info['raw_root'][0]-base['root_positions'][0]).max())}
                if arm == 'T':
                    t_raw = o
                write(xt.run_dir('kimodo', t['id'], s, arm, d), joints, names, fps,
                      receipt('kimodo', t, s, arm, d, g, joints, names, constraints=len(constraints), cfg_weight=[2.0, weight],
                              post_processing=post, wall_seconds=sec, **extra), posed_joints=o['posed_joints'][0])
    verify_unchanged(binding)
    json_write(out/f'{stage}_timing.json', {k: float(np.mean(v)) for k, v in timing.items()})


# ---------------------------------------------------------------- OmniControl

def omni_generate(args, stage):
    import torch
    from motiogen.adapters import create_adapter
    from motiogen.config import load_config, repo_root
    from motiogen.motion import canonicalize
    from motiogen.weight_identity import bind, verify_unchanged
    cfg = load_config()
    repo = repo_root(cfg, 'omnicontrol'); os.chdir(repo); sys.path.insert(0, str(repo))
    reg = xt.prereg()
    out = xt.OUT/'omnicontrol'; out.mkdir(parents=True, exist_ok=True)
    scene = {'name': 'x3', 'kind': 'position', 'duration_seconds': 9.8, 'prompt': reg['prompts'][0], 'constraints': []}
    adapter = create_adapter(cfg, 'omnicontrol', out, scene, 0)
    adapter.preflight(); binding = bind(cfg, 'omnicontrol', out); adapter.load()
    switch = {'realism': 1., 'spatial': 1.}
    calls = {}
    def count(key):
        calls[key] = calls.get(key, 0)+1
    native_cmdm = adapter.model.model.cmdm_forward
    def realism(x, timesteps, y=None, **k):
        count('realism')
        if switch['realism']:
            count('realism_active')
        elif y is not None and 'hint' in y:
            y = {**y, 'hint': torch.zeros_like(y['hint'])}
        return native_cmdm(x, timesteps, y, **k)
    adapter.model.model.cmdm_forward = realism
    native_guide = adapter.diffusion.guide
    def spatial(x, *a, **k):
        count('spatial')
        if switch['spatial']:
            count('spatial_active'); return native_guide(x, *a, **k)
        return x
    adapter.diffusion.guide = spatial
    fps, names = adapter.fps, adapter.names
    arms = {'C0': (0, 0), 'T': (1, 0), 'S': (0, 1), 'TS': (1, 1)}
    def run(prompt, seed, arm, hint):
        switch['realism'], switch['spatial'] = arms[arm]
        calls.clear()
        adapter.scene = {**scene, 'prompt': prompt, 'constraints': [{'time_seconds': 6.0, 'joint': 'pelvis', 'position_m': hint}]}
        adapter.seed_all(seed)
        start = time.perf_counter()
        joints = adapter.generate()
        seconds = time.perf_counter()-start
        joints, _ = canonicalize(joints, adapter.spec['axes'], None, units=adapter.spec.get('position_units', 'm'))
        return joints, seconds, dict(calls)
    timing = {}
    for t, s in xt.new_cells('omnicontrol', stage):
        path = xt.run_dir('omnicontrol', t['id'], s, 'C0')
        if done(path):
            m = load_motion(path/'motion.npz'); c0 = m['joints']
            rows = json.loads(goal_file('omnicontrol', t, s).read_text())
        else:
            c0, sec, info = run(t['prompt'], s, 'C0', [0., 0., .9])
            timing.setdefault('C0', []).append(sec)
            rows = save_goals('omnicontrol', t, s, c0, names, fps)
            write(path, c0, names, fps, receipt('omnicontrol', t, s, 'C0', None, None, c0, names, condition=dict(zip(('realism', 'spatial'), arms['C0'])),
                  calls=info, hint='placeholder; with realism and spatial off the ControlNet sees the native zero hint (unconstrained OmniControl)',
                  wall_seconds=sec))
        for d in reg['offsets_m']:
            g = rows[f'{d:g}']
            hint = [*g[0]['target_xy'], g[0]['free_height_m']]
            for arm in ('T', 'S', 'TS'):
                path = xt.run_dir('omnicontrol', t['id'], s, arm, d)
                if done(path):
                    continue
                joints, sec, info = run(t['prompt'], s, arm, hint)
                timing.setdefault(arm, []).append(sec)
                write(path, joints, names, fps, receipt('omnicontrol', t, s, arm, d, g, joints, names,
                      condition=dict(zip(('realism', 'spatial'), arms[arm])), calls=info, hint=hint, wall_seconds=sec))
    verify_unchanged(binding)
    json_write(out/f'{stage}_timing.json', {k: float(np.mean(v)) for k, v in timing.items()})


# ---------------------------------------------------------------- output layer (Kimodo env, CPU)

def post(args, stage):
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(DATA/'baselines/kimodo'))
    from motiogen.x3_correct import Rig
    method = args.method
    for t, s in xt.new_cells(method, stage):
        rows = json.loads(goal_file(method, t, s).read_text())
        c0 = load_motion(xt.run_dir(method, t['id'], s, 'C0')/'motion.npz')
        names, fps = list(c0['joint_names']), float(c0['fps'])
        rig0 = Rig(c0['joints'], names, fps)
        for d in xt.prereg()['offsets_m']:
            g = rows[f'{d:g}']
            frames, targets = [x['frame'] for x in g], [x['target_xy'] for x in g]
            todo = {'O': rig0}
            if method == 'omnicontrol':
                tm = load_motion(xt.run_dir(method, t['id'], s, 'T', d)/'motion.npz')
                todo['TO'] = Rig(tm['joints'], names, fps)
            for arm, rig in todo.items():
                path = xt.run_dir(method, t['id'], s, arm, d)
                if done(path):
                    continue
                start = time.perf_counter()
                local, root = rig.correct(frames, targets)
                joints = rig.to_canonical(rig.joints_y(local, root))
                optional = {}
                if method == 'dart':
                    optional['chunk_boundaries'] = c0['chunk_boundaries']
                write(path, joints, names, fps, receipt(method, t, s, arm, d, g, joints, names, base='C0' if arm == 'O' else 'T',
                      wrapper_fk=rig.fk, correction='native post_process_motion, Root2D at goal frames, whole clip',
                      wall_seconds=time.perf_counter()-start), **optional)
            if method != 'dart':
                continue
            boundaries = [int(b) for b in c0['chunk_boundaries']]
            edges = boundaries+[len(c0['joints'])]
            local, root = rig0.local, rig0.root
            windows = []
            for f, target in zip(frames, targets):
                i = max(i for i, b in enumerate(edges[:-1]) if b <= f)
                a, b = edges[i], edges[i+1]
                local, root = rig0.correct([f], [target], start=a, stop=b, local=local, root=root)
                windows.append([a, b])
            w0 = (local, root)
            xf = rig0.crossfade(local, root, boundaries)
            end_shift = {}
            for arm, (lo, ro) in (('W0', w0), ('W0X', xf)):
                path = xt.run_dir(method, t['id'], s, arm, d)
                if done(path):
                    continue
                joints = rig0.to_canonical(rig0.joints_y(lo, ro))
                write(path, joints, names, fps, receipt(method, t, s, arm, d, g, joints, names, base='C0', wrapper_fk=rig0.fk,
                      windows=windows, correction='native post_process_motion on the primitive window holding each goal; no write-back'
                      + ('; ARDY X0 crossfade at every primitive boundary' if arm == 'W0X' else '')),
                      chunk_boundaries=c0['chunk_boundaries'])
            path = xt.run_dir(method, t['id'], s, 'W1', d)
            equivariance = xt.OUT/'dart'/f'{stage}_w1_equivariance.json'
            if not done(path) and equivariance.exists() and json.loads(equivariance.read_text())['pass']:
                base, pelvis = c0['joints'], names.index('pelvis')
                shift = np.zeros(3)
                out = base.copy()
                local1, root1 = rig0.local, rig0.root
                for i, ((a, b), f, target) in enumerate(zip(windows, frames, targets)):
                    local1, root1 = rig0.correct([f], [(np.asarray(target)-shift[:2]).tolist()], start=a, stop=b,
                                                 local=local1, root=root1)
                    segment = rig0.to_canonical(rig0.joints_y(local1[:, a:b], root1[:, a:b]))
                    out[a:b] = segment+shift
                    shift = shift+(segment[-1, pelvis]-base[b-1, pelvis])*np.array([1., 1., 0.])
                    stop = windows[i+1][0] if i+1 < len(windows) else len(base)
                    out[b:stop] = base[b:stop]+shift
                write(path, out, names, fps, receipt(method, t, s, 'W1', d, g, out, names, base='C0', wrapper_fk=rig0.fk, windows=windows,
                      correction='W0 windows; corrected root translation written back (DART history canonicalization makes later frames a rigid XY shift, verified in w1_equivariance)'),
                      chunk_boundaries=c0['chunk_boundaries'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, choices=['dart', 'kimodo', 'omnicontrol'])
    p.add_argument('--stage', required=True, choices=['pilot', 'confirm', 'top_up'])
    p.add_argument('--part', default='generate', choices=['generate', 'post', 'equivariance'])
    args = p.parse_args()
    if args.part == 'post':
        return post(args, args.stage)
    if args.part == 'equivariance':
        return dart_equivariance(args.stage)
    {'dart': dart_generate, 'kimodo': kimodo_generate, 'omnicontrol': omni_generate}[args.method](args, args.stage)


if __name__ == '__main__':
    main()
