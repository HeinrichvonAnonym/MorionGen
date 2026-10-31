"""Paired PRIMAL inference using official weights and a documented DART seed."""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(json.loads((ROOT / 'config/baselines.json').read_text())['data_root'])


def main():
    setup_started=time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--output-dir', type=Path, help='Fresh output directory for native component experiments')
    parser.add_argument('--ground-ablation', action='store_true', help='Native 2x2x2 ground snap / reprojection / inertialization factorial')
    parser.add_argument('--omega-sweep', help='Comma-separated inertialization omegas (1/s), ground snap and reprojection on, plus inertialization off')
    parser.add_argument('--initial-history',type=int,choices=[0,1,2],default=0,
                        help='0=adapted standing seed, 1/2=model-generated warm history (not independent real motion)')
    parser.add_argument('--initial-yaw',type=float,default=0.)
    parser.add_argument('--initial-seed',type=int,default=9000,
                        help='Warm-history sampling seed, independent of evaluation seed')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--suite', action='store_true', help='Run the preregistered v4 multi-scenario replication')
    parser.add_argument('--clock', choices=['paused', 'realtime'], default='paused')
    parser.add_argument('--goal-feedback', type=int, choices=[0, 1], default=1)
    parser.add_argument('--horizon-primitives', type=int, choices=[1, 2, 4], default=1)
    parser.add_argument('--handoff', choices=['observed', 'committed'], default='committed')
    args = parser.parse_args()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.resolve()
        if args.online or args.suite:
            parser.error('--output-dir is reserved for native component experiments')
        if args.output_dir.exists():
            parser.error('Output directory already exists; refusing to overwrite experiment evidence')
    if args.ground_ablation and args.output_dir is None:
        parser.error('--ground-ablation requires a fresh --output-dir')
    if args.omega_sweep and (args.output_dir is None or args.ground_ablation):
        parser.error('--omega-sweep requires a fresh --output-dir and excludes --ground-ablation')
    repo = DATA / 'baselines/PRIMAL'
    registry = DATA / 'outputs/assets/primal-registry'
    body = registry / 'models/SMPLX/neutral/SMPLX_neutral.npz'
    body.parent.mkdir(parents=True, exist_ok=True)
    source = Path('/home/heinrich/TactileLab/data/GRAB/smplx/models/smplx/SMPLX_NEUTRAL.npz')
    if not body.exists():
        body.symlink_to(source)
    os.environ['MODEL_REGISTRY_PATH'] = str(registry)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from scipy.spatial.transform import Rotation
    from primal.models.motion_diffuser import ARDiffusion
    sys.path.insert(0, str(ROOT / 'utils'))
    from motiogen.motion import HUMANML_NAMES, canonicalize, save_motion
    from motiogen.metrics import evaluate
    from motiogen.primal_interventions import NativeBranchCounter,NativePrimitiveClock
    weight_binding=None
    if args.output_dir is not None and (args.ground_ablation or args.omega_sweep):
        from motiogen.config import load_config
        from motiogen.weight_identity import bind,verify_unchanged
        weight_binding=bind(load_config(),'primal',args.output_dir.parent)

    modeldir = DATA / 'outputs/assets/primal-official/motion_diffuser_ar/runs/silu_anchor/tensorboard/version_0'
    model_load_started=time.perf_counter()
    cfg = OmegaConf.load(modeldir / 'hparams.yaml')
    model = ARDiffusion(cfg)
    checkpoint = torch.load(modeldir / 'checkpoints/epoch=29999-step=480000.ckpt', map_location='cpu')
    result = model.load_state_dict(checkpoint['state_dict'], strict=False)
    missing_learned = sorted(set(result.missing_keys) & set(dict(model.named_parameters())))
    # Published checkpoint retains an unused causal mask; this config is noncausal.
    unexpected = set(result.unexpected_keys)
    if not cfg.network.is_causal:
        unexpected.discard('causal_mask')
    if missing_learned or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: {missing_learned}, {result.unexpected_keys}')
    ema = checkpoint['ema_state_dict']
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in ema:
                raise RuntimeError(f'Missing EMA parameter: {name}')
            parameter.copy_(ema[name])
    model = model.cuda().eval()
    model.requires_grad_(False)
    torch.cuda.synchronize()
    model_load_seconds=time.perf_counter()-model_load_started

    with open(DATA / 'outputs/assets/dart-official/data/stand.pkl', 'rb') as handle:
        initial = pickle.load(handle)
    # DART Z-up -> PRIMAL Y-up. This is seed adaptation, not an official benchmark.
    rotation = Rotation.from_euler('x', -90, degrees=True)
    trans = rotation.apply(initial['transl'][:2])
    orient = (rotation * Rotation.from_rotvec(initial['global_orient'][:2])).as_rotvec()
    pose = np.concatenate([trans, orient, initial['body_pose'][:2]], axis=-1)
    batch = {'xb': torch.as_tensor(pose, dtype=torch.float32, device='cuda')[None],
             'betas': torch.zeros(1, 2, 16, device='cuda'),
             'ori': torch.tensor([[[1., 0., 0.]]], device='cuda')}
    if args.online or args.suite:
        from motiogen.feedback import run_episode
        from motiogen.motion import native_positions
        initial_joints = model._fwd_smplx_seq(batch['betas'], model.aa2rotcont(batch['xb']))
        initial_root = canonicalize(initial_joints[0, -1:, :22].detach().cpu().numpy(), 'Y-up')[0][0, 0]
        goal0 = initial_root + np.array([1., 0., 0.])
        goal1 = initial_root + np.array([-1., .5, 0.])
        goals = [{'frame': 0, 'position': goal0.tolist()},
                 {'frame': args.frames // 2, 'position': goal1.tolist()}]
        def generate(observation):
            torch.manual_seed(args.seed * 100000 + observation.frame)
            history = torch.as_tensor(observation.human_history, dtype=torch.float32, device='cuda')[None]
            body_shape = torch.zeros(1, 2, 16, device='cuda')
            joints = model._fwd_smplx_seq(body_shape, model.aa2rotcont(history))
            goal_native = torch.as_tensor(native_positions(observation.goal, 'Y-up'),
                                          dtype=torch.float32, device='cuda')
            vector = goal_native - joints[0, -1, 0]
            vector[1] = 0
            distance = vector.norm().clamp(min=1e-6)
            velocity = vector / distance * distance.clamp(max=.8)
            result = model.generate_perpetual_navigation(
                {'xb': history, 'betas': body_shape, 'ori': velocity[None, None]},
                n_inference_steps=int(cfg.scheduler.num_train_timesteps),
                nt_max=(int(cfg.data.seq_len) - 1) * args.horizon_primitives,
                snap_to_ground=True, reproj_kpts=True, switch_on_inertialization=True,
                switch_on_control=True)
            return result[1][0][0].detach().cpu().numpy()
        if args.suite:
            if args.horizon_primitives != 1:
                raise ValueError('The v4 replication fixes PRIMAL to one primitive')
            from motiogen.feedback_benchmark import run_suite
            def suite_joints(poses):
                tensor = torch.as_tensor(poses, dtype=torch.float32, device='cuda')[None]
                shape = torch.zeros(1, len(poses), 16, device='cuda')
                joints = model._fwd_smplx_seq(shape, model.aa2rotcont(tensor))
                return canonicalize(joints[0, :, :22].detach().cpu().numpy(), 'Y-up')[0]
            return run_suite(generate, pose, suite_joints, 'primal', args, ROOT)
        from motiogen.feedback import Observation
        warmup = lambda: generate(Observation(0, 30, pose, goal0))
        poses, trace = run_episode(generate, pose, goals, fps=30, seconds=args.frames / 30,
                                   mode=args.clock, feedback=bool(args.goal_feedback), warmup=warmup,
                                   handoff=args.handoff)
        out = ROOT / 'develop_logs/test_runs' / f'20261024_primal_online_v2_{args.clock}_feedback{args.goal_feedback}_seed{args.seed}'
        if args.handoff == 'committed':
            out = out.with_name(out.name.replace('_v2_', '_v3_'))
        if args.horizon_primitives != 1:
            out = out.with_name(out.name + f'_h{args.horizon_primitives}')
        out.mkdir(parents=True, exist_ok=True)
        pose_tensor = torch.as_tensor(poses, dtype=torch.float32, device='cuda')[None]
        shape = torch.zeros(1, len(poses), 16, device='cuda')
        joints = model._fwd_smplx_seq(shape, model.aa2rotcont(pose_tensor))
        xyz, _ = canonicalize(joints[0, :, :22].detach().cpu().numpy(), 'Y-up')
        save_motion(out / 'motion.npz', xyz, HUMANML_NAMES, 30,
            {'units': 'm', 'axes': 'right-handed Z-up', 'model': 'PRIMAL',
             'variant': 'project_online_wrapper', 'seed': args.seed})
        assessment = evaluate({'joints': xyz, 'joint_names': HUMANML_NAMES, 'fps': 30},
                              {'name': 'root_goal_change'})
        from motiogen.feedback import score_root_goals
        task = score_root_goals(xyz, goals)
        trace.update({'task': task, 'goal_events': goals, 'seed': args.seed,
                      'horizon_primitives': args.horizon_primitives,
                      'wrapper': 'native short-horizon generation with observed pose seed and geometric goal-to-velocity controller',
                      'gpu_peak_bytes': torch.cuda.max_memory_allocated()})
        (out / 'trace.json').write_text(json.dumps(trace, indent=2))
        (out / 'metrics.json').write_text(json.dumps(assessment, indent=2))
        print({'output': str(out), 'task': task, 'hold_fraction': trace['hold_fraction']}, flush=True)
        return
    out = args.output_dir or ROOT / 'develop_logs/test_runs' / f'20261024_primal_feedback_f{args.frames}_seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    initial_history_started=time.perf_counter()
    if args.initial_history:
        # Warm histories use a fixed seed independent of evaluation noise and
        # the same all-on native geometric pipeline for every intervention.
        torch.manual_seed(args.initial_seed+args.initial_history)
        warm=model.generate_perpetual_navigation(
            {key:value.clone() for key,value in batch.items()},
            n_inference_steps=int(cfg.scheduler.num_train_timesteps),
            nt_max=140*args.initial_history,snap_to_ground=True,reproj_kpts=True,
            switch_on_inertialization=True,switch_on_control=False)
        warm_pose=warm[1][0] if isinstance(warm[1],list) else warm[1]
        batch['xb']=warm_pose[:,-2:].detach().clone()
    if args.initial_yaw:
        yaw=Rotation.from_euler('y',args.initial_yaw,degrees=True)
        initial_pose=batch['xb'][0].cpu().numpy().copy()
        initial_pose[:,:3]=yaw.apply(initial_pose[:,:3])
        initial_pose[:,3:6]=(yaw*Rotation.from_rotvec(initial_pose[:,3:6])).as_rotvec()
        batch['xb']=torch.as_tensor(initial_pose,device='cuda',dtype=torch.float32)[None]
    np.savez(out/'initial_history.npz',pose=batch['xb'].cpu().numpy(),betas=batch['betas'].cpu().numpy())
    torch.cuda.synchronize()
    initial_history_seconds=time.perf_counter()-initial_history_started
    records = []
    warmup_started=time.perf_counter()
    model.generate_perpetual_navigation(
        {key: value.clone() for key, value in batch.items()},
        n_inference_steps=int(cfg.scheduler.num_train_timesteps), nt_max=32,
        snap_to_ground=True, reproj_kpts=True, switch_on_inertialization=True,
        switch_on_control=False)
    torch.cuda.synchronize()
    from motiogen.runtime import json_write
    json_write(out/'setup_timing.json',{'cold_start_seconds':time.perf_counter()-setup_started,
        'model_load_seconds':model_load_seconds,'initial_history_seconds':initial_history_seconds,
        'inference_warmup_seconds':time.perf_counter()-warmup_started,
        'scope':'Once per eight-condition worker, not once per factorial cell. Cold start includes imports, weight identity, model loading, initial-history preparation and native warmup. Generation timing excludes this setup.'})
    if args.omega_sweep:
        conditions = [(True, True, True, float(w)) for w in args.omega_sweep.split(',')]+[(True, True, False, None)]
    else:
        conditions = [(ground, reproj, inertial, None)
                      for ground in ([False, True] if args.ground_ablation else [True])
                      for reproj, inertial in [(False, False), (True, False), (False, True), (True, True)]]
    import contextlib
    from unittest import mock
    import primal.models.motion_diffuser as diffuser_module
    native_inertialize = diffuser_module.inertialize
    omega_calls = [0]

    def omega_patch(omega):
        # Native call sites pass omega=10, dt=1/30; omega 10 runs unpatched.
        if omega is None or omega == 10.:
            return contextlib.nullcontext()
        def swept(seq1, seq2, omega=10., dt=1./30.):
            if omega != 10.:
                raise RuntimeError(f'Unexpected native inertialization omega {omega}')
            omega_calls[0] += 1
            return native_inertialize(seq1, seq2, omega=swept_omega, dt=dt)
        swept_omega = omega
        return mock.patch.object(diffuser_module, 'inertialize', swept)
    # Randomize execution order, preserving common random numbers across conditions.
    np.random.default_rng(args.seed).shuffle(conditions)
    for ground, reproject, inertialize, omega in conditions:
        omega_calls[0] = 0
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with omega_patch(omega), NativeBranchCounter(model) as branch_counter, NativePrimitiveClock(model,int(cfg.scheduler.num_train_timesteps),int(cfg.data.seq_len)-2,args.clock) as native_clock:
            result = model.generate_perpetual_navigation(
            {key: value.clone() for key, value in batch.items()},
            n_inference_steps=int(cfg.scheduler.num_train_timesteps),
            # Upstream tests tt+(seq_len-1)<=nt_max, emits seq_len-2 frames,
            # and increments tt by seq_len-2 (one lookahead frame is removed).
            nt_max=((args.frames+int(cfg.data.seq_len)-3)//(int(cfg.data.seq_len)-2))*(int(cfg.data.seq_len)-2)+1,
            snap_to_ground=ground, reproj_kpts=reproject, switch_on_control=False,
            switch_on_inertialization=inertialize,
        )
        branch_calls = branch_counter.validate(ground, reproject, inertialize)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        nonwait_seconds=elapsed-native_clock.wait_seconds
        if nonwait_seconds<0:raise RuntimeError('Measured pacing exceeds native generation interval')
        betas, xb = result[:2]
        motion = xb[0] if isinstance(xb, list) else xb
        generated_frames=int(motion.shape[1])
        if generated_frames < args.frames:
            raise RuntimeError(f'Native primitive output too short: {generated_frames} < {args.frames}')
        motion=motion[:,:args.frames]
        joints = model._fwd_smplx_seq(betas.repeat(1, motion.shape[1], 1), model.aa2rotcont(motion))
        name = f'reproject{int(reproject)}_inertialize{int(inertialize)}'
        if args.ground_ablation:
            name = f'ground{int(ground)}_' + name
        if args.omega_sweep:
            name = f'omega{omega:g}' if inertialize else 'inertialize0'
            if omega is not None and omega != 10. and omega_calls[0] != branch_calls.get('inertialization', omega_calls[0]):
                raise RuntimeError('Swept inertialization did not replace every native call')
            if omega is not None and omega != 10. and not omega_calls[0]:
                raise RuntimeError('Swept inertialization never executed')
        np.savez(out / f'{name}.npz', pose=motion.detach().cpu().numpy(),
                 joints=joints.detach().cpu().numpy())
        xyz, _ = canonicalize(joints[0, :, :22].detach().cpu().numpy(), 'Y-up')
        metadata = {'units': 'm', 'axes': 'right-handed Z-up', 'model': 'PRIMAL',
                    'variant': name, 'seed': args.seed, 'post_process': True}
        save_motion(out / f'{name}_canonical.npz', xyz, HUMANML_NAMES, 30, metadata)
        assessment = evaluate({'joints': xyz, 'joint_names': HUMANML_NAMES, 'fps': 30},
                              {'name': 'primal_standing_seed_feedback'})
        (out / f'{name}_metrics.json').write_text(json.dumps(assessment, indent=2))
        records.append({'variant': name, 'seconds': elapsed, 'frames': int(motion.shape[1]),
                        'paced_wait_seconds':native_clock.wait_seconds,
                        'nonwait_generation_seconds':nonwait_seconds,
                        'nonwait_scope':'Full native generation wall interval minus directly measured pacing sleep; includes CPU/GPU work and host scheduling, not utilization-integrated CUDA kernel time. No sleep duration inferred from nominal schedule.',
                        'native_branch_calls': branch_calls,
                        'clock':args.clock,'primitive_clock_trace':native_clock.trace,
                        'ground_snap': ground, 'reprojection': reproject, 'inertialization': inertialize,
                        'inertialization_omega': (10. if omega is None else omega) if inertialize else None,
                        'swept_inertialization_calls': omega_calls[0],
                        'native_control': False,
                        'initial_history':args.initial_history,'initial_yaw_degrees':args.initial_yaw,
                        'initial_seed':args.initial_seed,'generated_frames_before_trim':generated_frames,
                        'fps': 30, 'gpu_peak_bytes': torch.cuda.max_memory_allocated(),
                        'seed': args.seed, 'seed_source': 'DART stand.pkl adapted to Y-up; optional native model warm history; neutral zero shape',
                        'scope': 'paired projection/inertialization inference; not goal-change benchmark'})
        (out / 'timing.json').write_text(json.dumps(records, indent=2))
        print({k:v for k,v in records[-1].items() if k!='primitive_clock_trace'}, flush=True)
    if weight_binding is not None:verify_unchanged(weight_binding)


if __name__ == '__main__':
    main()
