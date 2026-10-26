"""Versioned, resident-model, scenario-clustered feedback replication.

Initial conditions are standing plus two model-generated warm starts, not three
independent real motion recordings. All future goals are hidden until their event.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import resource
import time
import numpy as np
from motiogen.feedback import Observation, run_episode, score_root_goals
from motiogen.motion import HUMANML_NAMES, save_motion
from motiogen.metrics import evaluate


def scenarios():
    # Determined before viewing replication outcomes; distances are in meters.
    directions = {'continue': [2., 0., 0.], 'left': [1., 1.5, 0.],
                  'right': [1., -1.5, 0.], 'reverse': [-1., .5, 0.]}
    return [{'id': f'{initial}_{direction}', 'initial': initial, 'direction': direction,
             'goal0_offset': [1., 0., 0.], 'goal1_offset': position,
             'switch_frame': 150, 'frames': 450, 'fps': 30}
            for initial in ['standing', 'walk_warmstart', 'turn_warmstart']
            for direction, position in directions.items()]


def run_suite(generate, initial_pose, body_joints, method, args, root):
    import torch
    out = root / f'develop_logs/test_runs/20261024_feedback_replication_v4/{method}'
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    started = datetime.now(timezone.utc).isoformat()
    def xyz(poses):
        with torch.no_grad():
            value = body_joints(poses)
        return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
    def predict(obs):
        with torch.no_grad():
            return generate(obs)

    initial_root = xyz(initial_pose)[-1, 0]
    initial_states = {'standing': np.array(initial_pose, copy=True)}
    # Warm starts use held-out construction seeds, disjoint from test seeds.
    for name, seed, offset in [('walk_warmstart', 901, [3., 0., 0.]),
                               ('turn_warmstart', 902, [0., 3., 0.])]:
        path = out / f'{name}.npz'
        if path.exists():
            initial_states[name] = np.load(path)['pose']
            continue
        args.seed = seed
        history = initial_pose.copy()
        generated = []
        frame = 0
        while frame < 60:
            future = predict(Observation(frame, 30, history[-2:], initial_root + offset))
            generated.extend(future)
            history = np.concatenate([history, future])[-2:]
            frame += len(future)
        initial_states[name] = history.copy()
        np.savez(path, pose=history, warmup_pose=np.asarray(generated), seed=seed,
                 source='model-generated 2-second warm start; not independent real data')
    manifest = {'protocol': 4, 'method': method, 'scenarios': scenarios(), 'seeds': [10, 11, 12],
                'modes': ['paused', 'realtime'], 'feedback': [0, 1], 'handoff': 'committed',
                'lookahead_seconds': .2 if method == 'dart' else .4,
                'lookahead_basis': 'fixed before replication from preceding p95 compute + margin; not outcome-tuned',
                'initial_conditions': 'standing + two model-generated warm starts (seeds 901 and 902)',
                'primary_metric': 'post-switch mean XY root distance over 10 seconds',
                'secondary_metrics': ['final_error', 'time_in_target', 'sustained_recovery', 'jerk', 'posture', 'hold_fraction'],
                'inference': 'scenario-cluster bootstrap; exact within-scenario seed pairing; no cross-model causal ranking',
                'device': os.environ.get('CUDA_VISIBLE_DEVICES'), 'started_at': started}
    manifest['initial_sha256'] = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in initial_states.items()}
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert previous['initial_sha256'] == manifest['initial_sha256']
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2))
    jobs = [(case, seed, mode) for case in scenarios() for seed in manifest['seeds'] for mode in manifest['modes']]
    rng = np.random.default_rng(20261024)
    rng.shuffle(jobs)
    completed = 0
    for case, seed, mode in jobs:
        flags = rng.permutation([0, 1]).tolist()
        pose = initial_states[case['initial']].copy()
        base = xyz(pose)[-1, 0]
        goals = [{'frame': 0, 'position': (base + case['goal0_offset']).tolist()},
                 {'frame': case['switch_frame'], 'position': (base + case['goal1_offset']).tolist()}]
        args.seed = seed
        for flag in flags:
            directory = out / f"{case['id']}_seed{seed}_{mode}_feedback{flag}"
            receipt = directory / 'trace.json'
            if receipt.exists():
                completed += 1
                continue
            directory.mkdir(parents=True, exist_ok=True)
            torch.cuda.reset_peak_memory_stats()
            poses, trace = run_episode(predict, pose, goals, fps=30, seconds=15,
                mode=mode, feedback=bool(flag), handoff='committed',
                lookahead_seconds=manifest['lookahead_seconds'],
                warmup=lambda: predict(Observation(0, 30, pose, np.asarray(goals[0]['position']))))
            joints = xyz(poses)
            assert joints.shape == (450, 22, 3) and np.isfinite(joints).all()
            save_motion(directory / 'motion.npz', joints, HUMANML_NAMES, 30,
                {'model': method, 'seed': seed, 'scenario': case['id'], 'protocol': 4,
                 'units': 'm', 'axes': 'right-handed Z-up'})
            np.savez(directory / 'native_pose.npz', pose=poses)
            metrics = evaluate({'joints': joints, 'joint_names': HUMANML_NAMES, 'fps': 30},
                               {'name': case['id']})
            (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2))
            trace.update({'protocol_version': 4, 'method': method, 'case': case, 'seed': seed,
                'goal_events': goals, 'task': score_root_goals(joints, goals),
                'gpu_peak_bytes': torch.cuda.max_memory_allocated(),
                'process_rss_highwater_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'rss_scope': 'resident process high-water mark; not independent case peak',
                'completed_at': datetime.now(timezone.utc).isoformat()})
            receipt.write_text(json.dumps(trace, indent=2))
            completed += 1
            print(json.dumps({'method': method, 'completed': completed, 'total': 144,
                              'case': case['id'], 'seed': seed, 'mode': mode,
                              'feedback': flag, 'hold': trace['hold_fraction']}, ensure_ascii=False), flush=True)
    (out / 'completion.json').write_text(json.dumps({'completed': completed,
        'wall_seconds_this_invocation': time.perf_counter() - start,
        'completed_at': datetime.now(timezone.utc).isoformat(),
        'accounting': 'exclusive GPU process wall time, including CPU and realtime waiting; not GPU kernel utilization'}, indent=2))
