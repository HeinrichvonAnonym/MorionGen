"""Kinematic online execution with explicit observation and result timestamps.

Adapters own their pose representation. This executor does not simulate forces.
Future actors are carried separately from the controlled human and object state.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class Observation:
    frame: int
    fps: float
    human_history: np.ndarray
    goal: np.ndarray
    objects: dict = field(default_factory=dict)
    participants: tuple = ()
    observed_frame: int | None = None
    observed_history: np.ndarray | None = None
    history_kind: str = 'executed'


def score_root_goals(xyz, goals, fps=30, radius=.3, hold_seconds=.5):
    results = []
    hold = max(1, round(fps * hold_seconds))
    for index, goal in enumerate(goals):
        start = goal['frame']
        end = goals[index + 1]['frame'] if index + 1 < len(goals) else len(xyz)
        errors = np.linalg.norm(xyz[start:end, 0, :2] - np.asarray(goal['position'])[:2], axis=-1)
        hits = errors <= radius
        recovered = next((i / fps for i in range(len(hits) - hold + 1)
                          if hits[i:i + hold].all()), None)
        results.append({'goal': goal, 'final_error_m': float(errors[-1]),
                        'recovery_seconds': recovered, 'success': recovered is not None,
                        'radius_m': radius, 'hold_seconds': hold_seconds})
    return results


def run_episode(generate, initial_history, goals, *, fps=30, seconds=10,
                mode='paused', feedback=True, lookahead_seconds=None, warmup=None, prefill=True,
                handoff='observed'):
    """Generate returns future native poses, beginning at Observation.frame.

    Real-time mode advances independently of inference, drops expired predictions,
    and holds the last pose on buffer exhaustion. Paused mode waits at request time.
    feedback=False freezes only the goal input; both variants receive live history.
    """
    if mode not in ('paused', 'realtime'):
        raise ValueError(mode)
    if handoff not in ('observed', 'committed'):
        raise ValueError(handoff)
    total = int(round(fps * seconds))
    history = [np.array(x, copy=True) for x in initial_history]
    if len(history) < 2 or not goals or goals[0]['frame'] != 0:
        raise ValueError('Need two history frames and an initial goal')
    events = sorted(goals, key=lambda event: event['frame'])
    queue, motions, trace = {}, [], []
    pending = None
    target = np.array(events[0]['position'], dtype=float)
    first_target = target.copy()
    holds = 0
    with ThreadPoolExecutor(max_workers=1) as worker:
        if warmup is not None:
            worker.submit(warmup).result()
        startup_seconds = 0.
        if prefill:
            start = time.perf_counter()
            initial_plan = np.asarray(worker.submit(generate, Observation(
                0, fps, np.stack(history[-2:]), first_target.copy())).result())
            startup_seconds = time.perf_counter() - start
            if not len(initial_plan) or not np.isfinite(initial_plan).all():
                raise ValueError('Invalid initial plan')
            queue.update({i: pose.copy() for i, pose in enumerate(initial_plan)})
        if lookahead_seconds is None:
            lookahead_seconds = min((len(initial_plan) - 1) / fps,
                                    startup_seconds * 1.25 + 1 / fps) if prefill else 0.
        origin = time.perf_counter()
        for frame in range(total):
            for event in events:
                if event['frame'] == frame:
                    target = np.array(event['position'], dtype=float)

            def receive():
                nonlocal pending
                if pending is None:
                    return
                future, requested_frame, requested_wall, goal, observed_frame = pending
                poses, compute_seconds = future.result()
                if len(poses) == 0 or not np.isfinite(poses).all():
                    raise ValueError('Generator returned an empty or non-finite plan')
                expired = max(0, frame - requested_frame)
                for offset in range(expired, len(poses)):
                    queue[requested_frame + offset] = poses[offset].copy()
                trace.append({'request_frame': requested_frame, 'apply_frame': frame,
                              'observed_frame': observed_frame,
                              'request_wall_seconds': requested_wall,
                              'compute_seconds': compute_seconds,
                              'expired_frames': min(expired, len(poses)),
                              'plan_frames': len(poses), 'goal': goal.tolist()})
                pending = None

            if pending is not None and pending[0].done():
                receive()
            remaining = sum(index >= frame for index in queue)
            if pending is None and remaining <= max(0, round(fps * lookahead_seconds)):
                observed_goal = target.copy() if feedback else first_target.copy()
                observed_history = np.stack(history[-2:])
                request_frame = frame
                plan_history = observed_history
                if handoff == 'committed' and remaining:
                    # Every queued frame is already committed and will execute unchanged.
                    # This is a forecasted handoff state, never a future measurement.
                    indices = sorted(index for index in queue if index >= frame)
                    if indices != list(range(frame, indices[-1] + 1)):
                        raise ValueError('Committed handoff requires a contiguous queue')
                    request_frame = indices[-1] + 1
                    plan_history = np.stack((history + [queue[index] for index in indices])[-2:])
                observation = Observation(request_frame, fps, plan_history, observed_goal,
                    observed_frame=frame, observed_history=observed_history,
                    history_kind='committed_forecast' if request_frame > frame else 'executed')
                def predict(snapshot=observation):
                    start = time.perf_counter()
                    poses = np.asarray(generate(snapshot))
                    return poses, time.perf_counter() - start
                pending = (worker.submit(predict), request_frame, time.perf_counter() - origin, observed_goal, frame)
                if mode == 'paused':
                    receive()
            if frame in queue:
                pose = queue.pop(frame)
            else:
                pose = history[-1].copy()
                holds += 1
            history.append(pose)
            motions.append(pose)
            if mode == 'realtime':
                delay = origin + (frame + 1) / fps - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
        elapsed = time.perf_counter() - origin
        trailing = pending is not None
        trailing_compute = pending[0].result()[1] if trailing else 0.
    return np.stack(motions), {'mode': mode, 'feedback': feedback, 'fps': fps,
        'frames': total, 'wall_seconds': elapsed, 'hold_frames': holds,
        'hold_fraction': holds / total, 'requests': trace,
        'final_request_not_applied': trailing,
        'trailing_compute_seconds': trailing_compute,
        'handoff': handoff,
        'startup_seconds': startup_seconds, 'lookahead_seconds': lookahead_seconds,
        'protocol_version': 3 if handoff == 'committed' else 2,
        'environment': 'kinematic playback; no force or contact solver',
        'fallback': 'hold last pose', 'participants': []}
