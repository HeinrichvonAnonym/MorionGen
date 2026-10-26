"""Diagnostic only: replay goal timing and isolate the native marker collider.

Reuse the completed native runner without editing it or the upstream checkout.
Every exact in-memory source overlay is retained with its source in the receipt.
These runs are not members of the registered primary capability matrix.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import runpy
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(json.loads((ROOT / 'config/baselines.json').read_text())['data_root'])
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--probe-goal-frame', type=int, required=True)
parser.add_argument('--probe-marker', choices=['native', 'isolated'], required=True)
probe, remaining = parser.parse_known_args()
if not 0 < probe.probe_goal_frame < 600:
    raise ValueError('Diagnostic goal frame must be inside the saved rollout')
if os.environ.get('MOTIONGEN_CLOSD_ENV_READY') != '1':
    raise ValueError('Use the registered probe plan with the isolated CLoSD environment')
repo = DATA / 'baselines/CLoSD'
sys.path[:0] = [str(repo), str(repo / 'closd'), str(DATA / 'staged_dependencies/pytorch3d'), str(ROOT / 'utils')]
from isaacgym import gymapi  # Must precede torch.
import closd.run as upstream
import closd.env.tasks.closd_multitask as multitask
import motiogen.closd_native as native
from motiogen.runtime import json_write, checksum

overlays = []
def overlay(function, replacements, namespace):
    source = textwrap.dedent(inspect.getsource(function))
    changed = source
    for old, new in replacements:
        if changed.count(old) != 1:
            raise ValueError('Source overlay no longer has one exact match: ' + old)
        changed = changed.replace(old, new, 1)
    overlays.append({'function': function.__module__ + '.' + function.__qualname__,
                     'original_sha256': hashlib.sha256(source.encode()).hexdigest(),
                     'overlay_sha256': hashlib.sha256(changed.encode()).hexdigest(),
                     'replacements': replacements, 'executed_source': changed})
    scope = dict(namespace)
    exec(compile(changed, '<recorded-cl osd-marker-probe>'.replace(' ', ''), 'exec'), scope)
    return scope[function.__name__]

if probe.probe_marker == 'isolated':
    multitask.CLoSDMultiTask._build_target = overlay(
        multitask.CLoSDMultiTask._build_target,
        [('col_group = env_id', 'col_group = env_id + self.num_envs')], vars(multitask))

frame = probe.probe_goal_frame
scope_text = 'Diagnostic schedule replay; marker collision mode=' + probe.probe_marker + '; not a primary native capability run'
replayed = overlay(native.run, [
    ("if not phase['changed'] and (wall>=5. if args.clock=='realtime' else frame>=150):",
     "if not phase['changed'] and frame>=" + str(frame) + ':'),
    ("{'frame':150,'position':[*args.goal_next,0.]}", "{'frame':" + str(frame) + ",'position':[*args.goal_next,0.]}"),
    ("'goal_request_seconds':5", "'goal_request_seconds':" + repr(frame / 30)),
    ("'variant':'native DiP + PHC + IsaacGym human'", "'variant':" + repr(scope_text)),
    ("'scope':'Native fixed pretrained human simulation; explicit absolute root goals; native failure resets retained'",
     "'scope':" + repr(scope_text)),
], vars(native))

records = []
player = upstream.im_amp_players.IMAMPPlayerContinuous
original_step = player.env_step
def observe(self, env, action):
    result = original_step(self, env, action)
    task = env.task
    if task.num_envs != 1:
        raise ValueError('Diagnostic requires one human environment')
    records.append({'frame': len(records),
        'marker_contact_force_n': task._tar_contact_forces[0].detach().cpu().tolist(),
        'marker_position_m': task._target_states[0, :3].detach().cpu().tolist(),
        'human_body_count': task.num_bodies,
        'actors_per_environment': task.get_num_actors_per_env()})
    return result
player.env_step = observe

def instrumented(args, upstream_entry, root, data):
    if args.clock != 'paused':
        raise ValueError('Replay uses fixed simulation frames, not a new wall-clock schedule')
    try:
        return replayed(args, upstream_entry, root, data)
    finally:
        json_write(args.run_dir / 'marker_probe.json', {
            'scope': scope_text, 'goal_update_frame': frame, 'marker_mode': probe.probe_marker,
            'complete': len(records) == args.frames, 'records': records, 'overlays': overlays,
            'native_runner_sha256': checksum(ROOT / 'utils/motiogen/closd_native.py'),
            'native_marker_sha256': checksum(repo / 'closd/data/assets/urdf/location_marker.urdf'),
            'collision_group_contract': 'IsaacGym preview4: bodies collide only within the same group; native target group=env_id, isolated target group=env_id+num_envs. Human group is unchanged.',
            'timing_scope': 'Per-step observer copies contact forces; do not interpret these diagnostic timings as primary H estimates.'})

native.run = instrumented
entry = ROOT / 'python/scripts/closd_native.py'
sys.argv = [str(entry), *remaining]
runpy.run_path(str(entry), run_name='__main__')
