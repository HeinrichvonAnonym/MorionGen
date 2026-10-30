"""Body-frame task geometry for the 20261030 next-round protocol.

Report frame: right-handed Z-up, converted from native Y-up by (x, -z, y).
HumanML-style generators canonicalise frame 0 to face native +Z with the body
right side at native -X, so in the report frame the body faces -Y and its right
side is -X. Tasks are authored in that body frame and converted here once; the
pilot audit measures the realised frame-0 heading instead of assuming it.

Nominal body dimensions come from frame 0 of the 20261024 MotionLCM control
matrix (shoulder 0.20 m lateral, 1.37 m high, shoulder-elbow-wrist 0.54 m).
They define task tiers only; per-run reachability is recomputed from the
generated skeleton.
"""
import math

import numpy as np

BODY_FORWARD = np.array([0., -1., 0.])
BODY_RIGHT = np.array([-1., 0., 0.])
UP = np.array([0., 0., 1.])
SHOULDER_LATERAL_M = .20
SHOULDER_HEIGHT_M = 1.37
ARM_LENGTH_M = .54
ELEVATIONS_DEG = {'low': -30., 'mid': 0., 'high': 30.}
LEGACY_HEIGHTS_M = {'low': .9, 'mid': 1.2, 'high': 1.5}


def body_to_world(forward, right, up):
    point = forward*BODY_FORWARD+right*BODY_RIGHT+up*UP
    return [round(float(v), 8) for v in point]


def world_to_body(point):
    point = np.asarray(point, float)
    return [float(point@BODY_FORWARD), float(point@BODY_RIGHT), float(point@UP)]


def _sign(side):
    if side not in ('right', 'left'):
        raise ValueError('side must be right or left')
    return 1. if side == 'right' else -1.


def _forward_target(side, distance, elevation_deg, shoulder_side=None):
    shoulder_side = shoulder_side or side
    angle = math.radians(elevation_deg)
    lateral = _sign(shoulder_side)*SHOULDER_LATERAL_M
    return (distance*math.cos(angle), lateral, SHOULDER_HEIGHT_M+distance*math.sin(angle))


def reach_tasks():
    """24 single-wrist targets: 3 tiers x 3 heights x 2 sides + 6 mirror targets."""
    tasks = []
    for tier, ratio in [('reachable', .8), ('stepping', 1.3), ('legacy', None)]:
        for height in ELEVATIONS_DEG:
            for side in ('right', 'left'):
                if ratio is None:
                    # Original 20261024 world target, kept verbatim as the replication tier.
                    world = [_sign(side)*.35, .6, LEGACY_HEIGHTS_M[height]]
                    body = world_to_body(world)
                else:
                    body = list(_forward_target(side, ratio*ARM_LENGTH_M, ELEVATIONS_DEG[height]))
                    world = body_to_world(*body)
                tasks.append({'tier': tier, 'height': height, 'side': side, 'joint': side+'_wrist',
                              'target_body_m': [round(v, 8) for v in body], 'position_m': world})
    for height in ELEVATIONS_DEG:
        for side in ('right', 'left'):
            # Cross-body target 0.10 m past the midline: about 0.50 m from the
            # constrained shoulder (inside 0.54 m) and closer to the other hand's side.
            angle = math.radians(ELEVATIONS_DEG[height])
            body = [.40*math.cos(angle), -_sign(side)*.10, SHOULDER_HEIGHT_M+.40*math.sin(angle)]
            tasks.append({'tier': 'mirror', 'height': height, 'side': side, 'joint': side+'_wrist',
                          'target_body_m': [round(v, 8) for v in body], 'position_m': body_to_world(*body)})
    for index, task in enumerate(tasks):
        shoulder = np.array([0., _sign(task['side'])*SHOULDER_LATERAL_M, SHOULDER_HEIGHT_M])
        task['id'] = index
        task['nominal_shoulder_distance_ratio'] = round(float(np.linalg.norm(np.asarray(task['target_body_m'])-shoulder)/ARM_LENGTH_M), 4)
        task['prompt'] = f"A person reaches forward with the {task['side']} hand."
    return tasks


def reach_direction_prompt(task):
    """Target-direction wording for the V1b text cross; None when identical to the default."""
    if task['height'] == 'mid':
        return None
    word = 'up' if task['height'] == 'high' else 'down'
    return f"A person reaches forward and {word} with the {task['side']} hand."


ARDY_PHRASES = {0: 'forward', 45: 'forward and to the right', 90: 'to the right', 135: 'backward and to the right',
                180: 'backward', 225: 'backward and to the left', 270: 'to the left', 315: 'forward and to the left'}
ARDY_SPEEDS = [(.4, -.2), (.2, -.1), (.6, -.3)]
ARDY_SCHEDULES = {'none': None, 'reverse_5s': 5., 'reverse_10s': 10.}


def native_direction(heading_deg):
    """Body heading (0 forward, 90 right) to a unit native XZ vector (forward +Z, right -X)."""
    angle = math.radians(heading_deg)
    vector = np.array([-math.sin(angle), math.cos(angle)])
    vector[np.abs(vector) < 1e-12] = 0.
    return [float(v) for v in vector/np.linalg.norm(vector)]


def walk_prompt(heading_deg):
    return f'A person walks {ARDY_PHRASES[int(heading_deg) % 360]}.'


def ardy_path_tasks():
    """24 fixed root paths: 8 body headings x 3 speed profiles, reversing at 5 s."""
    tasks = []
    for before, after in ARDY_SPEEDS:
        for heading in sorted(ARDY_PHRASES):
            tasks.append({'id': len(tasks), 'heading_deg': heading, 'direction_xz': native_direction(heading),
                          'speed_before_m_s': before, 'speed_after_m_s': after, 'switch_seconds': 5.,
                          'aligned_prompt': walk_prompt(heading), 'aligned_update': walk_prompt(heading+180),
                          'conflict_prompt': 'A person walks forward.', 'conflict_update': 'A person walks backward.'})
    return tasks


def ardy_offset_tasks():
    """24 offset tasks: 8 headings x 3 text schedules; lateral offset is toward body right of the heading."""
    tasks = []
    for schedule, seconds in ARDY_SCHEDULES.items():
        for heading in sorted(ARDY_PHRASES):
            tasks.append({'id': len(tasks), 'heading_deg': heading, 'direction_xz': native_direction(heading),
                          'lateral_xz': native_direction(heading+90), 'schedule': schedule, 'update_seconds': seconds,
                          'aligned_prompt': walk_prompt(heading),
                          'aligned_update': walk_prompt(heading+180) if seconds is not None else None})
    return tasks


def path_distance(seconds, before, after, switch=5.):
    seconds = np.asarray(seconds, float)
    return before*np.minimum(seconds, switch)+after*np.maximum(seconds-switch, 0.)
