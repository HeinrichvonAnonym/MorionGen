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
NOMINAL_BODY = {'shoulder_forward_m': 0., 'shoulder_lateral_m': SHOULDER_LATERAL_M, 'shoulder_height_m': SHOULDER_HEIGHT_M,
                'arm_length_m': ARM_LENGTH_M, 'mirror_forward_m': .40, 'mirror_cross_m': .10, 'mirror_from_shoulder': False}
# Amendment 1 (config/next_round_20261030_amend1.json): frame 0 of the 18 V1
# pilot runs puts the shoulders 0.03 m behind the pelvis and gives
# shoulder-elbow-wrist sums of 0.43-0.56 m (OmniControl median below 0.50 m).
MEASURED_BODY_20261031 = {'shoulder_forward_m': -.03, 'shoulder_lateral_m': .20, 'shoulder_height_m': 1.37,
                          'arm_length_m': .47, 'mirror_forward_m': .30, 'mirror_cross_m': .05, 'mirror_from_shoulder': True}


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


def _forward_target(side, distance, elevation_deg, body=NOMINAL_BODY):
    angle = math.radians(elevation_deg)
    lateral = _sign(side)*body['shoulder_lateral_m']
    return (body['shoulder_forward_m']+distance*math.cos(angle), lateral, body['shoulder_height_m']+distance*math.sin(angle))


def reach_tasks(body=None):
    """24 single-wrist targets: 3 tiers x 3 heights x 2 sides + 6 mirror targets.

    body defaults to NOMINAL_BODY (the registered 20261030 tasks);
    MEASURED_BODY_20261031 gives the amendment-1 tasks.
    """
    body = body or NOMINAL_BODY
    arm = body['arm_length_m']
    tasks = []
    for tier, ratio in [('reachable', .8), ('stepping', 1.3), ('legacy', None)]:
        for height in ELEVATIONS_DEG:
            for side in ('right', 'left'):
                if ratio is None:
                    # Original 20261024 world target, kept verbatim as the replication tier.
                    world = [_sign(side)*.35, .6, LEGACY_HEIGHTS_M[height]]
                    point = world_to_body(world)
                else:
                    point = list(_forward_target(side, ratio*arm, ELEVATIONS_DEG[height], body))
                    world = body_to_world(*point)
                tasks.append({'tier': tier, 'height': height, 'side': side, 'joint': side+'_wrist',
                              'target_body_m': [round(v, 8) for v in point], 'position_m': world})
    for height in ELEVATIONS_DEG:
        for side in ('right', 'left'):
            # Cross-body target past the midline, inside arm length from the
            # constrained shoulder and closer to the other hand's side.
            angle = math.radians(ELEVATIONS_DEG[height])
            forward = body['mirror_forward_m']*math.cos(angle)+(body['shoulder_forward_m'] if body['mirror_from_shoulder'] else 0.)
            point = [forward, -_sign(side)*body['mirror_cross_m'], body['shoulder_height_m']+body['mirror_forward_m']*math.sin(angle)]
            tasks.append({'tier': 'mirror', 'height': height, 'side': side, 'joint': side+'_wrist',
                          'target_body_m': [round(v, 8) for v in point], 'position_m': body_to_world(*point)})
    for index, task in enumerate(tasks):
        shoulder = np.array([body['shoulder_forward_m'], _sign(task['side'])*body['shoulder_lateral_m'], body['shoulder_height_m']])
        task['id'] = index
        task['nominal_shoulder_distance_ratio'] = round(float(np.linalg.norm(np.asarray(task['target_body_m'])-shoulder)/arm), 4)
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
