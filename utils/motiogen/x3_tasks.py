"""Experiment 3 (X3): one goal definition shared by DART, Kimodo and OmniControl."""
import json
import os
from pathlib import Path

import numpy as np

from .config import ROOT

PREREG = ROOT/'config/x3_cross_method_layers_20261104.json'
OUT = Path(os.environ.get('X3_OUT', ROOT/'develop_logs/test_runs/20261104_x3'))
GENERATED = {'dart': ['G0', 'G4', 'S'], 'kimodo': ['T', 'T+'], 'omnicontrol': ['T', 'S', 'TS']}
CORRECTED = {'dart': ['W0', 'W0X', 'O'], 'kimodo': ['O', 'TO'], 'omnicontrol': ['O', 'TO']}
PILOT_TASKS = [0, 3, 6, 9]


def prereg():
    return json.loads(PREREG.read_text())


def tasks(method):
    reg = prereg()
    if method == 'dart':
        rows = []
        for i in range(12):
            text, stream = ('walk', 'run')[i // 6], i % 6
            sign = 1. if stream % 2 == 0 else -1.
            rows.append({'id': i, 'text': text, 'stream': stream,
                         'goal_seconds': reg['methods']['dart']['goal_seconds'],
                         'signs': [sign*s for s in (1., -1., 1., -1.)]})
        return rows
    return [{'id': i, 'prompt': p, 'goal_seconds': reg['methods'][method]['goal_seconds'],
             'signs': [1. if i % 2 == 0 else -1.]} for i, p in enumerate(reg['prompts'])]


def seeds(stage):
    reg = prereg()
    return reg['seeds'][:1] if stage == 'pilot' else reg['seeds']+([reg['top_up_seed']] if stage == 'top_up' else [])


def job_tasks(method, stage):
    rows = tasks(method)
    rows = [t for t in rows if t['id'] in PILOT_TASKS] if stage == 'pilot' else rows
    if os.environ.get('X3_TASKS'):
        rows = [t for t in rows if t['id'] in {int(x) for x in os.environ['X3_TASKS'].split(',')}]
    return rows


def new_cells(method, stage):
    """(task, seed) cells a stage generates: confirm reuses the pilot cells, top_up adds only its seed."""
    reg = prereg()
    if stage == 'top_up':
        return [(t, reg['top_up_seed']) for t in job_tasks(method, stage)]
    rows = [(t, s) for t in job_tasks(method, stage) for s in seeds(stage)]
    if stage == 'confirm' and not os.environ.get('X3_INCLUDE_PILOT'):
        rows = [(t, s) for t, s in rows if not (t['id'] in PILOT_TASKS and s == reg['seeds'][0])]
    return rows


def base_dir(method, task, seed):
    return OUT/method/f'task{task:02d}_seed{seed}'


def run_dir(method, task, seed, arm, offset=None):
    return base_dir(method, task, seed)/(arm if arm == 'C0' else f'{arm}_d{offset:g}')


def lateral(joints, names, fps, frame, window_seconds=.5, min_speed=.2):
    """Unit XY vector to the body right of the root heading at frame (Z-up world)."""
    names = list(names)
    pelvis = joints[:, names.index('pelvis'), :2]
    back = max(0, frame-int(round(window_seconds*fps)))
    velocity = (pelvis[frame]-pelvis[back])/max((frame-back)/fps, 1e-9)
    if np.linalg.norm(velocity) >= min_speed:
        forward = velocity/np.linalg.norm(velocity)
        return np.array([forward[1], -forward[0]]), 'velocity'
    right = joints[frame, names.index('right_hip'), :2]-joints[frame, names.index('left_hip'), :2]
    return right/np.linalg.norm(right), 'hip_facing'


def goals(joints, names, fps, task, offset):
    names = list(names)
    pelvis = joints[:, names.index('pelvis')]
    rows = []
    for seconds, sign in zip(task['goal_seconds'], task['signs']):
        frame = int(round(seconds*fps))
        side, rule = lateral(joints, names, fps, frame)
        free = pelvis[frame, :2]
        rows.append({'seconds': seconds, 'frame': frame, 'sign': sign, 'lateral_xy': side.tolist(), 'lateral_rule': rule,
                     'free_xy': free.tolist(), 'free_height_m': float(pelvis[frame, 2]),
                     'target_xy': (free+sign*offset*side).tolist()})
    return rows


def goal_receipt(joints, names, goal_rows):
    pelvis = joints[:, list(names).index('pelvis'), :2]
    out = []
    for g in goal_rows:
        reached = pelvis[g['frame']]
        side = np.asarray(g['lateral_xy'])
        offset = np.linalg.norm(np.asarray(g['target_xy'])-np.asarray(g['free_xy']))
        out.append({**g, 'reached_xy': reached.tolist(),
                    'error_m': float(np.linalg.norm(reached-np.asarray(g['target_xy']))),
                    'fraction_toward_target': float(np.dot(reached-np.asarray(g['free_xy']), g['sign']*side)/offset) if offset else None})
    return out
