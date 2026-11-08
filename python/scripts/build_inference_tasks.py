#!/usr/bin/env python
"""Inference task table of the section-14 batch: 24 tasks per method, text table, development / held-out split, rows.

Reads only inputs and receipts, never an outcome:
  - develop_logs/test_runs/20261109_task_reachability/{reachability,overhead_limit}.json (task_reachability.py, nominal body)
  - the native matrix rows (20261024_native_matrix_analysis/analysis.json: task_spec, seed, condition, clock, run): seen
    inputs and reuse candidates; the V2 CHOIS plans and queues (20261101_v2_chois): seen windows and reuse candidates
  - the 10-31 M0 ARDY plans (20261030_next_round/m0*): seen root-path directions
  - the contract scan (20261108_contract/scan.json): layer 4 state per method
Writes config/inference_tasks_20261109.json. Registration: config/inference_registration_20261109.json.
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.task_geometry import body_to_world, native_direction, world_to_body  # noqa: E402

RUNS = ROOT/'develop_logs/test_runs'
REACH = RUNS/'20261109_task_reachability'
MATRIX = RUNS/'20261024_native_matrix_analysis/analysis.json'
V2 = RUNS/'20261101_v2_chois'
M0 = [RUNS/'20261030_next_round/m0/pilot_plan.json', RUNS/'20261030_next_round/m0_amend2/pilot_plan.json',
      RUNS/'20261030_next_round/m0_amend2/batch_plan.json', RUNS/'20261030_next_round/m0_amend2/continuation_plan.json']
SCAN = RUNS/'20261108_contract/scan.json'
OUT = ROOT/'config/inference_tasks_20261109.json'

SEEDS = [20, 21, 22, 23, 24]
HELD_OUT_SEED, HELD_OUT_PARAPHRASE = 24, 4
CONDITIONS = {
    'motionlcm': [{'control_residual': 0}, {'control_residual': 1}],
    'omnicontrol': [{'realism': r, 'spatial': s} for r in (0, 1) for s in (0, 1)],
    'closd': [{'execution_context': 0}, {'execution_context': 1}],
    'dart': [{'goal_update': 0}, {'goal_update': 1}],
    'ardy': [{'history_frames': h, 'correction': c} for h in (32, 160) for c in (0, 1)],
    'primal': [{'ground_snap': g, 'reprojection': r, 'inertialization': i} for g in (0, 1) for r in (0, 1) for i in (0, 1)],
    'chois': [{'guidance': 0}, {'guidance': 1}]}
L4_REUSABLE = ('ok', 'legacy')
# development / held-out: target positions (conventions; interleaved so both sets cover every level)
HELD_OUT_TASKS = {'motionlcm': [12, 17, 19, 20], 'omnicontrol': [12, 17, 19, 20], 'closd': [12, 17, 19, 22], 'dart': [12, 17, 19, 22],
                  'primal': [15, 17, 19, 21], 'ardy': [], 'chois': []}
# PRIMAL: primal_feedback_smoke.py seeds the history with initial_seed + initial_history and generates 140 x history frames,
# so a short and a long history with the same effective seed share their first 140 frames; held out by whole groups
PRIMAL_HELD_OUT_RULE = ('held out by resampling group (effective seed = initial_seed + initial_history): groups 9009 (tasks 15, 19) '
                        'and 9011 (tasks 17, 21), both history lengths twice; no held-out initial state shares a history prefix with '
                        'a development one (tasks 12, 13 share theirs with matrix tasks 10, 11)')
NO_HELD_OUT_POSITION = {'ardy': 'all 8 root-path directions were run and inspected in the 10-31 M0 experiments (seeds 20-24); held out only through seed 24',
                        'chois': 'the 12 added windows were run and inspected in V2 (seeds 20-22); held out only through seed 24, which tests generalisation across seeds only (to be confirmed by the user)'}
# text table (conventions, not derived; for user review)
TEMPLATES = ['A person reaches {z} with the {side} hand.', 'A person extends the {side} hand {z}.',
             'Someone reaches {z} using the {side} hand.', 'A person stretches the {side} arm {z}.',
             'The person moves the {side} hand {z}.']
NEUTRAL = 'forward'
LEGACY_HEIGHT_WORDS = {.9: 'at hip height', 1.2: 'at chest height', 1.5: 'at shoulder height'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def r6(v):
    return [round(float(x), 6) + 0. for x in v]


def zone_phrase(task):
    side = task['side']
    if task['tier'] == 'legacy':
        across = 'left' if side == 'right' else 'right'
        return f"behind and to the {across} {LEGACY_HEIGHT_WORDS[task['position_m'][2]]}"
    out = {('floor', 'front'): 'down to the floor in front', ('floor', 'side90'): f'down to the floor on the {side} side',
           ('knee', 'front'): 'forward at knee height', ('knee', 'side90'): f'out to the {side} side at knee height',
           ('overhead', 'front'): 'up high in front', ('overhead', 'side90'): f'up high on the {side} side'}
    return out[(task['zone'], task['direction'])]


def reach_tasks(matrix, reach):
    by_target = {tuple(round(v, 4) for v in t['target_body_m']): t for t in reach['tasks']}
    tasks = []
    for tid in range(12):
        spec = next(r['task_spec'] for r in matrix if r['method'] == 'motionlcm' and r['task'] == tid)
        c = spec['constraints'][0]; side = c['joint'].split('_')[0]
        body = world_to_body(c['position_m'])
        rr = by_target[tuple(round(v, 4) for v in body)]
        tasks.append({'id': tid, 'tier': 'legacy', 'side': side, 'joint': c['joint'], 'position_m': c['position_m'],
                      'target_body_m': r6(body), 'native_matrix_task': tid, 'reachability_flat_residual_m': rr['flat']['residual_m']})
    risk = [t for t in reach['tasks'] if t['tier'] == 'risk']
    for k, t in enumerate(sorted(risk, key=lambda t: (t['side'] != 'right', ['floor', 'knee', 'overhead'].index(t['zone']), t['direction'] != 'front'))):
        tasks.append({'id': 12+k, 'tier': 'risk', 'side': t['side'], 'zone': t['zone'], 'direction': t['direction'], 'joint': t['side']+'_wrist',
                      'position_m': r6(body_to_world(*t['target_body_m'])), 'target_body_m': r6(t['target_body_m']),
                      'native_matrix_task': None, 'reachability_flat_residual_m': t['flat']['residual_m']})
    for t in tasks:
        t['time_seconds'], t['duration_seconds'] = 2.5, 5
        t['consistent_phrase'] = zone_phrase(t)
    return tasks


def root_goal_tasks(matrix, method, next_start, next_turn):
    tasks = []
    for tid in range(12):
        spec = next(r['task_spec'] for r in matrix if r['method'] == method and r['task'] == tid)
        tasks.append({'id': tid, 'tier': 'legacy', 'goal_start': spec['goal_start'], 'goal_next': spec['goal_next'],
                      'native_matrix_task': tid})
    levels = []
    for t in tasks:
        d = round(math.hypot(*t['goal_start']), 6)
        t['distance_m'] = d; t['direction_deg'] = round(math.degrees(math.atan2(t['goal_start'][1], t['goal_start'][0])) % 360)
        if d not in levels:
            levels.append(d)
    for a, d in enumerate(levels):
        for b, angle in enumerate((45, 135, 225, 315)):
            u = [math.cos(math.radians(angle)), math.sin(math.radians(angle))]
            start = [d*u[0], d*u[1]]
            nxt = [next_start*start[0]-next_turn*u[1], next_start*start[1]+next_turn*u[0]]
            tasks.append({'id': 12+4*a+b, 'tier': 'added', 'goal_start': r6(start), 'goal_next': r6(nxt), 'distance_m': d,
                          'direction_deg': angle, 'native_matrix_task': None})
    for t in tasks:
        t['update_seconds'], t['duration_seconds'] = 5, 20
    return sorted(tasks, key=lambda t: t['id'])


def ardy_tasks(matrix):
    tasks = []
    for tid in range(12):
        spec = next(r['task_spec'] for r in matrix if r['method'] == 'ardy' and r['task'] == tid)
        tasks.append({'id': tid, 'tier': 'legacy', 'native_root_path': spec['native_root_path'], 'prompt': spec['prompt'],
                      'updates': spec['updates'], 'native_matrix_task': tid})
    speeds = []
    for t in tasks:
        p = t['native_root_path']; s = (p['speed_before_m_s'], p['speed_after_m_s'])
        if s not in speeds:
            speeds.append(s)
    for a, (before, after) in enumerate(speeds):
        for b, heading in enumerate((315, 225, 135, 45)):
            tasks.append({'id': 12+4*a+b, 'tier': 'added', 'heading_deg': heading,
                          'native_root_path': {'direction_xz': native_direction(heading), 'speed_before_m_s': before, 'speed_after_m_s': after},
                          'prompt': tasks[0]['prompt'], 'updates': tasks[0]['updates'], 'native_matrix_task': None})
    return sorted(tasks, key=lambda t: t['id'])


def primal_tasks(matrix):
    tasks = []
    for tid in range(12):
        spec = next(r['task_spec'] for r in matrix if r['method'] == 'primal' and r['task'] == tid)
        tasks.append({'id': tid, 'tier': 'legacy', 'family': spec['family'], 'initial_history': spec['initial_history'],
                      'initial_seed': spec['initial_seed'], 'native_matrix_task': tid})
    for k, s in enumerate(range(9005, 9011)):
        tasks.append({'id': 12+k, 'tier': 'added', 'family': 1, 'initial_history': 1, 'initial_seed': s, 'native_matrix_task': None})
    for k, s in enumerate(range(9006, 9012)):
        tasks.append({'id': 18+k, 'tier': 'added', 'family': 2, 'initial_history': 2, 'initial_seed': s, 'native_matrix_task': None})
    for t in tasks:
        t['frames'] = 600
    return tasks


def chois_tasks(matrix):
    v2 = json.loads((V2/'batch_plan.json').read_text())['tasks']
    tasks = []
    for t in v2[:24]:
        legacy = t['task_id'] < 12
        if legacy:
            spec = next(r['task_spec'] for r in matrix if r['method'] == 'chois' and r['task'] == t['task_id'])
            assert (spec['native_window_index'], spec['native_sequence']) == (t['native_window_index'], t['sequence'])
        tasks.append({'id': t['task_id'], 'tier': 'legacy' if legacy else 'added', 'native_window_index': t['native_window_index'],
                      'native_sequence': t['sequence'], 'object': t['object'], 'v2_task': t['task_id'],
                      'native_matrix_task': t['task_id'] if legacy else None, 'duration_seconds': 4})
    return tasks


def input_key(method, task, seed, condition, prompt=None):
    if method in ('motionlcm', 'omnicontrol'):
        x = [prompt, task['joint'], task['position_m']]
    elif method in ('closd', 'dart'):
        x = [task['goal_start'], task['goal_next']]
    elif method == 'ardy':
        x = [task['prompt'], task['updates'], task['native_root_path']]
        # the native matrix analysis records history_frames 32 / 160 as history_long 0 / 1
        condition = {'history_long': int(condition['history_frames'] == 160), 'correction': condition['correction']}
    elif method == 'primal':
        x = [task['family'], task['initial_history'], task['initial_seed']]
    else:
        x = [task['native_window_index'], task['native_sequence']]
    return json.dumps([method, x, seed, condition], sort_keys=True)


def matrix_key(row):
    s, m = row['task_spec'], row['method']
    if m in ('motionlcm', 'omnicontrol'):
        c = s['constraints'][0]; x = [s['prompt'], c['joint'], c['position_m']]
    elif m in ('closd', 'dart'):
        x = [s['goal_start'], s['goal_next']]
    elif m == 'ardy':
        # task 0 seed 20 pilots have no native_root_path in their spec: not treated as identical
        x = [s['prompt'], s['updates'], s.get('native_root_path')]
    elif m == 'primal':
        x = [s['family'], s['initial_history'], s['initial_seed']]
    else:
        # task 0 seed 20 pilots lack native_window_index: not treated as identical
        x = [s.get('native_window_index'), s['native_sequence']]
    return json.dumps([m, x, row['seed'], row['condition']], sort_keys=True)


def v2_seen():
    """V2 CHOIS: identical-input runs (condition exactly {'guidance': 0}) with their run directories, and all seen windows."""
    runs, windows = {}, set()
    for plan in sorted(V2.glob('*_plan.json')):
        queue = V2/plan.name.replace('_plan.json', '_queue')/'state.json'
        state = json.loads(queue.read_text())['jobs'] if queue.exists() else {}
        for j in json.loads(plan.read_text())['jobs']:
            sc = j['scene']; windows.add(sc['native_window_index'])
            st = state.get(j['id'], {})
            if st.get('status') == 'success':
                key = json.dumps(['chois', [sc['native_window_index'], sc['native_sequence']], j['seed'], j['config']['_intervention']], sort_keys=True)
                runs[key] = str(Path(st['run_dir']).relative_to(ROOT))
    return runs, windows


def m0_seen():
    directions = set()
    for plan in M0:
        for j in json.loads(plan.read_text())['jobs']:
            if j['method'] == 'ardy':
                directions.add(tuple(round(v, 6) for v in j['scene']['native_root_path']['direction_xz']))
    return directions


def main():
    reach, limit = json.loads((REACH/'reachability.json').read_text()), json.loads((REACH/'overhead_limit.json').read_text())
    matrix_all = json.loads(MATRIX.read_text())['rows']
    matrix = [r for r in matrix_all if r['clock'] in ('paused', None)]
    goals = {(r['method'], r['task']): r['task_spec'] for r in matrix if 'goal_start' in r['task_spec']}
    for r in matrix:
        if r['method'] in ('closd', 'dart') and 'goal_start' not in r['task_spec']:
            # task 0 seed 20 pilots ran on the script defaults; the matrix analysis files them as task 0
            r['task_spec'] = {**r['task_spec'], **{k: goals[(r['method'], r['task'])][k] for k in ('goal_start', 'goal_next')}}
    seen = {}
    for r in matrix:
        seen.setdefault(matrix_key(r), r)
    v2_runs, v2_windows = v2_seen()
    m0_dirs = m0_seen()
    l4 = {m: s['L4_native']['states'] for m, s in json.loads(SCAN.read_text())['summary'].items()}
    reach_t = reach_tasks(matrix, reach)
    methods = {
        'motionlcm': {'family': 'reach (wrist position target at 2.5 s of a 5 s clip)', 'tasks': reach_t, 'text': 'neutral / consistent x 5 paraphrases'},
        'omnicontrol': {'family': 'reach, same 24 tasks as motionlcm', 'tasks': reach_t, 'text': 'neutral / consistent x 5 paraphrases'},
        'closd': {'family': 'native root-goal change (REACH task: ground goal, second goal at 5 s of 20 s); does not cover risk-zone reaching (design section 2 lists CLoSD among the reach methods; to be confirmed by the user)',
                  'tasks': root_goal_tasks(matrix, 'closd', -.75, 1.), 'goal_next_rule': 'goal_next = -0.75 goal_start + 1.0 R90(u), R90(x, y) = (-y, x) (native matrix rule)',
                  'text': 'native fixed text "A person is walking." (no text factor, no paraphrase rotation)'},
        'dart': {'family': 'native goal reaching (reach_location policy), second goal at 5 s of 20 s', 'tasks': root_goal_tasks(matrix, 'dart', -1., .5),
                 'goal_next_rule': 'goal_next = -goal_start + 0.5 R90(u) (native matrix rule)', 'text': 'native fixed goal text "walk"'},
        'ardy': {'family': 'streaming root path: 8 directions x 3 speed profiles, text switch at 5 s', 'tasks': ardy_tasks(matrix),
                 'text': 'native matrix texts "A person walks forward." -> "A person walks backward." at 5 s for every direction (no paraphrase rotation)'},
        'primal': {'family': 'unconditioned continuation from 24 initial states; one run writes all 8 geometric switch combinations', 'tasks': primal_tasks(matrix),
                   'text': 'no text input'},
        'chois': {'family': 'official OMOMO test windows (first two eligible windows per object, native order; V2 tasks 0-23)', 'tasks': chois_tasks(matrix),
                  'text': 'the OMOMO annotation of each window (the task_spec prompt "Lift and carry the box." is metadata only)'}}
    for t in methods['ardy']['tasks']:
        t['direction_seen_in_m0'] = tuple(round(v, 6) for v in t['native_root_path']['direction_xz']) in m0_dirs
    for t in methods['chois']['tasks']:
        t['window_seen_in_v2'] = t['native_window_index'] in v2_windows
    paraphrases = {}
    for m, spec in methods.items():
        held = set(HELD_OUT_TASKS[m])
        for t in spec['tasks']:
            t['split_position'] = 'held_out' if t['id'] in held else 'development'
        spec['conditions'] = CONDITIONS[m]
        spec['l4_state_of_old_runs'] = l4.get(m)
        spec['held_out_positions'] = sorted(held)
        if m in NO_HELD_OUT_POSITION:
            spec['no_held_out_position'] = NO_HELD_OUT_POSITION[m]
    methods['primal']['held_out_rule'] = PRIMAL_HELD_OUT_RULE
    for t in methods['primal']['tasks']:
        t['resampling_group'] = 'stand' if t['family'] == 0 else t['initial_seed']+t['initial_history']
    header = ['method', 'task', 'seed', 'condition', 'text_level', 'paraphrase', 'prompt', 'split', 'split_reason', 'identical_seen_run', 'reuse_candidate']
    rows = []
    for m, spec in methods.items():
        levels = ('neutral', 'consistent') if m in ('motionlcm', 'omnicontrol') else (None,)
        conds = [None] if m == 'primal' else spec['conditions']
        for t in spec['tasks']:
            hand_index = None
            if m in ('motionlcm', 'omnicontrol'):
                same = [x['id'] for x in spec['tasks'] if x['side'] == t['side']]
                hand_index = same.index(t['id'])
            for seed in SEEDS:
                para = None
                if hand_index is not None:
                    para = HELD_OUT_PARAPHRASE if seed == HELD_OUT_SEED else (hand_index+seed-SEEDS[0]) % 4
                for ci, cond in enumerate(conds):
                    for level in levels:
                        prompt = None
                        if level:
                            prompt = TEMPLATES[para].format(z=NEUTRAL if level == 'neutral' else t['consistent_phrase'], side=t['side'])
                            paraphrases.setdefault(t['side'], {}).setdefault(level, {}).setdefault(t['id'], {})[para] = prompt
                        if m == 'primal':
                            keys = [input_key(m, t, seed, c) for c in CONDITIONS['primal']]
                            hit = [seen.get(k) for k in keys]
                            same_run = hit[0]['run'] if all(hit) and len({h['run'] for h in hit}) == 1 else None
                        elif m == 'chois' and t['tier'] == 'added':
                            same_run = v2_runs.get(input_key(m, t, seed, cond))
                        else:
                            h = seen.get(input_key(m, t, seed, cond, prompt)); same_run = h['run'] if h else None
                        if same_run:
                            split, reason = 'development', 'identical input already run'
                        elif t['split_position'] == 'held_out':
                            split, reason = 'held_out', 'held-out target position'
                        elif seed == HELD_OUT_SEED:
                            split, reason = 'held_out', 'held-out seed' + (' and paraphrase' if para == HELD_OUT_PARAPHRASE else '')
                        else:
                            split, reason = 'development', 'development position and seed'
                        reuse = bool(same_run) and set((l4.get(m) or {}).keys()) <= set(L4_REUSABLE)
                        rows.append([m, t['id'], seed, ci, level, para, prompt, split, reason, same_run, reuse])
    count = {}
    for r in rows:
        c = count.setdefault(r[0], {'rows': 0, 'development': 0, 'held_out': 0, 'identical_seen': 0, 'reuse_candidates': 0, 'to_generate': 0})
        c['rows'] += 1; c[r[7]] += 1; c['identical_seen'] += bool(r[9]); c['reuse_candidates'] += r[10]; c['to_generate'] += not r[10]
    count['primal']['note'] = 'PRIMAL rows are runs; each run writes the 8 switch combinations (x 8 evaluation rows)'
    out = {
        'version': 'inference-tasks-draft-1', 'created': time.strftime('%Y-%m-%d %H:%M'),
        'status': 'draft; frozen with config/inference_registration_20261109.json and the other three section-14 files after user review',
        'generator': {'script': 'python/scripts/build_inference_tasks.py', 'sha256': sha(__file__)},
        'inputs_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [REACH/'reachability.json', REACH/'overhead_limit.json', MATRIX, V2/'batch_plan.json', SCAN, *M0]},
        'frames': {
            'reach': 'position_m: report frame, Z up, metres; the body faces -Y and its right side is -X at frame 0 (utils/motiogen/task_geometry.py); origin: the method\'s initial pelvis joint projected on the floor. target_body_m: [forward, right, up] in the same frame. The nominal Rajagopal body stands for the pelvis joint with its hip-centre midpoint (convention, not derived; its pelvis origin is 5.6 cm ahead of that midpoint); in HumanML3D standing frames the pelvis joint is 1.2 cm ahead of the hip midpoint, so a method receives each target 1.2 cm further forward relative to its hips than the nominal check assumed (sensitivity: reachability_origin_shift_+0.012.json)',
            'reach_nominal_body': 'risk-zone targets from the nominal Rajagopal body (task_reachability.py): front = shoulder + 0.8 arm forward at the shoulder lateral offset, side90 = shoulder + 0.8 arm outward; heights above the floor',
            'root_goal': 'goal_start / goal_next: native goal coordinates of the method, metres, same frame as the native matrix (cardinal directions 0, 90, 180, 270 deg; added diagonals 45, 135, 225, 315 deg)',
            'ardy': 'direction_xz: native XZ unit vector (forward +Z, body right -X), utils/motiogen/task_geometry.native_direction'},
        'risk_zone_heights_m': reach['conventions_not_derived']['risk_heights_m'],
        'overhead_height': limit['overhead_height'],
        'seeds': SEEDS,
        'split': {
            'dimensions': 'target position, paraphrase, seed (design section 14 item 1)',
            'rule': 'a row is held out if any of its dimensions is held out, except that a row whose input is identical to an already-run input (method, task input, seed, condition) is always development',
            'held_out_seed': HELD_OUT_SEED, 'held_out_paraphrase': HELD_OUT_PARAPHRASE,
            'held_out_positions': 'per method: 4 of the 12 added tasks, interleaved so that both sets cover every zone / direction / distance / speed / family level (methods.<m>.held_out_positions); ARDY and CHOIS have none (methods.<m>.no_held_out_position)',
            'paraphrase_rotation': 'nested Latin square (user 2026-11-09): within each hand the 12 tasks are indexed i = 0..11 (native matrix tasks first, then the risk-zone tasks); seed s in 20-23 uses paraphrase (i + s - 20) mod 4, seed 24 uses paraphrase 4; the same paraphrase index applies to the neutral and the consistent text'},
        'text_table': {'templates_convention_not_derived': TEMPLATES, 'neutral_phrase': NEUTRAL,
                       'paraphrase_0_neutral': 'reproduces the native matrix sentence exactly ("A person reaches forward with the <side> hand.")',
                       'consistent_phrases': {str(t['id']): t['consistent_phrase'] for t in reach_t}, 'rendered': paraphrases},
        'methods': {m: {k: v for k, v in s.items()} for m, s in methods.items()},
        'kimodo': {'tasks': 'reuse of the 416 deck pairs (E4), not extended', 'split': 'deck_recheck', 'source': 'config/e4_cross_method_20261103.json', 'l4_state_of_old_runs': l4.get('kimodo')},
        'counts': count,
        'rows': {'header': header, 'conditions': 'condition = index into methods.<m>.conditions (PRIMAL: null, all 8 in one run)', 'data': rows}}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(count, indent=1))


if __name__ == '__main__':
    main()
