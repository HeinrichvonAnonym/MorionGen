#!/usr/bin/env python
"""Generation smoke and timing of the section-14 inference registration (premise check 3): one development row per method.

Rows come from config/inference_tasks_20261109.json and must be development rows; the job of each row is the native
matrix job (CHOIS: 20261024_native_followups) of the same method and condition with the task input replaced. Only receipts, intervention calls
and timings of these runs are inspected, never an outcome metric.
Templates are paused-clock jobs only (the ARDY matrix also holds realtime jobs with the same command).
Writes develop_logs/test_runs/20261109_inference_smoke/plan.json (--round2: plan_round2.json, a second development row per
method for warm timing; --ardy-paused: plan_ardy_paused.json, the two ARDY rows again, their first runs having taken a
realtime template); run it with python/scripts/experiments.py run.
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'develop_logs/test_runs'
OUT = RUNS/'20261109_inference_smoke'
TEMPLATES = {'motionlcm': RUNS/'20261024_control_matrix/plan.json', 'omnicontrol': RUNS/'20261024_control_matrix/plan.json',
             'dart': RUNS/'20261024_dart_matrix/plan.json', 'closd': RUNS/'20261024_closd_matrix/plan.json',
             'ardy': RUNS/'20261024_ardy_matrix/plan.json', 'primal': RUNS/'20261024_primal_matrix/plan.json',
             'chois': RUNS/'20261024_native_followups/plan.json'}
# (method, task, seed, condition, text_level)
SMOKE = [('motionlcm', 16, 20, {'control_residual': 1}, 'consistent'),
         ('omnicontrol', 13, 20, {'realism': 1, 'spatial': 1}, 'neutral'),
         ('dart', 13, 20, {'goal_update': 1}, None),
         ('closd', 13, 20, {'execution_context': 1}, None),
         ('ardy', 13, 20, {'history_frames': 160, 'correction': 1}, None),
         ('primal', 12, 20, {'ground_snap': 0, 'reprojection': 0, 'inertialization': 0}, None),
         ('chois', 13, 23, {'guidance': 1}, None)]
# second timed row per method (warm page cache), development rows again
SMOKE_ROUND2 = [('motionlcm', 21, 21, {'control_residual': 1}, 'neutral'),
                ('omnicontrol', 14, 21, {'realism': 1, 'spatial': 1}, 'consistent'),
                ('dart', 14, 21, {'goal_update': 0}, None),
                ('closd', 14, 21, {'execution_context': 0}, None),
                ('ardy', 14, 21, {'history_frames': 32, 'correction': 0}, None),
                ('primal', 18, 21, {'ground_snap': 0, 'reprojection': 0, 'inertialization': 0}, None),
                ('chois', 14, 23, {'guidance': 0}, None)]


def arg(command, flag, values):
    i = command.index(flag)
    return command[:i+1]+[str(v) for v in values]+command[i+1+len(values):]


def template(method, condition):
    jobs = [j for j in json.loads(TEMPLATES[method].read_text())['jobs'] if j['scene'].get('clock', 'paused') == 'paused']
    if method in ('motionlcm', 'omnicontrol', 'ardy', 'chois'):
        return next(j for j in jobs if j['method'] == method and j['config']['_intervention'] == condition)
    if method == 'dart':
        return next(j for j in jobs if j['method'] == 'dart'
                    and j['command'][j['command'].index('--goal-feedback')+1] == str(condition['goal_update']))
    if method == 'closd':
        source = 'simulation' if condition['execution_context'] else 'prediction'
        return next(j for j in jobs if j['command'][j['command'].index('--context-source')+1] == source)
    return next(j for j in jobs if j['method'] == 'primal' and j['scene'].get('family') == 1)


def job(method, task, seed, condition, prompt, run_tag):
    j = copy.deepcopy(template(method, condition))
    j['id'] = f'smoke_{method}_task{task["id"]}_seed{seed}'
    j['seed'] = seed; j['task_id'] = task['id']
    c = j['command']
    if '--seed' in c:
        c = arg(c, '--seed', [seed])
    if method in ('motionlcm', 'omnicontrol'):
        j['scene'] = {**j['scene'], 'prompt': prompt,
                      'constraints': [{'time_seconds': task['time_seconds'], 'joint': task['joint'], 'position_m': task['position_m']}]}
    elif method in ('dart', 'closd'):
        j['scene'] = {**j['scene'], 'goal_start': task['goal_start'], 'goal_next': task['goal_next']}
        c = arg(arg(c, '--goal-start', task['goal_start']), '--goal-next', task['goal_next'])
    elif method == 'ardy':
        j['scene'] = {**j['scene'], 'native_root_path': task['native_root_path'], 'prompt': task['prompt'], 'updates': task['updates']}
    elif method == 'primal':
        j['scene'] = {**j['scene'], 'initial_history': task['initial_history'], 'family': task['family'], 'initial_seed': task['initial_seed']}
        c = arg(arg(c, '--initial-history', [task['initial_history']]), '--initial-seed', [task['initial_seed']])
    elif method == 'chois':
        j['scene'] = {**j['scene'], 'native_window_index': task['native_window_index'], 'native_sequence': task['native_sequence']}
    j['command'] = c
    j['inference_row'] = run_tag
    return j


def main():
    second = '--round2' in sys.argv
    ardy_paused = '--ardy-paused' in sys.argv
    tasks = json.loads((ROOT/'config/inference_tasks_20261109.json').read_text())
    header = tasks['rows']['header']
    rows = [dict(zip(header, r)) for r in tasks['rows']['data']]
    jobs = []
    smoke = [s for s in SMOKE+SMOKE_ROUND2 if s[0] == 'ardy'] if ardy_paused else SMOKE_ROUND2 if second else SMOKE
    for method, tid, seed, condition, level in smoke:
        cond_index = tasks['methods'][method]['conditions'].index(condition)
        row = next(r for r in rows if r['method'] == method and r['task'] == tid and r['seed'] == seed
                   and r['condition'] == cond_index and r['text_level'] == level)
        if row['split'] != 'development':
            raise SystemExit(f'{method} task {tid} seed {seed}: {row["split"]} row, smoke must use development rows')
        task = next(t for t in tasks['methods'][method]['tasks'] if t['id'] == tid)
        jobs.append(job(method, task, seed, condition, row['prompt'], {k: row[k] for k in header}))
    plan = {'stage': 'feasibility_and_throughput', 'devices': [0, 1],
            'scope': 'section-14 inference registration premise check 3: one development row per method; receipts, intervention calls and timings only',
            'source': 'config/inference_tasks_20261109.json', 'templates': {m: str(p.relative_to(ROOT)) for m, p in TEMPLATES.items()},
            'jobs': jobs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/('plan_ardy_paused.json' if ardy_paused else 'plan_round2.json' if second else 'plan.json')).write_text(json.dumps(plan, indent=1))
    for j in jobs:
        print(j['id'], json.dumps(j['scene'])[:200], [x for x in j['command'] if '/' not in x])


if __name__ == '__main__':
    main()
