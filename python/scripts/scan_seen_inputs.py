#!/usr/bin/env python
"""Held-out integrity of config/inference_tasks_20261109.json against every earlier run (read only, no outcome read).

Scans every develop_logs/test_runs/**/<stamp>_<method>_demo_<id> run (manifest scenario, seed, status, command; config
_intervention) and reports, per method:
  - held-out target positions that appear in any earlier run (any seed, any condition)
  - held-out rows whose input equals an earlier run (same task input, seed, condition, text; script entry points: the same
    command arguments without paths)
Runs created by the section-14 smoke (develop_logs/test_runs/20261109_inference_smoke plans) are development rows by
construction and are listed separately. Output: develop_logs/test_runs/20261109_inference_premises/seen_inputs.json
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'develop_logs/test_runs'
OUT = RUNS/'20261109_inference_premises/seen_inputs.json'
NAME = re.compile(r'^\d{8}_\d{6}_([a-z]+)_demo_[0-9a-f]+$')
ARDY_DEFAULTS = {'waypoints': 'sparse', 'target_mode': 'path', 'offset_m': 0., 'root_margin_m': .04, 'constraint_cfg': 2.,
                 'boundary': 'none', 'boundary_writeback': True, 'renoise_fraction': 0.}


def r6(v):
    return [round(float(x), 4)+0. for x in v]


def arg(command, flag, n=1, default=None):
    if flag not in command:
        return default
    i = command.index(flag)
    return command[i+1:i+1+n] if n > 1 else command[i+1]


def script_args(command):
    out = []
    skip = False
    for x in command[2:]:
        if skip:
            skip = False; continue
        if x in ('--run-dir', '--output-dir'):
            skip = True; continue
        out.append(x)
    return out


def position(method, scene, command):
    if method in ('motionlcm', 'omnicontrol'):
        return [json.dumps([c['joint'], r6(c['position_m'])]) for c in scene.get('constraints', [])]
    if method in ('dart', 'closd'):
        start = scene.get('goal_start') or arg(command, '--goal-start', 2)
        nxt = scene.get('goal_next') or arg(command, '--goal-next', 2)
        return [json.dumps([r6(start), r6(nxt)])] if start and nxt else []
    if method == 'primal':
        h = scene.get('initial_history', arg(command, '--initial-history'))
        s = scene.get('initial_seed', arg(command, '--initial-seed', default=9000))
        return [json.dumps([int(h), int(s)])] if h is not None else []
    if method == 'chois':
        return [json.dumps(scene['native_window_index'])] if 'native_window_index' in scene else []
    if method == 'ardy':
        p = scene.get('native_root_path')
        return [json.dumps([r6(p['direction_xz']), p['speed_before_m_s'], p['speed_after_m_s']])] if p else []
    return []


def ardy_condition(c):
    if any(k not in ('history_frames', 'correction', 'text') and ARDY_DEFAULTS.get(k, object()) != v for k, v in c.items()):
        return None
    return {'history_frames': int(c['history_frames']), 'correction': int(c['correction'])}


def identity(method, scene, seed, condition, command):
    if method in ('dart', 'closd', 'primal'):
        return json.dumps([method, script_args(command)])
    if method in ('motionlcm', 'omnicontrol'):
        x = [scene.get('prompt'), [[c['joint'], r6(c['position_m']), c['time_seconds']] for c in scene.get('constraints', [])],
             scene.get('duration_seconds')]
    elif method == 'chois':
        x = [scene.get('native_window_index'), scene.get('native_sequence')]
    else:
        condition = ardy_condition(condition)
        if condition is None or 'native_root_path' not in scene:
            return None
        p = {k: v for k, v in scene['native_root_path'].items() if not (k == 'switch_seconds' and v == 5.)}
        p['direction_xz'] = r6(p['direction_xz'])
        x = [scene['prompt'], [[float(u['time_seconds']), u['prompt']] for u in scene['updates']], p, scene.get('clock', 'paused')]
    return json.dumps([method, x, seed, condition], sort_keys=True)


def main():
    tasks = json.loads((ROOT/'config/inference_tasks_20261109.json').read_text())
    smoke_ids = set()
    for plan in (RUNS/'20261109_inference_smoke').glob('plan*.json'):
        smoke_ids |= {j['id'] for j in json.loads(plan.read_text())['jobs']}
    smoke_runs = set()
    for state in (RUNS/'20261109_inference_smoke').glob('queue*/state.json'):
        smoke_runs |= {Path(v['run_dir']).name for v in json.loads(state.read_text())['jobs'].values() if v.get('run_dir')}
    seen_pos, seen_id = {}, {}
    scanned = 0
    for d in RUNS.rglob('*_demo_*'):
        mt = NAME.match(d.name)
        if not mt or not (d/'manifest.json').is_file() or d.name in smoke_runs:
            continue
        method = mt.group(1)
        try:
            man = json.loads((d/'manifest.json').read_text())
            cfg = json.loads((d/'config.json').read_text()) if (d/'config.json').is_file() else {}
        except (json.JSONDecodeError, OSError):
            continue
        scanned += 1
        scene = man.get('scenario') or {}
        command = man.get('command') or []
        cond = cfg.get('_intervention') or {}
        for p in position(method, scene, command):
            seen_pos.setdefault((method, p), []).append(str(d.relative_to(RUNS)))
        if man.get('status') == 'success':
            key = identity(method, scene, man.get('seed'), cond, command)
            if key:
                seen_id.setdefault(key, str(d.relative_to(RUNS)))
    smoke_plan = {}
    for plan in (RUNS/'20261109_inference_smoke').glob('plan*.json'):
        for j in json.loads(plan.read_text())['jobs']:
            smoke_plan[j['id']] = j
    header = tasks['rows']['header']
    rows = [dict(zip(header, r)) for r in tasks['rows']['data']]
    report = {'scanned_runs': scanned, 'excluded_smoke_runs': len(smoke_runs), 'methods': {}}
    for m, spec in tasks['methods'].items():
        T = {t['id']: t for t in spec['tasks']}
        held_pos = []
        for tid in spec['held_out_positions']:
            t = T[tid]
            if m in ('motionlcm', 'omnicontrol'):
                keys = [json.dumps([t['joint'], r6(t['position_m'])])]
            elif m in ('dart', 'closd'):
                keys = [json.dumps([r6(t['goal_start']), r6(t['goal_next'])])]
            else:
                keys = [json.dumps([t['initial_history'], t['initial_seed']])]
            hits = sorted({r for k in keys for r in seen_pos.get((m, k), [])})
            held_pos.append({'task': tid, 'earlier_runs': len(hits), 'examples': hits[:3]})
        # identity of the held-out rows: rebuild each row's input as an earlier run would have recorded it
        template = {j['method']: j for j in smoke_plan.values()}
        held_ident, dev_hits = [], 0
        for r in rows:
            if r['method'] != m:
                continue
            t = T[r['task']]; cond = spec['conditions'][r['condition']] if m != 'primal' else None
            if m in ('motionlcm', 'omnicontrol'):
                scene = {'prompt': r['prompt'], 'constraints': [{'joint': t['joint'], 'position_m': t['position_m'], 'time_seconds': t['time_seconds']}],
                         'duration_seconds': t['duration_seconds']}
                key = identity(m, scene, r['seed'], cond, [])
            elif m == 'chois':
                key = identity(m, {'native_window_index': t['native_window_index'], 'native_sequence': t['native_sequence']}, r['seed'], cond, [])
            elif m == 'ardy':
                scene = {'prompt': t['prompt'], 'updates': t['updates'], 'native_root_path': t['native_root_path'], 'clock': 'paused'}
                key = identity(m, scene, r['seed'], cond, [])
            else:
                c = list(template[m]['command'])
                if '--seed' in c:
                    c[c.index('--seed')+1] = str(r['seed'])
                if m in ('dart', 'closd'):
                    i = c.index('--goal-start'); c[i+1:i+3] = [str(v) for v in t['goal_start']]
                    i = c.index('--goal-next'); c[i+1:i+3] = [str(v) for v in t['goal_next']]
                    if m == 'dart':
                        c[c.index('--goal-feedback')+1] = str(cond['goal_update'])
                    else:
                        c[c.index('--context-source')+1] = 'simulation' if cond['execution_context'] else 'prediction'
                else:
                    c[c.index('--initial-history')+1] = str(t['initial_history']); c[c.index('--initial-seed')+1] = str(t['initial_seed'])
                key = identity(m, {}, r['seed'], None, c)
            if key and key in seen_id and r['split'] == 'development':
                dev_hits += 1
            elif key and key in seen_id:
                held_ident.append({'task': r['task'], 'seed': r['seed'], 'condition': cond, 'earlier_run': seen_id[key]})
        dev_pos = sum(bool(seen_pos.get((m, k))) for t in spec['tasks'] if t['id'] not in spec['held_out_positions']
                      for k in ([json.dumps([t['joint'], r6(t['position_m'])])] if m in ('motionlcm', 'omnicontrol') else
                                [json.dumps([r6(t['goal_start']), r6(t['goal_next'])])] if m in ('dart', 'closd') else
                                [json.dumps([t['initial_history'], t['initial_seed']])] if m == 'primal' else
                                [json.dumps(t['native_window_index'])] if m == 'chois' else
                                [json.dumps([r6(t['native_root_path']['direction_xz']), t['native_root_path']['speed_before_m_s'],
                                             t['native_root_path']['speed_after_m_s']])]))
        report['methods'][m] = {'held_out_positions_seen_earlier': held_pos,
                                'positive_control': {'development_rows_identical_to_earlier_runs': dev_hits,
                                                     'generator_identical_seen': sum(r['method'] == m and r['identical_seen_run'] is not None for r in rows),
                                                     'development_positions_seen_earlier': dev_pos},
                                'held_out_rows': sum(r['method'] == m and r['split'] == 'held_out' for r in rows),
                                'held_out_rows_identical_to_earlier_runs': held_ident}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print('scanned', scanned, 'excluded smoke', len(smoke_runs))
    for m, v in report['methods'].items():
        print(m, 'held-out rows', v['held_out_rows'], 'identical to earlier', len(v['held_out_rows_identical_to_earlier_runs']),
              v['held_out_rows_identical_to_earlier_runs'][:2], v['positive_control'])
        for p in v['held_out_positions_seen_earlier']:
            if p['earlier_runs']:
                print('   held-out position seen', p)


if __name__ == '__main__':
    main()
