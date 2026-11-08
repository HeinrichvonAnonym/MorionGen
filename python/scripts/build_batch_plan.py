#!/usr/bin/env python
"""Generation plans of the large batch from the frozen task table (config/inference_tasks_20261109.json, freeze-20261109).

Every frozen row becomes one queue job or, when reuse_candidate is true, a link to its identical earlier run (the link is
verified: the run exists, its manifest status is success and its input identity, python/scripts/scan_seen_inputs.py
identity(), equals that of the job the row would get). Held-out rows are written like any other row (analysis plan
split_use.enforcement). Jobs are the native matrix jobs of python/scripts/build_inference_smoke.py job() (paused-clock
templates) with the contract L2 fields added to the scene (task_id, paraphrase_id, text_condition, target_zone; DART,
CLoSD and PRIMAL also the unified kind / prompt / constraints of the contract L2 kinds, which their entry scripts do not read).

Self check (exit 1 on any difference): every selected frozen row is exactly one job or one link; each job's inference_row
equals its frozen row; method, condition, task, paraphrase, seed, text and target read back from the job's scene, command
and config equal the frozen row and task.

  build_batch_plan.py --out DIR [--rehearsal] [--deadline-hours H]
Writes DIR/plan_main.json (every method but CHOIS, devices 0 and 1), DIR/plan_chois.json (device 0, the other card idle,
as CHOIS was timed in the smoke), DIR/links.json and DIR/self_check.json.
--rehearsal: per method one development (task, seed, text level, paraphrase) group with all its conditions (rows of the
group that are reuse candidates become links), plus one more link for each method that has reuse candidates.
"""
import argparse
import copy
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_inference_smoke as smoke  # noqa: E402
from scan_seen_inputs import identity  # noqa: E402

TASKS_PATH = ROOT/'config/inference_tasks_20261109.json'
TEXT_METHODS = ('motionlcm', 'omnicontrol')
ZONE = {'dart': 'path', 'closd': 'path', 'ardy': 'path', 'primal': 'initial_state', 'chois': 'official_window'}


def load_tasks():
    tasks = json.loads(TASKS_PATH.read_text())
    header = tasks['rows']['header']
    rows = [dict(zip(header, r)) for r in tasks['rows']['data']]
    return tasks, header, rows


def row_key(r):
    return (r['method'], r['task'], r['seed'], r['condition'], r['text_level'])


def job_id(r):
    return f"batch_{r['method']}_t{r['task']}_s{r['seed']}_c{r['condition']}_{r['text_level'] or 'na'}"


def target_zone(method, task):
    if method in TEXT_METHODS:
        return task['zone'] if task['tier'] == 'risk' else 'ppt_original'
    return ZONE[method]


def world_fps(method):
    return json.loads((ROOT/f'config/motion_world/{method}.json').read_text())['fps']['value']


def batch_job(tasks, header, r):
    m = r['method']
    spec = tasks['methods'][m]
    task = next(t for t in spec['tasks'] if t['id'] == r['task'])
    j = smoke.job(m, task, r['seed'], spec['conditions'][r['condition']], r['prompt'], {k: r[k] for k in header})
    j['id'] = job_id(r)
    s = {**j['scene'], 'task_id': r['task'], 'paraphrase_id': r['paraphrase'], 'text_condition': r['text_level'],
         'target_zone': target_zone(m, task)}
    if m in ('dart', 'closd'):
        frame = 'native goal coordinates of the method (inference tasks frames.root_goal); z not constrained'
        s.update(kind='root_goal_change', prompt=None,
                 constraints=[{'time_seconds': 0, 'joint': 'pelvis', 'position_m': [*task['goal_start'], None], 'frame': frame},
                              {'time_seconds': task['update_seconds'], 'joint': 'pelvis', 'position_m': [*task['goal_next'], None], 'frame': frame}])
    elif m == 'primal':
        s.update(kind='continuation', prompt=None, constraints=[], duration_seconds=task['frames']/world_fps('primal'),
                 initial_state={'history': task['initial_history'], 'family': task['family'], 'seed': task['initial_seed']})
    j['scene'] = s
    return j


def arg(command, flag, n=1):
    i = command.index(flag)
    return command[i+1:i+1+n] if n > 1 else command[i+1]


def readback(job):
    """Method, condition, task input, text and seed as the job itself carries them."""
    m, s, c = job['method'], job['scene'], job['command']
    out = {'method': m, 'seed': job['seed'], 'task': job['task_id'], 'scene_task_id': s.get('task_id'),
           'paraphrase': s.get('paraphrase_id'), 'text_level': s.get('text_condition'),
           'clock': s.get('clock', arg(c, '--clock') if '--clock' in c else 'paused')}
    if '--seed' in c:
        out['command_seed'] = int(arg(c, '--seed'))
    if m in ('motionlcm', 'omnicontrol', 'ardy', 'chois'):
        out['condition'] = job['config']['_intervention']
    elif m == 'dart':
        out['condition'] = {'goal_update': int(arg(c, '--goal-feedback'))}
    elif m == 'closd':
        out['condition'] = {'execution_context': int(arg(c, '--context-source') == 'simulation')}
    if m in TEXT_METHODS:
        k = s['constraints'][0]
        out.update(prompt=s['prompt'], target={'joint': k['joint'], 'position_m': k['position_m'], 'time_seconds': k['time_seconds'],
                                               'duration_seconds': s['duration_seconds']})
    elif m in ('dart', 'closd'):
        out['target'] = {'goal_start': s['goal_start'], 'goal_next': s['goal_next'],
                         'command_goal_start': [float(x) for x in arg(c, '--goal-start', 2)],
                         'command_goal_next': [float(x) for x in arg(c, '--goal-next', 2)]}
    elif m == 'ardy':
        out.update(prompt=s['prompt'], target={'native_root_path': s['native_root_path'], 'updates': s['updates']})
    elif m == 'primal':
        out['target'] = {'initial_history': s['initial_history'], 'initial_seed': s['initial_seed'], 'family': s['family'],
                         'command_initial_history': int(arg(c, '--initial-history')), 'command_initial_seed': int(arg(c, '--initial-seed'))}
    elif m == 'chois':
        out['target'] = {'native_window_index': s['native_window_index'], 'native_sequence': s['native_sequence'],
                         'duration_seconds': s['duration_seconds']}
    return out


def expected(tasks, r):
    m = r['method']
    spec = tasks['methods'][m]
    t = next(x for x in spec['tasks'] if x['id'] == r['task'])
    e = {'method': m, 'seed': r['seed'], 'task': r['task'], 'scene_task_id': r['task'], 'paraphrase': r['paraphrase'],
         'text_level': r['text_level'], 'clock': 'paused'}
    if m in ('dart', 'closd', 'primal'):
        e['command_seed'] = r['seed']
    if m != 'primal':
        e['condition'] = spec['conditions'][r['condition']]
    if m in TEXT_METHODS:
        e.update(prompt=r['prompt'], target={'joint': t['joint'], 'position_m': t['position_m'], 'time_seconds': t['time_seconds'],
                                             'duration_seconds': t['duration_seconds']})
    elif m in ('dart', 'closd'):
        e['target'] = {'goal_start': t['goal_start'], 'goal_next': t['goal_next'],
                       'command_goal_start': [float(x) for x in t['goal_start']], 'command_goal_next': [float(x) for x in t['goal_next']]}
    elif m == 'ardy':
        e.update(prompt=t['prompt'], target={'native_root_path': t['native_root_path'], 'updates': t['updates']})
    elif m == 'primal':
        e['target'] = {'initial_history': t['initial_history'], 'initial_seed': t['initial_seed'], 'family': t['family'],
                       'command_initial_history': t['initial_history'], 'command_initial_seed': t['initial_seed']}
    elif m == 'chois':
        e['target'] = {'native_window_index': t['native_window_index'], 'native_sequence': t['native_sequence'],
                       'duration_seconds': t['duration_seconds']}
    return e


def job_identity(job):
    m = job['method']
    return identity(m, job['scene'], job['seed'], job.get('config', {}).get('_intervention'), job['command'])


def verify_link(job, run):
    d = ROOT/run
    try:
        man = json.loads((d/'manifest.json').read_text())
        cfg = json.loads((d/'config.json').read_text()) if (d/'config.json').is_file() else {}
    except (OSError, ValueError) as e:
        return f'{run}: {type(e).__name__}'
    if man.get('status') != 'success':
        return f'{run}: status {man.get("status")}'
    old = identity(job['method'], man.get('scenario') or {}, man.get('seed'), cfg.get('_intervention') or {}, man.get('command') or [])
    new = job_identity(job)
    return None if old is not None and old == new else f'{run}: input identity differs'


def rehearsal_rows(rows):
    """Per method one development (task, seed, text level, paraphrase) group with all conditions, preferring added /
    risk-zone tasks and groups with the most rows to generate; then one extra reuse row for methods that have them."""
    groups = {}
    for r in rows:
        groups.setdefault((r['method'], r['task'], r['seed'], r['text_level'], r['paraphrase']), []).append(r)
    tasks, _, _ = load_tasks()
    out = []
    for m, spec in tasks['methods'].items():
        n = len(spec['conditions']) if m != 'primal' else 1
        tier = {t['id']: t['tier'] for t in spec['tasks']}
        cand = [g for k, g in groups.items() if k[0] == m and len(g) == n and all(r['split'] == 'development' for r in g)]
        cand.sort(key=lambda g: (-sum(not r['reuse_candidate'] for r in g), tier[g[0]['task']] == 'legacy', g[0]['task'], g[0]['seed']))
        chosen = cand[0]
        out += chosen
        if any(r['method'] == m and r['reuse_candidate'] for r in rows):
            out.append(next(r for r in rows if r['method'] == m and r['reuse_candidate'] and r not in chosen))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--rehearsal', action='store_true')
    p.add_argument('--deadline-hours', type=float, default=96.)
    args = p.parse_args()
    tasks, header, rows = load_tasks()
    keys = [row_key(r) for r in rows]
    problems = [] if len(keys) == len(set(keys)) else ['frozen rows are not unique on (method, task, seed, condition, text_level)']
    selected = rehearsal_rows(rows) if args.rehearsal else rows
    jobs, links = [], []
    for r in selected:
        j = batch_job(tasks, header, r)
        if j['inference_row'] != r:
            problems.append(f'{j["id"]}: inference_row differs from the frozen row')
        got, want = readback(j), expected(tasks, r)
        if got != want:
            problems.append(f'{j["id"]}: ' + ', '.join(k for k in sorted(set(got) | set(want)) if got.get(k) != want.get(k)))
        if r['reuse_candidate']:
            bad = verify_link(j, r['identical_seen_run'])
            if bad:
                problems.append(f'{j["id"]} link {bad}')
            links.append({'row': r, 'run': r['identical_seen_run'], 'job_identity_sha256': hashlib.sha256(job_identity(j).encode()).hexdigest()})
        else:
            jobs.append(j)
    covered = [row_key(j['inference_row']) for j in jobs] + [row_key(x['row']) for x in links]
    want = [row_key(r) for r in selected]
    if sorted(covered) != sorted(want) or len(covered) != len(set(covered)):
        problems.append('rows and jobs + links do not correspond one to one')
    if not args.rehearsal and sorted(covered) != sorted(keys):
        problems.append('not every frozen row is covered')
    counts = {m: {'rows': sum(r['method'] == m for r in selected), 'jobs': sum(j['method'] == m for j in jobs),
                  'links': sum(x['row']['method'] == m for x in links),
                  'held_out_jobs': sum(j['method'] == m and j['inference_row']['split'] == 'held_out' for j in jobs)}
              for m in tasks['methods']}
    for m, c in counts.items():
        frozen = tasks['counts'][m]
        if not args.rehearsal and (c['links'] != frozen['reuse_candidates'] or c['jobs'] != frozen['rows'] - frozen['reuse_candidates']):
            problems.append(f'{m}: jobs / links {c["jobs"]} / {c["links"]} differ from the frozen counts')
    now = datetime.datetime.now(datetime.timezone.utc)
    common = {'source': str(TASKS_PATH.relative_to(ROOT)),
              'source_sha256': hashlib.sha256(TASKS_PATH.read_bytes()).hexdigest(),
              'builder': str(Path(__file__).resolve().relative_to(ROOT)), 'rehearsal': args.rehearsal,
              'deadline': (now + datetime.timedelta(hours=args.deadline_hours)).isoformat(), 'report_reserve_seconds': 0,
              'templates': {m: str(t.relative_to(ROOT)) for m, t in smoke.TEMPLATES.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'created': now.isoformat(), 'rehearsal': args.rehearsal, 'counts': counts, 'problems': problems,
              'checked': ['frozen rows unique', 'each selected row exactly one job or link', 'inference_row equals the frozen row',
                          'method, condition, task, paraphrase, seed, text, target and clock read back from scene / command / '
                          'config equal the frozen row and task', 'links: run exists, status success, input identity equal']}
    (args.out/'self_check.json').write_text(json.dumps(report, indent=1, ensure_ascii=False) + '\n')
    if problems:
        print(json.dumps(report, indent=1, ensure_ascii=False)[:4000])
        raise SystemExit(1)
    main_jobs = [j for j in jobs if j['method'] != 'chois']
    chois = [j for j in jobs if j['method'] == 'chois']
    for name, js, devices in (('plan_main.json', main_jobs, [0, 1]), ('plan_chois.json', chois, [0])):
        plan = {**copy.deepcopy(common), 'devices': devices, 'jobs': js,
                'scope': 'large batch generation (design section 14), ' + ('rehearsal' if args.rehearsal else 'formal')}
        (args.out/name).write_text(json.dumps(plan, indent=1, ensure_ascii=False) + '\n')
    (args.out/'links.json').write_text(json.dumps({**common, 'links': links}, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({'counts': counts, 'problems': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
