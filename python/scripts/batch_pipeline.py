#!/usr/bin/env python
"""Thin orchestration of the large batch (plan section 4; evaluation protocol execution_order: the whole generation first,
evalkit only after it).

  batch_pipeline.py start  --batch NAME [--rehearsal | --reuse-batch NAME] [--workers 10] [--kimodo all|N] [--deadline-hours 96]
  batch_pipeline.py resume --batch NAME
  batch_pipeline.py status --batch NAME

The batch directory develop_logs/test_runs/NAME is a link to <data_root>/batch_runs/NAME (runs, queues, evalkit rows and
logs all on the data disk; run paths stay inside the repository, as the analysis reads them). start: freeze check
(build_freeze_manifest.py --check), plans with their self check (build_batch_plan.py, run_root NAME/runs), disk and memory
check, then the runner detaches (nohup setsid nice -n 19, PYTHONNOUSERSITE=1). Stages in order, state in NAME/stages.json,
finished stages skipped on resume, each stage resumable by itself (queue resume, evalkit batch key, contract integrity):

  generate_main    plans/plan_main.json, every method but CHOIS, devices 0 and 1      -> queue_main
  generate_chois   plans/plan_chois.json, device 0                                    -> queue_chois
  generate_retry   jobs whose last status is resource_gate, the same job and seed, device 0 (plans/retry_resource_gate.json)
                   -> queue_retry; failures after the model ran stay recorded and are never rerun or replaced
  shadow           rehearsal: plans/shadow.json -> shadow_queue; shadow_compare.json (identical motion sha256 count)
  contract         complete_run_contract.py complete, then scan (exit 1 on a gap)
  manifest         build_evalkit_manifest.py (development and held-out manifests apart)
  seed_evalkit     --reuse-batch: rehearsal evalkit row directories copied in for rows of equal id and input sha256;
                   the evalkit batch key then skips them or recomputes them
  eval_method      evalkit_batch.py run, method floor, frozen offset reference          -> evalkit/development/method
  eval_self        manifest-self-floor, run                                            -> evalkit/development/self
  eval_lumbar      manifest-lumbar-sample on the self-floor manifest (1 per stratum, deck_recheck and held_out deferred),
                   run with upper added (evalkit ADDABLE: no refit)
  neck_rotation    evalkit_batch.py neck-rotation on both arms
  analysis         analyze_batch.py (rehearsal: --structure-only, plus the held-out refusal probe)

Held-out rows are not evaluated here: evalkit_batch.py run writes and prints the median offset of its manifest
(_batch_median), a held-out summary.
status prints counts and elapsed times only (queue states, presence of evalkit.json / error.json), never a metric.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.config import data_root, load_config  # noqa: E402

PY = sys.executable
OSIM_PY = os.environ.get('OSIM_PY', str(Path.home()/'.local/share/opensim-api/bin/python'))
OFFSET_REFERENCE = 'config/evalkit_offset_reference_20261108.json'
DIMS = 'residual,ergo,muscle,kinematic'
DEFER = 'deck_recheck,held_out'
MIN_FREE_GIB = {'data': 100, 'system': 20, 'memory': 6}
GENERATION = ('generate_main', 'generate_chois', 'generate_retry')
QUEUES = {'generate_main': ('plan_main.json', 'queue_main'), 'generate_chois': ('plan_chois.json', 'queue_chois'),
          'generate_retry': ('retry_resource_gate.json', 'queue_retry'), 'shadow': ('shadow.json', 'shadow_queue')}
EVAL_DIRS = {'eval_method': 'method', 'eval_self': 'self', 'eval_lumbar': 'self', 'neck_rotation': 'self'}


def batch_dir(name):
    return ROOT/'develop_logs/test_runs'/name


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec='seconds')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p, default=None):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else default


def write(p, obj):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + '\n')
    tmp.replace(p)


def stage_names(settings):
    names = list(GENERATION)
    if settings['rehearsal']:
        names.append('shadow')
    names += ['contract', 'manifest']
    if settings['reuse_batch']:
        names.append('seed_evalkit')
    return names + ['eval_method', 'eval_self', 'eval_lumbar', 'neck_rotation', 'analysis']


def resources(d):
    data, system = shutil.disk_usage(d.resolve()), shutil.disk_usage(ROOT)
    mem = {k: int(v.split()[0]) for k, v in (l.split(':') for l in Path('/proc/meminfo').read_text().splitlines())}
    out = {'data_free_gib': round(data.free/2**30, 1), 'system_free_gib': round(system.free/2**30, 1),
           'memory_available_gib': round(mem['MemAvailable']/2**20, 1)}
    short = [k for k, key in (('data', 'data_free_gib'), ('system', 'system_free_gib'), ('memory', 'memory_available_gib'))
             if out[key] < MIN_FREE_GIB[k]]
    return out, short


def call(cmd, log, **kw):
    with open(log, 'a') as f:
        f.write(f'\n[{now()}] {" ".join(map(str, cmd))}\n'); f.flush()
        return subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kw).returncode


def freeze_check(log):
    return call([PY, 'python/scripts/build_freeze_manifest.py', '--check'], log)


# ---------------------------------------------------------------- stages

def queue_stage(d, name, log):
    plan, qdir = QUEUES[name]
    plan, qdir = d/'plans'/plan, d/qdir
    if name == 'generate_retry' and not plan.exists():
        gated = {}
        for q in ('queue_main', 'queue_chois'):
            for job, rec in read(d/q/'state.json', {'jobs': {}})['jobs'].items():
                if rec.get('status') == 'resource_gate':
                    gated[job] = q
        base = read(d/'plans/plan_main.json')
        jobs = [j for f in ('plan_main.json', 'plan_chois.json') for j in read(d/'plans'/f)['jobs'] if j['id'] in gated]
        write(plan, {**{k: v for k, v in base.items() if k not in ('jobs', 'devices', 'scope')}, 'devices': [0], 'jobs': jobs,
                     'scope': 'resource_gate rerun: the same job and seed on one card', 'created': now()})
    if not read(plan)['jobs']:
        return 0, {'jobs': 0}
    action = 'resume' if (qdir/'state.json').exists() else 'run'
    rc = call([PY, 'python/scripts/experiments.py', action, '--plan', plan, '--directory', qdir], log)
    state = read(qdir/'state.json', {})
    if rc != 0 or state.get('status') != 'finished':
        return rc or 1, {'queue_status': state.get('status')}
    if name == 'shadow':
        write(d/'shadow_compare.json', shadow_compare(d))
    return 0, queue_counts(qdir)


def motion_files(run, method):
    return sorted((run/'factorial').glob('*.npz')) if method == 'primal' else [run/'motion.npz']


def shadow_compare(d):
    plan, state = read(d/'plans/shadow.json'), read(d/'shadow_queue/state.json')
    out = {'jobs': len(plan['jobs']), 'succeeded': 0, 'identical': 0, 'per_job': {}}
    for j in plan['jobs']:
        rec = state['jobs'][j['id']]
        if rec.get('status') != 'success':
            out['per_job'][j['id']] = rec.get('status'); continue
        out['succeeded'] += 1
        new, old = Path(rec['run_dir']), ROOT/j['shadow_of']
        a, b = motion_files(new, j['method']), motion_files(old, j['method'])
        same = [p.name for p in a] == [p.name for p in b] and all(sha(x) == sha(y) for x, y in zip(a, b))
        out['identical'] += same
        out['per_job'][j['id']] = 'identical' if same else 'differs'
    return out


def seed_evalkit(d, settings, log):
    src = batch_dir(settings['reuse_batch'])/'evalkit/development'
    dst = d/'evalkit/development'
    old = {r['id']: r['input_sha256'] for r in read(src.parent/'manifest_development.json')['rows']}
    copied = {'method': 0, 'self': 0}
    for r in read(d/'evalkit/manifest_development.json')['rows']:
        if old.get(r['id']) != r['input_sha256']:
            continue
        for arm, rid in (('method', r['id']), ('self', r['id'] + '__selffloor')):
            if (src/arm/rid/'evalkit.json').exists() and not (dst/arm/rid).exists():
                (dst/arm).mkdir(parents=True, exist_ok=True)
                shutil.copytree(src/arm/rid, dst/arm/rid)
                copied[arm] += 1
    with open(log, 'a') as f:
        f.write(f'[{now()}] copied {json.dumps(copied)} from {src}\n')
    return 0, copied


def eval_counts(manifest, rows_dir, upper=False):
    rows = read(manifest, {'rows': []})['rows']
    if upper:
        rows = [r for r in rows if r.get('upper')]
    done = err = 0
    for r in rows:
        e = Path(rows_dir)/r['id']
        if (e/'error.json').exists():
            err += 1
        elif (e/'evalkit.json').exists() and (not upper or 'upper' in read(e/'evalkit.json').get('batch', {}).get('dims', [])):
            done += 1
    return {'rows': len(rows), 'evaluated': done, 'errors': err}


def eval_stage(d, name, settings, log):
    e = d/'evalkit/development'
    e.mkdir(parents=True, exist_ok=True)
    m, w = d/'evalkit/manifest_development.json', str(settings['workers'])
    B = [OSIM_PY, 'python/scripts/evalkit_batch.py']
    if name == 'eval_method':
        rc = call(B + ['run', '--manifest', m, '--out', e/'method', '--dims', DIMS, '--workers', w,
                       '--offset-reference', OFFSET_REFERENCE], log)
        return rc, eval_counts(m, e/'method')
    if name == 'eval_self':
        rc = call(B + ['manifest-self-floor', '--manifest', m, '--rows', e/'method', '--out', e/'self_manifest.json'], log)
        rc = rc or call(B + ['run', '--manifest', e/'self_manifest.json', '--out', e/'self', '--dims', DIMS, '--workers', w], log)
        return rc, eval_counts(e/'self_manifest.json', e/'self')
    if name == 'eval_lumbar':
        rc = call(B + ['manifest-lumbar-sample', '--manifest', e/'self_manifest.json', '--out', e/'lumbar_manifest.json',
                       '--per-cell', '1', '--defer-split', DEFER], log)
        rc = rc or call(B + ['run', '--manifest', e/'lumbar_manifest.json', '--out', e/'self', '--dims', DIMS + ',upper',
                             '--workers', w], log)
        return rc, eval_counts(e/'lumbar_manifest.json', e/'self', upper=True)
    rc = call(B + ['neck-rotation', '--manifest', m, '--rows', e/'method', '--workers', w], log)
    rc = rc or call(B + ['neck-rotation', '--manifest', e/'lumbar_manifest.json', '--rows', e/'self', '--workers', w], log)
    return rc, neck_counts(d)


def neck_counts(d):
    e, out = d/'evalkit/development', {}
    for arm, manifest in (('method', d/'evalkit/manifest_development.json'), ('self', d/'evalkit/development/lumbar_manifest.json')):
        c = {}
        for r in read(manifest, {'rows': []})['rows']:
            p = e/arm/r['id']/'evalkit.json'
            if p.exists():
                s = read(p).get('neck_rotation', {}).get('status', 'not run')
                c[s] = c.get(s, 0) + 1
        out[arm] = c
    return out


def held_out_probe(d, log):
    """analyze_batch.py must refuse a development manifest holding a held-out row (analysis plan split_use)."""
    probe = d/'analysis/held_out_probe'
    (probe/'evalkit').mkdir(parents=True, exist_ok=True)
    man = read(d/'evalkit/manifest_development.json')
    row = next(r for r in man['rows'] if r['method'] != 'kimodo')
    write(probe/'evalkit/manifest_development.json', {**man, 'rows': man['rows'] + [{**row, 'id': row['id'] + '_probe', 'split': 'held_out'}]})
    with open(log, 'a') as f:
        f.write(f'\n[{now()}] held-out refusal probe\n')
    r = subprocess.run([PY, 'python/scripts/analyze_batch.py', '--batch', str(probe), '--structure-only'], cwd=ROOT,
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    refused = r.returncode != 0 and "refused: rows of split ['held_out']" in r.stderr
    with open(log, 'a') as f:
        f.write(f'returncode {r.returncode}; refused {refused}\n')
    return refused


def analysis_stage(d, settings, log):
    if settings['rehearsal']:
        rc = call([PY, 'python/scripts/analyze_batch.py', '--batch', d, '--structure-only'], log)
        refused = held_out_probe(d, log)
        return (rc or (0 if refused else 1)), {'structure': (d/'analysis/structure.json').exists(), 'held_out_refused': refused}
    rc = call([PY, 'python/scripts/analyze_batch.py', '--batch', d], log)
    return rc, {'development_json': (d/'analysis/development.json').exists()}


def run_stage(d, name, settings, log):
    if name in QUEUES:
        return queue_stage(d, name, log)
    if name == 'contract':
        rc = call([PY, 'python/scripts/complete_run_contract.py', 'complete', '--batch', d], log)
        rc = rc or call([PY, 'python/scripts/complete_run_contract.py', 'scan', '--batch', d], log)
        return rc, read(d/'contract_scan.json', {}).get('summary', {})
    if name == 'manifest':
        rc = call([PY, 'python/scripts/build_evalkit_manifest.py', '--batch', d, '--kimodo', settings['kimodo']], log)
        return rc, {s: len(read(d/f'evalkit/manifest_{s}.json', {'rows': []})['rows']) for s in ('development', 'held_out')}
    if name == 'seed_evalkit':
        return seed_evalkit(d, settings, log)
    if name in EVAL_DIRS:
        return eval_stage(d, name, settings, log)
    return analysis_stage(d, settings, log)


def runner(d):
    lock = open(d/'pipeline.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = read(d/'settings.json')
    stages = read(d/'stages.json')
    stages['runner'] = {'pid': os.getpid(), 'started': now()}
    write(d/'stages.json', stages)
    for name in stage_names(settings):
        st = stages['stages'][name]
        if st['status'] == 'done':
            continue
        res, short = resources(d)
        if short:
            st.update(status='blocked', reason=f'below the free-space / memory floor: {short}', resources=res, finished=now())
            write(d/'stages.json', stages)
            return 1
        st.update(status='running', started=now(), resources_at_start=res)
        st.pop('finished', None)
        write(d/'stages.json', stages)
        t = time.time()
        try:
            rc, counts = run_stage(d, name, settings, d/'logs'/f'{name}.log')
        except Exception as e:
            rc, counts = 1, {'exception': f'{type(e).__name__}: {e}'}
        st.update(status='done' if rc == 0 else 'failed', returncode=rc, finished=now(),
                  seconds=round(st.get('seconds', 0) + time.time() - t), counts=counts)
        write(d/'stages.json', stages)
        if rc:
            return rc
    stages['runner']['finished'] = now()
    write(d/'stages.json', stages)
    return 0


def detach(d):
    env = {**os.environ, 'PYTHONNOUSERSITE': '1'}
    with open(d/'logs/pipeline.log', 'a') as f:
        subprocess.Popen(['nohup', 'setsid', 'nice', '-n', '19', PY, str(Path(__file__).resolve()), '_run', '--batch', d.name],
                         cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def runner_alive(d):
    if not (d/'pipeline.lock').exists():
        return False
    with open(d/'pipeline.lock', 'a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
    return False


def start(a):
    d = batch_dir(a.batch)
    if d.exists() or d.is_symlink():
        raise SystemExit(f'{d.relative_to(ROOT)} exists; use resume')
    if a.rehearsal and a.reuse_batch:
        raise SystemExit('--reuse-batch is for the formal batch')
    target = data_root(load_config())/'batch_runs'/a.batch
    target.mkdir(parents=True)
    d.symlink_to(target, target_is_directory=True)
    (d/'logs').mkdir()
    log = d/'logs/preflight.log'
    if freeze_check(log):
        raise SystemExit(f'freeze check failed (see {log.relative_to(ROOT)})')
    cmd = [PY, 'python/scripts/build_batch_plan.py', '--out', d/'plans', '--run-root', f'develop_logs/test_runs/{a.batch}/runs',
           '--deadline-hours', a.deadline_hours]
    cmd += ['--rehearsal'] if a.rehearsal else []
    cmd += ['--reuse-batch', f'develop_logs/test_runs/{a.reuse_batch}'] if a.reuse_batch else []
    if call(cmd, log) or read(d/'plans/self_check.json')['problems']:
        raise SystemExit(f'plan self check failed (see {log.relative_to(ROOT)})')
    res, short = resources(d)
    if short:
        raise SystemExit(f'below the free-space / memory floor {MIN_FREE_GIB}: {res}')
    settings = {'batch': a.batch, 'rehearsal': a.rehearsal, 'reuse_batch': a.reuse_batch, 'workers': a.workers,
                'kimodo': a.kimodo if not a.rehearsal or a.kimodo != 'all' else '1', 'created': now(),
                'commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                'plans_sha256': {p.name: sha(p) for p in sorted((d/'plans').glob('*.json'))}, 'resources': res,
                'min_free_gib': MIN_FREE_GIB}
    write(d/'settings.json', settings)
    write(d/'stages.json', {'stages': {n: {'status': 'pending'} for n in stage_names(settings)}})
    detach(d)
    print(json.dumps({'batch': a.batch, 'stages': stage_names(settings), 'resources': res}, ensure_ascii=False))


def resume(a):
    d = batch_dir(a.batch)
    if runner_alive(d):
        raise SystemExit('the runner is alive')
    settings = read(d/'settings.json')
    changed = [n for n, h in settings['plans_sha256'].items() if sha(d/'plans'/n) != h]
    if changed:
        raise SystemExit(f'plans changed since start: {changed}')
    if freeze_check(d/'logs/preflight.log'):
        raise SystemExit('freeze check failed (see logs/preflight.log)')
    stages = read(d/'stages.json')
    for st in stages['stages'].values():
        if st['status'] in ('running', 'failed', 'blocked'):
            st['status'] = 'pending'
    write(d/'stages.json', stages)
    detach(d)
    print(json.dumps({'batch': a.batch, 'resumed': [n for n, s in stages['stages'].items() if s['status'] == 'pending']}))


def queue_counts(qdir):
    state = read(Path(qdir)/'state.json')
    if not state:
        return {}
    out = {}
    for job, rec in state['jobs'].items():
        method = job.split('_')[1]
        c = out.setdefault(method, {'success': 0, 'failed': 0, 'timeout': 0, 'resource_gate': 0, 'pending': 0, 'running': 0,
                                    'other': 0, 'attempts': 0, 'gpu_seconds': 0.})
        s = rec.get('status')
        c[s if s in c else 'other'] += 1
        c['attempts'] += len(rec.get('attempts', []))
        c['gpu_seconds'] += sum(x.get('elapsed_seconds') or 0 for x in rec.get('attempts', []))
    for c in out.values():
        c['gpu_seconds'] = round(c['gpu_seconds'])
    return {'queue_status': state.get('status'), 'heartbeat': state.get('heartbeat'), 'methods': out}


def status(a):
    d = batch_dir(a.batch)
    settings, stages = read(d/'settings.json'), read(d/'stages.json')
    out = {'batch': a.batch, 'runner_alive': runner_alive(d), 'runner': stages.get('runner'), 'resources': resources(d)[0], 'stages': {}}
    for name in stage_names(settings):
        st = stages['stages'][name]
        row = {k: st[k] for k in ('status', 'started', 'finished', 'seconds', 'reason') if k in st}
        if name in QUEUES and (d/QUEUES[name][1]/'state.json').exists():
            row['queue'] = queue_counts(d/QUEUES[name][1])
        elif name in EVAL_DIRS and st['status'] in ('running', 'done', 'failed'):
            e = d/'evalkit/development'
            row['rows'] = (eval_counts(d/'evalkit/manifest_development.json', e/'method') if name == 'eval_method' else
                           eval_counts(e/'self_manifest.json', e/'self') if name == 'eval_self' else
                           eval_counts(e/'lumbar_manifest.json', e/'self', upper=True) if name == 'eval_lumbar' else neck_counts(d))
        elif 'counts' in st:
            row['counts'] = st['counts']
        out['stages'][name] = row
    print(json.dumps(out, indent=1, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['start', 'resume', 'status', '_run'])
    p.add_argument('--batch', required=True)
    p.add_argument('--rehearsal', action='store_true')
    p.add_argument('--reuse-batch')
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--kimodo', default='all')
    p.add_argument('--deadline-hours', type=float, default=96.)
    a = p.parse_args()
    if a.action == 'start':
        start(a)
    elif a.action == 'resume':
        resume(a)
    elif a.action == 'status':
        status(a)
    else:
        raise SystemExit(runner(batch_dir(a.batch)))


if __name__ == '__main__':
    main()
