"""Independent supervisor. No torch imports or model allocations in parent."""
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import psutil

from .config import ROOT, data_root, repo_root

GIB = 1024 ** 3


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def event(path, kind, **fields):
    with Path(path).open('a') as handle:
        handle.write(json.dumps({'event': kind, 'wall_time': dt.datetime.now(dt.timezone.utc).isoformat(), 'monotonic_seconds': time.monotonic(), **fields}, ensure_ascii=False, allow_nan=False) + '\n')


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            h.update(part)
    return h.hexdigest()


def command_output(command, cwd=None):
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
        return {'returncode': result.returncode, 'stdout': result.stdout.strip(), 'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'returncode': None, 'stdout': '', 'stderr': f'{type(exc).__name__}: {exc}'}


def gpu_snapshot():
    try:
        result = command_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,memory.used', '--format=csv,noheader,nounits'])
        if result['returncode']:
            return {'available': False, 'error': result['stderr']}
        # GPU 0 is the explicit device used by this single-GPU framework.
        line = result['stdout'].splitlines()[0]
        index, total, free, used = [int(x.strip()) for x in line.split(',')]
        return {'available': True, 'index': index, 'total_bytes': total * 1024 ** 2, 'free_bytes': free * 1024 ** 2, 'used_bytes': used * 1024 ** 2}
    except (OSError, ValueError, subprocess.TimeoutExpired, IndexError) as exc:
        return {'available': False, 'error': str(exc)}


def memory_controller():
    for path in (Path('/sys/fs/cgroup/cgroup.controllers'), Path('/sys/fs/cgroup/unified/cgroup.controllers')):
        if path.exists() and 'memory' in path.read_text().split():
            return {'controller_present': True, 'hard_limit_enforced': False, 'reason': 'presence is not proof of delegated/enforced limits'}
    return {'controller_present': False, 'hard_limit_enforced': False, 'reason': 'memory controller unavailable; watchdog cannot guarantee hard isolation'}


def budgets(cfg):
    r = cfg['resources']
    ram = psutil.virtual_memory()
    gpu = gpu_snapshot()
    storage = data_root(cfg)
    storage.mkdir(parents=True, exist_ok=True)
    disk = psutil.disk_usage(storage)
    return {'ram_bytes': max(0, min(r['ram_max_gib'] * GIB, ram.available - r['ram_reserve_gib'] * GIB)),
            'gpu_bytes': max(0, min(r['gpu_max_gib'] * GIB, gpu.get('free_bytes', 0) - r['gpu_reserve_gib'] * GIB)),
            'available_ram_bytes': ram.available, 'gpu': gpu, 'free_disk_bytes': disk.free,
            'storage_root': str(storage),
            'disk_ok': disk.free > r['disk_reserve_gib'] * GIB, 'isolation': memory_controller()}


class Run:
    def __init__(self, cfg, baseline, action, scene=None, seed=None, command=None):
        timestamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        self.path = ROOT / cfg['run_root'] / f'{timestamp}_{baseline}_{action}_{uuid.uuid4().hex[:8]}'
        self.path.mkdir(parents=True, exist_ok=False)
        self.events = self.path / 'events.jsonl'
        self.manifest = {'schema_version': 1, 'baseline': baseline, 'action': action, 'scenario': scene, 'seed': seed,
                         'status': 'running', 'command': command, 'timezone': 'Asia/Shanghai', 'budgets': budgets(cfg)}
        repo = repo_root(cfg, baseline)
        self.manifest['git'] = {key: command_output(cmd, repo) for key, cmd in {
            'commit': ['git', 'rev-parse', 'HEAD'], 'status': ['git', 'status', '--short'],
            'remote': ['git', 'remote', '-v'], 'patch': ['git', 'diff', '--no-ext-diff']}.items()}
        self.manifest['framework_git']={key:command_output(cmd,ROOT) for key,cmd in {
            'commit':['git','rev-parse','HEAD'],'status':['git','status','--short']}.items()}
        implementation=[*ROOT.joinpath('utils/motiogen').glob('*.py'),*ROOT.joinpath('python/scripts').glob('*.py'),*ROOT.joinpath('scripts').glob('*.sh')]
        json_write(self.path/'implementation.json',{str(file.relative_to(ROOT)):checksum(file) for file in implementation})
        from .config import asset_root
        assets=asset_root(cfg)/baseline
        receipts=[*assets.glob('*.extracted.json'),*assets.rglob('source_receipt.json')] if assets.exists() else []
        json_write(self.path/'asset_receipts.json',[{'path':str(file),'receipt':json.loads(file.read_text())} for file in receipts])
        json_write(self.path / 'config.json', cfg)
        # Capture metric/transport/resource contracts, not only the launch config.
        contracts={}
        for name in ('metrics.json','evaluation.json','downloads.json','language_models.json','inference_evidence.json','training_evidence.json'):
            source=ROOT/'config'/name
            if source.exists():
                contracts[name]={'sha256':checksum(source),'content':json.loads(source.read_text())}
        json_write(self.path/'contracts.json',contracts)
        json_write(self.path / 'manifest.json', self.manifest)
        if scene is not None:
            json_write(self.path / 'scenario.json', scene)
        # Ensure logs exist even when preflight rejects the process.
        (self.path / 'stdout.log').touch()
        (self.path / 'stderr.log').touch()
        (self.path / 'resources.jsonl').touch()
        event(self.events, 'run_created', action=action)

    def finish(self, status, reason=None, **fields):
        self.manifest.update(status=status, reason=reason, **fields)
        json_write(self.path / 'manifest.json', self.manifest)
        event(self.events, 'run_finished', status=status, reason=reason)
        return {'status': status, 'reason': reason, 'run_dir': str(self.path), **fields}


@contextlib.contextmanager
def gpu_lock(cfg):
    path = ROOT / cfg['run_root'] / '.gpu.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def terminate_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def supervise(run, command, cfg, cwd=ROOT, gpu=False, extra_env=None, timeout=None, require_protocol=True):
    """Watch process tree and global headroom, not other users' GPU processes."""
    budget = budgets(cfg)
    run.manifest.update(command=command, budgets=budget)
    json_write(run.path / 'manifest.json', run.manifest)
    if not budget['disk_ok']:
        return run.finish('resource_terminated', 'disk reserve unavailable')
    if budget['ram_bytes'] <= 0 or (gpu and budget['gpu_bytes'] <= 0):
        return run.finish('resource_terminated', 'RAM/VRAM reserve unavailable')
    env = os.environ.copy()
    env.update({'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1', 'TOKENIZERS_PARALLELISM': 'false',
                'MPLCONFIGDIR': str(run.path / 'cache/matplotlib'), 'WANDB_MODE': 'offline', 'CUDA_VISIBLE_DEVICES': '0'})
    if run.manifest['action']!='prepare':
        env.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'})
    if extra_env:
        env.update(extra_env)
    start = time.monotonic()
    timeout = timeout if timeout is not None else cfg['resources']['timeout_seconds']
    peak_ram = 0
    peak_gpu_used = 0
    status, reason = 'success', None
    with (run.path / 'stdout.log').open('w') as stdout, (run.path / 'stderr.log').open('w') as stderr, (run.path / 'resources.jsonl').open('w') as samples:
        proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        event(run.events, 'process_started', pid=proc.pid)
        while proc.poll() is None:
            try:
                parent = psutil.Process(proc.pid)
                processes = [parent] + parent.children(recursive=True)
                rss = sum(p.memory_info().rss for p in processes if p.is_running())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                rss = 0
            ram = psutil.virtual_memory()
            gs = gpu_snapshot() if gpu else {'available': False}
            elapsed = time.monotonic() - start
            peak_ram = max(peak_ram, rss)
            peak_gpu_used = max(peak_gpu_used, gs.get('used_bytes', 0))
            sample = {'elapsed_seconds': elapsed, 'rss_tree_bytes': rss, 'available_ram_bytes': ram.available, 'gpu': gs}
            samples.write(json.dumps(sample) + '\n')
            samples.flush()
            if elapsed > timeout:
                status, reason = 'timeout', 'worker timeout'
            elif rss > budget['ram_bytes'] or ram.available < cfg['resources']['ram_reserve_gib'] * GIB:
                status, reason = 'resource_terminated', 'RAM budget/reserve crossed'
            elif gpu and (not gs['available'] or gs['free_bytes'] < cfg['resources']['gpu_reserve_gib'] * GIB or gs.get('used_bytes',0)-budget['gpu'].get('used_bytes',0)>budget['gpu_bytes']):
                status, reason = 'resource_terminated', 'VRAM telemetry unavailable or reserve crossed'
            elif psutil.disk_usage(data_root(cfg)).free < cfg['resources']['disk_reserve_gib'] * GIB:
                status, reason = 'resource_terminated', 'disk reserve crossed'
            if reason:
                event(run.events, 'termination_requested', reason=reason)
                terminate_group(proc)
                break
            time.sleep(cfg['resources']['sample_seconds'])
        returncode = proc.wait()
    if not reason and returncode:
        status, reason = 'failed', f'worker exit {returncode}'
    worker_result = run.path / 'worker_result.json'
    fields = {'returncode': returncode, 'elapsed_seconds': time.monotonic() - start,
              'peak_rss_tree_bytes': peak_ram, 'peak_global_gpu_used_bytes': peak_gpu_used,
              'gpu_peak_scope': 'device-wide sampled usage; worker allocator peaks recorded separately'}
    if worker_result.exists():
        payload = json.loads(worker_result.read_text())
        fields['worker'] = payload
        if status == 'success':
            status, reason = payload['status'], payload.get('reason')
    elif status == 'success' and require_protocol:
        status, reason = 'failed', 'worker did not produce result protocol'
    return run.finish(status, reason, **fields)
