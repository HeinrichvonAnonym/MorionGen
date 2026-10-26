"""Two-device queue with durable task receipts and a shared wall deadline."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import psutil

from .config import ROOT, load_config
from .runtime import Run, checksum, gpu_lock, json_write, supervise


def fingerprint(plan):
    from .execution_identity import identity_payload
    payload = identity_payload(plan,load_config())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def execute(plan, directory, resume=False):
    directory.mkdir(parents=True, exist_ok=True)
    identity = fingerprint(plan)
    state_path = directory/'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if not resume or state['identity'] != identity:
            raise ValueError('Existing queue requires resume with unchanged plan, code and config')
        for row in state['jobs'].values():
            if row['status'] == 'running':
                raise ValueError('Interrupted running job requires process/evidence reconciliation before resume')
    else:
        state = {'identity': identity, 'jobs': {j['id']: {'status': 'pending'} for j in plan['jobs']}}
    state.update(pid=os.getpid(), process_create_time=psutil.Process().create_time(),
                 status='running', deadline=plan['deadline'])
    deadline = datetime.fromisoformat(plan['deadline'].replace('Z','+00:00')).timestamp()
    admission = deadline - float(plan.get('report_reserve_seconds',14400))
    mutex = threading.Lock()
    stop = threading.Event()
    def persist():
        state['heartbeat'] = datetime.now(timezone.utc).isoformat()
        json_write(state_path, state)
    def heartbeat():
        while not stop.wait(5):
            with mutex:
                persist()
    persist()
    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    def lane(device):
        cfg = load_config()
        cfg['gpu_index'] = device
        while True:
            # Do not wait behind unrelated queues merely to discover that this
            # queue is drained. Recheck after acquiring the lease for races.
            with mutex:
                if not any(r['status']=='pending' for r in state['jobs'].values()):
                    return
            # The lease belongs to one heavy task, not the entire method queue.
            # Other queued methods can acquire the device at task boundaries.
            with gpu_lock(cfg):
                with mutex:
                    job = next((j for j in plan['jobs'] if state['jobs'][j['id']]['status'] == 'pending'), None)
                    if job is None or time.time()+float(job.get('estimated_seconds',0)) >= admission:
                        return
                    previous=state['jobs'][job['id']]
                    state['jobs'][job['id']] = {'status': 'running', 'device': device,
                        'attempts':previous.get('attempts',[]),'started_at':datetime.now(timezone.utc).isoformat()}
                    persist()
                try:
                    local = copy.deepcopy(cfg)
                    local.update(job.get('config', {}))
                    method = job['method']
                    if method not in local['baselines']:
                        local['baselines'][method] = job['method_spec']
                    run = Run(local, method, job.get('action','demo'), job.get('scene'), job.get('seed'), job.get('command'))
                    command = [arg.replace('{run_dir}', str(run.path)) for arg in job['command']]
                    with mutex:
                        state['jobs'][job['id']]['run_dir'] = str(run.path)
                        persist()
                    result = supervise(run, command, local, gpu=True,
                        timeout=max(.01,min(job.get('timeout_seconds',900),deadline-time.time())),
                        extra_env=job.get('environment'),
                        require_protocol=job.get('require_protocol',True))
                    if result['status'] == 'success':
                        for filename in job.get('required_outputs', []):
                            p = Path(filename.replace('{run_dir}',str(run.path)))
                            if not p.is_file():
                                result = run.finish('failed',f'Missing expected output: {p}')
                                break
                except Exception as exc:
                    result = {'status':'failed','reason':f'{type(exc).__name__}: {exc}'}
                with mutex:
                    compact = {key: result[key] for key in ('status','reason','run_dir','returncode','elapsed_seconds') if key in result}
                    state['jobs'][job['id']].update(compact)
                    row=state['jobs'][job['id']]
                    row['attempts'].append(dict(compact))
                    # Retry only declared recoverable outcomes, never missing dependencies
                    # or resource-gate failures. Preserve every attempt, including failures.
                    if compact['status'] in job.get('retryable_statuses',['timeout']) and len(row['attempts'])<2:
                        row['status']='pending'
                    persist()
                print(json.dumps({'job':job['id'],'device':device,**compact},ensure_ascii=False),flush=True)
    try:
        with ThreadPoolExecutor(max_workers=len(plan['devices'])) as pool:
            futures = [pool.submit(lane,d) for d in plan['devices']]
            for future in futures:
                future.result()
        with mutex:
            state['status'] = 'admission_closed' if any(r['status']=='pending' for r in state['jobs'].values()) else 'finished'
            persist()
    finally:
        stop.set()
        pulse.join()
    return state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['plan','run','status','resume'])
    p.add_argument('--plan', required=True, type=Path)
    p.add_argument('--directory', required=True, type=Path)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    ids = [j['id'] for j in plan['jobs']]
    if len(ids) != len(set(ids)) or len(set(plan['devices'])) != len(plan['devices']):
        raise ValueError('Duplicate job/device identity')
    if args.action == 'plan':
        print(json.dumps({'jobs':len(ids),'devices':plan['devices'],'deadline':plan['deadline'],'mutations':False}))
    elif args.action == 'status':
        state = json.loads((args.directory/'state.json').read_text())
        try:
            process=psutil.Process(state['pid'])
            state['process_alive'] = process.is_running() and process.create_time()==state.get('process_create_time')
        except psutil.NoSuchProcess:
            state['process_alive'] = False
        state['heartbeat_age_seconds'] = time.time()-datetime.fromisoformat(state['heartbeat']).timestamp()
        print(json.dumps(state,ensure_ascii=False,indent=2))
    else:
        args.directory.mkdir(parents=True,exist_ok=True)
        with (args.directory/'queue.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            execute(plan,args.directory,args.action=='resume')

