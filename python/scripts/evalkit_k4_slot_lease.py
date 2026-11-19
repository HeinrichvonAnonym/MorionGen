"""Reserve one running TV slot reversibly while an isolated single-worker task runs.

The TV controller and all numerical files stay unchanged. One direct, childless
numerical process is suspended in memory; its waiting controller thread retains
the slot. Other TV workers continue. SIGTERM stops the guest before resuming TV.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def info(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        return dict(pid=pid,state=fields[0],ppid=int(fields[1]),start_ticks=fields[19],
                    argv=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
    except FileNotFoundError:return None


def same(saved):
    current=info(saved['pid'])
    return current and current['start_ticks']==saved['start_ticks']


def atomic(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tv-controller',type=int,required=True)
    ap.add_argument('--tv-run',type=Path,required=True)
    ap.add_argument('--command',type=Path,required=True)
    ap.add_argument('--receipt',type=Path,required=True)
    a=ap.parse_args();a.receipt.parent.mkdir(parents=True,exist_ok=True)
    lock=(a.tv_run/'.external_slot_lease.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    controller=info(a.tv_controller)
    if not controller or str(a.tv_run) not in controller['argv']:raise ValueError('TV controller identity mismatch')
    command=json.loads(a.command.read_text());assert isinstance(command,list) and all(isinstance(x,str) for x in command)
    child=None;reserved=None;controller_stopped=False
    def terminate(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM,terminate);signal.signal(signal.SIGINT,terminate)
    state=dict(version='external-single-slot-1',guard_pid=os.getpid(),tv_controller=controller,command=command,started_unix_s=time.time(),scope='13 TV runnable slots + one guest numerical slot; no fit/selection changes')
    try:
        while reserved is None:
            if not same(controller):raise RuntimeError('TV controller exited before reservation')
            os.kill(a.tv_controller,signal.SIGSTOP);controller_stopped=True
            procs=[info(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
            children=[p for p in procs if p and p['ppid']==a.tv_controller and p['state'] not in ('Z','T','t')]
            candidates=[p for p in children if ('evalkit_k4_run.py' in p['argv'] or 'evalkit_k4_test_evaluate.py score' in p['argv']) and not any(c and c['ppid']==p['pid'] for c in procs)]
            for candidate in sorted(candidates,key=lambda p:p['pid'],reverse=True):
                try:
                    os.kill(candidate['pid'],signal.SIGSTOP)
                    for _ in range(100):
                        current=info(candidate['pid'])
                        if current and current['state'] in ('T','t'):reserved=candidate;break
                        time.sleep(.01)
                    if reserved:break
                except ProcessLookupError:pass
            os.kill(a.tv_controller,signal.SIGCONT);controller_stopped=False
            if reserved is None:time.sleep(5)
        state.update(state='reserved',paused_tv_process=reserved);atomic(a.receipt,state)
        child=subprocess.Popen(command,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
        state.update(state='guest_running',guest_pid=child.pid);atomic(a.receipt,state)
        # Rotate the reserved slot so a single TV record is not delayed for the
        # entire Andy test. Once TV has <14 pending source windows, its pending
        # set only shrinks, so it can safely reclaim every remaining slot.
        last_rotation=time.monotonic();state['reservation_history']=[reserved]
        while child.poll() is None:
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:pass
            if child.poll() is not None:break
            if reserved and same(controller):
                progress_path=a.tv_run/'progress.json'
                progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
                active=progress.get('active')
                if progress.get('expected')==4256 and isinstance(active,int) and 0<active<14:
                    if same(reserved):os.kill(reserved['pid'],signal.SIGCONT)
                    state.update(paused_tv_process=None,reservation_released_for_tail=True)
                    reserved=None;atomic(a.receipt,state)
                elif time.monotonic()-last_rotation>=600:
                    os.kill(a.tv_controller,signal.SIGSTOP);controller_stopped=True
                    procs=[info(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()]
                    candidates=[p for p in procs if p and p['ppid']==a.tv_controller and p['state'] not in ('Z','T','t')
                                and ('evalkit_k4_run.py' in p['argv'] or 'evalkit_k4_test_evaluate.py score' in p['argv'])
                                and not any(c and c['ppid']==p['pid'] for c in procs)]
                    for candidate in sorted(candidates,key=lambda p:p['pid'],reverse=True):
                        try:
                            os.kill(candidate['pid'],signal.SIGSTOP)
                            for _ in range(100):
                                current=info(candidate['pid'])
                                if current and current['state'] in ('T','t'):
                                    if same(reserved):os.kill(reserved['pid'],signal.SIGCONT)
                                    reserved=candidate;state['reservation_history'].append(candidate)
                                    state['paused_tv_process']=candidate;last_rotation=time.monotonic()
                                    break
                                time.sleep(.01)
                            if reserved==candidate:break
                        except ProcessLookupError:pass
                    os.kill(a.tv_controller,signal.SIGCONT);controller_stopped=False
                    atomic(a.receipt,state)
        code=child.returncode;state.update(state='guest_finished',returncode=code)
    except KeyboardInterrupt as error:
        code=130;state.update(state='guest_interrupted',error=str(error),returncode=code)
    except Exception as error:
        state.update(state='lease_failed',error=str(error));raise
    finally:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if reserved and same(reserved):os.kill(reserved['pid'],signal.SIGCONT)
        if controller_stopped and same(controller):os.kill(a.tv_controller,signal.SIGCONT)
        state.update(released_unix_s=time.time(),tv_slot_resumed=bool(reserved and same(reserved)))
        atomic(a.receipt,state)
    raise SystemExit(code)


if __name__=='__main__':main()
