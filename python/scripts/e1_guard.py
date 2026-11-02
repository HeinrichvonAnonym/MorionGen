"""Unattended guard for the E1 biomechanics batch.

Every minute: pause E1 workers when the CPU package is hot, resume them once it
cools, relaunch the runner if it died before finishing, and after the runner
exits with all runs evaluated, run the post hoc marker pass and analyze_e1.
Writes guard.log and guard_status.json next to the E1 outputs; touches
guard_done when the post-processing finished (a root helper waits on it to
re-enable turbo).
"""
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'develop_logs/test_runs/20261101_e1_biomech'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen')
OPENSIM_PY = Path.home()/'.local/share/opensim-api/bin/python'
ARDY_PY = DATA/'envs/motiogen-ardy/bin/python'
TOTAL = 1080
PAUSE_C, RESUME_C = 88, 78
STALE_S = 1800
MAX_RELAUNCH = 3


def log(msg):
    with open(OUT/'guard.log', 'a') as f:
        f.write(f'{datetime.now().astimezone().isoformat(timespec="seconds")} {msg}\n')


def package_temp():
    for hw in Path('/sys/class/hwmon').iterdir():
        if (hw/'name').read_text().strip() != 'coretemp':
            continue
        for label in hw.glob('temp*_label'):
            if label.read_text().startswith('Package'):
                return int((hw/label.name.replace('label', 'input')).read_text())/1000
    return None


def mem_available_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable'):
            return int(line.split()[1])/2**20
    return None


def runner_pids():
    r = subprocess.run(['pgrep', '-f', '[p]ython/scripts/run_e1_biomech.py'], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]


def worker_pids(parents):
    pids = []
    for p in parents:
        r = subprocess.run(['pgrep', '-P', str(p)], capture_output=True, text=True)
        pids += [int(c) for c in r.stdout.split()]
    return pids


def counts():
    runs = OUT/'runs'
    return sum(1 for _ in runs.glob('*/metrics.json')), sum(1 for _ in runs.glob('*/error.json'))


def progress_age():
    try:
        stamp = json.loads((OUT/'progress.json').read_text())['updated_at']
        return (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()
    except Exception:
        return None


def env():
    return {**os.environ, 'PYTHONNOUSERSITE': '1'}


def launch_runner():
    with open(OUT/'worker.log', 'a') as f:
        f.write(f'=== guard relaunch {datetime.now().astimezone().isoformat()} ===\n')
        subprocess.Popen([str(OPENSIM_PY), 'python/scripts/run_e1_biomech.py', '--workers', '8'], cwd=ROOT, env=env(),
                         stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def post_process():
    for name, cmd in [('marker_posthoc', [str(OPENSIM_PY), 'python/scripts/e1_marker_posthoc.py']),
                      ('analyze_e1', [str(ARDY_PY), 'python/scripts/analyze_e1.py'])]:
        log(f'start {name}')
        with open(OUT/f'{name}_final.log', 'w') as f:
            code = subprocess.run(['nice', '-n', '10', *cmd], cwd=ROOT, env=env(), stdout=f, stderr=subprocess.STDOUT).returncode
        log(f'{name} exit={code}')
        if code:
            return False
    return True


def main():
    paused, relaunches, hot = False, 0, 0
    log('guard start')
    while True:
        temp, mem = package_temp(), mem_available_gb()
        done, err = counts()
        runners = runner_pids()
        state = {'time': datetime.now().astimezone().isoformat(timespec='seconds'), 'pkg_c': temp, 'mem_avail_gb': mem,
                 'evaluated': done, 'errors': err, 'runner_alive': bool(runners), 'paused': paused, 'relaunches': relaunches}
        (OUT/'guard_status.json').write_text(json.dumps(state, indent=1))

        if runners:
            hot = hot + 1 if temp is not None and temp >= PAUSE_C else 0
            if not paused and hot >= 2:
                for p in worker_pids(runners):
                    os.kill(p, signal.SIGSTOP)
                paused = True
                log(f'ALERT_TEMP pkg={temp} paused workers')
            elif paused and temp is not None and temp <= RESUME_C:
                for p in worker_pids(runners):
                    os.kill(p, signal.SIGCONT)
                paused = False
                log(f'resumed workers pkg={temp}')
            age = progress_age()
            if not paused and age is not None and age > STALE_S:
                log(f'ALERT_STALE progress age={age:.0f}s')
            if mem is not None and mem < 3:
                log(f'ALERT_MEM avail={mem:.1f}G')
        elif done + err >= TOTAL:
            log(f'E1 finished evaluated={done} errors={err}')
            ok = post_process()
            (OUT/'guard_done').write_text('ok\n' if ok else 'post_process_failed\n')
            log('guard exit')
            return
        elif relaunches < MAX_RELAUNCH:
            relaunches += 1
            log(f'ALERT_EXIT runner gone at evaluated={done} errors={err}; relaunch {relaunches}')
            launch_runner()
        else:
            log('ALERT_EXIT relaunch limit reached; giving up')
            (OUT/'guard_done').write_text('runner_failed\n')
            return
        time.sleep(60)


if __name__ == '__main__':
    main()
