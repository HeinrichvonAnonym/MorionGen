"""Unattended P0 finish: wait for the batch queue, re-queue failures once, then run the registered analysis.

Writes finish.log next to the P0 plans.
"""
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'develop_logs/test_runs/20261102_p0'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen')
TOOLS = DATA/'envs/motiogen-tools/bin/python'
ARDY = DATA/'envs/motiogen-ardy/bin/python'
ENV = {**os.environ, 'PYTHONNOUSERSITE': '1'}


def log(msg):
    with open(OUT/'finish.log', 'a') as f:
        f.write(f'{datetime.now().astimezone().isoformat(timespec="seconds")} {msg}\n')


def wait(queue, jobs):
    while True:
        state = json.loads((OUT/queue/'state.json').read_text())['jobs'] if (OUT/queue/'state.json').exists() else {}
        statuses = [state.get(j, {}).get('status') for j in jobs]
        if all(s not in (None, 'pending', 'running') for s in statuses):
            return sum(s == 'success' for s in statuses)
        time.sleep(60)


def run(cmd, out):
    with open(OUT/out, 'a') as f:
        return subprocess.run(cmd, cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT).returncode


def main():
    log('finish start')
    jobs = [j['id'] for j in json.loads((OUT/'batch_plan.json').read_text())['jobs']]
    done = wait('batch_queue', jobs)
    log(f'batch finished success={done}/{len(jobs)}')
    if done < len(jobs):
        code = run([str(TOOLS), 'python/scripts/plan_p0.py', '--stage', 'continuation'], 'finish_plan.log')
        log(f'continuation plan exit={code}')
        if code == 0:
            cont = [j['id'] for j in json.loads((OUT/'continuation_plan.json').read_text())['jobs']]
            run([str(TOOLS), 'python/scripts/experiments.py', 'run', '--plan', str(OUT/'continuation_plan.json'),
                 '--directory', str(OUT/'continuation_queue')], 'continuation_queue.log')
            log(f'continuation finished success={wait("continuation_queue", cont)}/{len(cont)}')
    code = run([str(ARDY), 'python/scripts/analyze_p0.py', '--folder', str(OUT), '--analyze'], 'analysis.out')
    log(f'analysis exit={code}')


if __name__ == '__main__':
    main()
