"""Versioned simulation event protocol under the existing experiment GPU lease."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--run-dir', type=Path, required=True)
p.add_argument('--events', type=Path, required=True)
p.add_argument('--seed', type=int, default=0)
p.add_argument('--clock', choices=['paused', 'realtime'], default='realtime')
p.add_argument('--ignore', action='store_true')
p.add_argument('--load', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
a.run_dir.mkdir(parents=True, exist_ok=True)
socket = f'/tmp/mg-sim-event-{os.getpid()}.sock'
worker = [sys.executable, str(root/'python/scripts/closd_runtime_worker.py'), '--run-dir', str(a.run_dir),
          '--socket', socket, '--seed', str(a.seed), '--clock', a.clock]
if a.ignore:
    worker.append('--ignore')
commands = {'producer': [sys.executable, str(root/'python/scripts/reveal_runtime_event.py'), '--socket', socket,
    '--ready', str(a.run_dir/'ready.json'), '--events', str(a.events), '--output', str(a.run_dir/'producer.json')],
    'generator': worker}
if a.load:
    commands['load'] = [sys.executable, str(root/'python/scripts/runtime_compute_load.py'), '--ready',
        str(a.run_dir/'ready.json'), '--output', str(a.run_dir/'compute_load.json')]
children = {}
try:
    for name, command in commands.items():
        children[name] = subprocess.Popen(command)
    (a.run_dir/'processes.json').write_text(json.dumps({
        'children': {k: {'pid': v.pid, 'argv': commands[k]} for k, v in children.items()},
        'lease_gpu': os.getenv('CUDA_VISIBLE_DEVICES')}))
    if children['generator'].wait():
        raise RuntimeError('Native simulation worker failed; see failure.json')
    for name, child in children.items():
        if child.wait(timeout=30):
            raise RuntimeError(f'{name} failed')
finally:
    for child in children.values():
        if child.poll() is None:
            child.terminate()
    for child in children.values():
        child.wait()
    Path(socket).unlink(missing_ok=True)
