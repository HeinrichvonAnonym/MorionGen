"""Exercise scheduling with sleeping dummy children; no numerical workloads."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('/home/heinrich/MorionGen')
OUT=ROOT/'develop_logs/test_runs/20261120_k4_andy/lease_check'/str(time.time_ns())
OUT.mkdir(parents=True)
(OUT/'evalkit_k4_run.py').write_text('import time\ntime.sleep(120)\n')
(OUT/'controller.py').write_text("import subprocess,sys,time,pathlib\np=subprocess.Popen([sys.executable,sys.argv[1]])\npathlib.Path(sys.argv[2]).write_text(str(p.pid))\np.wait()\n")
checks=[]
for mode in ('normal','termination'):
    run=OUT/mode;run.mkdir(exist_ok=True)
    c=subprocess.Popen([sys.executable,str(OUT/'controller.py'),str(OUT/'evalkit_k4_run.py'),str(run/'child.pid'),str(run)],start_new_session=True)
    guard=None
    try:
        while not (run/'child.pid').exists():time.sleep(.02)
        pid=int((run/'child.pid').read_text())
        command=[sys.executable,'-c',f'import time;time.sleep({2 if mode=="normal" else 30})']
        (run/'command.json').write_text(json.dumps(command))
        guard=subprocess.Popen([sys.executable,str(ROOT/'python/scripts/evalkit_k4_slot_lease.py'),'--tv-controller',str(c.pid),'--tv-run',str(run),'--command',str(run/'command.json'),'--receipt',str(run/'lease.json')])
        for _ in range(200):
            if (run/'lease.json').exists() and json.loads((run/'lease.json').read_text()).get('state')=='guest_running':break
            time.sleep(.02)
        else:raise AssertionError('No reservation')
        raw=Path(f'/proc/{pid}/stat').read_text();assert raw[raw.rfind(')')+2:].startswith('T')
        if mode=='termination':guard.terminate()
        guard.wait(timeout=10)
        raw=Path(f'/proc/{pid}/stat').read_text();assert not raw[raw.rfind(')')+2:].startswith('T')
        record=json.loads((run/'lease.json').read_text());assert record['tv_slot_resumed']
        checks.append(dict(mode=mode,suspended_during_guest=True,resumed_after_guest=True))
    finally:
        if guard and guard.poll() is None:guard.terminate();guard.wait(timeout=10)
        os.killpg(c.pid,signal.SIGTERM);c.wait(timeout=5)
(OUT/'result.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks))
