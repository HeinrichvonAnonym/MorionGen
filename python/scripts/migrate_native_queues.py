"""One-time, lossless drain of pre-fairness queue; no generated evidence is deleted."""
import json
import os
from pathlib import Path
import signal
import sys
import time
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write,event

base=ROOT/'develop_logs/test_runs'
waiting=['20261024_native_pilots/ardy_queue_cfg_corrected','20261024_native_followups/queue']
def owned(directory):
    state=json.loads((directory/'state.json').read_text()); proc=psutil.Process(state['pid'])
    if proc.create_time()!=state['process_create_time'] or 'python/scripts/experiments.py' not in proc.cmdline():
        raise ValueError('Queue process identity mismatch')
    return state,proc
for name in waiting:
    directory=base/name
    previous=json.loads((directory/'state.json').read_text())
    if previous.get('status')=='migrated_before_admission' and not psutil.pid_exists(previous['pid']): continue
    state,proc=owned(directory)
    if proc.children(recursive=True) or any(r['status']!='pending' for r in state['jobs'].values()):
        raise ValueError('Waiting queue is no longer empty; stop and reconcile explicitly')
    proc.terminate();proc.wait(timeout=10)
    state.update(status='migrated_before_admission',migration_reason='Release old whole-queue leases without competing new admission')
    json_write(directory/'state.json',state)
directory=base/'20261024_control_matrix/queue';state,parent=owned(directory)
parent.suspend()
terminated=False
try:
    children=parent.children()
    workers={}
    for child in children:
        command=child.cmdline()
        if command and Path(command[0]).name=='nvidia-smi': continue
        if '--run-dir' not in command or not any(arg.endswith('/baseline_worker.py') for arg in command):
            raise ValueError('Unexpected child; refusing to drain an unrecognized process')
        workers[command[command.index('--run-dir')+1]]=child
    print(json.dumps({'stage':'draining','parent':parent.pid,'workers':[p.pid for p in workers.values()]}),flush=True)
    started=time.monotonic()
    while any(p.is_running() and p.status()!=psutil.STATUS_ZOMBIE for p in workers.values()):
        if time.monotonic()-started>240: raise TimeoutError('Drain exceeded 240s; original queue will resume')
        time.sleep(1)
    codes={}
    for run,p in workers.items():
        fields=Path(f'/proc/{p.pid}/stat').read_text().rsplit(') ',1)[1].split()
        codes[run]=os.waitstatus_to_exitcode(int(fields[49]))
    parent.kill();parent.wait(timeout=10);terminated=True
    # The parent was suspended before it could reap workers; verify OS exit,
    # protocol and required artifacts before reconciling the original receipt.
    for job,row in state['jobs'].items():
        if row['status']!='running': continue
        run=Path(row['run_dir']); result=json.loads((run/'worker_result.json').read_text())
        ok=codes.get(str(run))==0 and result.get('status')=='success' and all((run/n).is_file() for n in ['motion.npz','metrics.json','intervention.json'])
        status='success' if ok else 'interrupted_infrastructure'
        row.update(status=status,returncode=codes.get(str(run)),reconciliation='OS exit + worker protocol + required artifacts after lease migration drain')
        row.setdefault('attempts',[]).append({'status':status,'run_dir':str(run),'returncode':codes.get(str(run))})
        manifest=json.loads((run/'manifest.json').read_text());manifest.update(status=status,reconciliation=row['reconciliation'])
        json_write(run/'manifest.json',manifest);event(run/'events.jsonl','queue_drain_reconciled',status=status,returncode=codes.get(str(run)))
    state['status']='migrated';json_write(directory/'state.json',state)
    plan=json.loads((base/'20261024_control_matrix/plan.json').read_text())
    plan['migration_parent']=str(directory);plan['reused_completed_jobs']={k:v for k,v in state['jobs'].items() if v['status']=='success'}
    plan['jobs']=[j for j in plan['jobs'] if state['jobs'][j['id']]['status']!='success']
    destination=base/'20261024_control_matrix/continuation_plan.json'
    if destination.exists(): raise ValueError('Continuation plan already exists')
    json_write(destination,plan)
    print(json.dumps({'stage':'drained','preserved_completed':len(plan['reused_completed_jobs']),'remaining':len(plan['jobs'])}),flush=True)
finally:
    if not terminated and parent.is_running(): parent.resume()
