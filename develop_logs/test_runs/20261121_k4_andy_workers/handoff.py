"""Retain current subprocess checkpoint, then resume same Andy run with 14 slots."""
import hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen')
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
RUN=B/'andy_head_test_20261120_v2';LOG=ROOT/'develop_logs/test_runs/20261121_k4_andy_workers'
PY='/home/heinrich/.local/share/opensim-api/bin/python'
def read(p):return json.loads(p.read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def proc(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();s=raw[raw.rfind(')')+2:].split()
        return dict(pid=pid,ppid=int(s[1]),state=s[0],start=s[19])
    except FileNotFoundError:return None
television=read(B/'test_budget_20261120_v1/completion.json')
assert television['finished']==4256 and television['active']==0
lease=read(RUN/'slot_lease.json');old=lease['guest_pid'];guard=lease['guard_pid']
assert str(RUN) in Path(f'/proc/{old}/cmdline').read_bytes().decode()
entry=RUN/'control/evalkit_k4_andy_parallel_20261121.py'
assert not entry.exists(),'Do not repeat an existing handoff'
shutil.copyfile(ROOT/'python/scripts/evalkit_k4_andy.py',entry)
execution=dict(version='andy-parallel-execution-1',workers=14,
               television_completion=str(B/'test_budget_20261120_v1/completion.json'),
               controller=str(entry),controller_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
               numerical_protocol=str(RUN/'protocol.json'),numerical_policy_changed=False,
               hash_policy='No tree rehash; reuse existing task checkpoints and registered identities')
save(RUN/'execution_parallel_14.json',execution)
save(ROOT/'config/evalkit_k4_andy_head_20261120/execution_parallel_14.json',execution)
before=read(RUN/'progress.json');save(LOG/'usage_before.json',read(RUN/'usage.json'))
record=dict(started_unix_s=time.time(),old_controller=old,old_guard=guard,progress_before={k:before[k] for k in ['state','evaluated','passed','pending']})
os.kill(old,signal.SIGSTOP);terminated=False
try:
    children=[p for f in Path('/proc').iterdir() if f.name.isdigit() for p in [proc(int(f.name))] if p and p['ppid']==old]
    record['checkpoint_children']=children;save(LOG/'handoff.json',record)
    print('Waiting for current stage checkpoint:',children,flush=True)
    while any((current:=proc(p['pid'])) and current['start']==p['start'] and current['state']!='Z' for p in children):time.sleep(2)
    # No numerical child is interrupted. SIGTERM stops the paused controller
    # before it can schedule the next subprocess; the old guard then exits.
    os.kill(old,signal.SIGTERM)
    try:os.kill(old,signal.SIGCONT)
    except ProcessLookupError:pass
    terminated=True
    for _ in range(150):
        if (current:=proc(old)) is None or current['state']=='Z':break
        time.sleep(.1)
    else:raise RuntimeError('Previous controller still active')
    for _ in range(150):
        if 'released_unix_s' in read(RUN/'slot_lease.json'):break
        time.sleep(.1)
    else:raise RuntimeError('Old guard did not finish')
    command=[PY,str(entry),'run','--run',str(RUN),'--execution-config',str(RUN/'execution_parallel_14.json')]
    with (LOG/'run_14.log').open('a') as stream:
        controller=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    record.update(new_controller=controller.pid,switched_unix_s=time.time(),workers=14,current_stage_finished=True,command=command)
    save(LOG/'handoff.json',record);save(RUN/'execution_handoff_20261121.json',record)
    print('Started',controller.pid,'workers 14',flush=True)
finally:
    if not terminated:
        try:os.kill(old,signal.SIGCONT)
        except ProcessLookupError:pass
