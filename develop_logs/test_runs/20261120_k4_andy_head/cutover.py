"""One-time registered cutover; no protection/input/source tree hash scans."""
import hashlib,json,os,signal,shutil,subprocess,time
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen')
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
OLD=B/'andy_head_test_20261120_v1';NEW=B/'andy_head_test_20261120_v2'
LOG=ROOT/'develop_logs/test_runs/20261120_k4_andy_head'
PY='/home/heinrich/.local/share/opensim-api/bin/python'
def read(p):return json.loads(p.read_text())
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
verification=read(LOG/'verification.json');assert len(verification['checks'])==2
assert all(x['early'] and x['missing_rejected'] and x['export_jump_rejected'] for x in verification['checks'])
assert not NEW.exists(),'Use existing cutover, do not repeat'
NEW.mkdir();control=NEW/'control';control.mkdir()
for name in ('evalkit_k4_andy.py','evalkit_k4_head_run.py','evalkit_k4_slot_lease.py'):
    shutil.copyfile(ROOT/'python/scripts'/name,control/name)
shutil.copyfile(ROOT/'utils/motiogen/evalkit/k/head_task_selection.py',control/'head_task_selection.py')
shutil.copyfile(ROOT/'config/evalkit_k4_andy_head_20261120/pool.json',NEW/'pool.json')
old=read(OLD/'protocol.json');tv=read(B/'test_budget_20261120_v1/protocol.json')
save(NEW/'registered_numerical_identity.json',dict(identity=tv['numerical_identity'],policy='reuse registered frozen evidence, no rehash'))
lease=read(OLD/'slot_lease.json');guard=lease['guard_pid']
assert str(OLD) in Path(f'/proc/{guard}/cmdline').read_bytes().replace(b'\0',b' ').decode()
before=read(OLD/'progress.json');at=time.time()
os.kill(guard,signal.SIGTERM)
for _ in range(200):
    ended=read(OLD/'slot_lease.json')
    if 'released_unix_s' in ended:break
    time.sleep(.1)
else:raise RuntimeError('Old slot guard did not release; no new controller started')
# The old process group has terminated before this file-level snapshot.
completed=[]
for row in old['records']:
    w=row['window_key']
    for method in old['methods']:
        key=w+'_'+method;source=OLD/'tasks'/key/'result.json'
        if source.exists():
            value=read(source);value.update(selector_version='whole_body_v1',origin=str(source))
            save(NEW/'tasks'/key/'result.json',value);completed.append(key)
    for folder in ('prepared','runtime_inputs'):
        source=OLD/folder/w
        if source.exists():
            target=NEW/folder/w;target.parent.mkdir(exist_ok=True);target.symlink_to(source,target_is_directory=True)
pending=next((row['window_key']+'_'+m for row in old['records'] for m in old['methods'] if row['window_key']+'_'+m not in completed),None)
record=dict(cutover_unix_s=at,old_run=str(OLD),new_run=str(NEW),old_progress=before['state'],resume_task=pending,legacy_completed=completed,old_slot_released=ended['tv_slot_resumed'],hash_policy='no tree hash scans; known receipts and registered identities reused; only new control files fingerprinted',new_control_sha256={str(f):digest(f) for f in control.iterdir()})
save(OLD/'interruption_for_head_selector_v2.json',record);save(NEW/'cutover.json',record);save(LOG/'cutover.json',record)
shutil.copyfile(ROOT/'docs/20261120_k4_andy_test.html',OLD/'report_at_cutover.html')
p=dict(old,version='andy-head-optical-k4-2',run=str(NEW),selection_scope='optical_head_task_d_only_v1',
       previous_run=str(OLD),legacy_completed=completed,resume_without_tree_rehash=True,
       fit_runner=str(control/'evalkit_k4_head_run.py'),selector_module=str(control/'head_task_selection.py'),
       registered_identity=str(NEW/'registered_numerical_identity.json'),fit_config=str(NEW/'pool.json'),
       selector='Head-task D-only early accept; original margin/order/candidates; template body diagnostic only',
       implementation_transition=record)
save(NEW/'protocol.json',p)
cmd=[PY,str(control/'evalkit_k4_andy.py'),'run','--run',str(NEW),'--lease',str(NEW/'slot_lease.json')]
save(NEW/'launch_command.json',cmd)
with (LOG/'run.log').open('a') as log:
    process=subprocess.Popen([PY,str(control/'evalkit_k4_slot_lease.py'),'--tv-controller',str(lease['tv_controller']['pid']),'--tv-run',str(B/'test_budget_20261120_v1'),'--command',str(NEW/'launch_command.json'),'--receipt',str(NEW/'slot_lease.json')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
save(LOG/'launch.json',dict(guard_pid=process.pid,new_run=str(NEW),resume_task=pending))
save(ROOT/'config/evalkit_k4_andy_head_20261120/registration.json',dict(run=str(NEW),previous_run=str(OLD),resume_task=pending,legacy_completed=len(completed),selector_scope=p['selection_scope'],thresholds=read(NEW/'pool.json')['selection_thresholds'],margin=.8,code_identity=record['new_control_sha256'],hash_policy=record['hash_policy'],development_verification=str(LOG/'verification.json')))
print(json.dumps(dict(guard_pid=process.pid,resume_task=pending,legacy_completed=len(completed))),flush=True)
