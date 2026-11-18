"""Safe candidate-boundary controller handoff; numerical implementation remains frozen."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
run=B/'full_candidate_ablation_20261118_v1'
lock=(run/'.handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old_pid=307857
while Path(f'/proc/{old_pid}').exists():
    time.sleep(2)
# The old pool has joined every running candidate before process exit.
controller_lock=(run/'.controller.lock').open('a');fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
selector_lock=(B/'full_regression_20261118_budget_v3/.controller.lock').open('a');fcntl.flock(selector_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
progress=json.loads((B/'full_regression_20261118_budget_v3/progress.json').read_text())
assert progress['state']=='validation_stage_failed'
assert (run/'pause_requested').exists()
(run/'pause_requested').rename(run/f'pause_ack_v3_{time.time_ns()}.txt')
command=[sys.executable,str(B/'candidate_ablation_controller_v3_checkout/python/scripts/evalkit_k4_candidate_ablation.py'),'run','--run',str(run),'--workers','14','--independent-after-selector-stop','--controller-amendment',str(run/'controller_scheduling_amendment_v3.json')]
fcntl.flock(controller_lock,fcntl.LOCK_UN);fcntl.flock(selector_lock,fcntl.LOCK_UN)
with (run/'full_ablation_v3_workers14.log').open('a') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    record=dict(pid=child.pid,previous_pid=old_pid,workers=14,started_unix_s=time.time(),command=command,scope='Independent ablation; original selector gate preserved')
    (run/'handoff_v3.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    time.sleep(3)
    if child.poll() is not None:raise RuntimeError(f'Controller exited early: {child.returncode}')
