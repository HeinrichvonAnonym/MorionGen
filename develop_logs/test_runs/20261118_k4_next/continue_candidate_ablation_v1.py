"""Continue the same candidate matrix after read-only existing-artifact review."""
import fcntl,json,os,subprocess,time
from pathlib import Path
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
RUN=B/'full_candidate_ablation_20261118_v1';OLD=306332
lock=(RUN/'.continuation.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
print(json.dumps({'state':'waiting_for_existing_artifact_review','pid':os.getpid(),'previous_pid':OLD}),flush=True)
while Path('/proc',str(OLD)).exists():
 cmd=Path('/proc',str(OLD),'cmdline').read_bytes()
 if b'evalkit_k4_candidate_ablation.py' not in cmd:raise RuntimeError('Old PID was reused')
 time.sleep(5)
with (RUN/'.controller.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
failed=[]
for p in RUN.glob('cells/*/*/state.json'):
 if json.loads(p.read_text())['status']=='execution_failed':failed.append(str(p))
if failed:raise RuntimeError('Preserved execution failures require diagnosis before filling missing fits: '+str(failed))
entry=B/'candidate_ablation_controller_v2_checkout/python/scripts/evalkit_k4_candidate_ablation.py'
args=['/home/heinrich/.local/share/opensim-api/bin/python',str(entry),'run','--run',str(RUN),'--workers','3','--controller-amendment',str(RUN/'controller_progress_amendment.json')]
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
log=RUN/'full_ablation.log'
with log.open('a') as out:p=subprocess.Popen(args,cwd='/home/heinrich/MorionGen',env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'state':'started_full_candidate_fill','pid':p.pid,'workers':3,'command':args,'log':str(log),'time':time.time()}
(RUN/'continuation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
