"""One-shot, explicit stage-boundary scheduler handoff; never interrupts fitting."""
import argparse,datetime,fcntl,hashlib,json,os,signal,subprocess,time,shutil
from pathlib import Path
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
R=Path('/home/heinrich/MorionGen');RUN=B/'full_regression_20261118_budget_v3';CODE=B/'closure_budget_v3_checkout';PY='/home/heinrich/.local/share/opensim-api/bin/python'
NEW_RUNNER=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117/scheduler_workers14_20261118/evalkit_k4_full_regression.py')
NEW_RUNNER_SHA='fbfc2f049960ede836f6692c96e8ace91f0651add0e733816b192b1fcf3bf15e'
PROTOCOL=R/'config/evalkit_k4_20261118/full_regression_budget_v3/protocol.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def processes():
 out={}
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   s=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes().decode().split('\0');out[int(p.name)]={'ppid':int(s[1]),'state':s[0],'cmd':cmd}
  except (OSError,ValueError):continue
 return out
def descendants(procs,pid):
 found=set();todo=[pid]
 while todo:
  children={n for n,v in procs.items() if v['ppid'] in todo};children-=found;found|=children;todo=list(children)
 return {n:procs[n] for n in found}
def inspect(pid,stage):
 procs=processes();parent=procs.get(pid);assert parent and str(CODE/'python/scripts/evalkit_k4_full_regression.py') in parent['cmd'],'Controller identity mismatch'
 p=RUN/'stages'/f'{stage}.json';assert p.exists(),'Stage file not yet saved';d=json.loads(p.read_text());assert d['passed'] and all(x['status']=='completed' and x['passed'] for x in d['results']),'Stage did not pass'
 stagekeys={f"{x['window']}_{x['method']}" for x in d['results']}
 tasks=list((RUN/'tasks').glob('*/state.json'));assert all((p.parent/'result.json').exists() for p in tasks),'Active/interrupted task exists'
 prog=json.loads((RUN/'progress.json').read_text());assert prog['stage']==stage,'Next stage started'
 children=descendants(procs,pid)
 assert all(str(CODE/'python/scripts/evalkit_k4_full_report.py') in c['cmd'] and '--record' not in c['cmd'] for c in children.values()),'Numerical or unknown child exists'
 return {'parent':parent,'children':children,'stage':stage,'stage_count':len(stagekeys),'progress':{k:prog[k] for k in ['stage','finished','completed','passed','failed']}}
def main():
 a=argparse.ArgumentParser();a.add_argument('--pid',type=int,default=301370);a.add_argument('--stage',default='format_completion38');a.add_argument('--execute',action='store_true');a.add_argument('--wait',action='store_true');args=a.parse_args()
 waiter_lock=(B/'monitor_cache_20261118/handoff_workers12_wait.lock').open('a')
 fcntl.flock(waiter_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 print(json.dumps({'event':'waiting_started','pid':os.getpid(),'target_pid':args.pid,'stage':args.stage,'time':datetime.datetime.now().isoformat()}),flush=True)
 assert not (B/'monitor_cache_20261118/handoff_hold').exists(),'Handoff held by parent for short numerical test'
 if args.wait:
  while True:
   try:inspect(args.pid,args.stage);break
   except AssertionError:
    if args.pid not in processes():raise RuntimeError('Controller exited before safe handoff')
    progress=json.loads((RUN/'progress.json').read_text())
    if progress.get('state') in ('validation_stage_failed','completed'):raise RuntimeError('Stage gate stopped or run completed; preserve status')
    if progress.get('stage')=='full_remaining':raise RuntimeError('Safe format_completion38 boundary missed; full_remaining already started, leave fitting untouched')
    if (RUN/'stage_failure.json').exists():raise RuntimeError('Stage failure preserved; do not cross gate')
    time.sleep(5)
 assert sha(NEW_RUNNER)==NEW_RUNNER_SHA,'New scheduler changed'
 initial=inspect(args.pid,args.stage);print(json.dumps(initial),flush=True)
 if not args.execute:return
 out=B/'monitor_cache_20261118'/('budget_worker_handoff_'+datetime.datetime.now().strftime('%H%M%S'));out.mkdir();record={'started':datetime.datetime.now().isoformat(),'initial':initial,'status':'preparing'}
 stopped=False;terminated=False
 def save(): (out/'handoff.json').write_text(json.dumps(record,indent=2)+'\n')
 try:
  os.kill(args.pid,signal.SIGSTOP);stopped=True
  for _ in range(100):
   if processes().get(args.pid,{}).get('state')=='T':break
   time.sleep(.01)
  assert processes().get(args.pid,{}).get('state')=='T','STOP not confirmed'
  record['stopped_check']=inspect(args.pid,args.stage)
  manifest=R/'config/evalkit_k4_20261117/preservation.json';protected=json.loads(manifest.read_text())['files'];actual={n:sha(R/n) for n in protected};assert actual==protected,'Protected file changed'
  paths=[PROTOCOL,RUN/'protocol.json',RUN/'identity.json',manifest,*list((RUN/'stages').glob('*.json')),*list((RUN/'tasks').glob('*/result.json')),*list((RUN/'tasks').glob('*/summary.json')),*list((RUN/'scores').glob('*/*.json'))]
  before={str(p):sha(p) for p in paths};record.update(protected_actual_sha256=actual,preserved_sha256=before,new_holdout_uses=0)
  ident=json.loads((RUN/'identity.json').read_text());assert sha(PROTOCOL)==ident['protocol_sha256'] and sha(CODE/'python/scripts/evalkit_k4_full_regression.py')==ident['runner_sha256'],'Identity changed'
  assert shutil.disk_usage(RUN).free>=80*1024**3,'Insufficient disk reserve'
  amendment={'version':'k4-execution-amendment-2','protocol_sha256':ident['protocol_sha256'],'original_runner_sha256':ident['runner_sha256'],'output':str(RUN),'max_workers':13,'total_worker_limit':14,'reserved_workers':1,'authorization':'User authorized fourteen total workers. Thirteen regression plus one reserved numerical worker at a safe passed-stage boundary. Scientific protocol unchanged.'}
  (out/'execution_config.json').write_text(json.dumps(amendment,indent=2)+'\n');save()
  childpids=list(record['stopped_check']['children']);os.kill(args.pid,signal.SIGTERM);os.kill(args.pid,signal.SIGCONT);stopped=False;terminated=True
  deadline=time.monotonic()+55
  while time.monotonic()<deadline:
   procs=processes()
   if all(n not in procs or procs[n]['state']=='Z' for n in [args.pid,*childpids]):break
   time.sleep(.1)
  else:raise RuntimeError('Old controller/report exit timeout; do not restart yet')
  with (RUN/'.controller.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert all(sha(p)==h for p,h in before.items()),'Preserved protocol/result/score hashes changed'
  env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
  with (out/'resume.log').open('w') as log:
   proc=subprocess.Popen([PY,str(NEW_RUNNER),'--protocol',str(PROTOCOL),'--execution-config',str(out/'execution_config.json')],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  record.update(status='resumed',new_controller_pid=proc.pid,ended=datetime.datetime.now().isoformat(),preserved_hashes_equal=True);save();print(json.dumps({'evidence':str(out),'new_pid':proc.pid}),flush=True)
 except Exception as e:
  record.update(status='aborted_after_termination' if terminated else 'handoff_cancelled',error=str(e));save();raise
 finally:
  if stopped:os.kill(args.pid,signal.SIGCONT)
if __name__=='__main__':main()
