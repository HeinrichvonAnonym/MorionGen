"""Explicit one-off lease approved by root; finally resumes waiting full controller."""
import datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen');O=ROOT/'develop_logs/test_runs/20261118_k4_monitor';B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');FULL=B/'full_regression_20261118_unified_v2';PID=288821;PROC=Path('/proc')/str(PID);OUT=O/'requested_audit_lease_v1.json'
def stamp():return datetime.datetime.now().astimezone().isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def state():return next(s for s in (PROC/'status').read_text().splitlines() if s.startswith('State:'))
def numerical():
 rows=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   args=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
   if not args or not args[0].endswith('/python'):continue
   names=('evalkit_k4_run.py','evalkit_k4_compose_bank.py','evalkit_k4_native_export_views.py','evalkit_k4_ablation_score.py','evalkit_k4_export_audit.py','evalkit_k4_trajectory_replay.py')
   scripts=[Path(x).name for x in args if x.endswith(names)]
   if scripts:rows.append({'pid':int(p.name),'script':scripts[0]})
  except OSError:pass
 return rows
record={'started_at':stamp(),'controller_pid':PID,'scope':'Pause only waiting full-v2 controller; existing fitting children untouched. Sequential explicit candidate audit, no fitting.'};paused=False
full_reviews=[O/'pool_candidates/review.json',O/'composed_pool_candidates/review.json'];record['full_review_before_sha256']={str(p):sha(p) for p in full_reviews}
def save():
 temp=OUT.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(OUT)
try:
 progress=json.loads((FULL/'progress.json').read_text());record['progress_before']={k:v for k,v in progress.items() if k!='results'};record['process_state_before']=state();record['process_starttime']=(PROC/'stat').read_text().split()[21];record['controller_cmdline_sha256']=sha(PROC/'cmdline');record['children_before']=(PROC/'task'/str(PID)/'children').read_text().split();record['numerical_before']=numerical()
 if progress['state']!='waiting_for_registered_runs' or record['children_before']:raise RuntimeError('Controller no longer safely waiting without children')
 if len(record['numerical_before'])>=8:raise RuntimeError('No numerical slot available')
 if 'evalkit_k4_full_regression.py' not in (PROC/'cmdline').read_bytes().decode():raise RuntimeError('Controller PID changed')
 os.kill(PID,signal.SIGSTOP);paused=True;record['stop_sent_at']=stamp()
 deadline=time.monotonic()+2
 while 'T (' not in state() and time.monotonic()<deadline:time.sleep(.02)
 record['process_state_stopped']=state();record['children_after_stop']=(PROC/'task'/str(PID)/'children').read_text().split()
 if 'T (' not in record['process_state_stopped'] or record['children_after_stop']:raise RuntimeError('Controller did not stop safely before scheduling')
 record['commands']=[];save();print('Waiting full controller paused; existing numerical children:',len(record['numerical_before']),flush=True)
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 for script in ['pool_candidate_audit.py','composed_pool_candidate_audit.py']:
  active=numerical()
  if len(active)>=8:raise RuntimeError('Numerical capacity changed before requested audit')
  command=[sys.executable,str(O/script),'--requests',str(O/'candidate_budget_requests_v1.json')];item={'script':script,'started_at':stamp(),'numerical_before':active};record['commands'].append(item);save();result=subprocess.run(command,cwd=ROOT,env=env);item.update(finished_at=stamp(),returncode=result.returncode);save()
  if result.returncode:raise RuntimeError('Requested audit entry failed: '+script)
 record['requested_outputs']={}
 for folder in ['pool_candidates','composed_pool_candidates']:
  p=O/folder/'requested_review.json';d=json.loads(p.read_text());record['requested_outputs'][str(p)]={'sha256':sha(p),'expected':d['expected_requests'],'completed':d['completed'],'passed':d['passed'],'metric_failed':d['metric_failed'],'pending':d['pending']}
 record['full_review_after_sha256']={str(p):sha(p) for p in full_reviews};record['full_reviews_unchanged']=record['full_review_before_sha256']==record['full_review_after_sha256'];record['status']='completed'
 if not record['full_reviews_unchanged']:raise RuntimeError('Full-pool review changed during request lease')
except BaseException as e:
 record.update(status='failed',error=repr(e));raise
finally:
 if paused:
  try:
   if (PROC/'stat').read_text().split()[21]!=record['process_starttime']:raise RuntimeError('Controller PID was reused; do not signal unrelated process')
   os.kill(PID,signal.SIGCONT);record['continue_sent_at']=stamp();record['resumed']=True
  except BaseException as e:record.update(resumed=False,resume_error=repr(e))
 record['finished_at']=stamp();save();print('Lease finished; waiting controller resumed:',record.get('resumed'),flush=True)
