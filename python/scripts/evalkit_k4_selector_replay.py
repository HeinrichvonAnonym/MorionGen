"""Read-only replay of frozen D-only selectors; independent cached R audit follows selection."""
import argparse, fcntl, hashlib, json, os, shutil, statistics, subprocess, sys, time
from pathlib import Path

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def worker(job_path):
 start=time.perf_counter();j=read(job_path);os.nice(10)
 sys.path[:0]=[str(Path(j['code_root'])/'utils'),str(Path(j['code_root'])/'utils/motiogen')]
 import numpy as np
 from motiogen.evalkit.k.motion import load
 from motiogen.evalkit.k.candidate_quality import assess,choose
 from motiogen.evalkit.k.candidate_budget import accepts
 for p,h in j['input_sha256'].items():
  if sha(p)!=h:raise ValueError('D identity changed: '+p)
 motion=load(j['input'],j['method'],config=Path(j['code_root'])/'config/evalkit_k4_20261117/input.json')
 loaded=time.perf_counter();qualities={};timings={}
 for n,c in j['candidates'].items():
  t=time.perf_counter();folder=Path(c['fit'])
  for p,h in c['sha256'].items():
   if sha(p)!=h:raise ValueError('Fit identity changed: '+p)
  with np.load(folder/'fit.npz',allow_pickle=False) as z:values={k:z[k] for k in z.files}
  head=None
  if (folder/'head_neck.npz').exists():
   with np.load(folder/'head_neck.npz',allow_pickle=False) as z:head=z['head_relative_thorax']
  qualities[n]=assess(motion,values,head_rotation=head,thresholds=j['settings']['selection_thresholds'],include_export_path=j['settings']['selection_include_export_path'],ranking_policy=j['settings']['selection_policy'])
  timings[n]=time.perf_counter()-t
 full=choose(qualities);visited=[];early=None
 for n,q in qualities.items():
  visited.append(n)
  if accepts(q,j['settings']['candidate_budget']):early=n;break
 budget=early or full
 write(j['output'],dict(status='completed',job_sha256=sha(job_path),full_selected=full,budget_selected=budget,early_accepted=early,visited=visited,available=len(qualities),qualities=qualities,timing=dict(load_D_s=loaded-start,candidate_s=timings,total_s=time.perf_counter()-start),scope='Frozen D-only rules, complete candidate bank; no R/score fields in this request; no fitting'))

def allocation():
 # Reserve configured capacity, not transient idle gaps, to avoid launch races.
 cap=0;active=0
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:a=(p/'cmdline').read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if any(x.endswith('evalkit_k4_candidate_ablation.py') for x in a) and 'run' in a and '--workers' in a:cap+=int(a[a.index('--workers')+1])
  if any(x.endswith(('evalkit_k4_run.py','evalkit_k4_candidate_ablation.py','evalkit_k4_ablation_score.py','evalkit_k4_export_audit.py')) for x in a) and ('--job' in a or 'variant' in a or '--fits' in a or '--fit' in a):active+=1
 return max(cap,active),active

def prepare(route,p,run,out,settings):
 key=route['window']+'_'+route['method'];items={};states={}
 for n,c in route['cells'].items():
  if c['status']=='inapplicable' or c.get('alias'):continue
  path=run/'cells'/key/n/'state.json'
  if not path.exists():return None
  s=read(path)
  if s['status']!='evaluated':return None
  f=Path(s['fit']);checks={str(f/k):v for k,v in s['source_sha256'].items()};checks[str(f/'receipt.json')]=s['receipt_sha256']
  if (f/'head_neck.npz').exists():checks[str(f/'head_neck.npz')]=sha(f/'head_neck.npz')
  items[n]={'fit':str(f),'sha256':checks};states[n]={'file':str(path),'sha256':sha(path)}
 if not items:return None
 payload=dict(code_root=p['code_root'],input=route['input'],method=route['method'],input_sha256=route['expected_key']['inputs'],settings=settings,candidates=items,output=str(out/'routes'/key/'selection.json'))
 job=out/'jobs'/(key+'.json');write(job,payload);write(out/'audit_sources'/(key+'.json'),states)
 return job

def audit(key,out,frozen):
 result=read(out/'routes'/key/'selection.json');src=read(out/'audit_sources'/(key+'.json'));states={}
 for n,s in src.items():
  if sha(s['file'])!=s['sha256']:raise ValueError('Cached audit changed')
  d=read(s['file'])
  for path,h in {**d['score_sha256'],d['actual_export_audit_file']:d['actual_export_audit_sha256']}.items():
   if sha(path)!=h:raise ValueError('Audit artifact changed')
  states[n]=d
 passing=[n for n,d in states.items() if d['passed']];row={'key':key,'available':len(states),'passing_candidates':passing,'candidate_set_has_pass':bool(passing),'timing_s':result['timing']['total_s']}
 for label in ['full','budget']:
  n=result[label+'_selected'];s=states[n];row[label]={'selected':n,'passed':s['passed'],'missed_passing':bool(passing) and not s['passed'],'acceptance':s['acceptance'],'failures':s['failures'],'score_files':s['score_sha256']}
 row['early_stop_loss']=row['full']['passed'] and not row['budget']['passed'];row['visited']=len(result['visited'])
 old=frozen/'tasks'/key/'result.json'
 if old.exists() and read(old).get('status')=='completed':
  old=read(old);row['historical_selection']=old.get('selected');row['historical_budget_match']=old.get('selected')==result['budget_selected']
 write(out/'routes'/key/'audit.json',row);return row

def run(a):
 out=a.output;out.mkdir(parents=True,exist_ok=True);lock=(out/'.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 p=read(a.run/'protocol.json');frozen=Path(p['frozen_run']);fp=read(frozen/'protocol.json');cfg=read(fp['fit_config']);settings={k:cfg[k] for k in ['selection_thresholds','selection_include_export_path','selection_policy','candidate_budget']}
 if sha(fp['fit_config'])!=fp['fit_config_sha256']:raise ValueError('Frozen config changed')
 for rel,h in p['numerical_implementation'].items():
  if sha(Path(p['code_root'])/'utils/motiogen/evalkit/k'/rel)!=h:raise ValueError('Frozen implementation changed')
 protected=read(p['preservation_manifest'])['files']
 if any(sha(Path(p['workspace_root'])/n)!=h for n,h in protected.items()):raise ValueError('Protected file changed')
 identity={'version':'D-only-replay-1','script_sha256':sha(__file__),'protocol_sha256':sha(a.run/'protocol.json'),'settings':settings,'fit_code_root':p['code_root'],'expected':656,'protected_checked':len(protected),'new_holdout_uses':0}
 ip=out/'identity.json'
 if ip.exists() and read(ip)!=identity:raise ValueError('Replay identity changed')
 write(ip,identity);running={};rows={};errors={};start=time.time()
 for f in (out/'routes').glob('*/audit.json') if (out/'routes').exists() else []:rows[f.parent.name]=read(f)
 while True:
  for key,(proc,log,t0) in list(running.items()):
   if proc.poll() is None:continue
   log.close();del running[key]
   if proc.returncode:errors[key]={'returncode':proc.returncode};continue
   try:rows[key]=audit(key,out,frozen)
   except Exception as e:errors[key]={'audit_error':str(e)}
  progress=read(a.run/'progress.json');done=progress['fully_evaluated_routes']==656
  reserved,active=allocation();ceiling=14 if done else (1 if len(rows)<3 else 4);slots=max(0,min(ceiling,14-reserved))
  mem=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
  resource_ok=mem>=8*1024**3 and shutil.disk_usage(out).free>=80*1024**3
  if resource_ok:
   for r in p['routes']:
    if len(running)>=slots:break
    key=r['window']+'_'+r['method']
    if key in rows or key in running or key in errors:continue
    job=prepare(r,p,a.run,out,settings)
    if job is None:continue
    (out/'routes'/key).mkdir(parents=True,exist_ok=True);log=(out/'routes'/key/'worker.log').open('a');env={**os.environ,**{k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}
    proc=subprocess.Popen([sys.executable,__file__,'worker','--job',str(job)],stdout=log,stderr=subprocess.STDOUT,env=env);running[key]=(proc,log,time.time())
  summary={'expected':656,'completed':len(rows),'execution_or_integrity_errors':errors,'active':{k:v[0].pid for k,v in running.items()},'reserved_ablation_capacity':reserved,'replay_limit':slots,'ablation_complete':done,'elapsed_s':time.time()-start,'median_route_s':statistics.median(r['timing_s'] for r in rows.values()) if rows else None,'full_passed':sum(r['full']['passed'] for r in rows.values()),'budget_passed':sum(r['budget']['passed'] for r in rows.values()),'full_missed_passing':sum(r['full']['missed_passing'] for r in rows.values()),'budget_missed_passing':sum(r['budget']['missed_passing'] for r in rows.values()),'historical_mismatches':[k for k,r in rows.items() if r.get('historical_budget_match') is False],'updated_unix_s':time.time()}
  write(out/'progress.json',summary)
  if len(rows)+len(errors)==656 and not running:write(out/'completion.json',summary);break
  if a.once and not running:break
  time.sleep(3)

if __name__=='__main__':
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['run','worker']);ap.add_argument('--run',type=Path);ap.add_argument('--output',type=Path);ap.add_argument('--job',type=Path);ap.add_argument('--once',action='store_true');args=ap.parse_args()
 if args.action=='worker':worker(args.job)
 else:run(args)
