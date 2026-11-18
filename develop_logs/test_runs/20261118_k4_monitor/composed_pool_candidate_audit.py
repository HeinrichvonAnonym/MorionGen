"""One-shot read-only fit audit: no fitting, runtime choice, or R-based configuration."""
import hashlib,json,os,subprocess,sys,datetime
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen')
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
RUN=B/'closure_composed_pool_20261118_v1';CODE=B/'closure_composed_pool_v1_checkout';SCORER=B/'candidate_pool_v1_checkout'
OUT=B/'monitor_cache_20261118/composed_pool_candidates'
if '--requests' in sys.argv:
 from candidate_requested_audit import run as requested_audit
 requested_audit(ROOT,B,RUN,CODE,SCORER,OUT)
 raise SystemExit(0)
OUT.mkdir(parents=True,exist_ok=True)
sys.path[:0]=[str(SCORER/'utils'),str(SCORER/'utils/motiogen')]
from motiogen.evalkit.k.strict_acceptance import assess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
policy=read(SCORER/'config/evalkit_k4_20261117/strict_v1/acceptance.json')
for rel in ['utils/motiogen/evalkit/k/scoring.py','utils/motiogen/evalkit/k/strict_acceptance.py','config/evalkit_k4_20261117/strict_v1/acceptance.json']:
 if sha(CODE/rel)!=sha(SCORER/rel):raise RuntimeError('Frozen scoring mismatch: '+rel)
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(CODE/'utils')+':'+str(CODE/'utils/motiogen')}
rows=[];panel=read(RUN/'panel.json')['records']
for entry in panel:
 w,m=entry['window'],entry['method'];key=w+'_'+m;task=RUN/'tasks/pool'/key;status=task/'status.json'
 if not status.exists() or read(status)['status']!='completed':continue
 fit=RUN/'fits/pool'/w/m;selection=read(fit/'candidate_selection.json');runtime=read(task/'runtime_job.json');row={'window':w,'method':m,'selected':selection['selected'],'selection_file':str(fit/'candidate_selection.json'),'selection_sha256':sha(fit/'candidate_selection.json'),'fit_failures':selection['failures'],'inapplicable':selection['inapplicable'],'candidates':{}}
 for name in selection['qualities']:
  if (OUT/'pause_requested').exists():
   print('Paused at candidate boundary; cached evidence retained.',flush=True);sys.exit(0)
  folder=fit/'candidates'/name;dest=OUT/key/name;dest.mkdir(parents=True,exist_ok=True)
  fingerprints={str(folder/n):sha(folder/n) for n in ['receipt.json','model.osim','coordinates_refined_native.mot','fit.npz','head_neck.npz','head_neck_model.json','coordinates.mot']}
  auditfile=dest/'audit_summary.json'
  if auditfile.exists():
   summary=read(auditfile)
   if summary['source_sha256']!=fingerprints:raise RuntimeError('Candidate changed after audit: '+str(folder))
  else:
   def call(args,label):
    with (dest/(label+'.log')).open('w') as log:subprocess.run([sys.executable,*map(str,args)],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
   call([CODE/'python/scripts/evalkit_k4_native_export_views.py','--code-root',CODE,'--fit',folder,'--input',runtime['input'],'--method',m,'--window',w,'--output',dest/'views'],'views')
   acceptance={};failures=[];scores={}
   for view in ['native','export']:
    call([CODE/'python/scripts/evalkit_k4_ablation_score.py','--fits',dest/'views'/view,'--baseline',B/'common_v1','--output',dest/'scores'/view,'--window',w,'--method',m,'--strict'],'score_'+view)
    sf=dest/'scores'/view/(key+'.json');s=read(sf);scores[str(sf)]=sha(sf);acceptance[view]={}
    for clock,data in [('D_clock',s),('R_clock',s['original_clock'])]:
     a=assess(data,None,policy);acceptance[view][clock]=a['status']
     failures.extend(dict(view=view,clock=clock,**x) for x in a['required']+a['temporal']+a['full_window_coverage'] if x['status'] not in ['passed','unobservable'])
   call([CODE/'python/scripts/evalkit_k4_export_audit.py','--fit',folder,'--output',dest/'actual_export'],'export_audit')
   export=read(dest/'actual_export/audit.json')
   summary={'source_sha256':fingerprints,'score_sha256':scores,'acceptance':acceptance,'failures':failures,'actual_export_passed':export['passed'],'actual_export_audit_file':str(dest/'actual_export/audit.json'),'actual_export_audit_sha256':sha(dest/'actual_export/audit.json'),'passed':export['passed'] and all(s=='passed' for c in acceptance.values() for s in c.values())}
   dump(auditfile,summary)
  row['candidates'][name]=summary
 runtime_export=read(task/'actual_export/audit.json');runtime_acceptance={};runtime_failures=[];runtime_hashes={str(task/'actual_export/audit.json'):sha(task/'actual_export/audit.json')}
 for view in ['native','export']:
  sf=RUN/'scores/pool'/view/(key+'.json');data=read(sf);runtime_hashes[str(sf)]=sha(sf);runtime_acceptance[view]={}
  for clock,data_clock in [('D_clock',data),('R_clock',data['original_clock'])]:
   assessment=assess(data_clock,None,policy);runtime_acceptance[view][clock]=assessment['status']
   runtime_failures.extend(dict(view=view,clock=clock,**x) for x in assessment['required']+assessment['temporal']+assessment['full_window_coverage'] if x['status'] not in ['passed','unobservable'])
 row['actual_runtime_output']={'acceptance':runtime_acceptance,'failures':runtime_failures,'export_audit_passed':runtime_export['passed'],'source_sha256':runtime_hashes}
 row['any_candidate_passed']=any(x['passed'] for x in row['candidates'].values());row['selected_candidate_passed']=row['candidates'][row['selected']]['passed']
 row['runtime_selected_passed']=runtime_export['passed'] and all(s=='passed' for c in runtime_acceptance.values() for s in c.values())
 row['selected_candidate_runtime_qualification_agree']=row['selected_candidate_passed']==row['runtime_selected_passed'];row['missed_passing_candidate']=row['any_candidate_passed'] and not row['runtime_selected_passed']
 rows.append(row)
 result={'timestamp':datetime.datetime.now().isoformat(),'expected':len(panel),'reviewed':len(rows),'any_candidate_passed':sum(r['any_candidate_passed'] for r in rows),'runtime_selected_passed':sum(r['runtime_selected_passed'] for r in rows),'missed_passing_candidate':sum(r['missed_passing_candidate'] for r in rows),'records':rows,'policy_sha256':sha(SCORER/'config/evalkit_k4_20261117/strict_v1/acceptance.json'),'script_sha256':sha(__file__),'scope':'All successfully saved applicable candidates; original-R independent native/export x D/R plus actual_export. Actual frozen runtime choice read unchanged; no fitting and no R-driven choice.'}
 dump(OUT/'review.json',result)
 print(key,'candidates',len(row['candidates']),'any_pass',row['any_candidate_passed'],'selected_pass',row['runtime_selected_passed'],'missed',row['missed_passing_candidate'],flush=True)
if not rows:print('No completed pool route yet; nothing scored.')
