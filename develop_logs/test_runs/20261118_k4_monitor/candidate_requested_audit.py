"""Explicit candidate requests; never changes fitting or full-pool audit aggregates."""
import argparse,datetime,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def dump(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def run(root,base,run_root,code,scorer,cache):
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--requests',type=Path,required=True);ap.add_argument('--output',type=Path);ap.add_argument('--cached-only',action='store_true');a=ap.parse_args()
 output=a.output or cache/'requested_review.json'
 if output.exists():raise ValueError('Preserve completed requested review; specify a new --output')
 if output.name=='review.json':raise ValueError('Cannot overwrite full-pool review.json')
 document=read(a.requests)
 if 'requests' in document:requested=document['requests']
 elif 'rows' in document:requested=[dict(run=r['run'],window=r['window'],method=r['method'],candidate=r['chosen']) for r in document['rows'] if r['independent_pass'] is None]
 else:raise ValueError('Expected explicit requests or diagnostic rows with independent_pass=null')
 requests=[r for r in requested if r['run']==run_root.name];keys=[(r['window'],r['method'],r['candidate']) for r in requests]
 if len(keys)!=len(set(keys)):raise ValueError('Duplicate candidate request')
 implementation={}
 for rel in ['utils/motiogen/evalkit/k/scoring.py','utils/motiogen/evalkit/k/strict_acceptance.py','config/evalkit_k4_20261117/strict_v1/acceptance.json']:
  if sha(code/rel)!=sha(scorer/rel):raise RuntimeError('Frozen scoring mismatch: '+rel)
  implementation[str(scorer/rel)]=sha(scorer/rel)
 acceptance_file=scorer/'utils/motiogen/evalkit/k/strict_acceptance.py';spec=importlib.util.spec_from_file_location('requested_frozen_acceptance',acceptance_file);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);assess=module.assess
 policy=read(scorer/'config/evalkit_k4_20261117/strict_v1/acceptance.json');panel={(r['window'],r['method']) for r in read(run_root/'panel.json')['records']}
 env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':str(code/'utils')+':'+str(code/'utils/motiogen')}
 records=[dict(r,status='pending',independent_pass=None,failures=None,reason='not_processed') for r in requests]
 def save():
  result={'timestamp':datetime.datetime.now().isoformat(),'run':run_root.name,'expected_requests':len(requests),'document_requests':len(requested),'ignored_other_run_requests':len(requested)-len(requests),'completed':sum(r['status']=='completed' for r in records),'passed':sum(r.get('independent_pass') is True for r in records),'metric_failed':sum(r.get('independent_pass') is False for r in records),'pending':sum(r['status']!='completed' for r in records),'records':records,'requests_sha256':sha(a.requests),'implementation_sha256':implementation,'entry_sha256':sha(sys.argv[0]),'helper_sha256':sha(__file__),'cached_only':a.cached_only,'scope':'Only explicit requested candidate bodies. Full independent native/export x D/R + actual_export; unknown remains null. No fit or reranking, no full-pool candidate-existence/selection claim. Full review.json untouched.'};dump(output,result)
 save()
 for request,row in zip(requests,records):
  w,m,name=request['window'],request['method'],request['candidate'];row.pop('reason',None)
  try:
   if (w,m) not in panel:row['reason']='not_in_registered_panel';save();continue
   task=run_root/'tasks/pool'/(w+'_'+m);status=task/'status.json'
   if not status.exists() or read(status)['status']!='completed':row['reason']='route_not_completed';save();continue
   fit=run_root/'fits/pool'/w/m;selection=read(fit/'candidate_selection.json')
   if name not in selection['qualities']:row['reason']='candidate_not_successfully_saved';row['fit_failure']=selection.get('failures',{}).get(name);save();continue
   folder=fit/'candidates'/name;dest=cache/(w+'_'+m)/name;auditfile=dest/'audit_summary.json'
   required=['receipt.json','model.osim','coordinates_refined_native.mot','fit.npz','head_neck.npz','head_neck_model.json','coordinates.mot'];missing=[n for n in required if not (folder/n).exists()]
   if missing:row.update(reason='missing_source_files',missing_files=missing);save();continue
   fingerprints={str(folder/n):sha(folder/n) for n in required};row['candidate_source_sha256']=fingerprints
   if auditfile.exists():
    summary=read(auditfile)
    if summary['source_sha256']!=fingerprints:row.update(status='invalid_cache',reason='candidate_source_hash_mismatch');save();continue
    evidence=dict(summary['score_sha256']);evidence[summary['actual_export_audit_file']]=summary['actual_export_audit_sha256']
    changed=[p for p,h in evidence.items() if not Path(p).exists() or sha(p)!=h]
    if changed:row.update(status='invalid_cache',reason='cached_evidence_hash_mismatch',changed_files=changed);save();continue
    row['reused_cache']=True
   elif a.cached_only:row['reason']='missing_audit_cache';save();continue
   elif (cache/'pause_requested').exists():row['reason']='paused_before_candidate';save();continue
   else:
    if dest.exists() and any(dest.iterdir()):row['reason']='incomplete_prior_audit_artifacts_preserved';save();continue
    dest.mkdir(parents=True,exist_ok=True);runtime=read(task/'runtime_job.json')
    def call(args,label):
     with (dest/(label+'.log')).open('w') as log:subprocess.run([sys.executable,*map(str,args)],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
    call([code/'python/scripts/evalkit_k4_native_export_views.py','--code-root',code,'--fit',folder,'--input',runtime['input'],'--method',m,'--window',w,'--output',dest/'views'],'views')
    acceptance={};failures=[];scores={}
    for view in ['native','export']:
     call([code/'python/scripts/evalkit_k4_ablation_score.py','--fits',dest/'views'/view,'--baseline',base/'common_v1','--output',dest/'scores'/view,'--window',w,'--method',m,'--strict'],'score_'+view)
     sf=dest/'scores'/view/(w+'_'+m+'.json');data=read(sf);scores[str(sf)]=sha(sf);acceptance[view]={}
     for clock,score in [('D_clock',data),('R_clock',data['original_clock'])]:
      assessment=assess(score,None,policy);acceptance[view][clock]=assessment['status'];failures.extend(dict(view=view,clock=clock,**x) for x in assessment['required']+assessment['temporal']+assessment['full_window_coverage'] if x['status'] not in ['passed','unobservable'])
    call([code/'python/scripts/evalkit_k4_export_audit.py','--fit',folder,'--output',dest/'actual_export'],'export_audit');export=read(dest/'actual_export/audit.json')
    if any(sha(p)!=h for p,h in fingerprints.items()):raise RuntimeError('Candidate changed during audit')
    summary={'source_sha256':fingerprints,'score_sha256':scores,'acceptance':acceptance,'failures':failures,'actual_export_passed':export['passed'],'actual_export_audit_file':str(dest/'actual_export/audit.json'),'actual_export_audit_sha256':sha(dest/'actual_export/audit.json'),'passed':export['passed'] and all(s=='passed' for clocks in acceptance.values() for s in clocks.values())};dump(auditfile,summary);row['reused_cache']=False
   row.update(status='completed',independent_pass=summary['passed'],failures=summary['failures'],acceptance=summary['acceptance'],actual_export_passed=summary['actual_export_passed'],audit_summary_file=str(auditfile),audit_summary_sha256=sha(auditfile));save()
  except Exception as e:row.update(status='audit_error',reason=str(e));save()
 save();print(json.dumps({'run':run_root.name,'requested':len(requests),'completed':sum(r['status']=='completed' for r in records),'unknown':sum(r['independent_pass'] is None for r in records),'output':str(output)}))
