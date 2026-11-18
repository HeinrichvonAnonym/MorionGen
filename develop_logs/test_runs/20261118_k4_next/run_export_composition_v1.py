import sys,json,hashlib,os,subprocess,shutil
from pathlib import Path
CODE=Path(sys.argv[1]);B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');SCORER=B/'candidate_pool_v1_checkout';CFG=CODE/'config/evalkit_k4_20261118/closure_export_composition_v1';OUT=B/'closure_export_composition_20261118_v1'
OUT.mkdir(exist_ok=False);p=json.loads((CFG/'protocol.json').read_text());shutil.copyfile(CFG/'protocol.json',OUT/'protocol.json');(OUT/'panel.json').write_text(json.dumps({'records':[{k:r[k] for k in ('window','method')} for r in p['records']]},indent=2))
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
(OUT/'identity.json').write_text(json.dumps({'code_commit':subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD']).decode().strip(),'configuration_sha256':{f.name:sha(f) for f in CFG.glob('*.json')},'workers':1,'new_holdout_uses':0},indent=2))
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};results=[]
for r in p['records']:
 w,m=r['window'],r['method'];source=Path(r['source_fit'])
 for condition in p['conditions']:
  folder=OUT/'tasks'/condition/(w+'_'+m);folder.mkdir(parents=True);result=dict(window=w,method=m,condition=condition,status='failed')
  def call(script,args,log):
   with (folder/(log+'.log')).open('w') as stream:subprocess.run([sys.executable,str(script),*map(str,args)],check=True,stdout=stream,stderr=subprocess.STDOUT,env=env)
  try:
   if shutil.disk_usage(OUT).free<80*1024**3:raise RuntimeError('80GiB reserve required')
   assert all(sha(source/n)==d for n,d in r['source_sha256'].items())
   fit=source if condition=='control' else OUT/'fits'/condition/w/m
   if condition=='composed':call(CODE/'python/scripts/evalkit_k4_export_variant.py',['--input',r['input'],'--fit',source,'--method',m,'--policy',CFG/'path.json','--output',fit],'compose')
   views=OUT/'views'/condition/w/m
   call(CODE/'python/scripts/evalkit_k4_native_export_views.py',['--code-root',CODE,'--fit',fit,'--input',r['input'],'--method',m,'--window',w,'--output',views],'views')
   for view in ('native','export'):
    call(SCORER/'python/scripts/evalkit_k4_ablation_score.py',['--fits',views/view,'--baseline',B/'common_v1','--output',OUT/'scores'/condition/view,'--window',w,'--method',m,'--strict'],'score_'+view)
   call(CODE/'python/scripts/evalkit_k4_export_audit.py',['--fit',fit,'--output',folder/'actual_export'],'actual_export')
   assert all(sha(source/n)==d for n,d in r['source_sha256'].items())
   result['status']='completed'
  except Exception as e:
   import traceback
   result.update(error=str(e),traceback=traceback.format_exc())
  (folder/'status.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
  (OUT/'progress.json').write_text(json.dumps(dict(expected=4,results=results),indent=2)+'\n');print(result,flush=True)
(OUT/'completion.json').write_text(json.dumps(dict(expected=4,results=results,promotion='none'),indent=2)+'\n')
