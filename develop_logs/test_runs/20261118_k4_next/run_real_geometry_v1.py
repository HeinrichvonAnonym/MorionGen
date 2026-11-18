import sys,json,os,subprocess,hashlib,shutil
from pathlib import Path
CODE=Path(sys.argv[1]);CFG=CODE/'config/evalkit_k4_20261118/closure_real_geometry_v1';B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');OUT=B/'closure_real_geometry_20261118_v1';OUT.mkdir(exist_ok=False)
p=json.loads((CFG/'protocol.json').read_text());shutil.copyfile(CFG/'protocol.json',OUT/'protocol.json');sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
(OUT/'identity.json').write_text(json.dumps(dict(code_commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD']).decode().strip(),config_sha256={f.name:sha(f) for f in CFG.glob('*.json')},workers=1,new_holdout_uses=0),indent=2)+'\n')
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'};results=[]
for row in p['records']:
 for condition in p['conditions']:
  method=row['method'];folder=OUT/condition/method;folder.mkdir(parents=True);fit=folder/'fit';status=dict(method=method,condition=condition,status='failed')
  def call(script,args,label):
   with (folder/(label+'.log')).open('w') as log:subprocess.run([sys.executable,str(CODE/'python/scripts'/script),*map(str,args)],check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
  try:
   if shutil.disk_usage(OUT).free<80*1024**3:raise RuntimeError('80GiB reserve required')
   if any(sha(f)!=h for f,h in row['input_hashes'].items()):raise ValueError('Actual input changed')
   job=dict(input=row['input'],method=method,output=str(fit),mass_kg=75.,floor_m=0.,fit_config=str(CFG/(condition+'.json')));jp=folder/'runtime_job.json';jp.write_text(json.dumps(job,indent=2)+'\n')
   call('evalkit_k4_run.py',['--code-root',CODE,'--job',jp],'fit')
   call('evalkit_k4_native_export_views.py',['--code-root',CODE,'--fit',fit,'--input',row['input'],'--method',method,'--window','actual','--output',folder/'views'],'views')
   call('evalkit_k4_real_fidelity.py',['score','--input',row['input'],'--method',method,'--views',folder/'views','--output',folder/'fidelity.json'],'score')
   call('evalkit_k4_export_audit.py',['--fit',fit,'--output',folder/'actual_export'],'actual_export')
   if any(sha(f)!=h for f,h in row['input_hashes'].items()):raise ValueError('Actual input changed')
   status['status']='completed'
  except Exception as error:
   import traceback
   status.update(error=str(error),traceback=traceback.format_exc())
  (folder/'status.json').write_text(json.dumps(status,indent=2)+'\n');results.append(status);(OUT/'progress.json').write_text(json.dumps(dict(expected=2,results=results),indent=2)+'\n');print(status,flush=True)
(OUT/'completion.json').write_text(json.dumps(dict(expected=2,results=results,promotion='none',scope='ActualF-D only'),indent=2)+'\n')
