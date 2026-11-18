"""Registered12 controlled +4 actual serial replay comparisons, separate reports."""
import sys,json,subprocess,hashlib,shutil,os
from pathlib import Path
CODE=Path(sys.argv[1]);CFG=CODE/'config/evalkit_k4_20261118/closure_solver_trust_v1'
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
protocol=json.loads((CFG/'protocol.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
for scope,suffix in [('controlled_R','controlled'),('actual_FD','actual')]:
 rows=[r for r in protocol['records'] if r['scope']==scope]
 out=B/('closure_solver_trust_20261118_v1_'+suffix);out.mkdir(exist_ok=False)
 (out/'protocol.json').write_text(json.dumps(dict(protocol,records=rows,scope=scope),indent=2)+'\n')
 (out/'panel.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n')
 (out/'identity.json').write_text(json.dumps(dict(code_commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD']).decode().strip(),config_sha256={p.name:sha(p) for p in CFG.glob('*.json')},worker_limit=1,new_holdout_uses=0),indent=2)+'\n')
 results=[];expected=len(rows)*len(protocol['conditions'])
 def progress(name='progress.json'):
  temp=out/(name+'.tmp');temp.write_text(json.dumps(dict(results=results,expected=expected,scope=scope),indent=2)+'\n');temp.replace(out/name)
 progress()
 for row in rows:
  for condition in protocol['conditions']:
   w,m=row['window'],row['method'];key=w+'_'+m;task=out/'tasks'/condition/key;task.mkdir(parents=True)
   fit=out/'fits'/condition/w/m;views=out/'views'/condition/key
   status=dict(window=w,method=m,condition=condition,status='failed',scope=scope)
   def call(script,args,label):
    with (task/(label+'.log')).open('w') as log:subprocess.run([sys.executable,str(CODE/'python/scripts'/script),*map(str,args)],check=True,stdout=log,stderr=subprocess.STDOUT,env=env)
   try:
    if shutil.disk_usage(out).free<80*1024**3:raise RuntimeError('80GiB reserve required')
    for file,h in row['input_hashes'].items():
     if sha(file)!=h:raise ValueError('Changed registered input')
    for file,h in row['source_sha256'].items():
     if sha(Path(row['source_fit'])/file)!=h:raise ValueError('Changed model or seed')
    args=['--fit',row['source_fit'],'--input',row['input'],'--method',m,'--settings',CFG/(condition+'.json'),'--output',fit]
    if condition=='control':args+=['--require-control-reproduction']
    call('evalkit_k4_trajectory_replay.py',args,'replay')
    call('evalkit_k4_native_export_views.py',['--code-root',CODE,'--fit',fit,'--input',row['input'],'--method',m,'--window',w,'--output',views],'views')
    if scope=='controlled_R':
     for view in ['native','export']:
      call('evalkit_k4_ablation_score.py',['--fits',views/view,'--baseline',B/'common_v1','--output',out/'scores'/condition/view,'--window',w,'--method',m,'--strict'],view+'_score')
    else:
     call('evalkit_k4_real_fidelity.py',['score','--input',row['input'],'--method',m,'--views',views,'--output',task/'fidelity.json'],'score')
    call('evalkit_k4_export_audit.py',['--fit',fit,'--output',task/'actual_export'],'actual_export')
    status['status']='completed'
   except Exception as e:
    import traceback
    status.update(error=str(e),traceback=traceback.format_exc())
   (task/'status.json').write_text(json.dumps(status,indent=2)+'\n');results.append(status);progress();print(status,flush=True)
   if condition=='control' and status['status']!='completed':raise RuntimeError('Control did not reproduce: retain candidates pending and investigate')
 progress('completion.json')
