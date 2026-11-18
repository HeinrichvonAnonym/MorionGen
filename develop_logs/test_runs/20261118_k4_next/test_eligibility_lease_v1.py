"""Borrow idle composed scheduler capacity without interrupting a numeric child."""
import os,signal,time,subprocess,json
from pathlib import Path
pid=282105
log=Path('develop_logs/test_runs/20261118_k4_next/eligibility_test_lease.json')
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'evalkit_k4_optimize.py' in cmd and 'closure_composed_pool_20261118_v1' in cmd
start=time.time();record={'controller_pid':pid,'controller_cmd':cmd,'purpose':'Borrow idle waiting capacity; no running numerical child interrupted'}
os.kill(pid,signal.SIGSTOP)
try:
 for _ in range(100):
  state=Path(f'/proc/{pid}/status').read_text()
  if '\nState:\tT' in state:break
  time.sleep(.02)
 else:raise RuntimeError('Controller did not stop')
 children=set()
 for p in Path(f'/proc/{pid}/task').glob('*/children'):children.update(p.read_text().split())
 alive=[int(p) for p in children if Path(f'/proc/{p}/status').exists() and '\nState:\tZ' not in Path(f'/proc/{p}/status').read_text()]
 if alive:raise RuntimeError('Numeric child present; cancel lease: '+str(alive))
 record['numeric_children_at_pause']=alive
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
 tests=['python/tests/test_evalkit_k4_candidate_eligibility.py','python/tests/test_evalkit_k4_candidate_pool.py','python/tests/test_evalkit_k4_candidate_quality.py','python/tests/test_evalkit_k4_native_cache.py']
 with Path('develop_logs/test_runs/20261118_k4_next/eligibility_v1_test.log').open('w') as f:
  result=subprocess.run(['/home/heinrich/.local/share/opensim-api/bin/python','-m','pytest','-q',*tests,'--basetemp=develop_logs/test_runs/20261118_k4_next/eligibility_pytest'],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
 record['test_exit_code']=result.returncode
finally:
 os.kill(pid,signal.SIGCONT);record['controller_resumed']=True;record['elapsed_s']=time.time()-start;log.write_text(json.dumps(record,indent=2)+'\n');print(record)
