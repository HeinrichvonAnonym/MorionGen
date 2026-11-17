import sys,json,subprocess,hashlib
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts']
from evalkit_k35_legs import ROOT,BASE,read,atomic,digest,implementation_key
from evalkit_k37_summary import main
main()
a=read(ROOT/'config/evalkit_k37_20261117/acceptance.json');runtime=read(ROOT/'develop_logs/test_runs/20261117_k37/runtime_check.json');assert runtime['implementation']==implementation_key()
assert runtime['config_hash']==digest(ROOT/'config/evalkit_k37_20261117/kinematic_config.json')
assert '82 passed, 16 subtests passed' in (ROOT/'develop_logs/test_runs/20261117_k37/tests.log').read_text()
assert 'Frozen experiment differs' in (ROOT/'develop_logs/test_runs/20261117_k37/freeze_guard.log').read_text()
for name in ['probe','chart']:
 old=read(BASE/f'evalkit_k37_{name}_20261117/freeze.json')['runner']
 snapshot=subprocess.check_output(['git','show',f'df7344a:python/scripts/evalkit_k37_{name}.py'])
 assert hashlib.sha256(snapshot).hexdigest()==old
verification={'runtime':runtime,'tests':{'passed':82,'subtests':16},'protected_unchanged':a['protected_unchanged'],'references_unchanged':a['references_unchanged'],'new_holdout_runs':0,'regression_flags':a['regression_flags'],'ablation_runner_snapshot':'df7344a','rerun_guard':'changed implementation/runner is rejected before overwriting frozen experiment; use a fresh --out','scope':a['scope']}
atomic(ROOT/'config/evalkit_k37_20261117/verification.json',verification)
