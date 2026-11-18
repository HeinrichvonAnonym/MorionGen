"""One-shot filesystem status snapshot, no scheduling, fitting, or scoring."""
import json,datetime,shutil
from pathlib import Path
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');O=B/'monitor_cache_20261118/snapshots';O.mkdir(parents=True,exist_ok=True);prior_path=O/'watch_latest.json';prior=json.loads(prior_path.read_text()) if prior_path.exists() else {};current={}
for n in ['pool_transfer','real_fidelity','composed_pool','real_centres']:
 p=B/f'closure_{n}_20261118_v1';q=p/'progress.json';d=json.loads(q.read_text()) if q.exists() else {'results':[]};current[n]={'completed_records':len(d['results']),'execution_failed':sum(r.get('status')=='failed' for r in d['results']),'completion':(p/'completion.json').exists()}
p=B/'full_regression_20261118_unified_v2';d=json.loads((p/'progress.json').read_text());current['full_v2']={k:d.get(k) for k in ['state','stage','finished','completed','passed','failed']};current['full_v2'].update(stage_failure=(p/'stage_failure.json').exists(),completion=(p/'completion.json').exists())
p=B/'full_regression_20261118_unified_v2/scheduling_superseded.json';current['full_v2']['superseded_before_fitting']=p.exists()
p=B/'full_regression_20261118_budget_v3';d=json.loads((p/'progress.json').read_text());current['full_budget_v3']={k:d.get(k) for k in ['state','stage','finished','completed','passed','failed']};current['full_budget_v3'].update(stage_failure=(p/'stage_failure.json').exists(),completion=(p/'completion.json').exists())
for suffix in ['controlled','actual']:
 p=B/('closure_solver_trust_20261118_v1_'+suffix);q=p/'progress.json';d=json.loads(q.read_text()) if q.exists() else {'results':[]}
 current['solver_trust_'+suffix]={'started':p.exists(),'completed_records':len(d['results']),'execution_failed':sum(r.get('status')=='failed' for r in d['results']),'completion':(p/'completion.json').exists()}
for suffix in ['controlled','actual']:
 p=B/('closure_solver_trust_20261118_v2_'+suffix);q=p/'progress.json';d=json.loads(q.read_text()) if q.exists() else {'results':[]}
 current['solver_trust_v2_'+suffix]={'started':p.exists(),'completed_records':len(d['results']),'execution_failed':sum(r.get('status')=='failed' for r in d['results']),'completion':(p/'completion.json').exists()}
controllers={'pool_transfer':(277867,'evalkit_k4_optimize.py'),'real_fidelity':(278611,'evalkit_k4_real_fidelity.py'),'composed_pool':(282105,'evalkit_k4_optimize.py'),'real_centres':(291017,'evalkit_k4_real_ablation.py'),'full_v2':(288821,'evalkit_k4_full_regression.py'),'solver_trust_controlled':(292847,'run_solver_trust_v1.py'),'solver_trust_actual':(292847,'run_solver_trust_v1.py'),'solver_trust_v2_controlled':(293942,'run_solver_trust_v2.py'),'solver_trust_v2_actual':(293942,'run_solver_trust_v2.py'),'full_budget_v3':(296024,'evalkit_k4_full_regression.py')}
for n,(pid,script) in controllers.items():
 try:running=script in Path('/proc',str(pid),'cmdline').read_bytes().decode(errors='replace')
 except OSError:running=False
 current[n]['controller_running']=running
 if not running and not current[n]['completion'] and not current[n].get('stage_failure') and not current[n].get('superseded_before_fitting') and prior.get('states',{}).get(n,{}).get('controller_running') is not False:print('ALERT controller missing before completion',n,pid)
for k,v in current.items():
 if prior.get('states',{}).get(k)!=v:print(k,json.dumps(v))
free=shutil.disk_usage(B).free/2**30
if free<80:print('ERROR disk reserve',free)
prior_path.write_text(json.dumps({'timestamp':datetime.datetime.now().isoformat(),'states':current,'free_GiB':free},indent=2)+'\n')
