"""Archive full-window scores separately from updated input-fidelity gates."""
from evalkit_k34_development import *
from motiogen.evalkit.k.bones import NAMES
from motiogen.evalkit.k.fitting3 import support
from motiogen.evalkit.k.quality import metric_fidelity,apply_metric_gates

def main():
 summary=read(OUT/'summary.json');old=read(ROOT/'config/evalkit_k33_20261117/summary.json');old={(r['method'],r['window_key']):r for r in old['fits']};selected=[];regressions=[];costs=[]
 for row in summary['fits']:
  if row['variant']!='baseline':continue
  record=read(Path(row['output'])/'result.json');z=np.load(Path(row['output'])/'comparison.npz');q=support(z['D'],z['F:new'],NAMES);mq=metric_fidelity(z['D'],z['F:new'],NAMES,10.);apply_metric_gates(q,mq)
  entry={**row,'metric_coverage':{n:{'observed':float(v['observed'].mean()),'model_fit_valid':float(v['model_fit_valid'].mean())} for n,v in mq.items()},'region_coverage':{n:{'direction_only':float(v['direction_valid'].mean()),'direction_and_metrics':float(v['angle_valid'].mean())} for n,v in q.items()},'trunk':record['scores']['raw']['angles']['trunk_incline']}
  previous=old[(row['method'],row['window_key'])]
  entry['previous_trunk_end_to_end']=previous['regions']['torso']['new']['pose_p95_deg']
  for reg,v in row['metrics']['regions'].items():
   o=previous['regions'][reg]['new'];delta=v['end_to_end']-o['pose_p95_deg']
   if delta>1 or v['maximum_step_deg']>o['max_step_deg']+5:regressions.append({'method':row['method'],'window':row['window_key'],'region':reg,'angle_delta':delta,'step_delta':v['maximum_step_deg']-o['max_step_deg']})
  for space,positions in row['metrics']['positions'].items():
   for joint,value in positions.items():
    before=previous['positions'][space][joint]['end_to_end']['world']['p95_abs'];costs.append({'method':row['method'],'window':row['window_key'],'space':space,'joint':joint,'before_m':before,'after_m':value,'delta_m':value-before})
  selected.append(entry)
 audit=read(ROOT/'develop_logs/test_runs/20261117_k34/audit.json')
 report={'conversion':summary['light'],'selected':selected,'new_pose_or_step_regressions':regressions,'position_costs':costs,'solver_pairs':summary['solver_pairs'],'solver_selection':'not selected: same-input experiments show no material accuracy improvement; baseline solver retained',
 'scope':'82x8 conversion checks; 16 full-window paired fits, two historical-input diagnostics; no full 82x8 new fit',
 'protected_files_unchanged':audit['protected_files_unchanged'],'reference_files_unchanged':audit['reference_files_unchanged'],'new_holdout_runs':0,'experiment_commit':'5ee8c63',
 'full_data':str(OUT),'score_policy':'R as final reference; main errors use all original finite frames. New support masks are reported separately, not used to improve scores.'}
 contacts=[];velocity_flags=[]
 for r in selected:
  previous=old[(r['method'],r['window_key'])]
  before=read(Path(previous['output'])/'result.json');after=read(Path(r['output'])/'result.json')
  contacts.append({'method':r['method'],'window':r['window_key'],'before':before['scores']['contact']['F:new'],'after':after['scores']['contact']['F:new']})
  for reg,v in r['metrics']['regions'].items():
   prior=previous['regions'][reg]['new']['velocity_p95_deg_s']
   if v['velocity_error_p95']>prior+10:velocity_flags.append({'method':r['method'],'window':r['window_key'],'region':reg,'before':prior,'after':v['velocity_error_p95']})
 report.update(contact_checks=contacts,velocity_flags=velocity_flags)
 atomic(ROOT/'config/evalkit_k34_20261117/acceptance.json',report)
 print('selected',len(selected),'regressions',regressions)
 print('largest position increases',sorted(costs,key=lambda r:r['delta_m'],reverse=True)[:5])
 for r in selected:
  if r['method']=='kimodo' and r['window_key']=='010':print('010 knee coverage',r['metric_coverage']['left_knee_flexion'],'leg',r['region_coverage']['left_leg'])
if __name__=='__main__':main()
