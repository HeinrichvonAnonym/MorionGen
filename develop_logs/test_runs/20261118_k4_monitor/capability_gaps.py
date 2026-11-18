import json,hashlib,datetime,collections
from pathlib import Path
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');OUT=Path('/home/heinrich/MorionGen/develop_logs/test_runs/20261118_k4_monitor')
base=json.loads((B/'full_regression_20261118_v1/completion.json').read_text())['results'];failed={(r['window'],r['method']):r for r in base if not r['passed']};passed=collections.defaultdict(list);tested=collections.defaultdict(list);sources={}
def add(name,rows):
 for r in rows:
  if r.get('status')!='completed':continue
  key=r['window'],r['method'];tested[key].append(name+'__'+r['condition'])
  if r.get('passed'):passed[key].append(name+'__'+r['condition'])
d=json.loads((OUT/'latest.json').read_text())
for n,g in d['groups'].items():add('closure_'+n+'_20261118_v1',g['records'])
for n in ['closure_torso_20261118_v1','closure_legs_20261118_v1','closure_path_runtime_20261118_v1','foot_temporal_20261118_v1','foot_geometry_20261118_v1','failure_followup_20261118_v1','failure_followup_regression_20261118_v1','closure_trajectory_chart_20261118_v1','closure_position_foot_chain_20261118_v1']:
 p=B/n/'review.json'
 if not p.exists():continue
 raw=p.read_bytes();r=json.loads(raw);sources[str(p)]=hashlib.sha256(raw).hexdigest();add(n,r.get('records',[]))
# Completed native/coordinated replay with both clocks and real roundtrip.
p=B/'closure_path_alias_20261118_v1/review.json';raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest()
for r in json.loads(raw)['rows']:
 key=r['window'],r['method'];cs=r['conditions'];ok=all(cs[c][a]['status']=='passed' for c in ['native','coordinated'] for a in ['acceptance_relative_D','acceptance_original_clock_relative_D'])
 audit=json.loads((p.parent/'replays'/r['window']/r['method']/'roundtrip/audit.json').read_text());ok=ok and audit['passed'];tested[key].append('closure_path_alias_20261118_v1__coordinated')
 if ok:passed[key].append('closure_path_alias_20261118_v1__coordinated')
rows=[]
for key,r in sorted(failed.items()):rows.append({'window':key[0],'method':key[1],'any_completed_candidate_passed':bool(passed[key]),'passing_candidates':sorted(set(passed[key])),'completed_candidates_reviewed':sorted(set(tested[key])),'frozen_failures':r['failed_metrics']})
gaps=[r for r in rows if not r['any_completed_candidate_passed']]
result={'timestamp':datetime.datetime.now().isoformat(),'frozen_failed':len(rows),'covered_by_at_least_one_completed_candidate':len(rows)-len(gaps),'remaining_capability_gaps':len(gaps),'records':rows,'source_sha256':sources,'scope':'Existence of any fully accepted completed diagnostic candidate/replay, not uniform bank transfer or D-only selection success. No inference for untested method/window combinations.'}
(OUT/'candidate_capability_gaps.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:result[k] for k in ['frozen_failed','covered_by_at_least_one_completed_candidate','remaining_capability_gaps']})
for w in sorted(set(r['window'] for r in gaps)):
 rr=[r for r in gaps if r['window']==w];print(w,'tested_no_pass',[r['method'] for r in rr if r['completed_candidates_reviewed']],'no_completed_candidate_review',[r['method'] for r in rr if not r['completed_candidates_reviewed']])
for w in ['010','043','044','055','073','077']:
 print('PRIORITY',w,[(r['method'],r['passing_candidates']) for r in rows if r['window']==w and r['method'] in ['motionlcm','omnicontrol']])
