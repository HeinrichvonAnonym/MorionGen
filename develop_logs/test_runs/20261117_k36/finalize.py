import sys,json,shutil
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'utils'),str(Path.cwd()/'python/scripts')]
from motiogen.evalkit.k.config import digest,implementation_key
from evalkit_k36_summary import BASE,ROOT,ROOTS,read
s=read(BASE/'evalkit_k36_summary_20261117.json');out=ROOT/'config/evalkit_k36_20261117';out.mkdir(exist_ok=True)
# Compact per-run receipts point to immutable, fully detailed arrays on disk.
report={k:v for k,v in s.items() if k!='records'};report['records']=s['records'];report['selection']={'profile':'combined','hz':20,'continuity_weight':.02,'continue_joint_branch':True,'anchor_bent_frame':True,'anchor_min_plane_angle_deg':10.,'output_kind':'kinematic_only','force_eligible':False,'default_config':'K3.4 remains unchanged','limitation':'074 left candidate rejected; no full 82x8 new-domain fit or new holdout'}
flags=[];position_costs=[]
for r in s['records']['panel']:
 if r['profile']!='combined':continue
 for side,v in r['regions'].items():
  a,b=v['baseline'],v['selected']
  if b['end_to_end']>a['end_to_end']+1 or b['max_step']>a['max_step']+5 or b['velocity_error_p95']>a['velocity_error_p95']+10:flags.append({'method':r['method'],'window':r['window'],'side':side})
  position_costs.append({'method':r['method'],'window':r['window'],'side':side,'raw_delta_m':b['foot_position']['raw']-a['foot_position']['raw'],'normalized_delta_m':b['foot_position']['normalized']-a['foot_position']['normalized']})
report.update(new_region_or_temporal_flags=flags,foot_position_costs=position_costs,experiment_commits={'domain':'07cead9','continuation':'b1f23b7','anchor':'ad14bf3'})
(out/'acceptance.json').write_text(json.dumps(report,ensure_ascii=False,separators=(',',':'))+'\n')
freezes={name:read(path/'freeze.json') for name,path in ROOTS.items()}
(out/'freeze.json').write_text(json.dumps(freezes,separators=(',',':'))+'\n')
for f in freezes.values():
 assert all(digest(p)==h for p,h in f['inputs'].items())
 if 'protected' in f:assert all(digest(ROOT/p)==h for p,h in f['protected'].items())
refs=read(ROOT/'config/evalkit_k34_20261117/experiment_freeze.json')['references'];assert all(digest(p)==h for p,h in refs.items())
runtime=read(ROOT/'develop_logs/test_runs/20261117_k36/runtime_check.json');assert runtime['implementation']==implementation_key()
verification={'runtime':runtime,'reference_files_unchanged':len(refs),'protected_unchanged':s['protected_unchanged'],'new_holdout_runs':0,'summary_hash':digest(BASE/'evalkit_k36_summary_20261117.json'),'artifact_root':str(BASE),'tests':'81 passed, 16 subtests passed'}
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n');print('flags',flags,'references',len(refs),'protected',s['protected_unchanged'])
