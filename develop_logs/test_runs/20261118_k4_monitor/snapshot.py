import json, hashlib, shutil, datetime
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen')
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
OUT=B/'monitor_cache_20261118/snapshots'
OUT.mkdir(parents=True,exist_ok=True)
prior=json.loads((OUT/'latest.json').read_text()) if (OUT/'latest.json').exists() else {}
stamp=datetime.datetime.now().astimezone().isoformat()
result={'timestamp':stamp,'free_GiB':shutil.disk_usage(B).free/2**30,'groups':{}}
for name in ['formats','upper_temporal','arm_domain','064','trajectory','trajectory_seed_domain','trajectory_bounded','trajectory_balance','trajectory_geometry','trajectory_seed_chain','position_foot','position_foot_chain','trajectory_chart','pool_transfer','position_transfer','export_composition','composed_pool','geometry_transfer']:
 run=B/f'closure_{name}_20261118_v1'
 progress=json.loads((run/'progress.json').read_text()) if (run/'progress.json').exists() else {'results': []}
 if not (run/'review.json').exists():
  print('NO EXISTING REVIEW',name);continue
 review=json.loads((run/'review.json').read_text())
 group={k:review[k] for k in ['expected','completed','passed','pending','execution_failed','policy_sha256']}
 group.update(progress_completed=len(progress['results']), completion=(run/'completion.json').exists(),review_file=str(run/'review.json'),review_sha256=hashlib.sha256((run/'review.json').read_bytes()).hexdigest(),records=[])
 for row in review['records']:
  reduced={k:row[k] for k in ['window','method','condition','status','passed','failures','error','frozen_passed'] if k in row}
  if row['status']=='completed':
   reduced['actual_export_passed']=row['export_audit']['passed']
   reduced['temporal']={v:{c:a['temporal'] for c,a in clocks.items()} for v,clocks in row['acceptance'].items()}
  group['records'].append(reduced)
 result['groups'][name]=group
 old={(r['window'],r['method'],r['condition']):r['status'] for r in prior.get('groups',{}).get(name,{}).get('records',[])}
 print(name, {k:group[k] for k in ['completed','expected','passed','execution_failed','completion']})
 for r in group['records']:
  if r['status']=='completed' and old.get((r['window'],r['method'],r['condition']))!='completed':
   print('NEW',r['window'],r['method'],r['condition'],r['passed'],[(f['view'],f['clock'],f['metric'],round(f.get('candidate',f.get('p95',0)),3)) for f in r.get('failures',[])])
manifest=json.loads((ROOT/'config/evalkit_k4_20261117/preservation.json').read_text())['files']
result['preservation']={'total':len(manifest),'mismatches':[p for p,h in manifest.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]}
print('preservation',result['preservation'],'free_GiB',round(result['free_GiB']))
filename='snapshot_'+datetime.datetime.now().strftime('%H%M%S')+'.json'
(OUT/filename).write_text(json.dumps(result,indent=2)+'\n')
(OUT/'latest.json').write_text(json.dumps(result,indent=2)+'\n')
