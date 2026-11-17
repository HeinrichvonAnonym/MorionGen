import sys,json,shutil,hashlib
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'utils'),str(Path.cwd()/'python/scripts')]
from motiogen.evalkit.k.config import digest
from evalkit_k35_legs import OUT,read,ROOT,SETTINGS
out=ROOT/'config/evalkit_k35_20261117';a=read(OUT/'acceptance.json');f=read(OUT/'freeze.json');records=[]
for r in a['records']:
 scores={}
 for side in ['left','right']:
  scores[side]={}
  for metric in ['knee_flexion','thigh_incline','foot_incline','ankle_chain_angle','hip_trunk_angle']:
   n=side+'_'+metric;scores[side][metric]={k:{stage:v['raw']['angles'][n][stage] for stage in ['degradation','fit_deviation','end_to_end']} for k,v in r['scores'].items()}
 records.append({k:r[k] for k in ['method','window','guards','summary','output']}|{'angles':scores})
assert all(digest(ROOT/p)==h for p,h in f['protected'].items());assert all(digest(p)==h for p,h in f['inputs'].items())
references=read(ROOT/'config/evalkit_k34_20261117/experiment_freeze.json')['references']
assert all(digest(p)==h for p,h in references.items())
result={'reference_files_unchanged':len(references),'records':records,'scope':a['scope'],'settings':SETTINGS,'foot_weight':.5,'selection':'experimental only; default K3.4 retained; four legs accepted in three of eleven development windows; target failures unresolved','protected_unchanged':len(f['protected']),'input_files_unchanged':len(f['inputs']),'new_holdout_runs':0,'experiment_commit':'e580516','full_results':str(OUT/'acceptance.json'),'full_results_sha256':digest(OUT/'acceptance.json')}
(out/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');shutil.copy2(OUT/'freeze.json',out/'freeze.json')
verification={name:read(ROOT/'develop_logs/test_runs/20261117_k35'/f'{name}.json') for name in ['mapping_audit','subtalar_probe','runtime_check']};(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print('protected',len(f['protected']),'inputs',len(f['inputs']),'accepted',sum(v['accepted'] for r in records for v in r['guards'].values()))
