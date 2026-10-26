"""Tie historical Kimodo generation to its recorded pretrained asset hashes."""
import json
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.runtime import json_write
from motiogen.weight_identity import snapshot

run=ROOT/'develop_logs/test_runs/20261008_014115_kimodo_evaluate-native_48f8e536'
evaluation=json.loads((run/'native_evaluation.json').read_text())
assert evaluation['raw_generation']['success']==1856
assert evaluation['postprocess_generation']['success']==416
assert evaluation['raw_generation']['model']==evaluation['postprocess_generation']['model']=='kimodo-soma-rp-v1.1'
receipts=json.loads((run/'asset_receipts.json').read_text())
recorded={str((Path(r['path']).parent/entry['file']).resolve()):entry['sha256']
          for r in receipts for entry in r['receipt']['files']}
events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
start=datetime.fromisoformat(next(e for e in events if e['event']=='process_started')['wall_time']).timestamp()*1e9
identity=snapshot(load_config(),'kimodo')
checks=[]
for dependency in identity['files']:
    original=recorded.get(dependency['path'])
    unchanged=max(dependency['stat']['ctime_ns'],dependency['stat']['mtime_ns'])<start
    checks.append({'path':dependency['path'],'sha256':dependency['sha256'],
        'matches_historical_hash':dependency['sha256']==original if original else None,
        'file_change_time_predates_generation':unchanged,
        'passed':dependency['sha256']==original if original else unchanged,
        'scope':'Original recorded generator/stats hash compared to current content' if original else 'Text dependency retrospective content hash and filesystem change-time evidence; no historical per-load text hash'})
result={'passed':all(c['passed'] for c in checks),'original_generation_run':str(run.relative_to(ROOT)),
    'identity':identity,'checks':checks,
    'generation':{k:evaluation[k] for k in ['raw_generation','postprocess_generation']},
    'scope':'Raw and correction generated in two stages of the same original run with the same resolved model. Historical generator hashes are available; text dependencies use explicitly retrospective evidence. No retraining claim.'}
destination=ROOT/'develop_logs/test_runs/20261024_kimodo_weight_audit/audit.json'
json_write(destination,result)
print(json.dumps({'passed':result['passed'],'dependencies':len(checks),'failures':[c for c in checks if not c['passed']],'output':str(destination)}))
