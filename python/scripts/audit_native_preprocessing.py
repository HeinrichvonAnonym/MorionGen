"""Retrospective normalization/config audit, without rewriting raw load receipts."""
import json
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.preprocessing_identity import snapshot
from motiogen.runtime import json_write
base=ROOT/'develop_logs/test_runs'
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
cfg=load_config();identities={};checks=[];seen=set()
entries=list(analysis['rows'])
kimodo=json.loads((base/'20261024_kimodo_weight_audit/audit.json').read_text())
entries.append({'method':'kimodo','run':kimodo['original_generation_run']})
for row in entries:
    if row['run'] in seen:continue
    seen.add(row['run']);method=row['method']
    if method not in identities:
        identities[method]=snapshot(cfg,method)
        print(json.dumps({'method':method,'preprocessing_files':len(identities[method]['files'])}),flush=True)
    events=[json.loads(line) for line in (ROOT/row['run']/'events.jsonl').read_text().splitlines()]
    start=next(e['wall_time'] for e in events if e['event']=='process_started')
    threshold=datetime.fromisoformat(start).timestamp()*1e9
    changed=[f['path'] for f in identities[method]['files'] if max(f['stat']['mtime_ns'],f['stat']['ctime_ns'])>threshold]
    checks.append({'run':row['run'],'method':method,'identity':identities[method]['identity'],
                   'passed':not changed,'changed_after_start':changed})
receipt={'identities':identities,'checks':checks,'passed':all(c['passed'] for c in checks),
         'scope':'Current content hashes with filesystem change times predating original process start; explicitly retrospective, not claimed as original load-time hashing.'}
json_write(base/'20261024_native_preprocessing/audit.json',receipt)
print(json.dumps({'passed':receipt['passed'],'runs':len(checks),'failures':[r for r in checks if not r['passed']]}))
