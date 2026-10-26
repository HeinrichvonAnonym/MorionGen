"""Add derived retrospective identity evidence, preserving original raw receipts."""
import json
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.weight_identity import snapshot
from motiogen.runtime import json_write
cfg=load_config();base=ROOT/'develop_logs/test_runs'
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
identities={};rows=[]
for entry in analysis['rows']:
    method=entry['method'];run=entry['run']
    if any(r['run']==run for r in rows):continue
    if method not in identities:
        identities[method]=snapshot(cfg,method)
        print(json.dumps({'method':method,'dependencies':len(identities[method]['files'])}),flush=True)
    events=[json.loads(line) for line in (ROOT/run/'events.jsonl').read_text().splitlines()]
    started=next(r for r in events if r['event']=='process_started')['wall_time']
    threshold=datetime.fromisoformat(started).timestamp()*1e9
    changed=[r['path'] for r in identities[method]['files'] if max(r['stat']['mtime_ns'],r['stat']['ctime_ns'])>threshold]
    trace_path=ROOT/run/'trace.json';trace=json.loads(trace_path.read_text()) if trace_path.exists() else {}
    observed={r['sha256'] for r in identities[method]['files']}
    discrepancies=[key for key in ['policy_sha256','tracker_sha256','dip_sha256'] if key in trace and trace[key] not in observed]
    rows.append({'run':run,'method':method,'identity':identities[method]['identity'],'passed':not changed and not discrepancies,
        'changed_after_start':changed,'trace_hash_discrepancies':discrepancies,'started_at':started,
        'scope':'Retrospective: current content hashes plus filesystem ctime/mtime predating process start; original native trace hashes cross-checked when present. Not claimed as historical load-time hashing.'})
destination=base/'20261024_native_weight_audit/audit.json'
json_write(destination,{'identities':identities,'rows':rows})
print(json.dumps({'runs':len(rows),'passed':sum(r['passed'] for r in rows),'failures':[r for r in rows if not r['passed']],'output':str(destination)}),flush=True)
