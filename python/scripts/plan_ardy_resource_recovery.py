"""Replay only observed stale full-LLM resource-gate refusals, preserving evidence."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
base=ROOT/'develop_logs/test_runs/20261024_ardy_matrix'
plan=json.loads((base/'plan.json').read_text());state=json.loads((base/'queue/state.json').read_text())
jobs=[];recovery=[]
for job in plan['jobs']:
    row=state['jobs'][job['id']]
    if row['status']!='resource_gate':continue
    if row['reason']!='Reviewed inference peak exceeds current RAM/VRAM budget':raise ValueError('Different refusal requires its own diagnosis')
    jobs.append(job);recovery.append({'job':job['id'],'prior_run':row['run_dir'],'prior_reason':row['reason']})
if not jobs:raise ValueError('No matching refused jobs to recover')
plan.update(jobs=jobs,reused=[],recovery=recovery,
    recovery_scope='Same scientific inputs and seeds. Required-hit cache memory profile now distinct from full LLM loading; source refusal receipts retained.')
destination=ROOT/'develop_logs/test_runs/20261024_ardy_resource_recovery/plan.json'
if destination.exists():raise ValueError('Recovery already registered')
json_write(destination,plan);print(json.dumps({'path':str(destination),'jobs':len(jobs)}))
