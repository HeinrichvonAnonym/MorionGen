"""Replay two pre-instrumentation pilots; preserve, do not double-count, originals."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
base = ROOT/'develop_logs/test_runs/20261024_primal_matrix'
plan = json.loads((base/'plan.json').read_text())
jobs = []; replacements = []
for old in plan['reused']:
    if old['task'] not in (1, 6):
        continue
    job = copy.deepcopy(next(j for j in plan['jobs'] if j['task_id'] == old['task']))
    job.update(id=f"primal_initial{old['task']}_seed20_receipt_replay", seed=20)
    job['command'][job['command'].index('--seed')+1] = '20'
    jobs.append(job)
    replacements.append({'job': job['id'], 'prior_run': old['run'], 'reason': 'Original exact-length warm-start pilot preceded branch instrumentation; replay same inputs and check trajectory equality.'})
plan.update(jobs=jobs, reused=[], replacements=replacements)
destination = ROOT/'develop_logs/test_runs/20261024_primal_receipt_recovery/plan.json'
if destination.exists():
    raise ValueError('Recovery already registered')
json_write(destination, plan)
print(destination)
