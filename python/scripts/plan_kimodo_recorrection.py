"""One shared-queue correction replay; no new motion generation or retraining."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config,environment_root
from motiogen.runtime import json_write
cfg=load_config()
revision=json.loads((ROOT/'config/survey_revision.json').read_text())
plan={'devices':[0,1],'deadline':revision['experiment_deadline_at'],'report_reserve_seconds':14400,
      'jobs':[{'id':'kimodo_official_correction_replay','method':'kimodo','action':'audit-correction',
          'command':[str(environment_root(cfg,'kimodo')/'bin/python'),'-W','ignore::FutureWarning',
              str(ROOT/'python/scripts/audit_kimodo_recorrection.py'),'--device','cuda','--receipt','{run_dir}/replay.json'],
          'required_outputs':['{run_dir}/replay.json'],'require_protocol':False,
          'estimated_seconds':90,'timeout_seconds':600}],
      'identity_files':[str(ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json')],
      'scope':'Reapply official correction on saved raw arrays using original GPU conversion/FK path; does not rerun diffusion or add independent samples.'}
destination=ROOT/'develop_logs/test_runs/20261024_kimodo_recorrection/plan.json'
if destination.exists():raise ValueError('Correction replay is already registered')
json_write(destination,plan)
print(destination)
