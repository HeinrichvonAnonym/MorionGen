"""Evidence acceptance gates remain separate from human validity and coverage."""
import json,re,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
B=ROOT/'develop_logs/test_runs/20261028_supplement';current=ROOT/'develop_logs/notes/motiongen_survey_revision/src/data.json'
q=json.loads(current.read_text())['queries'];frozen=ROOT/'develop_logs/notes/motiongen_survey_delivery/dist/index.html'
chunks=re.findall(r'<script[^>]*type="application/json"[^>]*>(.*?)</script>',frozen.read_text(),re.S)
old=json.loads(''.join(chunks))['queries']
diff=[k for k,v in old.items() if q['historical_candidate_methods' if k=='candidate_methods' else k]!=v]
assert not diff,diff
posture=q['supplement_posture']['rows'];assert len(posture)==3416
assert len({r['motion_path'] for r in posture})==len(posture)
assert all(Path(r['arrays']).exists() for r in posture)
assert sum(v.get('complete_total_frames',0) for r in posture for v in r['ergonomic'].values())==0
runtime=q['supplement_runtime']['rows'];assert len(runtime)==200
primary=[r for r in runtime if r['cohort_role']=='primary_planned_block'];assert collections.Counter(r['method'] for r in primary)=={'ardy':64,'dart':64,'closd':64}
assert all(r['status']=='success' for r in primary)
assert not any(r.get('event_chain_order_pass') is False for r in runtime)
assert len(q['supplement_tracking']['rows'])==24
assert len(q['supplement_tracking_previews']['rows'])==24
assert all(r['identical_trajectory'] for r in q['supplement_tracking_timing']['rows'])
assert len(q['supplement_generation']['rows'])==32
assert len(q['supplement_reference']['rows'])==8
checks=dict(report_migration=dict(original_query_count=len(old),differences=diff,frozen_report_sha256=checksum(frozen)),
 posture=dict(records=len(posture),by_method=dict(collections.Counter(r['method'] for r in posture)),formal_total_frames=0),
 runtime=dict(recorded_conditions=len(runtime),primary_conditions=len(primary),process_states=dict(collections.Counter(r['status'] for r in runtime)),
  primary_response_failures=[{'method':r['method'],'job':r['job']} for r in primary if r.get('update_response_failed_within_observed_execution')]),
 gates={'automatic_rule_implementation':'source tables and exhaustive lookup checked; angle conventions retain human-validation gap',
 'independent_human_validity':'not performed: no raters; blind package delivered',
 'method_runnability':'MotionGPT/UHC/PHC/SMPLSim run; PhysDiff lacks obtained official entry; PHC_MJX lacks compatible policy',
 'statistical_coverage':'finite phase-one pilot only; expansion proposal separate',
 'stage_one_evidence_delivery':'reanalysis, rules, blind package, route pilots or concrete blockers, runtime chains and measured capacity present'},
 scope='No combined scientific-validity pass. Browser receipt is a separate artifact.')
json_write(B/'phase1_acceptance.json',checks);json_write(B/'report_migration_check.json',checks['report_migration']);print(json.dumps(checks,ensure_ascii=False))
