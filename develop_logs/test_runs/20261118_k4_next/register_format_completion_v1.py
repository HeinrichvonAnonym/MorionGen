"""Preregister missing format evidence; no fitting or R-dependent runtime rule."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[3]
b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
source=root/'develop_logs/test_runs/20261118_k4_monitor/candidate_capability_gaps_direct_131203.json'
d=json.loads(source.read_text());records=[{'window':r['window'],'method':r['method']} for r in d['records'] if not r['any_completed_candidate_passed']]
assert len(records)==38
assert all(not r['completed_candidates_reviewed'] for r in d['records'] if not r['any_completed_candidate_passed'])
existing=json.loads((root/'config/evalkit_k4_20261118/closure_pool_transfer_v1/sentinels.json').read_text())['records']
assert not ({(r['window'],r['method']) for r in records}&{(r['window'],r['method']) for r in existing})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=root/'config/evalkit_k4_20261118/closure_format_completion_v1';out.mkdir(exist_ok=False)
frozen=b/'format_completion_evidence_20261118_v1.json'
assert not frozen.exists();shutil.copyfile(source,frozen)
inputs={}
for r in records:
 key=r['window']+'_'+r['method'];job=b/'jobs'/(key+'.json')
 assert job.exists();inputs[key]={'job':str(job),'sha256':sha(job)}
(out/'sentinels.json').write_text(json.dumps({'records':records,'scope':'38 development format-window gaps, disjoint from the unchanged23 transfer panel'},indent=2)+'\n')
(out/'protocol.json').write_text(json.dumps({'version':'k4-format-completion-registration-1','status':'preregistered, not started','conditions':['pool'],'strict_scoring':True,'native_export_scoring':True,'new_holdout_uses':0,'promotion':'none','evidence_file':str(frozen),'evidence_sha256':sha(frozen),'evidence_counts':{'original_failed':94,'local_candidate_exists':56,'format_transfer_unverified':38},'input_jobs':inputs,'prerequisite':'Review current23 native/composed transfer first; use one frozen bank and selector for every row. Method changes require a new registered version; no method/window-specific setting. All input windows and masks remain frozen.','native_code_commit':'272f08e','native_pool_sha256':sha(root/'config/evalkit_k4_20261118/closure_pool_transfer_v1/pool.json'),'export_code_commit':'0d79bb0','export_pool_sha256':sha(root/'config/evalkit_k4_20261118/closure_composed_pool_v1/pool.json'),'scope':'These38 combinations have no completed new-method candidate in this evidence snapshot. They are missing migration evidence, not38 demonstrated failures of the new method. Local candidate existence is distinct from uniform runtime selection and full656 qualification.'},indent=2)+'\n')
print('Registered38 untested combinations;0 overlap with23 transfer; no execution launched')
