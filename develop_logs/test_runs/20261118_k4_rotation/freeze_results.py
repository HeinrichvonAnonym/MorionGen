from pathlib import Path
import hashlib,json,shutil,subprocess
root=Path('/home/heinrich/MorionGen');base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
protection=json.loads((root/'config/evalkit_k4_20261117/preservation.json').read_text());changed=[p for p,h in protection['files'].items() if digest(root/p)!=h];assert not changed
runs={};artifacts={}
for name in ['whole_rotation_20261118_v2','whole_rotation_20261118_stress_v2']:
 folder=base/name;protocol=json.loads((folder/'protocol.json').read_text());review=json.loads((folder/'review_final.json').read_text())
 for item in protocol['records']:
  receipt=json.loads((folder/'replays'/item['window']/item['method']/'receipt.json').read_text())
  assert digest(item['native_file'])==item['native_sha256']==receipt['native_coordinates_sha256']
  assert digest(Path(item['fit'])/'model.osim')==item['model_sha256']
  for path,h in receipt['source_fit_key']['inputs'].items():assert digest(path)==h
 for file in [folder/'protocol.json',folder/'review_final.json',folder/'completion.json',*sorted((folder/'scores').glob('*/*.json'))]:artifacts[str(file)]=digest(file)
 runs[name]={'expected_records':len(protocol['records']),'completed_primary_records':review['completed'],'score_files':len(list((folder/'scores').glob('*/*.json'))),
  'whole_so3_frozen_policy_passed':sum(r['conditions']['whole_so3']['acceptance_frozen_policy']['status']=='passed' for r in review['rows']),
  'fallback_joint_intervals':sum(v['fallback_intervals'] for r in review['rows'] for v in r['resampling']['counts'].values()),
  'expected_joint_intervals':sum(v['native_intervals'] for r in review['rows'] for v in r['resampling']['counts'].values()),
  'body_and_head_coverage_failures':sum(v['status']=='failed' for r in review['rows'] for v in r['conditions']['whole_so3']['acceptance_frozen_policy']['full_window_coverage'])}
for file in [base/'whole_rotation_audit_20261118_v1.json',base/'whole_rotation_replayed_fk_verification_20261118.json']:artifacts[str(file)]=digest(file)
result={'version':'k4-whole-rotation-results-20261118','source_commit_fixed_panel':'85a11af','core_interpolation_commit':'c9ffd3d',
 'runs':runs,'artifacts_sha256':artifacts,'protected_files_checked':len(protection['files']),'protected_files_changed':changed,
 'new_holdout_uses':0,'default_promoted':False,'force_qualification_extended':False,'data_available_GiB':shutil.disk_usage(base).free/1024**3,
 'implementation_sha256':{str(p.relative_to(root)):digest(p) for p in [root/'python/scripts/evalkit_k4_rotation_probe.py',root/'python/scripts/evalkit_k4_rotation_batch.py',root/'python/scripts/evalkit_k4_rotation_review.py',root/'python/scripts/evalkit_k4_rotation_figure.py',*sorted((root/'utils/motiogen/evalkit/k/resampling').glob('*.py'))]},
 'scope':'656 native model/coordinate audits, fixed20 and supplemental4 same-native interpolation replays; no new body fitting. D-R/F-D/F-R remain separate, native and export increments independently measured. Full K acceptance remains pending.'}
for name in ['all_k4_tests_final.log','ergo_tests.log','verify_replayed_fk.log']:
 file=root/'develop_logs/test_runs/20261118_k4_rotation'/name
 result.setdefault('validation_logs',{})[str(file.relative_to(root))]={'sha256':digest(file),'tail':file.read_text()[-250:]}
(root/'config/evalkit_k4_20261118/rotation_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(runs)
