"""Register a fixed-native actual-output path experiment; source is D-only."""
import hashlib,json
from pathlib import Path
root=Path.cwd();b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');code=b/'closure_budget_v3_checkout'
folder=root/'config/evalkit_k4_20261118/closure_actual_path_v1';folder.mkdir(exist_ok=False)
source=b/'closure_real_fidelity_20261118_v1/closd';selection=json.loads((source/'fit/candidate_selection.json').read_text())['selected'];native=source/'fit/candidates'/selection
policy=json.loads((code/'config/evalkit_k4_20261118/closure_budget_v1/pool.json').read_text())['coordinated_export_variants']
(folder/'policy.json').write_text(json.dumps(policy,indent=2)+'\n')
job=json.loads((source/'runtime_job.json').read_text());receipt=json.loads((native/'receipt.json').read_text())
files={str(native/name):hashlib.sha256((native/name).read_bytes()).hexdigest() for name in ['model.osim','coordinates_refined_native.mot','receipt.json','fit.npz','head_neck_model.json']}
files.update(receipt['key']['inputs'])
protocol=dict(code_root=str(code),code_commit='dfe82393261902b4b4003c6b0943ca146176d279',source_fit=str(native),source_baseline=str(source),source_sha256=files,input=job['input'],method='closd',policy=str(folder/'policy.json'),policy_sha256=hashlib.sha256((folder/'policy.json').read_bytes()).hexdigest(),output=str(b/'closure_actual_path_20261118_v1'),scope='Actual generated output only; fixed original native model/coordinates/head geometry; coordinated inserted FK path; no paired R. Same previously registered composed policy, no fitting or threshold change.',prespecified_comparison='native equality, F-D pose/coverage/timing, export path quality and actual file roundtrip',workers=1,promotion='none')
(folder/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(native)
