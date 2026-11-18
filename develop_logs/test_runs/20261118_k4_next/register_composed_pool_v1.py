import json,hashlib,subprocess
from pathlib import Path
root=Path('/home/heinrich/MorionGen');b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');native=b/'closure_pool_transfer_v1_checkout';run=b/'closure_pool_transfer_20261118_v1';out=root/'config/evalkit_k4_20261118/closure_composed_pool_v1';out.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
original=json.loads((native/'config/evalkit_k4_20261118/closure_pool_transfer_v1/pool.json').read_text())
settings=json.loads(json.dumps(original));settings['coordinated_export_variants']=settings['candidates']['forward_full_path']['coordinated_path']
(out/'pool.json').write_text(json.dumps(settings,indent=2)+'\n')
key=json.loads((run/'fits/pool/000/kimodo/receipt.json').read_text())['key']
contract={k:key[k] for k in ('config','implementation','input_config_sha256','biomech_sha256','model_sha256')};contract['config']=original
assert {str(p.relative_to(native/'utils/motiogen/evalkit/k')):sha(p) for p in sorted((native/'utils/motiogen/evalkit/k').rglob('*.py'))}==contract['implementation']
assert sha(native/'utils/motiogen/biomech.py')==contract['biomech_sha256']
assert sha(native/'config/evalkit_k4_20261117/input.json')==contract['input_config_sha256']
(out/'native_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
(out/'sentinels.json').write_text((root/'config/evalkit_k4_20261118/closure_pool_transfer_v1/sentinels.json').read_text())
protocol=dict(version='k4-composed-bank-transfer-1',conditions=['pool'],strict_scoring=True,native_export_scoring=True,new_holdout_uses=0,promotion='none',
 frozen_native_bank=dict(run=str(run),condition='pool',contract='native_contract.json',code_commit=subprocess.check_output(['git','-C',str(native),'rev-parse','HEAD']).decode().strip()),
 hypothesis='Compose fixed-native full-body export with every successful native candidate, preserving all original candidates and input-only whole-window ranking. Existing identical export policies are not recomputed. Validate that fixed native geometry choices can retain good interpolation without rerunning native fitting.',
 boundary='All23 predeclared windows unchanged. Native code/config/assets pinned independently from export code. Reuse every native candidate, never only the old winner and never filter by R acceptance. Frozen source failed and inapplicable states retained. New output/caches preserve original source and all candidates. R scoring remains a separate process.',
 sources={'native_bank_config_sha256':sha(native/'config/evalkit_k4_20261118/closure_pool_transfer_v1/pool.json'),'source_panel_sha256':sha(root/'config/evalkit_k4_20261118/closure_pool_transfer_v1/sentinels.json')})
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(out)
