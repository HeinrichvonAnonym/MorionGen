"""Freeze one implementation/config for all656; scheduling subsets change no split."""
import copy,hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[3];base=root/'config/evalkit_k4_20261118'
b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
code=b/'closure_unified_v2_checkout';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD']).decode().strip()
source=base/'full_regression_v1/protocol.json';protocol=copy.deepcopy(json.loads(source.read_text()))
config=code/'config/evalkit_k4_20261118/closure_unified_v2/pool.json'
paths=set((code/'utils').rglob('*.py'))|set((code/'python/scripts').glob('evalkit_k4_*.py'))
for folder in (code/'config').glob('evalkit*'):paths.update(folder.rglob('*.json'))
paths.update((code/'config/k_degradation_20261117').glob('*.json'))
paths.add(Path.home()/'.local/share/opensimcreator/resources/OpenSimCreator/models/RajagopalModel/RajagopalLaiUhlrich2023.osim')
protocol.update(version='k4-full-frozen-regression-2',fit_code_root=str(code),fit_code_commit=commit,
 control_code_root=str(code),control_code_commit=commit,output=str(b/'full_regression_20261118_unified_v2'),
 fit_config=str(config),fit_config_sha256=sha(config),workers=5,resource_budget=dict(max_workers=8,numerical_threads=1,minimum_free_GiB=80,
 authorization='User authorized at most8 CPU numerical workers. Full regression5 plus in-flight real-output2 and independent diagnostics/review1. Released slots remain for primary development.'),
 reuse_fit_roots=[],preflight_paths=[],report_title='K4统一候选v2：656路原动作验收',
 html_report=str(root/'docs/20261118_k4_unified_full_regression.html'),note_report=str(root/'develop_logs/notes/20261118_k4_unified_full_regression.md'),
 implementation_sha256={str(p):sha(p) for p in sorted(paths) if p.is_file()},
 source_protocol_sha256=sha(source),new_holdout_uses=0,promotion='none')
panel=json.loads((base/'closure_unified_v2/sentinels.json').read_text())['records']
gaps=json.loads((base/'closure_format_completion_v1/sentinels.json').read_text())['records']
key=lambda r:r['window']+'_'+r['method']
assert len(panel)==27 and len(gaps)==38 and not (set(map(key,panel))&set(map(key,gaps)))
assert len(protocol['records'])==656 and len(set(map(key,protocol['records'])))==656
assert set(map(key,panel+gaps))<=set(map(key,protocol['records']))
protocol['validation_stages']=[dict(name='registered_transfer27',paths=list(map(key,panel))),dict(name='format_completion38',paths=list(map(key,gaps)))]
prerequisites=[('closure_pool_transfer_20261118_v1',23,False),('closure_composed_pool_20261118_v1',23,True),('closure_geometry_transfer_20261118_v1',8,True)]
protocol['wait_for']=[str(b/name/'completion.json') for name,_,_ in prerequisites]
protocol['acceptance_prerequisites']=[dict(run=str(b/name),expected=count,require_all_passed=required,
 code_commit=json.loads((b/name/'identity.json').read_text())['code_commit']) for name,count,required in prerequisites]
protocol['acceptance']='Unchanged strict_v1 original-R native/export D/R clocks, exact masks/source strata, full coverage and actual files. Stages27+38+591 preserve all656; no per-window fit policy. Fresh fits all share the declared implementation/config; old cache/results retained.'
for record in protocol['records']:
 assert sha(Path(record['job_file']))==record['job_sha256']
assert sha(Path(protocol['window_manifest']))==protocol['window_manifest_sha256']
out=base/'full_regression_unified_v2';out.mkdir(exist_ok=False)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+'\n')
print({'fit_code':commit,'records':656,'stages':[27,38,591],'workers':5,'frozen_files':len(protocol['implementation_sha256'])})
