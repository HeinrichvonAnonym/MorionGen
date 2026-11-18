import copy,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[3];base=root/'config/evalkit_k4_20261118'
out=base/'closure_unified_v2';out.mkdir(exist_ok=False)
source=base/'closure_composed_pool_v1/pool.json';pool=json.loads(source.read_text())
extra=copy.deepcopy(pool['candidates']['position_trajectory_chart']);extra['position_trunk_strategy']='observed_chain_total_proxy_v1'
pool['candidates']['position_total_proxy']=extra
pool['candidate_requirements']['position_total_proxy']={'input_rig':'absent','geometry_alternative_to':'position_trajectory_chart'}
(out/'pool.json').write_text(json.dumps(pool,indent=2)+'\n')
records=json.loads((base/'closure_pool_transfer_v1/sentinels.json').read_text())['records']
changes=json.loads((base/'closure_geometry_transfer_v1/sentinels.json').read_text())['records']
keys={(r['window'],r['method']) for r in records}
records=records+[r for r in changes if (r['window'],r['method']) not in keys]
assert len(records)==27
(out/'sentinels.json').write_text(json.dumps({'records':records,'scope':'Unchanged23 transfer combinations plus all4 altered fixed geometry targets. No old panel removal.'},indent=2)+'\n')
(out/'protocol.json').write_text(json.dumps({'version':'k4-unified-closure-candidate-2','conditions':['pool'],'strict_scoring':True,'native_export_scoring':True,'new_holdout_uses':0,'promotion':'none','status':'Registered for final same-version development regression; requires ongoing transfer evidence before launch','source_pool_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'hypothesis':'Keep every original native and coordinated export candidate. Add total-chain geometry under the identical trajectory configuration; compute only when D establishes a different fixed target from the retained base. Same-target omission is an explicit equivalence rule, never based on R or fit errors.','boundaries':'One fresh frozen code/config for all656. Old source caches/results stay preserved; no final holdout use. All original and new native candidates, exports, failures and applicability decisions retained. All runtime choices use D only.','validation_stages':'27 registered transfer combinations, then38 pre-registered format gaps, then remaining591 of the fixed656. Each stage uses the same frozen implementation and full window, with unchanged scoring thresholds/support/time strata. These are development scheduling subsets, not dataset split changes.'},indent=2)+'\n')
print('Registered13 declared candidates (eligibility by D);27+38+591 validation scheduling')
