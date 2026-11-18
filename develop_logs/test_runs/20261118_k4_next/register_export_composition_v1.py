import json,hashlib
from pathlib import Path
root=Path('/home/heinrich/MorionGen');b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');out=root/'config/evalkit_k4_20261118/closure_export_composition_v1';out.mkdir(exist_ok=False)
policy=json.loads((root/'config/evalkit_k4_20261118/closure_pool_transfer_v1/pool.json').read_text())['candidates']['forward_full_path']['coordinated_path'];(out/'path.json').write_text(json.dumps(policy,indent=2)+'\n')
rows=[]
for run,condition,w,m in [('closure_position_transfer_20261118_v1','chain','010','motionlcm'),('closure_torso_20261118_v1','position_chain','032','motionlcm')]:
 source=b/run/'fits'/condition/w/m;job=json.loads((b/'jobs'/f'{w}_{m}.json').read_text());names=['receipt.json','model.osim','coordinates_refined_native.mot','head_neck_model.json','fit.npz']
 rows.append(dict(window=w,method=m,input=job['input'],source_fit=str(source),source_sha256={n:hashlib.sha256((source/n).read_bytes()).hexdigest() for n in names}))
(out/'protocol.json').write_text(json.dumps(dict(records=rows,conditions=['control','composed'],strict_scoring=True,native_export_scoring=True,new_holdout_uses=0,promotion='none',
 hypothesis='Input-chain geometry is already native-accepted for010/032 MotionLCM; isolate export-only composition with unchanged model, native coordinates and fixed head attachment. Reuse saved native fit without optimization or R access; independently score both controls and composed variants on frozen R.',
 scope='Two full-window mechanism/integration checks; not uniform23/656 qualification'),indent=2)+'\n')
