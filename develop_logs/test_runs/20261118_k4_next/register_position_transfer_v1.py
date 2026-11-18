import json,hashlib
from pathlib import Path
root=Path('/home/heinrich/MorionGen');out=root/'config/evalkit_k4_20261118/closure_position_transfer_v1';out.mkdir(exist_ok=False)
poolpath=root/'config/evalkit_k4_20261118/closure_pool_transfer_v1/pool.json';bank=json.loads(poolpath.read_text())
conditions={name:json.loads(json.dumps(bank['candidates'][key])) for name,key in [('chain','position_chain'),('trajectory_chart','position_trajectory_chart'),('trajectory_chart_path','position_trajectory_chart')]}
conditions['trajectory_chart_path']['coordinated_path']=bank['candidates']['forward_full_path']['coordinated_path']
for name,config in conditions.items():
 config['condition']=name
 (out/(name+'.json')).write_text(json.dumps(config,indent=2)+'\n')
records=[{'window':w,'method':m} for w,m in [('010','motionlcm'),('010','omnicontrol'),('073','omnicontrol'),('077','omnicontrol')]]
(out/'sentinels.json').write_text(json.dumps({'records':records,'new_holdout_uses':0,'scope':'Supplemental mechanism/format transfer; unchanged fixed development windows, not replacement of23 panel'},indent=2)+'\n')
protocol={'version':'k4-position-trajectory-transfer-1','conditions':list(conditions),'strict_scoring':True,'native_export_scoring':True,'new_holdout_uses':0,'promotion':'none',
 'hypothesis':'010 position formats still have native trunk/shoulder temporal failure under previous traversal candidates. Compare common position-chain geometry native solver with existing joint trajectory/equivalent chart, then add fixed-native full-body path. Transfer073/077 from MotionLCM to OmniControl without method-specific parameters.',
 'factors':'chain versus trajectory_chart changes whole-native trajectory solve and declared post-seed shoulder domain as an already independently tested bundle; no claim of single-factor attribution. trajectory_chart versus trajectory_chart_path isolates interpolation at identical native solution/geometry; verify native anchor hashes and FK. Native direction/position, export path and roundtrip independently accepted.',
 'boundary':'D only in fitting/selection; R only independent score; frozen5deg/5cm/full denominators/source jump strata. No action-ID runtime branch, no new range or structural intervention. All candidates/fallback retained.',
 'source_pool':str(poolpath.relative_to(root)),'source_pool_sha256':hashlib.sha256(poolpath.read_bytes()).hexdigest()}
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(out,len(records)*len(conditions))
