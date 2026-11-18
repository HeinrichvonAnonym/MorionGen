import json,hashlib
from pathlib import Path
root=Path('/home/heinrich/MorionGen');b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');out=root/'config/evalkit_k4_20261118/closure_real_geometry_v1';out.mkdir(exist_ok=False)
bank=json.loads((root/'config/evalkit_k4_20261118/closure_pool_transfer_v1/pool.json').read_text());cfg=bank['candidates']['position_trajectory_chart']
for name,policy in [('control','observed_chain_proxy_v1'),('total_proxy','observed_chain_total_proxy_v1')]:
 c=json.loads(json.dumps(cfg));c['position_trunk_strategy']=policy
 (out/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n')
audit=json.loads((b/'real_input_audit_20261118_v1.json').read_text());row=next(r for r in audit['rows'] if r['method']=='motionlcm')
protocol=dict(version='k4-real-geometry-1',records=[row],conditions=['control','total_proxy'],new_holdout_uses=0,promotion='none',
 hypothesis='Actual MotionLCM terminal spine3-to-shoulder-midpoint segment has22.93% spread while total chain spread5.34%. Treat this virtual endpoint uncertainty at the effective total length, keeping first3 spine-link and total-chain10% stability criteria; one clip-fixed median estimate, neutral length remains unidentified.',
 factors='Only fixed geometry eligibility policy differs; native solver, chart, domains, root, interpolation unchanged. Original model prior candidate retained. No reference or neutral-frame/max-distance calibration.',
 scoring='Actual baseline F-D only; raw/commonDlength, native/export/Dclock, full fixed input support and actual file readback. No pairedR or claim of anatomical/end-to-end accuracy.',
 evidence_file=str(b/'actual_motionlcm_chain_uncertainty_v1.json'),evidence_sha256=hashlib.sha256((b/'actual_motionlcm_chain_uncertainty_v1.json').read_bytes()).hexdigest())
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
