import json,pathlib
r=pathlib.Path('/home/heinrich/MorionGen');src=r/'config/evalkit_k4_20261118/closure_formats_v1/geometry_domain.json';print(src.exists())
if not src.exists():src=r/'config/evalkit_k4_20261118/foot_geometry_v1/geometry_domain.json'
base=json.loads(src.read_text());out=r/'config/evalkit_k4_20261118/closure_position_foot_v1';out.mkdir(exist_ok=False)
conditions=['control','fixed_endpoint','fixed_endpoint_full_domain']
for name in conditions:
 cfg=json.loads(json.dumps(base));cfg['condition']=name
 if name!='control':cfg['position_foot_geometry']=dict(version='input_polar_envelope_v1',length_tolerance_m=.02)
 if name=='fixed_endpoint_full_domain':
  for s in ('l','r'):
   for n in ('ankle_angle','subtalar_angle'):cfg['kinematic_domain']['coordinate_ranges_deg'][n+'_'+s]=[-180,180]
 (out/(name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
(out/'sentinels.json').write_text(json.dumps(dict(records=[dict(window=w,method=m) for w,m in [('002','motionlcm'),('002','omnicontrol'),('000','motionlcm'),('064','motionlcm')]]),indent=2)+'\n')
p=dict(version='k4-position-foot-1',strict_scoring=True,native_export_scoring=True,conditions=conditions,new_holdout_uses=0,promotion='none',hypothesis='002 position formats demand foot polar angles119..157deg about input-derived ankle axis; fixed original geometry full-domain envelope16..129deg. Estimate a clip-fixed MTP parent translation from all identifiable D knee planes, minimize translation subject to polar coverage and full-rotation foot length within observed median+/-2cm. Right foot geometry remains unchanged when already feasible. Neutral pose/length is not inferred as anatomical truth. Compare control, endpoint only, endpoint plus audited full PinJoint rotations, retaining original axes/coupling/structure.',parameter_source='2cm fixed-length guard is an explicit engineering geometry tradeoff below5cm position target, not derived anatomical tolerance. Five deterministic geometry starts, all valid D frames, no R. Existing full-domain audit at0.1deg reused; endpoint FK formula and unchanged XML outside MTP translation independently verified.',preregistration='Two002 position inputs plus000 and064 position regression controls,4x3=12. Inputs/time/masks/scoring unchanged; no motion-ID runtime logic. Full/native/export,D/R clocks independently score every candidate; no default promotion on diagnostic feasibility.')
(out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
