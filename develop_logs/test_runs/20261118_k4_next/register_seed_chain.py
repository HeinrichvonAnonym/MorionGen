import json,pathlib
r=pathlib.Path('/home/heinrich/MorionGen');src=r/'config/evalkit_k4_20261118/closure_trajectory_geometry_v1';out=r/'config/evalkit_k4_20261118/closure_trajectory_seed_chain_v1';out.mkdir(exist_ok=False)
base=json.loads((src/'balanced_chain.json').read_text());domain=json.loads((r/'config/evalkit_k4_20261118/closure_trajectory_seed_domain_v1/expanded_after_seed.json').read_text())['trajectory_kinematic_domain']
names=[]
for reverse,expand in [(False,False),(True,False),(False,True),(True,True)]:
 name=('reverse' if reverse else 'forward')+('_expanded' if expand else '_original');cfg=json.loads(json.dumps(base));cfg['condition']=name;cfg['refinement']['traversal']='reverse' if reverse else 'forward'
 if expand:cfg['trajectory_kinematic_domain']=domain
 (out/(name+'.json')).write_text(json.dumps(cfg,indent=2)+'\n');names.append(name)
(out/'sentinels.json').write_text((src/'sentinels.json').read_text())
p=dict(version='k4-trajectory-seed-chain-1',strict_scoring=True,native_export_scoring=True,conditions=names,new_holdout_uses=0,promotion='none',hypothesis='073 balanced_chain residual gradient blocked at arm_rot_r +90deg; expanded domain tested after common native seed, separately from reverse traversal. 077 chain seed has frame1 leftforearm47.34deg and source pelvis step132.32deg, native full-window P95 hides this short interval; no clipping occurred. Test reverse seed to avoid forward propagation, retain all source-jump/full-window scores and masks. Geometry fixed input-chain, same20 objective evaluation budget. Prior failed expanded-domain/TRF experiment retained as different solver/geometry evidence.',preregistration='073/077/000 MotionLCM x forward/reverse seed x original/expanded-after-seed arm axial domain =12. No window-specific runtime rules, D-only fitting; R independent scoring. Existing arm-domain coupling audit reused; seed FK remains checked exactly after range modification.')
(out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
