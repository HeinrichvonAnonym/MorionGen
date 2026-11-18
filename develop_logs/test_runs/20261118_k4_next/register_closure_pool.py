import json,pathlib,copy
r=pathlib.Path('config/evalkit_k4_20261118');out=r/'closure_pool_transfer_v1';out.mkdir(exist_ok=False)
def read(n):return json.loads((r/n).read_text())
pool=read('candidate_pool_v1/pool.json');pool['status']='Explicit closure transfer candidate bank; no default promotion';pool['selection_include_export_path']=True;pool['selection_policy']='observed_then_path_v1'
conditions={
 'rig_centres':('closure_v1/torso/registered_centres.json','required'),
 'rig_arm_domain_centres':('closure_arm_domain_v1/domain_centres.json','required'),
 'rig_reverse_leg_path':('closure_formats_v1/reverse_path.json','required'),
 'rig_reverse_rate_leg_path':('closure_formats_v1/reverse_rate1_path.json','required'),
 'reverse_full_path':('closure_upper_temporal_v1/reverse_fullpath.json',None),
 'position_chain':('closure_trajectory_geometry_v1/native_chain.json','absent'),
 'position_fixed_foot_chain':('closure_position_foot_chain_v1/chain.json','absent'),
 'position_trajectory':('closure_trajectory_chart_v1/control.json','absent'),
 'position_trajectory_chart':('closure_trajectory_chart_v1/interior_chart.json','absent')}
pool['candidate_requirements']={}
for name,(source,rig) in conditions.items():
 pool['candidates'][name]=read(source)
 if rig:pool['candidate_requirements'][name]={'input_rig':rig}
# Explicit interpolation-only sibling of the common forward native policy.
cfg=copy.deepcopy(pool['candidates']['balanced_targets']);cfg['coordinated_path']=read('closure_upper_temporal_v1/reverse_fullpath.json')['coordinated_path'];pool['candidates']['forward_full_path']=cfg
(out/'pool.json').write_text(json.dumps(pool,indent=2)+'\n');panel=read('closure_v1/transfer_panel.json');(out/'sentinels.json').write_text(json.dumps(panel,indent=2)+'\n')
p=dict(version='k4-closure-pool-transfer-1',strict_scoring=True,native_export_scoring=True,conditions=['pool'],new_holdout_uses=0,promotion='none',hypothesis='Validate one input-only candidate bank, including original2, independent rig centres/arm domain/forward and reverse coordinated paths, position geometry/foot geometry/joint trajectory and equivalent chart. Use observed-then-path ranking audited on17 diagnostic combinations. Candidate existence and actual runtime selection are independently checked. Pending077 mechanism candidates retain their experimental status; this is transfer evidence, not final656 freeze.',preregistration='Unchanged23 combination transfer panel from original20+8 dedup. One bank for every input; applicability only by declared input_rig, never window/method ID. Every applicable candidate is computed and retained. Same5deg/5cm/temporal selection limits, no reference metrics or source-jump reference labels in selector; R only external scoring.',sources={n:s for n,(s,_) in conditions.items()},old_candidates_preserved=['pelvis_continuity','balanced_targets'])
(out/'protocol.json').write_text(json.dumps(p,indent=2)+'\n');print('candidates',list(pool['candidates']))
