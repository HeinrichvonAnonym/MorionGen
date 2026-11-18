import subprocess,sys
from pathlib import Path
b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
args=[sys.executable,'python/scripts/evalkit_k4_closure_report.py','--frozen-run',str(b/'full_regression_20261118_v1')]
for name in ['closure_legs_20261118_v1','closure_path_runtime_20261118_v1','closure_torso_20261118_v1','closure_formats_20261118_v1','closure_upper_temporal_20261118_v1','closure_arm_domain_20261118_v1','closure_064_20261118_v1','closure_trajectory_20261118_v1','closure_trajectory_seed_domain_20261118_v1','closure_trajectory_bounded_20261118_v1','closure_trajectory_balance_20261118_v1','closure_trajectory_geometry_20261118_v1','closure_trajectory_seed_chain_20261118_v1','closure_position_foot_20261118_v1','closure_position_foot_chain_20261118_v1','closure_trajectory_chart_20261118_v1','closure_pool_transfer_20261118_v1','closure_position_transfer_20261118_v1','closure_export_composition_20261118_v1','closure_composed_pool_20261118_v1','closure_geometry_transfer_20261118_v1','closure_solver_trust_20261118_v2_controlled']:
 args+=['--candidate-run',str(b/name)]
for name in ['closure_path_residual_20261118_v1','closure_path_partial_20261118_v1','closure_path_partial_legs_20261118_v1','closure_path_alias_20261118_v1']:
 args+=['--rotation-replay',str(b/name)]
evidence=Path('docs/20261118_k4_budget_evidence.html')
if evidence.exists():args+=['--evidence-report',str(evidence)]
args+=['--budget-review','develop_logs/test_runs/20261118_k4_next/candidate_budget_acceptance_v1.json']
visual=Path('docs/20261118_k4_pool_comparison_v2.html')
if visual.exists():args+=['--pool-visual',str(visual)]
args+=['--unified-run',str(b/'full_regression_20261118_budget_v3'),'--unified-live-port','8768']
args+=['--actual-path-run',str(b/'closure_actual_path_20261118_v1')]
args+=['--actual-ablation-run',str(b/'closure_real_centres_20261118_v1')]
args+=['--real-run',str(b/'closure_real_fidelity_20261118_v1'),'--real-geometry-run',str(b/'closure_real_geometry_20261118_v1')]
args+=['--selection-review',str(b/'closure_selection_audit_20261118_v2/review.json'),'--reachability',str(b/'closure_reachability_20261118_v1'),'--output','docs/20261118_k4_failure_closure.html']
subprocess.run(args,check=True)
