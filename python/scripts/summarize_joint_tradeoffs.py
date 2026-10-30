"""Scenario-clustered conditional effects; no arbitrary cross-metric utility."""
import collections
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
source=BASE/'joint_correction_tradeoffs.json'
groups=collections.defaultdict(list)
for row in json.loads(source.read_text())['pairs']:
    family=str(Path(row['task']).parent) if row['method']=='kimodo' else 'native_registered_tasks'
    key=(row['method'],row['component'],row['clock'],json.dumps(row['other_conditions'],sort_keys=True),family)
    groups[key].append(row)
metrics=['hand_position_error','position_success_rate','contact_distance','object_goal_error','jerk_rms',
         'foot_sliding_given_5cm_contact_10cms','model_rtf','trunk_inclination','constraint_end_effector','rotation_geodesic_deg','foot_skate_ratio','foot_skate_from_height','foot_skate_from_pred_contacts','foot_contact_consistency']
rows=[]
for key,pairs in sorted(groups.items()):
    for metric in metrics:
        clusters=collections.defaultdict(list)
        for pair in pairs:
            value=pair['metrics'].get(metric,{}).get('on_minus_off')
            if value is not None and np.isfinite(value):clusters[pair['task']].append(value)
        values=np.array([np.mean(v) for v in clusters.values()])
        if not len(values):continue
        rng=np.random.default_rng(20261029)
        interval=np.quantile(values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1),[.025,.975]) if len(values)>1 else [None,None]
        rows.append(dict(method=key[0],component=key[1],clock=key[2],other_conditions=json.loads(key[3]),family=key[4],
            metric=metric,scenario_count=len(values),complete_pair_count=sum(map(len,clusters.values())),
            registered_pair_count=len(pairs),scenario_mean_on_minus_off=float(values.mean()),
            scenario_bootstrap95=list(interval),
            paired_values_below_zero=sum(v<0 for vals in clusters.values() for v in vals),
            paired_values_above_zero=sum(v>0 for vals in clusters.values() for v in vals)))
json_write(BASE/'joint_tradeoff_summary.json',dict(rows=rows,source_sha256=checksum(source),
    scope='Conditional component contrasts stratified by clock and all other switches; Kimodo constraint families remain separate. Average seeds within each registered task before resampling tasks. No frames-as-samples, no cross-clock sample doubling, no arbitrary utility weighting. These are descriptive marginal intervals, not multiplicity-adjusted confirmatory claims.'))
print(json.dumps({'conditional_metric_summaries':len(rows)}))
