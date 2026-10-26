"""Independent index-based check of common-history comparison denominators."""
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import json_write
base=ROOT/'develop_logs/test_runs'
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
checks=[];groups={};timing_checks=[]
for row in analysis['rows']:
    if row['method']!='ardy':continue
    worker=json.loads((ROOT/row['run']/'worker_result.json').read_text())
    durations=worker['timing']['replan']['samples_seconds']
    # Independent interpolation and accumulation, not the analysis helper.
    ordered=sorted(float(durations[i]) for i in range(20,50))
    def percentile(q):
        rank=29*q;lower=int(rank);upper=min(lower+1,29)
        return ordered[lower]+(rank-lower)*(ordered[upper]-ordered[lower])
    expected={'matched_full_history_request_count':30,
              'matched_full_history_request_mean_seconds':sum(ordered)/30,
              'matched_full_history_request_p50_seconds':percentile(.5),
              'matched_full_history_request_p95_seconds':percentile(.95),
              'matched_full_history_request_rtf':sum(ordered)/30/.4}
    timing_checks.append({'run':row['run'],'request_indices_1based':[21,50],
        'passed':len(durations)==50 and all(key in row['values'] and np.isclose(value,row['values'][key],rtol=1e-12,atol=1e-12)
                                          for key,value in expected.items()),'expected':expected})
    with np.load(ROOT/row['run']/'motion.npz') as motion:
        flat=motion['overlap_errors_m'];joints=len(motion['joint_names'])
        keep=160 if row['condition']['history_long'] else 32
        chunks=len(motion['joints'])//8
        cumulative=np.cumsum([0]+[min(step*8,keep)*joints for step in range(1,chunks)])
        indices=np.concatenate([np.arange(end-min(step*8,32)*joints,end)
                                for step,end in enumerate(cumulative[1:],1)])
        assert cumulative[-1]==len(flat)
        value=float(np.asarray(flat,dtype=np.float64)[indices].sum()/len(indices))
        passed=(len(indices)==row['values']['history_consistency_common32_samples']
                and np.isclose(value,row['values']['history_consistency_common32_m'],rtol=0,atol=1e-15))
        checks.append({'run':row['run'],'samples':len(indices),'recomputed_m':value,
            'difference_m':abs(value-row['values']['history_consistency_common32_m']), 'passed':bool(passed)})
        groups.setdefault((row['task'],row['seed'],row['clock']),set()).add(len(indices))
same_denominators=all(len(v)==1 for v in groups.values())
summary={'checked_runs':len(checks),'passed_runs':sum(c['passed'] for c in checks),
    'paired_denominators_equal':same_denominators,'samples_per_run':sorted({c['samples'] for c in checks}),
    'max_difference_m':max(c['difference_m'] for c in checks),
    'matched_timing_passed_runs':sum(c['passed'] for c in timing_checks),'matched_request_count':30}
receipt={'analysis_cutoff':analysis['updated_at'],'checks':checks,'timing_checks':timing_checks,'summary':[summary],
    'passed':bool(checks) and all(c['passed'] for c in checks+timing_checks) and same_denominators,
    'scope':'Post-generation fairness refinement: compare most recent min(32, available) frames at each native ARDY call. Reconstruct indices from saved flattened pre-correction overlap distances, not regeneration. Same 41040 joint-frame observations per 400-frame run; these are not independent inference units. Timing also compares identical requests21-50 (30 samples) using independently interpolated quantiles. Original whole-history metric and condition-specific full-buffer timings remain available.'}
json_write(base/'20261024_ardy_history_audit/audit.json',receipt)
print(json.dumps({'passed':receipt['passed'],**summary}))
if not receipt['passed']:raise SystemExit(1)
