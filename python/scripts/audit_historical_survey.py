"""Reconcile retained 1007/1008 evidence without regenerating pretrained outputs."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.runtime import checksum,json_write
from motiogen.native_analysis import clustered_interval

base=Path(load_config()['data_root'])/'benchmarks/kimodo-stratified-20261008'
cases=json.loads((base/'selection_manifest.json').read_text())['cases']
old_path=ROOT/'develop_logs/test_runs/20261008_064957_kimodo_evaluate-native_aa2faed4/supplemental_analysis/summary.json'
old=json.loads(old_path.read_text());rows=[];failures=[]
for item in cases:
    case=item['case'];folder=base/'generated_raw'/case
    try:
        with np.load(folder/'motion.npz',allow_pickle=False) as motion:
            for key in motion.files:
                value=motion[key]
                if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():raise ValueError('Nonfinite '+key)
        metrics=json.loads((folder/'metrics.json').read_text())['per_motion_mean_gen']
        seed=json.loads((base/'testsuite'/case/'seed_motion.json').read_text()) if (base/'testsuite'/case/'seed_motion.json').exists() else {}
        rows.append({'case':case,'cluster':seed.get('bvh_path',case),'metrics':metrics,'metrics_sha256':checksum(folder/'metrics.json')})
    except Exception as exc:failures.append({'case':case,'reason':str(exc)})
    if (len(rows)+len(failures))%256==0:print(json.dumps({'raw_checked':len(rows),'failures':len(failures)}),flush=True)
absolute=[]
for metric in ['constraint_root2d_err','constraint_end_effector','constraint_fullbody_keyframe']:
    selected=[r for r in rows if metric in r['metrics'] and np.isfinite(r['metrics'][metric])]
    values=np.array([r['metrics'][metric] for r in selected]);previous=old['raw_official']['generated'][metric]
    if len(values)!=previous['n'] or not np.isclose(values.mean(),previous['mean'],atol=1e-12):raise ValueError('Historical metric aggregation changed')
    absolute.append({'metric':metric,'n':len(values),'mean_m':float(values.mean()),'p95_m':float(np.quantile(values,.95)),
        'successes_5cm':int((values<=.05).sum()),'failures_5cm':int((values>.05).sum()),
        'source_cluster_mean':clustered_interval(values,[r['cluster'] for r in selected])})
timings={}
for label,filename in [('raw','raw_timing.jsonl'),('postprocess','postprocess_timing.jsonl')]:
    records=[json.loads(line) for line in (base/filename).read_text().splitlines() if line.strip()]
    accepted=[r for r in records if r['status']=='success']
    if len({r['case'] for r in accepted})!=len(accepted):raise ValueError('Duplicate timing case')
    timings[label]={r['case']:r for r in accepted}
pairs=json.loads((ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json').read_text())['rows'];paired=[]
for row in pairs:
    raw=timings['raw'][row['case']];post=timings['postprocess'][row['case']]
    if raw['frames']!=post['frames'] or raw['duration_seconds']!=post['duration_seconds']:raise ValueError('Timing duration differs')
    paired.append({'case':row['case'],'cluster':row['cluster'],'raw_seconds':raw['seconds'],'post_seconds':post['seconds'],
        'delta_seconds':post['seconds']-raw['seconds'],'delta_rtf':post['rtf']-raw['rtf']})
timing_summary={metric:clustered_interval([r[metric] for r in paired],[r['cluster'] for r in paired]) for metric in ['delta_seconds','delta_rtf']}
kit_path=ROOT/'develop_logs/test_runs/20261008_kit_ml_reference/summary.json'
kit=json.loads(kit_path.read_text());kit_rows=[json.loads(line) for line in (kit_path.parent/'per_case.jsonl').read_text().splitlines()]
if len(kit_rows)!=830 or len({r['id'] for r in kit_rows})!=830:raise ValueError('KIT count/duplicate mismatch')
for metric,summary in kit['metrics'].items():
    values=[r[metric] for r in kit_rows if r.get(metric) is not None]
    if not np.allclose(np.quantile(values,[.05,.5,.95]),[summary['p05'],summary['p50'],summary['p95']],atol=1e-12):raise ValueError('KIT quantile mismatch')
result={'raw_expected':len(cases),'raw_accepted':len(rows),'raw_failures':failures,'absolute':absolute,'raw_rows':rows,
    'timing_pairs':paired,'timing_delta':timing_summary,
    'timing_scope':'Historical separate raw/postprocess batches, same case and duration; includes native generation/decoding/correction, excludes text/model preload. Order was not randomized: paired observed latency difference is not an isolated causal correction timer.',
    'kit':kit,'kit_aggregation_check':'830 unique rows and all reported quantiles exactly reproduced; original official RIC recovery receipt retained, no new population sample',
    'sources':[str(old_path),str(base/'selection_manifest.json'),str(base/'raw_timing.jsonl'),str(base/'postprocess_timing.jsonl'),str(kit_path)]}
destination=ROOT/'develop_logs/test_runs/20261024_historical_survey_audit/audit.json';json_write(destination,result)
print(json.dumps({'raw_accepted':len(rows),'raw_failures':failures,'timing_pairs':len(paired),'kit':len(kit_rows),'output':str(destination)}),flush=True)
