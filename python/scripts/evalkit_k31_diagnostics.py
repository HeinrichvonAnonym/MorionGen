"""Explain flagged full-regression cases without changing configuration or masks."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.scoring import angles,region
from motiogen.evalkit.k.fitting3 import support
cfg=ROOT/'config/evalkit_k31_full_20261116';s=json.loads((cfg/'summary.json').read_text());out={'step_cases':[],'coverage_cases':[],'temporal_cases':[]}
for case in s['step_regressions']:
 r=json.loads(Path(case['result_path']).read_text())
 with np.load(Path(r['output'])/'comparison.npz') as z:t=z['times'];v={name:angles(z[key]) for name,key in [('R','R'),('D','D'),('K31','F:k31')]}
 with np.load(r['arrays']) as z:v['K3']=angles(z['F:optimized_raw'])
 metrics=[m for m in v['R'] if region(m)==case['region']];metric=max(metrics,key=lambda m:np.max(abs(np.diff(v['K31'][m]))));i=int(np.argmax(abs(np.diff(v['K31'][metric]))))
 out['step_cases'].append({'case':case,'metric':metric,'interval_s':t[i:i+2].tolist(),'step_at_interval_deg':{k:float(abs(np.diff(x[metric])[i])) for k,x in v.items()},'maximum_reference_step_deg':float(np.max(abs(np.diff(v['R'][metric]))))})
for case in s['coverage_drops']:
 r=json.loads(Path(case['result_path']).read_text());folder=Path(r['arrays']).parent
 item={'case':case,'range_summary':{}}
 for name,path in [('K3',folder/'optimized_raw/kinematics.npz'),('K31',Path(r['output'])/'kinematics.npz')]:
  with np.load(path) as z:q=support(z['source'],z['fitted'],list(z['names']))[case['region']]
  error=np.max(q['angle_error_deg'],axis=1);item['range_summary'][name]={'max_bone_direction_error_p05_median_p95_deg':np.percentile(error,[5,50,95]).tolist(),'fraction_within_0_1deg_of_threshold':float(np.mean(abs(error-10)<=.1))}
 out['coverage_cases'].append(item)
for case in s['temporal_regressions']:
 r=json.loads(Path(case['result_path']).read_text())
 with np.load(Path(r['output'])/'comparison.npz') as z:t=z['times'];R=angles(z['R']);D=angles(z['D']);F=angles(z['F:k31'])
 metrics=[m for m in R if region(m)==case['region']];metric=max(metrics,key=lambda m:np.percentile(abs(np.gradient(F[m],t)-np.gradient(R[m],t)),95))
 with np.load(Path(r['output'])/'kinematics.npz') as z:qt=z['times'];q=z['coordinates'];labels=list(z['labels'])
 # Inspect solver-grid endpoints separately from interpolation interiors.
 anchors=np.unique(np.r_[np.arange(0,len(qt),5),len(qt)-1]);on=np.min(abs(t[:,None]-qt[anchors][None,:]),axis=1)<1e-7
 inside=t>.1
 angular=abs(F[metric]-R[metric]);item={'case':case,'metric':metric,'after_0_1s_diagnostic_only':{
  'anchor_error_p95_deg':float(np.percentile(angular[on&inside],95)),
  'intermediate_error_p95_deg':float(np.percentile(angular[~on&inside],95))},'large_coordinate_jumps':[]}
 for name in ['arm_flex_r','arm_add_r','arm_rot_r','elbow_flex_r']:
  column=labels.index(name);delta=np.diff(q[anchors,column]);ids=np.flatnonzero(abs(delta)>90)
  for i in ids:item['large_coordinate_jumps'].append({'coordinate':name,'interval_s':qt[anchors[i:i+2]].tolist(),'values_deg':q[anchors[i:i+2],column].tolist()})
 item['scope']='grid/interior and coordinate jumps diagnose interpolation; full-window acceptance and masks remain unchanged'
 out['temporal_cases'].append(item)
(cfg/'flag_diagnostics.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
print('steps',[(r['case']['method'],r['interval_s'],r['step_at_interval_deg']) for r in out['step_cases']])
print('coverage',[(r['case']['method'],r['range_summary']) for r in out['coverage_cases']])
print('temporal',out['temporal_cases'])
