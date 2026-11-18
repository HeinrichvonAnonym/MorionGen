"""Real-output geometry ablation: F-D only; retain every observed denominator."""
import hashlib,json
from pathlib import Path
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117/closure_real_geometry_20261118_v1')
rows=[];sources={}
for condition in ('control','total_proxy'):
 folder=B/condition/'motionlcm';files=[folder/'fidelity.json',folder/'actual_export/audit.json',folder/'fit/receipt.json']
 for p in files:sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 fidelity,audit,receipt=[json.loads(p.read_text()) for p in files]
 views={}
 for view,m in fidelity['views'].items():
  failures=[];missing=[];metrics={}
  def check(name,stat,threshold,key='p95',baseline=0):
   n,den=stat['n'],stat['denominator'];value=stat[key]
   metrics[name]={'value':value,'n':n,'denominator':den,'limit':threshold,'baseline':baseline,'statistic':key}
   if n!=den:missing.append(name)
   if n and value>baseline+threshold:failures.append(name)
  for kind in ('directions_deg','joint_angles_deg'):
   for name,stat in m[kind].items():check(kind+'/'+name,stat,5)
  for name,value in m['positions_cm'].items():
   for geometry in ('world','pelvis_relative','common_length_world','common_length_pelvis_relative'):
    check('positions_cm/'+name+'/'+geometry,value[geometry],5)
  for name,value in m['temporal'].items():
   check('temporal/'+name+'/rate',value['direction_rate_vector_error_deg_s'],10)
   check('temporal/'+name+'/max_step',value['output_step_deg'],5,'max',value['input_step_deg']['max'])
  views[view]={'D_clock_passed':not failures and not missing,'failures':failures,'missing':missing,'metrics':metrics,'rotation_support':m['native_frame_support']}
 rows.append({'condition':condition,'views':views,'actual_export_passed':audit['passed'],'fixed_geometry_estimate':receipt['geometry']['targets'].get('effective_trunk_estimate'),'scope':'Actual MotionLCM input fidelity, no paired R. Export assessed on D timestamps; dense path relative to D interpolation assumptions is separate.'})
result={'records':rows,'sources_sha256':sources,'all_observable_D_clock_targets_passed':{r['condition']:r['actual_export_passed'] and all(v['D_clock_passed'] for v in r['views'].values()) for r in rows},'promotion':'none','new_holdout_uses':0,'scope':'Single-factor real F-D evidence; same control reproduced exactly. Not controlled F-R accuracy or full82-window transfer.'}
(B/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(result['all_observable_D_clock_targets_passed'])
for r in rows:print(r['condition'],{v:(m['failures'],m['missing']) for v,m in r['views'].items()})
