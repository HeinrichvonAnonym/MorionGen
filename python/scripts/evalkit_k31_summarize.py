"""Paired, participant-balanced K3.1 report; frozen windows and no retuning."""
import argparse,json,sys,hashlib,shutil,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.scoring import region
REGIONS=['pelvis','torso','left_arm','right_arm','left_leg','right_leg']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def mean(values):return float(np.mean(values)) if values else None
def aggregate(rows,fn):
 groups=defaultdict(list)
 for r in rows:groups[r['group_id']].append(fn(r))
 return mean([mean(v) for v in groups.values()])
def regstat(score,reg,stage,field='p95_abs'):
 values=[v[stage][field] for m,v in score['raw']['angles'].items() if region(m)==reg and v[stage][field] is not None]
 return mean(values)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();cfg=ROOT/'config/evalkit_k31_full_20261116';cfg.mkdir(exist_ok=True)
 frozen=read(a.out/'freeze.json');old_summary=read(ROOT/'config/evalkit_k3_20261116/results_summary.json');oldrows=[read(p) for p in old_summary['result_paths']]
 assert sha(ROOT/'config/evalkit_k_v3_1.json')==frozen['config_sha256']
 assert sha(ROOT/'python/scripts/evalkit_k31_full.py')==frozen['runner_sha256']
 for p,h in frozen['implementation'].items():assert sha(ROOT/p)==h,p
 checked_inputs={}
 helpers={}
 for name in ['python/scripts/evalkit_k3_rescue_accept.py','python/scripts/evalkit_k3_experiment.py','utils/motiogen/biomech.py']:
  baseline=subprocess.check_output(['git','show','c0018cf:'+name],cwd=ROOT)
  assert (ROOT/name).read_bytes()==baseline,name
  helpers[name]=sha(ROOT/name)
 rows=[];failed=[]
 for old in oldrows:
  path=a.out/old['window_key']/old['method']/'result.json';r=read(path)
  assert r['freeze']==frozen['freeze']
  if r['status']!='computed':failed.append(r);continue
  assert sha(r['reference'])==r['identity']['reference']
  assert sha(old['arrays'])==r['identity']['comparison']
  baseline=Path(old['arrays']).parent/'optimized_raw'
  assert sha(baseline/'kinematics.npz')==r['identity']['baseline_kinematics']
  assert sha(baseline/'model.osim')==r['identity']['baseline_model']
  for p,h in r['identity']['inputs'].items():
   if p not in checked_inputs:checked_inputs[p]=sha(p)
   assert checked_inputs[p]==h,p
  r['old_score']=old['scores']['optimized_raw'];r['result_path']=str(path)
  for space in ['raw','normalized']:
   for metric,v in r['scores'][space]['angles'].items():
    assert v['degradation']==r['old_score'][space]['angles'][metric]['degradation'],(r['id'],metric,space)
    assert v['end_to_end']['valid']==v['degradation']['valid'],(r['id'],metric,'changed scoring mask')
    if r['method']=='closd':assert v['end_to_end']==r['old_score'][space]['angles'][metric]['end_to_end'],(r['id'],metric,'CLoSD reuse drift')
  rows.append(r)
 assert len(rows)+len(failed)==656
 if failed:raise ValueError('Incomplete execution: '+str([(r['id'],r['method'],r['reason']) for r in failed]))
 methods={};local=[];temporal=[];steps=[];coverage=[];position_changes=[];allchanges=[]
 for method in sorted({r['method'] for r in rows}):
  pool=[r for r in rows if r['method']==method];assert len(pool)==82
  def body(r,old=False):return mean([regstat(r['old_score'] if old else r['scores'],reg,'end_to_end') for reg in REGIONS])
  entry={'records':len(pool),'groups':len({r['group_id'] for r in pool}),
   'K3_body_p95':aggregate(pool,lambda r:body(r,True)),'K31_body_p95':aggregate(pool,body),
   'subsets':{},'regions':{},'positions':{}}
  for scope,part in [('LAB',[r for r in pool if r['subset']=='LAB']),('industrial',[r for r in pool if r['subset']!='LAB'])]:
   entry['subsets'][scope]={'records':len(part),'K3':aggregate(part,lambda r:body(r,True)),'K31':aggregate(part,body)}
  for reg in REGIONS:
   e={stage:aggregate(pool,lambda r:regstat(r['scores'],reg,stage)) for stage in ['degradation','fit_deviation','end_to_end']}
   e['K3_end_to_end']=aggregate(pool,lambda r:regstat(r['old_score'],reg,'end_to_end'))
   e['median']=aggregate(pool,lambda r:regstat(r['scores'],reg,'end_to_end','median_abs'))
   e['signed_bias']=aggregate(pool,lambda r:regstat(r['scores'],reg,'end_to_end','signed_bias'))
   e['tiers']={str(k):aggregate(pool,lambda r:regstat(r['scores'],reg,'end_to_end','within_'+str(k))) for k in [5,10,20]}
   e['metric_frames']={key:sum(v['end_to_end'][key] for r in pool for m,v in r['scores']['raw']['angles'].items() if region(m)==reg) for key in ['valid','total']}
   e['coverage']={k:aggregate(pool,lambda r:r['coverage'][reg][k]) for k in ['angle','registered_angle','registered_position']}
   e['coverage']['angle']=aggregate(pool,lambda r:r['scores']['coverage'][reg]['share'])
   for version in ['K3','new']:
    e[version+'_velocity_p95']=aggregate(pool,lambda r:r['temporal']['regions'][reg][version]['velocity_error_p95_deg_s'])
   entry['regions'][reg]=e
  for joint in ['left_wrist','right_wrist','left_foot','right_foot']:
   entry['positions'][joint]={space:{stage:aggregate(pool,lambda r:r['scores'][space]['positions'][joint][stage]['world']['p95_abs']) for stage in ['degradation','fit_deviation','end_to_end']} for space in ['raw','normalized']}
  for joint in entry['positions']:
   for space in ['raw','normalized']:
    entry['positions'][joint][space]['K3_end_to_end']=aggregate(pool,lambda r:r['old_score'][space]['positions'][joint]['end_to_end']['world']['p95_abs'])
  for r in pool:
   for joint in ['left_wrist','right_wrist','left_foot','right_foot']:
    before=r['old_score']['raw']['positions'][joint]['end_to_end']['world']['p95_abs'];after=r['scores']['raw']['positions'][joint]['end_to_end']['world']['p95_abs']
    if after-before>.01:position_changes.append({'id':r['id'],'method':r['method'],'joint':joint,'K3_m':before,'K31_m':after,'delta_m':after-before,'result_path':r['result_path']})
   for reg in REGIONS:
    t=r['temporal']['regions'][reg];before=regstat(r['old_score'],reg,'end_to_end');after=regstat(r['scores'],reg,'end_to_end');delta=after-before
    v0=t['K3']['velocity_error_p95_deg_s'];v1=t['new']['velocity_error_p95_deg_s']
    change={k:r[k] for k in ['id','method','group_id','subset','window_key','result_path']}
    change.update(region=reg,K3=before,K31=after,delta_deg=delta,velocity_K3=v0,velocity_K31=v1,
       max_step_K3=t['K3']['maximum_step_deg'],max_step_K31=t['new']['maximum_step_deg'],
       input_direction_coverage_K3=r['old_score']['coverage'][reg]['share'],input_direction_coverage_K31=r['scores']['coverage'][reg]['share'])
    allchanges.append(change)
    if delta>1:local.append(change)
    if v1-v0>20 and v1>1.2*v0:temporal.append(change)
    if change['max_step_K31']>change['max_step_K3']+5 and change['max_step_K31']>1.2*change['max_step_K3']:steps.append(change)
    if change['input_direction_coverage_K31']<change['input_direction_coverage_K3']-.05:coverage.append(change)
  methods[method]=entry
 preserved=read(ROOT/'config/evalkit_k3_rescue_20261116/experiment.json');protected={**preserved['protected']['files'],**preserved['k3_sha256']}
 for p,h in protected.items():assert sha(ROOT/p)==h,p
 assert not subprocess.check_output(['git','diff','c0018cf','--','plans'],cwd=ROOT)
 oldhead=sha(ROOT/'config/evalkit_k3_rescue_20261116/acceptance.json')
 summary={'status':'complete','scope':'all fixed development windows; original reference; masks fixed; no configuration retuning',
  'freeze':frozen['freeze'],'config_sha256':frozen['config_sha256'],'unchanged_helpers':helpers,'completed':len(rows),'failed':failed,'windows':82,'methods':methods,
  'thresholds':{'local_pose_increase_deg':1,'velocity_increase_deg_s':20,'velocity_ratio':1.2,'coverage_drop':.05,'max_step_increase_deg':5,'max_step_ratio':1.2,'position_p95_increase_m':.01,
   'scope':'engineering review flags, not biomechanical acceptance standards; all differences retained'},
  'local_regressions':sorted(local,key=lambda r:r['delta_deg'],reverse=True),'temporal_regressions':sorted(temporal,key=lambda r:r['velocity_K31']-r['velocity_K3'],reverse=True),
  'position_regressions':sorted(position_changes,key=lambda r:r['delta_m'],reverse=True),'step_regressions':sorted(steps,key=lambda r:r['max_step_K31']-r['max_step_K3'],reverse=True),'coverage_drops':coverage,'all_region_changes':allchanges,'result_paths':[r['result_path'] for r in rows],
  'preservation':{'protected_files':len(protected),'all_unchanged':True,'holdout_runs':0,'k31_prior_acceptance_sha256':oldhead,'plans_unchanged':True},
  'resource':{'free_gib':shutil.disk_usage(a.out).free/1024**3,'max_workers':3,'numeric_threads':1},
  'head_scope':'native head/neck path unchanged; prior reference checks retained; full-body solver contains no independent head position fit'}
 assert summary['resource']['free_gib']>=80
 write(cfg/'summary.json',summary);shutil.copy2(a.out/'freeze.json',cfg/'freeze.json')
 print('complete',len(rows),'local flags',len(local),'temporal flags',len(temporal),'step flags',len(steps),'coverage drops',len(coverage))
if __name__=='__main__':main()
