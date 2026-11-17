"""Read-only diagnosis of the fixed first eleven failures; no fitting or selection."""
import json,sys,hashlib,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path('/home/heinrich/MorionGen')
P=json.loads((ROOT/'config/evalkit_k4_20261118/full_regression_v1/protocol.json').read_text());RUN=Path(P['output'])
sys.path[:0]=[str(Path(P['fit_code_root'])/'utils'),str(Path(P['fit_code_root'])/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion
from motiogen.degradation.reference import sample
from motiogen.degradation.metrics import points,angle
from motiogen.evalkit.k.scoring import DIRECTIONS
OUT=ROOT/'develop_logs/test_runs/20261118_k4_first11'
snapshot=json.loads((OUT/'snapshot.json').read_text())
assert len(snapshot['failed'])==11

def read_motion(path):
 with np.load(path) as z:return Motion(z['positions'],list(z['names']),z['times'],'analysis')
def storage(path):
 lines=path.read_text().splitlines();i=next(i for i,l in enumerate(lines) if l.strip()=='endheader')
 return lines[i+1].split(),np.loadtxt(lines[i+2:])
def stat(a):
 a=np.asarray(a);return dict(min=float(np.nanmin(a)),median=float(np.nanmedian(a)),p95=float(np.nanpercentile(a,95)),max=float(np.nanmax(a)))
def ranges(t,mask):
 ix=np.flatnonzero(mask);groups=np.split(ix,np.flatnonzero(np.diff(ix)>1)+1)
 return [[round(float(t[g[0]]),4),round(float(t[g[-1]]),4),int(len(g))] for g in groups if len(g)]
result=[]
for item in snapshot['failed']:
 key=item['key'];task=RUN/'tasks'/key;s=json.loads((task/'summary.json').read_text());row=next(r for r in P['records'] if r['window']+'_'+r['method']==key)
 fit=Path(item['result']['fit']);rec=json.loads((fit/'receipt.json').read_text());sel=json.loads((fit/'candidate_selection.json').read_text());score=json.loads(Path(s['score_files']['native']).read_text())
 d=dict(key=key,source=row,selected=s['selected'],failures=s['failed_metrics'],coverage_passed=all(v['status']=='passed' for c in s['acceptance'].values() for a in c.values() for v in a['full_window_coverage']),clock=s['clock'],original_clock=s['original_clock'],input_boundary_missing_frames=s['input_boundary_missing_frames'],native_export_max_abs_position_m=s['native_export_max_abs_position_m'],export_readback=s['export_audit'],metrics=s['metrics'],temporal={})
 for c in ['native','export']:
  sc=json.loads(Path(s['score_files'][c]).read_text());d['temporal'][c]={clock:{pair:block['comparisons'][pair]['continuous']['temporal'] for pair in ['D-R','F-D','F-R']} for clock,block in [('D_clock',sc),('R_clock',sc['original_clock'])]}
 d['candidate_quality']={n:{'rank':q['rank'],'passed':q['passed'],'failed':{k:v for k,v in q['metrics'].items() if not v['passed']}} for n,q in sel['qualities'].items()}
 d['geometry']={'targets':rec['geometry']['targets'],'after':rec['geometry']['after']}
 native=read_motion(task/'views/native'/row['window']/row['method']/'fk.npz');original=read_motion(score['original_reference_file']);r=points(sample(original,native.times),True);f=points(native)
 d['phases']={}
 names={x['metric'].split('/')[1] for x in s['failed_metrics'] if x['metric'].startswith('directions_deg/')}
 names|={'left_thigh','left_shank','left_foot_axis'} if row['window']=='010' else set()
 for name in sorted(names):
  a,b= ('left_hip','right_hip') if name=='pelvis_line' else DIRECTIONS[name]
  e=angle(r[b]-r[a],f[b]-f[a]);d['phases'][name]={'error':stat(e),'above_5deg':ranges(native.times,e>5),'max_time':float(native.times[np.nanargmax(e)])}
 d['original_pose']={}
 for side in ['left','right']:
  d['original_pose'][side+'_knee_flexion_deg']=stat(180-angle(r[side+'_hip']-r[side+'_knee'],r[side+'_ankle']-r[side+'_knee']))
  d['original_pose'][side+'_ankle_height_m']=stat(r[side+'_ankle'][:,1]);d['original_pose'][side+'_knee_height_m']=stat(r[side+'_knee'][:,1])
 d['original_pose']['trunk_tilt_deg']=stat(angle(r['shoulder_mid']-r['pelvis_mid'],np.tile([0,1,0],(len(native.times),1))))
 labels,q=storage(fit/'coordinates_refined_native.mot');xml=ET.parse(fit/'model.osim');bounds={c.attrib['name']:np.fromstring(c.findtext('range',''),sep=' ')*180/np.pi for c in xml.findall('.//Coordinate')}
 d['coordinate_limits']={}
 for name in ['hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l','subtalar_angle_l','lumbar_extension','lumbar_bending','lumbar_rotation']:
  if name not in labels:continue
  a=q[:,labels.index(name)];lo,hi=bounds[name];d['coordinate_limits'][name]={'range_deg':[float(lo),float(hi)],'native':stat(a),'near_limit_frames':int(np.sum((a<lo+.05)|(a>hi-.05))),'max_raw_step_deg':float(np.max(abs(np.diff(a))))}
 if key=='010_ardy':
  with np.load(fit/'fit.npz') as z:et=z['times'];eq=z['coords'];el=list(z['labels'])
  d['coordinate_steps']={}
  for name in labels[1:]:
   a=q[:,labels.index(name)];j=int(np.argmax(abs(np.diff(a))));step=abs(np.diff(a))[j]
   if step>10:d['coordinate_steps'][name]={'raw_step':float(step),'time_interval':q[j:j+2,0].tolist(),'values':a[j:j+2].tolist(),'export_between':eq[(et>=q[j,0]-1e-8)&(et<=q[j+1,0]+1e-8),el.index(name)].tolist()}
 ref=json.loads((fit/'refinement.json').read_text());attempts=[a for frame in ref['frames'] for a in frame.get('attempts',[])]
 d['solver_termination']={'frames':len(ref['frames']),'attempt_status_counts':{str(v):sum(a['status']==v for a in attempts) for v in set(a['status'] for a in attempts)},'after_loss':stat([f['after_loss'] for f in ref['frames']])}
 d['source_files_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [task/'summary.json',fit/'receipt.json',fit/'candidate_selection.json',fit/'coordinates_refined_native.mot',fit/'model.osim']}
 result.append(d)
(OUT/'analysis.json').write_text(json.dumps({'snapshot_total':snapshot['total'],'scope':'Fixed eleven failures; independent read-only R scoring analysis; no new fit, configuration or promotion','records':result},indent=2,allow_nan=False)+'\n')
for d in result:
 if d['key'] not in ['002_kimodo','008_motionlcm','010_ardy']:continue
 print(d['key'],'pose',d['original_pose'],'phases',d['phases'],'limits',d['coordinate_limits'],'steps',d.get('coordinate_steps'))
