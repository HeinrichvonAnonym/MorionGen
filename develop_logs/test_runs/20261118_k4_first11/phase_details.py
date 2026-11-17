"""Read saved poses only; phase diagnosis and static figure, no numerical fitting."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/heinrich/MorionGen');OUT=ROOT/'develop_logs/test_runs/20261118_k4_first11'
p=json.loads((ROOT/'config/evalkit_k4_20261118/full_regression_v1/protocol.json').read_text());run=Path(p['output']);code=Path(p['fit_code_root'])
sys.path[:0]=[str(code/'utils'),str(code/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion
from motiogen.degradation.reference import sample
from motiogen.degradation.metrics import points,angle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def motion(path):
 with np.load(path) as z:return Motion(z['positions'],list(z['names']),z['times'],'diagnostic')
def interp(m,t):
 return Motion(np.stack([np.interp(t,m.times,m.positions[:,j,k]) for j in range(len(m.names)) for k in range(3)],axis=1).reshape(len(t),len(m.names),3),m.names,t,'diagnostic')
def storage(path):
 lines=path.read_text().splitlines();i=lines.index('endheader');return lines[i+1].split(),np.loadtxt(lines[i+2:])
fig,axs=plt.subplots(3,2,figsize=(13,10),layout='constrained');results={}
for row,key in enumerate(['002_kimodo','008_motionlcm','010_ardy']):
 task=run/'tasks'/key;w,m=key.split('_');s=json.loads((task/'summary.json').read_text());fit=Path(json.loads((task/'result.json').read_text())['fit']);score=json.loads(Path(s['score_files']['native']).read_text());raw=motion(score['original_reference_file']);native=motion(task/'views/native'/w/m/'fk.npz');export=motion(task/'views/export'/w/m/'fk.npz');t=native.times;r=points(sample(raw,t),True);f=points(native);labels,q=storage(fit/'coordinates_refined_native.mot');results[key]={}
 if row==0:
  for n,a,b in [('thigh','left_hip','left_knee'),('shank','left_knee','left_ankle'),('foot','left_ankle','left_foot')]:axs[row,0].plot(t,angle(r[b]-r[a],f[b]-f[a]),label=n)
  bound=(q[:,labels.index('ankle_angle_l')]<-49.95)|(q[:,labels.index('subtalar_angle_l')]<-74.95)
  err=angle(r['left_foot']-r['left_ankle'],f['left_foot']-f['left_ankle']);results[key]['foot_error_over_5_frames']=int((err>5).sum());results[key]['over5_at_ankle_or_subtalar_bound']=int(((err>5)&bound).sum());results[key]['bound_frames']=int(bound.sum())
  for n in ['ankle_angle_l','subtalar_angle_l']:axs[row,1].plot(t,q[:,labels.index(n)],label=n)
  axs[row,1].axhline(-50,color='grey',ls=':');axs[row,1].axhline(-75,color='grey',ls=':')
  for when in [3.75,4.3,7.8,8.1]:
   i=int(abs(t-when).argmin());results[key][str(when)]={'time':float(t[i]),'trunk_tilt':float(angle((r['shoulder_mid']-r['pelvis_mid'])[i],[0,1,0])),'left_knee_flexion':float(180-angle((r['left_hip']-r['left_knee'])[i],(r['left_ankle']-r['left_knee'])[i]))}
 if row==1:
  for n,a,b in [('trunk','pelvis_mid','shoulder_mid'),('shoulder line','left_shoulder','right_shoulder'),('pelvis line','left_hip','right_hip')]:axs[row,0].plot(t,angle(r[b]-r[a],f[b]-f[a]),label=n)
  for n in ['left_shoulder','right_shoulder','shoulder_mid']:axs[row,1].plot(t,np.linalg.norm(f[n]-r[n],axis=1)*100,label=n)
  axs[row,1].set_ylabel('Position error (cm)');axs[row,1].axhline(5,color='grey',ls=':')
 if row==2:
  rt=raw.times[(raw.times>=t[0])&(raw.times<=t[-1])];rr=points(sample(raw,rt),True);nn=points(interp(native,rt));ee=points(interp(export,rt))
  for n,x in [('native FK, position interpolation',nn),('exported model FK',ee)]:
   err=angle(rr['left_foot']-rr['left_ankle'],x['left_foot']-x['left_ankle']);axs[row,0].plot(rt,err,label=n)
   idx=np.argmax(err);results[key][n]={'max_deg':float(err[idx]),'max_time_s':float(rt[idx]),'p95_deg':float(np.percentile(err,95)),'over5_frames':int((err>5).sum()),'frames':len(rt)}
  for n in ['hip_rotation_l','subtalar_angle_l']:axs[row,1].plot(t,q[:,labels.index(n)],marker='.',label=n)
  axs[row,1].set_xlim(.2,.7)
 for col in range(2):axs[row,col].set_title(key+(' / F-R directions' if col==0 else ' / fixed-model diagnostics'));axs[row,col].set_xlabel('Original recording time (s)');axs[row,col].legend(fontsize=8);axs[row,col].grid(alpha=.2)
 axs[row,0].axhline(5,color='grey',ls=':');axs[row,0].set_ylabel('Direction error (deg)')
 if row!=1:axs[row,1].set_ylabel('Model coordinate (deg)')
fig.suptitle('First 11 failures: original-motion acceptance, frozen configuration',fontsize=14);fig.savefig(OUT/'phase_diagnostics.png',dpi=150)
(OUT/'phase_details.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
