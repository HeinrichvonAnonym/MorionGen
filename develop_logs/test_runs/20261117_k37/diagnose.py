import sys,json
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts']
import numpy as np
from evalkit_k35_legs import BASE,NAMES,atomic
from motiogen.evalkit.k.geometry import unit
from motiogen.evalkit.k.bones import normalize
p=BASE/'evalkit_k36_panel_20261117/combined/074';z=np.load(p/'comparison.npz');q=np.load(p/'coordinates.npz');t=z['times'];idx=[NAMES.index('left_'+n) for n in ['hip','knee','ankle','foot']];out={}
for key in ['R','D','F:baseline','F:candidate']:
 x=z[key][:,idx];v=np.diff(x,axis=1);u=unit(v);steps=np.degrees(np.arccos(np.clip((u[1:]*u[:-1]).sum(-1),-1,1)));i,j=np.unravel_index(steps.argmax(),steps.shape)
 out[key]={'length_median':np.median(np.linalg.norm(v,axis=-1),axis=0).tolist(),'max_step':float(steps[i,j]),'step_frame':int(i),'step_time':t[i:i+2].tolist(),'step_bone':int(j)}
 if key.startswith('F'):
  d=x-z['R'][:,idx];norm=normalize(z[key],z['reference_lengths'])-normalize(z['R'],z['reference_lengths']);hip=d[:,0];chain=d-hip[:,None]
  out[key].update(world_p95=np.percentile(np.linalg.norm(d,axis=-1),95,axis=0).tolist(),hip_relative_p95=np.percentile(np.linalg.norm(chain,axis=-1),95,axis=0).tolist(),normalized_p95=np.percentile(np.linalg.norm(norm[:,idx],axis=-1),95,axis=0).tolist(),length_component_p95=np.percentile(np.linalg.norm(d-norm[:,idx],axis=-1),95,axis=0).tolist())
  if key=='F:candidate':
   labels=list(q['labels']);cols=[k for k,n in enumerate(labels) if n.endswith('_l') and any(s in n for s in ['hip','knee','ankle','subtalar'])]
   out[key]['jump_coordinates']={labels[k]:q['candidate'][max(0,i-3):i+5,k].tolist() for k in cols}
   out[key]['ranges']={labels[k]:[float(q['candidate'][:,k].min()),float(q['candidate'][:,k].max())] for k in cols}
   err=np.linalg.norm(d[:,-1],axis=-1);bad=np.argsort(err)[-10:];out[key]['worst_positions']=[{'frame':int(k),'time':float(t[k]),'error':float(err[k]),'hip_error':float(np.linalg.norm(hip[k]))} for k in bad]
print(json.dumps(out,indent=2));atomic(Path('develop_logs/test_runs/20261117_k37/diagnosis.json'),out)
# Exact vector closure: anchor + length along reference + direction at fitted length.
x=z['F:candidate'][:,idx];r=z['R'][:,idx];vf=np.diff(x,axis=1);vr=np.diff(r,axis=1);lf=np.linalg.norm(vf,axis=-1);lr=np.linalg.norm(vr,axis=-1);uf=unit(vf);ur=unit(vr)
anchor=x[:,0]-r[:,0];length=((lf-lr)[...,None]*ur).sum(1);direction=(lf[...,None]*(uf-ur)).sum(1)
np.testing.assert_allclose(anchor+length+direction,x[:,-1]-r[:,-1],atol=1e-12)
out['vector_decomposition']={k:{'median_m':float(np.median(np.linalg.norm(v,axis=-1))),'p95_m':float(np.percentile(np.linalg.norm(v,axis=-1),95))} for k,v in {'hip_anchor':anchor,'length':length,'direction':direction,'total':anchor+length+direction}.items()}
out['vector_decomposition']['scope']='exact per-frame vectors; scalar norms and percentiles do not add'
atomic(Path('develop_logs/test_runs/20261117_k37/diagnosis.json'),out)
print(json.dumps(out['vector_decomposition'],indent=2))
