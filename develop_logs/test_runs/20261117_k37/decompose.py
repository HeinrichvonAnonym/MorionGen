"""Full-window and fixed time-bin leg error attribution; reference used only here."""
import sys,json
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts']
import numpy as np
from evalkit_k35_legs import BASE,NAMES,atomic
from motiogen.evalkit.k.geometry import unit
from motiogen.evalkit.k.bones import normalize

def analyze(path):
 z=np.load(Path(path)/'comparison.npz');t=z['times'];idx=[NAMES.index('left_'+n) for n in ['hip','knee','ankle','foot']];out={}
 for name in ['baseline','candidate','selected']:
  x=z['F:'+name][:,idx];r=z['R'][:,idx];d=z['D'][:,idx];vf=np.diff(x,axis=1);vr=np.diff(r,axis=1);lf=np.linalg.norm(vf,axis=-1);lr=np.linalg.norm(vr,axis=-1);uf=unit(vf);ur=unit(vr)
  anchor=x[:,0]-r[:,0];length=((lf-lr)[...,None]*ur).sum(1);direction=(lf[...,None]*(uf-ur)).sum(1);total=x[:,-1]-r[:,-1]
  np.testing.assert_allclose(anchor+length+direction,total,atol=1e-12)
  terms={'hip_anchor':anchor,'length':length,'direction':direction,'total':total,'hip_fit':x[:,0]-d[:,0],'hip_degradation':d[:,0]-r[:,0]}
  record={k:{'p95_m':float(np.percentile(np.linalg.norm(v,axis=-1),95)),'median_m':float(np.median(np.linalg.norm(v,axis=-1))),'vector_bias_m':v.mean(0).tolist()} for k,v in terms.items()}
  steps=np.degrees(np.arccos(np.clip((uf[1:]*uf[:-1]).sum(-1),-1,1)));i,j=np.unravel_index(steps.argmax(),steps.shape)
  record['jump']={'degrees':float(steps[i,j]),'times':t[i:i+2].tolist(),'bone':int(j)}
  record['phases']={label:{k:float(np.percentile(np.linalg.norm(v[mask],axis=-1),95)) for k,v in terms.items()} for label,mask in [('0_to_0.5',t<.5),('0.5_to_1', (t>=.5)&(t<1)),('1_to_end',t>=1)]}
  normalized=normalize(z['F:'+name],z['reference_lengths'])-normalize(z['R'],z['reference_lengths'])
  record['normalized_foot_p95_m']=float(np.percentile(np.linalg.norm(normalized[:,idx[-1]],axis=-1),95));out[name]=record
 return out
if __name__=='__main__':
 paths={'k36':BASE/'evalkit_k36_panel_20261117/combined/074'}
 for stage in ['probe','chart']:
  paths.update({p.parent.name:p for p in (BASE/('evalkit_k37_'+stage+'_20261117')).glob('*/074') if (p/'comparison.npz').exists()})
 result={k:analyze(p) for k,p in paths.items()};atomic(Path('develop_logs/test_runs/20261117_k37/decomposition.json'),result)
 for k,v in result.items():print(k,json.dumps(v['candidate'],indent=2))
