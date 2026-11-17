import sys
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts']
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evalkit_k35_legs import BASE,NAMES,angles
from motiogen.evalkit.k.geometry import unit
old=np.load(BASE/'evalkit_k36_panel_20261117/combined/074/comparison.npz');new=np.load(BASE/'evalkit_k37_panel_20261117/hip_flexion_chart/074/comparison.npz');t=old['times'];i=[NAMES.index('left_'+n) for n in ['hip','knee','ankle','foot']]
curves={'Original R':(old['R'],'#25384a'),'Degraded D':(old['D'],'#d9a022'),'K3.6 fallback':(old['F:selected'],'#999999'),'K3.6 candidate':(old['F:candidate'],'#d05a62'),'K3.7 candidate / selected':(new['F:candidate'],'#118879')}
fig,axes=plt.subplots(2,2,figsize=(15,9),constrained_layout=True)
for label,(p,c) in curves.items():
 axes[0,0].plot(t,angles(p)['left_knee_flexion'],label=label,color=c,lw=1.6)
 if label!='Original R':axes[0,1].plot(t,np.linalg.norm(p[:,i[-1]]-old['R'][:,i[-1]],axis=-1)*100,label=label,color=c,lw=1.6)
 u=unit(np.diff(p[:,i],axis=1));step=np.degrees(np.arccos(np.clip((u[1:]*u[:-1]).sum(-1),-1,1))).max(1)
 axes[1,0].plot(t[1:],step,label=label,color=c,lw=1.6)
x=new['F:candidate'][:,i];r=old['R'][:,i];vf=np.diff(x,axis=1);vr=np.diff(r,axis=1);lf=np.linalg.norm(vf,axis=-1);lr=np.linalg.norm(vr,axis=-1)
terms={'Hip anchor':x[:,0]-r[:,0],'Length term':((lf-lr)[...,None]*unit(vr)).sum(1),'Direction term':(lf[...,None]*(unit(vf)-unit(vr))).sum(1),'Total foot error':x[:,-1]-r[:,-1]}
for n,v in terms.items():axes[1,1].plot(t,np.linalg.norm(v,axis=-1)*100,label=n)
for ax,title,ylabel in zip(axes.flat,['Left knee: all 497 frames','Foot position relative to original R','Maximum step across thigh / shin / foot','K3.7 exact vector terms (norms are not additive)'],['Degrees','cm','Degrees per original frame','cm']):
 ax.set(title=title,xlabel='Window time (s)',ylabel=ylabel);ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[1,0].set_xlim(.25,.75)
fig.suptitle('Kimodo 074: same input, same bone geometry; hip-flexion coordinate-domain change',fontsize=15)
out=Path('develop_logs/notes/20261117_k37');out.mkdir(exist_ok=True);fig.savefig(out/'074_diagnosis.png',dpi=150)
