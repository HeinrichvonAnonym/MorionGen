import sys,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.cwd()/'utils'))
from motiogen.evalkit.k.scoring import angles
from motiogen.evalkit.k.bones import PARENTS
root=Path.cwd();base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k34_20261117')
old=json.loads((root/'config/evalkit_k33_20261117/summary.json').read_text());old={(r['method'],r['window_key']):r for r in old['fits']}
fig,axs=plt.subplots(4,2,figsize=(12,13),constrained_layout=True)
for row,(method,w,metric) in enumerate([('motionlcm','031','trunk_incline'),('ardy','051','trunk_incline'),('primal','046','trunk_incline'),('kimodo','010','left_knee_flexion')]):
 a=np.load(Path(old[(method,w)]['output'])/'comparison.npz');b=np.load(base/f'fits/baseline/{w}/{method}/comparison.npz');np.testing.assert_array_equal(a['R'],b['R']);t=b['times']-b['times'][0]
 for p,label,color,style in [(b['R'],'R original','#234e70','-'),(a['D'],'previous D','#d8b269','--'),(a['F:new'],'previous F','#c6a9a6','--'),(b['D'],'selected D','#b87910','-'),(b['F:new'],'selected F','#bd3e3e','-')]:axs[row,0].plot(t,angles(p)[metric],label=label,color=color,ls=style)
 axs[row,0].set_title(f'{method} {w}: {metric}');axs[row,0].set_xlabel('Window time (s)');axs[row,0].set_ylabel('Degrees');axs[row,0].legend(fontsize=8);axs[row,0].grid(alpha=.2)
 i=int(np.argmax(angles(b['R'])[metric]));right=b['R'][0,6]-b['R'][0,2];right[1]=0;right/=np.linalg.norm(right);forward=np.cross([0,1,0],right);origin=b['R'][i,0]
 for key,label,color in [('R','Original','#234e70'),('D','Degraded','#b87910'),('F:new','Model','#bd3e3e')]:
  p=b[key][i]-origin;x=p@forward;y=p[:,1]
  for j,par in enumerate(PARENTS):
   if par>=0:axs[row,1].plot(x[[par,j]],y[[par,j]],color=color,alpha=.7,lw=1.8)
  axs[row,1].scatter(x,y,color=color,s=9,label=label)
 axs[row,1].set_title(f'Selected R / D / F, t={t[i]:.2f}s');axs[row,1].set_aspect('equal');axs[row,1].set_xlabel('Fixed sagittal projection (m)');axs[row,1].set_ylabel('Height relative to R pelvis (m)');axs[row,1].legend(fontsize=8);axs[row,1].grid(alpha=.2)
fig.savefig(root/'develop_logs/notes/20261117_k34/alignment.png',dpi=140)
