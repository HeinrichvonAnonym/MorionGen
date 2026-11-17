import sys
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts']
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evalkit_k35_legs import BASE
root=BASE/'evalkit_k38_root_20261117/074/kimodo';old=np.load(root/'old.npz');both=np.load(root/'both_anchors.npz');t=old['times'];R=old['R']
fig,axes=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
for name,p,c in [('Original R',R,'#263b50'),('Old D',old['D'],'#d99d25'),('Old fitted F',old['F:output'],'#b74c5f'),('Corrected D',both['D'],'#66aab3'),('Both corrected F',both['F:output'],'#118878')]:axes[0,0].plot(t,p[:,0,1],label=name,color=c)
axes[0,0].set(title='Pelvis midpoint: world vertical position',ylabel='m')
for name,v,c in [('Old D - R',old['D']-R,'#d99d25'),('Old F - D',old['F:output']-old['D'],'#b74c5f'),('Old F - R',old['F:output']-R,'#263b50')]:axes[0,1].plot(t,v[:,0,1]*100,label=name,color=c)
axes[0,1].axhline(0,color='gray',lw=.5);axes[0,1].set(title='Signed vertical errors: partial cancellation',ylabel='cm')
for name,filename,c in [('Old', 'old','#999999'),('Export anchor only','export_anchor_only','#d99d25'),('Fit anchor only','fit_anchor_only','#b74c5f'),('Both anchors','both_anchors','#118878')]:
 z=np.load(root/(filename+'.npz'));d=z['F:output']-R
 axes[1,0].plot(t,np.linalg.norm(d[:,0],axis=-1)*100,label=name,color=c)
 axes[1,1].plot(t,np.linalg.norm(d[:,5],axis=-1)*100,label=name,color=c)
axes[1,0].set(title='2 x 2 control: pelvis endpoint error vs original R',ylabel='cm')
axes[1,1].set(title='2 x 2 control: left foot error vs original R',ylabel='cm')
for ax in axes.flat:ax.set_xlabel('Window time (s)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Kimodo 074: separate root conventions from fitting translation; complete 497-frame window',fontsize=14)
p=Path('develop_logs/notes/20261117_k38');p.mkdir(exist_ok=True);fig.savefig(p/'074_root_stages.png',dpi=150)
