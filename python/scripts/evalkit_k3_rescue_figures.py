import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.scoring import angles,region
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k3_rescue_20261116');a=json.loads((base/'acceptance.json').read_text())
summary=json.loads((ROOT/'config/evalkit_k3_20261116/results_summary.json').read_text());rows=[json.loads(Path(p).read_text()) for p in summary['result_paths']]
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
fig,axs=plt.subplots(8,2,figsize=(15,26),layout='constrained')
for axes,r in zip(axs,[r for r in a['tail'] if r['variant']=='interior_seed']):
 row=next(x for x in rows if x['method']==r['method'] and x['id']==r['id'])
 with np.load(row['arrays']) as z:t=z['times'];points={'R':z['R'],'D':z['D'],'K3':z['F:optimized_raw']}
 with np.load(r['probe']) as z:p=z['fitted'];points['rescue']=np.stack([np.interp(t,z['times'],v) for v in p.reshape(len(p),-1).T],1).reshape(points['R'].shape)
 values={k:angles(p) for k,p in points.items()};metrics=[m for m in values['R'] if region(m)==r['focus']]
 metric=max(metrics,key=lambda m:np.percentile(abs(values['K3'][m]-values['R'][m]),95))
 for name in values:
  axes[0].plot(t,values[name][metric],label=name)
  axes[1].plot(t,np.gradient(values[name][metric],t)-np.gradient(values['R'][metric],t),label=name)
 axes[0].set_title(r['method']+' · '+metric);axes[1].set_title('相对原始参考的角速度误差')
 for ax in axes:ax.grid(alpha=.2);ax.set_xlabel('秒');ax.legend(ncol=4)
 axes[0].set_ylabel('度');axes[1].set_ylabel('度／秒')
folder=ROOT/'develop_logs/notes/20261116_k3_rescue';folder.mkdir(exist_ok=True)
fig.savefig(folder/'temporal.png',dpi=110)
