import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.cwd()/'utils'))
from motiogen.evalkit.k.scoring import angles
from motiogen.evalkit.k.geometry import unit
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k35_legs_20261117')
fig,axs=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
for row,(w,m,weight,label) in enumerate([('010','kimodo',.25,'Rejected candidate'),('074','kimodo',.5,'Rejected candidate'),('038','omnicontrol',.5,'Accepted candidate')]):
 z=np.load(base/f'foot_{weight}/{w}/{m}/comparison.npz');t=z['times']-z['times'][0]
 for key,name,color in [('R','Original R','#244f75'),('D','Degraded D','#b58418'),('F:baseline','Baseline','#777777'),('F:candidate',label,'#c34e52')]:axs[row,0].plot(t,angles(z[key])['left_knee_flexion'],label=name,color=color)
 target=unit(z['D'][:,5]-z['D'][:,4])
 for key,name,color in [('F:baseline','Baseline','#777777'),('F:candidate',label,'#c34e52')]:
  v=unit(z[key][:,5]-z[key][:,4]);error=np.degrees(np.arccos(np.clip((v*target).sum(-1),-1,1)));axs[row,1].plot(t,error,label=name,color=color)
 axs[row,0].set_title(f'{m} {w}: left knee geometry');axs[row,1].set_title('Foot direction error relative to input D')
 for ax in axs[row]:ax.set_xlabel('Window time (s)');ax.set_ylabel('Degrees');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig('develop_logs/notes/20261117_k35/tradeoffs.png',dpi=150)
