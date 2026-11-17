import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path.cwd()/'utils'))
from motiogen.evalkit.k.scoring import angles
from motiogen.evalkit.k.geometry import unit
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k36_panel_20261117/combined')
fig,axs=plt.subplots(4,2,figsize=(12,11),constrained_layout=True)
for row,(w,side) in enumerate([('010','left'),('074','right'),('074','left'),('077','right')]):
 z=np.load(base/w/'comparison.npz');t=z['times']-z['times'][0]
 for key,label,color in [('R','Original R','#244f75'),('D','Degraded D','#b58418'),('F:baseline','Nominal model','#777777'),('F:selected','Kinematic candidate / fallback','#c34e52')]:axs[row,0].plot(t,angles(z[key])[side+'_knee_flexion'],label=label,color=color)
 ankle,foot=(4,5) if side=='left' else (8,9);target=unit(z['D'][:,foot]-z['D'][:,ankle])
 for key,label,color in [('F:baseline','Nominal model','#777777'),('F:selected','Selected','#c34e52')]:
  v=unit(z[key][:,foot]-z[key][:,ankle]);axs[row,1].plot(t,np.degrees(np.arccos(np.clip((v*target).sum(-1),-1,1))),label=label,color=color)
 axs[row,0].set_title(f'Kimodo {w}: {side} knee'+(' (baseline retained)' if w=='074' and side=='left' else ''));axs[row,1].set_title('3D foot direction error relative to input D')
 for ax in axs[row]:ax.set_xlabel('Window time (s)');ax.set_ylabel('Degrees');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.savefig('develop_logs/notes/20261117_k36/knee_and_foot.png',dpi=150)
