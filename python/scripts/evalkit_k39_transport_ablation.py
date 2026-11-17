import sys,json
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts',str(Path(__file__).parent)]
import numpy as np
import opensim as osim
from motiogen.evalkit.k.limb_transport import transport
from evalkit_k39_prepare import BASE,atomic
from motiogen.evalkit.k.root_alignment import align_fitted_root
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.trustworthy import guard,fidelity
osim.Logger.setLevelString('error')
base=BASE/'evalkit_k39_ablation_20261117';out=BASE/'evalkit_k39_transport_ablation_20261117';out.mkdir(exist_ok=True)
for p in sorted(base.glob('*/*/result.json')):
 a=p.parent/'g0s0';z=np.load(a/'fit.npz');names=list(z['names']);labels=list(z['labels']);t=z['times'];D=common(z['source'],names);baseline=common(z['fitted'],names)
 for name in ['g0s1','g1s1']:
  b=p.parent/name;new=np.load(b/'fit.npz')
  for limbs in [('legs',),('legs','arms')]:
   target=out/p.parent.parent.name/p.parent.name/(name+'_'+'_'.join(limbs));target.mkdir(parents=True,exist_ok=True)
   if (target/'result.json').exists():continue
   q,receipt=transport(a/'kinematic_model.osim',b/'kinematic_model.osim',labels,z['coordinates'],new['coordinates'],limbs)
   q,F,valid,shift,root=align_fitted_root(b/'kinematic_model.osim',t,labels,q,new['source'],names)
   accepted,checks=guard(D,baseline,common(F,names),t)
   np.savez_compressed(target/'fit.npz',times=t,labels=labels,names=names,coordinates=q,fitted=F,source=new['source'])
   atomic(target/'result.json',{'accepted':accepted,'checks':checks,'fidelity':fidelity(D,common(F,names),t),'transport':receipt,'model':str(b/'kinematic_model.osim'),'root':root})
   print(target.relative_to(out),accepted,checks['failures'],flush=True)
