import sys,json
sys.path[:0]=['utils','utils/motiogen','python/scripts']
import numpy as np
import opensim as osim
import biomech
from evalkit_k35_legs import BASE,atomic
from motiogen.evalkit.k.geometry import unit
from pathlib import Path
osim.Logger.setLevelString('error');p=BASE/'evalkit_k37_panel_20261117/hip_flexion_chart/074';z=np.load(p/'coordinates.npz');labels=list(z['labels']);q=z['candidate'][0].copy();model=osim.Model(str(p/'model.osim'));state=model.initSystem();values=np.linspace(0,175,3501);angle=[]
for v in values:
 q[labels.index('knee_angle_l')]=v;biomech._set(model,state,labels,q)
 x=np.array([biomech._child_origin(model,state,j) for j in ['hip_l','walker_knee_l','ankle_l']]);u=unit(np.diff(x,axis=0));angle.append(float(np.degrees(np.arccos(np.clip(u[0]@u[1],-1,1)))))
i=int(np.argmax(angle));out={'coordinate_sweep_deg':[0,175],'spacing_deg':.05,'max_geometric_flexion_deg':angle[i],'at_coordinate_deg':float(values[i]),'at_upper_bound_deg':angle[-1],'scope':'geometric knee angle attainable by copied fixed-geometry model at fixed remaining pose, within selected coordinate range; no anatomy inference'}
atomic(Path('develop_logs/test_runs/20261117_k37/knee_sweep.json'),out);print(out)
