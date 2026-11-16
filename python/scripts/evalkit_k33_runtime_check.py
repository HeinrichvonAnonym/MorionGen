"""Full-window production refinement with model-coupler enforcement."""
import sys,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import biomech,opensim as osim
from motiogen.evalkit.k import load
from motiogen.evalkit.k.refinement import refine
from motiogen.evalkit.k.fitting3 import positions
osim.Logger.setLevelString('error')
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k33_conversion_20261117')
row=next(r for r in json.loads((BASE/'exports.json').read_text()) if r['method']=='ardy' and r['window_key']=='051')
folder=BASE/'fits/051/ardy/fit';work=BASE/'coupling_runtime';work.mkdir(exist_ok=True)
cfg=json.loads((ROOT/'config/evalkit_k_v3_1.json').read_text());motion=load(row['source'],'ardy',config=cfg)
t,labels,data=biomech.read_storage(folder/'base/coordinates.mot');inside=(t>=0)&(t<=motion.times[-1]-motion.times[0]+1e-8);t=t[inside];data=data[inside]
model=work/'model.osim';shutil.copy2(folder/'base/model.osim',model)
s=dict(cfg['fitting']);s.update(s['per_method']['ardy']);s['enforce_coordinate_couplers']=True
q,diag=refine(model,motion,t,labels,data,s);p,n=positions(model,t,labels,q)
with np.load(folder/'kinematics.npz') as z:
    ids=[i for i,name in enumerate(labels) if not name.endswith('_beta')]
    independent=float(np.max(abs(q[:,ids]-z['coordinates'][:,ids])));position=float(np.max(abs(p-z['fitted'])))
# Numerical equivalence tolerance: 0.0001 degree / one micrometre; not an accuracy tier.
assert independent<1e-4 and position<1e-6,(independent,position)
assert all(np.max(abs(np.deg2rad(q[:,labels.index('knee_angle_'+side)])-q[:,labels.index('knee_angle_'+side+'_beta')]))<1e-12 for side in 'lr')
result={'status':'passed','numerical_tolerance':{'independent_coordinate_deg':1e-4,'endpoint_m':1e-6},'independent_coordinate_max_difference':independent,'endpoint_max_difference_m':position,'coordinate_coupling':diag['coordinate_coupling'],'model_constraint_contract':diag['model_constraint_contract']}
(ROOT/'develop_logs/test_runs/20261117_k_conversion/coupling_runtime.json').write_text(json.dumps(result,indent=2));print('passed',independent,position)
