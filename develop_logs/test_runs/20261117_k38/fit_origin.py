import sys,json
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts']
import numpy as np
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from evalkit_k35_legs import BASE,atomic,read
from motiogen.evalkit.k.fitting3 import positions
from motiogen.evalkit.k.bones import common
from evalkit_k3_experiment import interp
folder=BASE/'evalkit_k3_corrected_20261116/fits/074/kimodo';last=BASE/'evalkit_k37_panel_20261117/hip_flexion_chart/074';z=np.load(last/'comparison.npz');t=z['times'];D=z['D'];R=z['R'];stages={}
for stage,sub in [('initial_ik','base_raw'),('k3_refinement','optimized_raw')]:
 p=folder/sub;tt,labels,q=biomech.read_storage(p/'coordinates.mot');xyz,names=positions(p/'model.osim',tt,labels,q);f=common(interp(tt,xyz,t),names);stages[stage]=(f,interp(tt,q,t)[:,[labels.index(n) for n in ['pelvis_tx','pelvis_ty','pelvis_tz']]])
p=BASE/'evalkit_k31_full_20261116/074/kimodo';q=np.load(p/'kinematics.npz');labels=list(q['labels']);stages['k31']=(common(q['fitted'],list(q['names'])),q['coordinates'][:,[labels.index(n) for n in ['pelvis_tx','pelvis_ty','pelvis_tz']]])
q=np.load(last/'coordinates.npz');stages['k37']=(z['F:selected'],q['selected'][:,[list(q['labels']).index(n) for n in ['pelvis_tx','pelvis_ty','pelvis_tz']]])
stats=lambda v:{'p95_cm':float(np.percentile(np.linalg.norm(v,axis=-1),95)*100),'bias_cm':(v.mean(0)*100).tolist()}
out={s:{'midpoint_fit':stats(f[:,0]-D[:,0]),'midpoint_e2e':stats(f[:,0]-R[:,0]),'root_coordinate_change_from_initial_ik':stats(q-stages['initial_ik'][1])} for s,(f,q) in stages.items()}
out['initial_calibration']=read(folder/'base_raw/receipt.json')['calibration'];atomic(Path('develop_logs/test_runs/20261117_k38/fit_origin.json'),out);print(json.dumps(out,indent=2))
