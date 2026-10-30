"""Uniform-scale executed/reference previews; full-precision metrics are separate."""
import json,sys
from pathlib import Path
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
from motiogen.motion import HUMANML_NAMES
B=ROOT/'develop_logs/test_runs/20261028_supplement'
mapping=['Pelvis','L_Hip','R_Hip','Torso','L_Knee','R_Knee','Spine','L_Ankle','R_Ankle','Chest','L_Toe','R_Toe','Neck','L_Thorax','R_Thorax','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist','L_Hand','R_Hand']
# Both native MuJoCo/PhysX traces use depth-first articulated body order.
native=['Pelvis','L_Hip','L_Knee','L_Ankle','L_Toe','R_Hip','R_Knee','R_Ankle','R_Toe','Torso','Spine','Chest','Neck','Head','L_Thorax','L_Shoulder','L_Elbow','L_Wrist','L_Hand','R_Thorax','R_Shoulder','R_Elbow','R_Wrist','R_Hand']
rows=[]
for r in json.loads((B/'common_tracking_analysis.json').read_text())['rows']:
 folder=Path(r['evidence'])
 if r['method']=='UHC':
  p=folder/'native_result.pkl';a=joblib.load(p);xyz=a['pred_jpos'].reshape(-1,24,3)[1:];ref=a['gt_jpos'].reshape(-1,24,3)[1:];names=native
 else:
  p=folder/'execution_trace.npz';a=np.load(p);xyz=a['joints'];ref=a['reference_joints'];names=a['body_names'].tolist()
 order=[names.index(n) for n in mapping];xyz=xyz[:,order];ref=ref[:,order]
 ids=np.unique(np.r_[np.arange(0,len(xyz),3),len(xyz)-1]);e=np.linalg.norm(xyz-ref,axis=-1).mean(axis=-1)
 rows.append(dict(method=r['method'],source=r['source'],rfc=r['rfc'],completion_fraction=r['completion_fraction'],
  names=HUMANML_NAMES+['left_hand','right_hand'],seconds=((ids+1)/30).tolist(),
  joints=np.round(xyz[ids],4).tolist(),reference=np.round(ref[ids],4).tolist(),mean_joint_error_m=e[ids].tolist(),
  full_precision_source=str(p),sha256=checksum(p),display_stride=3,observed_end_seconds=len(xyz)/30))
json_write(B/'tracking_previews.json',dict(rows=rows,scope='All eight sources, three execution conditions each. Preview every third native frame plus final frame; positions rounded to 0.1mm for display only. Native reference differs by body geometry. Frozen executed trajectories end where native termination occurs; no extrapolation.'))
print(len(rows))
