"""Subject-specific anatomy on saved UHC qpos FK, separate from cached-body metrics."""
import json,sys,time
from pathlib import Path
import numpy as np
import xml.etree.ElementTree as ET
for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'unicode':str,'str':str}.items():
 if name not in np.__dict__:setattr(np,name,value)
import torch,smplx
from smplx.vertex_ids import vertex_ids
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
from motiogen.motion import HUMANML_NAMES
from motiogen.posture_geometry import observable_angles
from motiogen.posture_rotations import recovered_angles
from motiogen.posture_scoring import score_angle_series
B=ROOT/'develop_logs/test_runs/20261028_supplement';O=B/'uhc_native_posture';O.mkdir(exist_ok=True)
D=Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
mapping=['Pelvis','L_Hip','R_Hip','Torso','L_Knee','R_Knee','Spine','L_Ankle','R_Ankle','Chest','L_Toe','R_Toe','Neck','L_Thorax','R_Thorax','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist','L_Hand','R_Hand']
torch.set_num_threads(2);rows=[];started=time.monotonic()
for row in json.loads((B/'uhc_executed_pose/export.json').read_text())['rows']:
 saved=np.load(row['path']);gender=str(saved['gender'].item());model=D/f'outputs/assets/closd-official/data/smpl/SMPL_{gender.upper()}.pkl'
 body=smplx.SMPLLayer(str(model),num_betas=16).eval()
 with torch.no_grad():neutral=body(betas=torch.tensor(saved['beta'],dtype=torch.float32).reshape(1,-1))
 joints=neutral.joints[0,:24].numpy();vertices=neutral.vertices[0].numpy()
 xml=ET.parse(row['xml']);native={node.get('name'):np.fromstring(node.get('pos'),sep=' ') for node in xml.findall('.//body')}
 native_joints=np.array([native[name] for name in mapping])
 cal=dict(names=HUMANML_NAMES+['left_hand','right_hand'],neutral_joints=native_joints.tolist(),parents=body.parents.numpy().tolist(),rotation_joint_count=24,sha256=checksum(row['xml']),
  hand_landmarks={side:{finger:(vertices[vertex_ids['smplh'][prefix+finger]]-joints[wrist]+native_joints[wrist]).tolist() for finger in ['thumb','middle']} for side,prefix,wrist in [('left','l',20),('right','r',21)]},
  geometry_scope='Native archived XML global joint centers calibrate bone offsets; SMPL subject mesh supplies hand-axis directions translated to the native wrist. No SMPL fitted pose replaces qpos.',
  smpl_to_native_max_joint_offset_m=float(np.linalg.norm(joints-native_joints,axis=-1).max()),smpl_model_sha256=checksum(model),
  beta=saved['beta'].tolist(),gender=gender)
 names=saved['body_names'].tolist();order=[names.index(n) for n in mapping]
 xyz=saved['joints'][:,order];rot=Rotation.from_quat(saved['rotations_xyzw'][:,order].reshape(-1,4)).as_matrix().reshape(len(xyz),24,3,3)
 result=dict(row,method='uhc',task=row['source'],seed=1,condition={'rfc':bool(row['rfc'])},motion_path=row['path'],frames=len(xyz),human_validation='pending_no_raters')
 try:
  angles=observable_angles(xyz,cal['names']);recovered,proof=recovered_angles(xyz,rot,cal);angles.update(recovered)
  arrays,scores=score_angle_series(angles,30);path=O/(Path(row['path']).stem+'.npz');np.savez_compressed(path,**arrays)
  cp=path.with_suffix('.calibration.json');json_write(cp,cal)
  result.update(status='rotation_components_computed',ergonomic=scores,arrays=str(path),calibration=proof,calibration_path=str(cp))
 except Exception as exc:result.update(status='recovery_failed',reason=f'{type(exc).__name__}: {exc}')
 rows.append(result);print(json.dumps({'count':len(rows),'status':result['status'],'reason':result.get('reason')}),flush=True)
json_write(O/'analysis.json',dict(rows=rows,elapsed_seconds=time.monotonic()-started,scope='Native saved qpos channel; original tracking metrics remain on cached body_xpos, whose discrepancy is retained. Neutral mesh landmarks calibrate axes, not observed finger motion. Subject betas and model gender preserved.'))
