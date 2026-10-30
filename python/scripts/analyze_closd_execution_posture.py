"""Recover anatomy from native PhysX body rotations, checking the SMPL frame map."""
import json
from pathlib import Path
import sys
import time

import numpy as np
for name,value in {'bool':bool,'int':int,'float':float,'complex':complex,'object':object,'unicode':str,'str':str}.items():
    if name not in np.__dict__:setattr(np,name,value)
import torch
import smplx
from smplx.vertex_ids import vertex_ids
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
from motiogen.posture_geometry import observable_angles
from motiogen.posture_rotations import recovered_angles
from motiogen.posture_scoring import score_angle_series
from motiogen.motion import HUMANML_NAMES
DATA=Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
OUT=BASE/'closd_execution_posture';OUT.mkdir(exist_ok=True)
modelpath=DATA/'outputs/assets/closd-official/data/smpl/SMPL_NEUTRAL.pkl'
torch.set_num_threads(2)
body=smplx.SMPLLayer(str(modelpath),num_betas=10).eval()
with torch.no_grad():neutral=body()
joints=neutral.joints[0,:24].numpy();vertices=neutral.vertices[0].numpy()
cal={'names':HUMANML_NAMES+['left_hand','right_hand'],'neutral_joints':joints.tolist(),
     'parents':body.parents.numpy().tolist(),'rotation_joint_count':24,'sha256':checksum(modelpath),
     'hand_landmarks':{side:{finger:vertices[vertex_ids['smplh'][prefix+finger]].tolist() for finger in ['thumb','middle']} for side,prefix in [('left','l'),('right','r')]},
     'hand_scope':'Neutral mesh landmarks calibrate axes; finger motion is not observed.'}
json_write(OUT/'calibration.json',cal)
mapping=['Pelvis','L_Hip','R_Hip','Torso','L_Knee','R_Knee','Spine','L_Ankle','R_Ankle','Chest','L_Toe','R_Toe','Neck','L_Thorax','R_Thorax','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist','L_Hand','R_Hand']
upright=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
rows=[];started=time.monotonic()
for statepath in [BASE/'closd_runtime/queue/state.json',BASE/'closd_runtime/full_block_queue/state.json',BASE/'closd_matrix/queue/state.json',BASE/'phc_common_tracking/queue_v2/state.json',BASE/'phc_common_tracking/remaining_queue/state.json']:
    state=json.loads(statepath.read_text())
    for job,status in state['jobs'].items():
        if status['status']!='success':continue
        tracking='phc_common_tracking' in str(statepath)
        run=Path(status['run_dir']);path=run/('execution_trace.npz' if tracking else 'native_physics.npz');saved=np.load(path)
        names=saved['body_names'].tolist();order=[names.index(n) for n in mapping]
        xyz=saved['joints'][:,order]
        rot=Rotation.from_quat(saved['rotations_xyzw'][:,order].reshape(-1,4)).as_matrix().reshape(len(xyz),24,3,3)@upright
        receipt=dict(job=job,run=str(run),source_sha256=checksum(path),method='phc_tracker' if tracking else 'closd')
        try:
            angles=observable_angles(xyz,HUMANML_NAMES+['left_hand','right_hand'])
            recovered,proof=recovered_angles(xyz,rot,cal);angles.update(recovered)
            arrays,summaries=score_angle_series(angles,30)
            outfile=OUT/(run.name+'.npz');np.savez_compressed(outfile,**arrays)
            receipt.update(status='rotation_components_computed',calibration=proof,ergonomic=summaries,arrays=str(outfile),
                frames=len(xyz),channel=('CLoSD PHC tracking layer' if tracking else 'CLoSD DiP + PHC')+' native PhysX executed rigid bodies',human_validation='pending_no_raters')
        except Exception as exc:
            receipt.update(status='recovery_failed',reason=f'{type(exc).__name__}: {exc}')
        rows.append(receipt)
        print(json.dumps({'completed':len(rows),'status':receipt['status'],'reason':receipt.get('reason')}),flush=True)
json_write(OUT/'analysis.json',dict(rows=rows,elapsed_seconds=time.monotonic()-started,
    scope='Snapshot of completed online CLoSD and isolated PHC tracking runs; pending queue jobs not included in this anatomy export. Neutral SMPL calibration requires source FK agreement; failed conversion remains a recovery failure, not a simulated task failure.'))
