"""Native pose-channel anatomy, explicitly distinct from predicted-joint playback."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import smplx
from smplx.joint_names import JOINT_NAMES

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
from motiogen.posture_geometry import observable_angles
from motiogen.posture_rotations import recovered_angles
from motiogen.posture_scoring import score_angle_series
from motiogen.motion import HUMANML_NAMES
DATA=Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
OUT=BASE/'dart_native_pose_posture';OUT.mkdir(exist_ok=True)
models=DATA/'outputs/runtime/dart/data/smplx_lockedhead_20230207/models_lockedhead'
torch.set_num_threads(2)
body=smplx.build_layer(models,model_type='smplx',gender='male',ext='npz',num_pca_comps=12).eval()
with torch.no_grad():neutral=body().joints[0,:55].numpy()
parents=body.parents.numpy().astype(int)
aliases={f'{side}_{finger}{i}':side.title()+'Hand'+finger.title()+str(i) for side in ['left','right'] for finger in ['index','middle','ring','pinky','thumb'] for i in [1,2,3]}
cal={'names':[aliases.get(n,n) for n in JOINT_NAMES[:55]],'neutral_joints':neutral.tolist(),
     'parents':parents.tolist(),'rotation_joint_count':22,'sha256':checksum(models/'smplx/SMPLX_MALE.npz'),
     'gender':'male','betas':[0.]*10,'scope':'Neutral axes from official locked-head model; no template hand articulation is treated as observed.'}
json_write(OUT/'calibration.json',cal)
rows=[];started=time.monotonic()
for statepath in [BASE/'dart_matrix/queue/state.json',BASE/'dart_load_queue/state.json']:
    state=json.loads(statepath.read_text())
    for job,status in state['jobs'].items():
        if status['status']!='success':continue
        run=Path(status['run_dir']);trace=json.loads((run/'runtime_trace.json').read_text())
        positions=[];rotations=[];prediction=[];source=[]
        for i,step in enumerate(trace['steps']):
            path=run/f'primitive_{i:03d}.npz';saved=np.load(path)
            assert np.all(saved['betas']==0)
            local=np.concatenate([saved['global_orient'][0,:,None],saved['body_pose'][0]],axis=1)
            rot=np.empty_like(local);xyz=np.empty((len(local),22,3))
            for j,parent in enumerate(parents[:22]):
                rot[:,j]=local[:,j] if parent<0 else rot[:,parent]@local[:,j]
                xyz[:,j]=neutral[0]+saved['transl'][0] if parent<0 else xyz[:,parent]+np.einsum('tij,j->ti',rot[:,parent],neutral[j]-neutral[parent])
            # Validate analytic FK against independent native SMPL-X skinning
            # once per run; the reference model generates actual posed joints.
            if i==0:
                with torch.no_grad():
                    reference=body(betas=torch.from_numpy(saved['betas'][0]),global_orient=torch.from_numpy(saved['global_orient'][0])[:,None],body_pose=torch.from_numpy(saved['body_pose'][0]),transl=torch.from_numpy(saved['transl'][0])).joints[:,:22].numpy()
                error=float(np.abs(reference-xyz).max())
                if error>2e-4:raise ValueError(f'Independent SMPL-X FK disagreement {error}')
            positions.append(xyz);rotations.append(rot);prediction.append(saved['joints'][0]);source.append({'path':str(path),'sha256':checksum(path)})
        initial=trace['steps'][0]['frame']
        total=len(np.load(run/'motion.npz')['joints'])
        xyz=np.concatenate(positions)[:total-initial];rot=np.concatenate(rotations)[:len(xyz)];pred=np.concatenate(prediction)[:len(xyz)]
        angles=observable_angles(xyz,HUMANML_NAMES);recovered,proof=recovered_angles(xyz,rot,cal);angles.update(recovered)
        arrays,scores=score_angle_series(angles,30)
        arrays.update(pose_channel_joints=xyz,predicted_channel_joints=pred)
        path=OUT/(run.name+'.npz');np.savez_compressed(path,**arrays)
        difference=np.linalg.norm(xyz-pred,axis=-1)
        rows.append(dict(job=job,run=str(run),source_primitives=source,frames=len(xyz),initial_frames_without_saved_pose=initial,
            independent_body_model_fk_max_error_m=error,calibration=proof,ergonomic=scores,arrays=str(path),
            mean_joint_channel_difference_m=float(difference.mean()),max_joint_channel_difference_m=float(difference.max()),
            channel='native SMPL-X pose channel, not predicted-joint playback',human_validation='pending_no_raters'))
        if len(rows)%8==0:print(json.dumps({'completed':len(rows),'elapsed_seconds':time.monotonic()-started}),flush=True)
json_write(OUT/'analysis.json',dict(rows=rows,elapsed_seconds=time.monotonic()-started,
    scope='64 frozen runtime pilots. Pose-channel anatomy is separate from the predicted joint channel used for task state/playback. Initial history poses were not saved and remain missing. No template fingers or unknown task conditions are treated as observations.'))
