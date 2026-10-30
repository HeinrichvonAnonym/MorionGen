"""Post-event motion diagnostics on consumed/executed frames, not queued futures."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.runtime import checksum,json_write
from motiogen.posture_geometry import observable_angles
BASE=ROOT/'develop_logs/test_runs/20261028_supplement'
source=BASE/'runtime_pilot_audit.json'
rows=[]
for run in json.loads(source.read_text())['rows']:
    row={k:run[k] for k in ['job','method','seed','clock','load','ignore','status','run_dir','resource_profile','cohort_role']}
    rows.append(row)
    if run['status']!='success':continue
    folder=Path(run['run_dir']);ready=json.loads((folder/'ready.json').read_text())
    trace=json.loads((folder/'runtime_trace.json').read_text())
    if run['method']=='closd':
        motion_path=folder/'motion.npz';steps=trace['steps'];fps=30
        times=np.array([s['simulation_step_end_ns'] for s in steps],dtype=np.int64)
        kind='native PhysX executed poses; native simulation time and wall time differ'
    else:
        motion_path=folder/'consumed_motion.npz';steps=json.loads((folder/'consumption.json').read_text())['rows']
        times=np.array([s['consume_ns'] for s in steps],dtype=np.int64)
        fps=20 if run['method']=='ardy' else 30
        kind='independent kinematic playback; held poses retained'
    xyz=np.load(motion_path)['joints'];names=np.load(folder/'motion.npz')['joint_names'].tolist()
    assert len(xyz)==len(times) and np.isfinite(xyz).all()
    wall=(times-ready['start_ns'])/1e9
    arrival=(trace['events'][0]['arrival_ns'] if run['method']=='closd' else next(json.loads(v)['arrival_ns'] for v in (folder/'input_events.jsonl').read_text().splitlines() if json.loads(v)['event']=='read'))
    after=times>=arrival
    before=~after
    speed=np.r_[0.,np.linalg.norm(np.diff(xyz[:,0,:2],axis=0),axis=-1)*fps]
    jerk=np.diff(xyz,n=3,axis=0)*fps**3
    row.update(channel=kind,frames=len(xyz),source=str(motion_path),source_sha256=checksum(motion_path),
        native_duration_seconds=len(xyz)/fps,observed_wall_seconds=float(wall[-1]),
        frames_consumed_within_wall_20s=int(np.sum(wall<=20)),
        root_speed_before_event_mps=float(speed[before].mean()) if before.any() else None,
        root_speed_after_event_mps=float(speed[after].mean()) if after.any() else None,
        consumed_motion_jerk_rms_mps3=float(np.sqrt(np.mean(jerk**2))),
        kinematic_time_basis='native frame interval; wall-clock delays reported separately')
    angles=observable_angles(xyz,names)
    for side in ['left','right']:
        row[side+'_arm_above90_postevent_fraction']=float(np.mean(angles[side+'_upper_arm_elevation_deg'][after]>90)) if after.any() else None
    if run['method'] in ['dart','closd']:
        key='goal_xy' if run['method']=='dart' else 'goal'
        target=np.asarray(run['event_payload'][0]['payload'][key])
        distance=np.linalg.norm(xyz[:,0,:2]-target,axis=-1)
        within=np.flatnonzero(wall<=20)
        row.update(revealed_goal_xy=target.tolist(),
            revealed_goal_final_distance_m=float(distance[-1]),
            revealed_goal_distance_at_wall_20s_m=float(distance[within[-1]]) if len(within) else None,
            revealed_goal_postevent_min_distance_m=float(distance[after].min()) if after.any() else None,
            revealed_goal_postevent_frame_mean_distance_m=float(distance[after].mean()) if after.any() else None,
            goal_scope='Common externally revealed goal used for evaluation even in ignored negative control; does not claim it was consumed. Continuous distance, no invented success threshold.')
        indices=np.unique(np.r_[np.arange(0,len(xyz),max(1,len(xyz)//120)),len(xyz)-1])
        row['distance_curve']=[[float(wall[i]),float(i/fps),float(distance[i])] for i in indices]
    else:
        row['semantic_scope']='Root speed and arm elevation are descriptive diagnostics, not an automatic text-task success classifier.'
json_write(BASE/'online_motion_response.json',dict(rows=rows,source_sha256=checksum(source),
    scope='All queue states retained. Completed motion diagnostics use executed/consumed frames, including playback holds. Frame averages are not task-level independent samples; simulation rollout endpoints beyond 20 wall seconds are distinguished.'))
print(json.dumps({'rows':len(rows),'completed':sum(r['status']=='success' for r in rows)}))
