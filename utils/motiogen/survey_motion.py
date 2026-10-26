"""Visualization rows derived from canonical motions; no metric aggregation here."""
import json
import numpy as np
from .config import ROOT
from .motion import load_motion

def accepted_pilot_entries(rows,originals):
    """Current task-0 paired block for the originally registered pilot cells.

    Historical pilot receipts remain available separately; the main motion
    viewer must use the accepted replacement, including its native geometry.
    """
    entries=[]
    for original in originals:
        condition=dict(original['condition'])
        if original['method']=='ardy':condition['history_long']=int(condition.pop('history_frames')==160)
        matches=[row for row in rows if row['method']==original['method'] and row['task']==0
                 and row['seed']==original['seed'] and row['clock']=='paused' and row['condition']==condition]
        if len(matches)!=1:raise ValueError('Current accepted pilot cell missing or duplicated: '+original['run'])
        row=matches[0]
        entries.append({**row,'scene':row['task_spec'],'case':'pilot_seed'+str(row['seed']),
                        'historical_pilot_run':original['run']})
    return entries

def component_jerk_series(xyz, fps):
    """Raw component RMS per four-frame window, aligned to its last frame."""
    xyz=np.asarray(xyz,dtype=float)
    if xyz.ndim!=3 or xyz.shape[-1]!=3 or not np.isfinite(xyz).all() or not np.isfinite(fps) or fps<=0:
        raise ValueError('Finite [frames,joints,3] coordinates and positive FPS required')
    result=np.full(len(xyz),np.nan)
    if len(xyz)>3:
        result[3:]=np.sqrt(np.mean((np.diff(xyz,n=3,axis=0)*fps**3)**2,axis=(1,2)))
    return result

def reviewed_motion_rows(entries):
    motions=[]; curves=[]
    geometry_path=ROOT/'develop_logs/test_runs/20261024_chois_survey_geometry/geometry.json'
    geometries={r['run']:r for r in json.loads(geometry_path.read_text())['rows']} if geometry_path.exists() else {}
    for entry in entries:
        run=ROOT/entry['run']; motion=load_motion(ROOT/entry['motion_path'] if entry.get('motion_path') else run/'motion.npz')
        visual_id=entry['run']+('#'+entry['variant'] if entry.get('variant') else '')
        xyz=motion['joints']; names=list(motion['joint_names']); fps=float(motion['fps'])
        label=json.dumps(entry['condition'],ensure_ascii=False)
        frames=len(xyz); case=entry.get('case','pilot_seed'+str(entry['seed']))
        targets=[c for c in entry['scene'].get('constraints',[]) if 'position_m' in c]
        pelvis=xyz[:,names.index('pelvis')]
        neck=xyz[:,names.index('neck')]
        axis=neck-pelvis; length=np.linalg.norm(axis,axis=-1)
        trunk=np.rad2deg(np.arccos(np.clip(axis[:,2]/np.maximum(length,1e-12),-1,1)))
        speed=np.r_[np.nan,np.linalg.norm(np.diff(pelvis,axis=0),axis=-1)*fps]
        jerk=component_jerk_series(xyz,fps)
        distance=np.full(frames,np.nan)
        root_targets=None
        if targets and entry['scene'].get('trajectory_family'):
            # Supplementary multi-anchor path: score only actual hint times.
            # Do not silently turn the first anchor into a constant target.
            for target in targets:
                frame=round(target['time_seconds']*fps)
                distance[frame]=np.linalg.norm(xyz[frame,names.index(target['joint'])]-target['position_m'])
        elif targets and targets[0]['joint'] in names:
            distance=np.linalg.norm(xyz[:,names.index(targets[0]['joint'])]-targets[0]['position_m'],axis=-1)
        elif entry['method'] in ('dart','closd'):
            trace=json.loads((run/'trace.json').read_text())
            goals=trace.get('requested_goals')
            if goals is None:
                goals=([{'frame':0,'position':[1,0,0]},{'frame':150,'position':[-1,.5,0]}] if entry['method']=='dart'
                       else [{'frame':0,'position':[2,0,0]},{'frame':150,'position':[-1.5,1,0]}])
            if trace.get('clock')=='realtime':
                observed=next(r for r in trace['steps'] if r['request_wall_seconds']>=5.)
                goals=[goals[0],{**goals[1],'frame':observed['frame']}]
            goal=np.tile(goals[0]['position'][:2],(frames,1)).astype(float)
            for item in goals[1:]:goal[item['frame']:]=item['position'][:2]
            distance=np.linalg.norm(pelvis[:,:2]-goal,axis=-1)
            root_targets=goal.tolist()
        elif entry['method']=='ardy':
            spec=json.loads((run/'intervention.json').read_text())['task']
            if isinstance(spec,str):
                if spec!='Root2D waypoints: native X=0, Z=.4*min(t,5)-.2*max(t-5,0), metres':raise ValueError('Unknown ARDY path')
                spec={'speed_before_m_s':.4,'speed_after_m_s':-.2,'root_direction_xz':[0.,1.]}
            t=np.arange(frames)/fps
            progress=spec['speed_before_m_s']*np.minimum(t,5)+spec['speed_after_m_s']*np.maximum(t-5,0)
            goal=progress[:,None]*(np.asarray(spec['root_direction_xz'])*[1,-1])
            distance=np.linalg.norm(pelvis[:,:2]-goal,axis=-1)
            root_targets=goal.tolist()
        motions.append({'method':entry['method'],'case':case,'condition':label,'seed':entry['seed'],
            'run':visual_id,'fps':fps,'frames':frames,'names':names,
            'joints':np.round(xyz,4).tolist(),'targets':targets,'root_targets_xy':root_targets,
            'diagnostic_selection':entry.get('diagnostic_selection'),
            'trajectory_family':entry['scene'].get('trajectory_family'),
            'object_geometry':geometries.get(entry['run']),
            'video':'../../../'+entry['run'].removeprefix('develop_logs/')+'/motion.mp4' if (run/'motion.mp4').exists() else None})
        for i in range(frames):
            curves.append({'method':entry['method'],'case':case,'condition':label,'run':visual_id,
                'frame':i,'seconds':i/fps,'trunk_degrees':float(trunk[i]),
                'root_speed_m_s':float(speed[i]) if np.isfinite(speed[i]) else None,
                'jerk_component_rms_m_s3':float(jerk[i]) if np.isfinite(jerk[i]) else None,
                'left_hand_object_distance_m':float(motion['hand_object_distances_m'][i,0]) if 'hand_object_distances_m' in motion else None,
                'right_hand_object_distance_m':float(motion['hand_object_distances_m'][i,1]) if 'hand_object_distances_m' in motion else None,
                'target_distance_m':float(distance[i]) if np.isfinite(distance[i]) else None})
    return motions,curves
