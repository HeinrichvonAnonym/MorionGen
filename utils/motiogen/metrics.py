"""Explicit denominators and missingness. No proxy silently substitutes a metric."""
import numpy as np

from .config import ROOT, read_json


def measured(value, unit, n, **extra):
    arr = np.asarray(value)
    if not np.isfinite(arr).all():
        return missing('non-finite metric input')
    return {'status': 'measured', 'value': float(arr), 'unit': unit, 'n': int(n), **extra}


def missing(reason):
    return {'status': 'unavailable', 'value': None, 'reason': reason}


def so3_degrees(actual, target):
    actual,target=np.asarray(actual),np.asarray(target)
    for rotation in (actual,target):
        if rotation.shape[-2:]!=(3,3) or not np.isfinite(rotation).all() or not np.allclose(rotation @ rotation.swapaxes(-1,-2),np.eye(3),atol=2e-3) or not np.allclose(np.linalg.det(rotation),1,atol=2e-3):
            raise ValueError('SO(3) error requires proper rotation matrices')
    rel = actual.swapaxes(-1,-2) @ target
    cosine = np.clip((np.trace(rel, axis1=-2, axis2=-1)-1)/2, -1, 1)
    return np.degrees(np.arccos(cosine))


def angle_degrees(a, b):
    norm = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.degrees(np.arccos(np.clip(np.sum(a*b, axis=-1)/np.maximum(norm,1e-12),-1,1))), norm > 1e-9


def evaluate(motion, scene, policy=None):
    policy = policy or read_json(ROOT / 'config/metrics.json')
    thresholds = policy['thresholds']
    xyz, fps = motion['joints'], float(motion['fps'])
    names = [str(x) for x in motion['joint_names']]
    index = {name: i for i, name in enumerate(names)}
    rot = motion.get('global_rotations')
    metrics = {name: missing(reason) for name, reason in policy['not_implemented_without_inputs'].items()}
    metrics.update({name: missing(reason) for name,reason in {
        'hand_position_error':'no position constraints', 'hand_orientation_error':'no orientation constraints',
        'position_success_rate':'no position constraints', 'orientation_success_rate':'no orientation constraints',
        'contact_distance':'requires hand-to-object surface distances', 'contact_retention':'requires commanded contact phase and distances',
        'contact_f1':'requires paired GT/predicted contact labels', 'object_goal_error':'requires generated object trajectory and explicit goal',
        'penetration':'requires human-object signed surface distances', 'history_consistency':'requires model pre-stitch overlap predictions',
        'boundary_position_jump':'no chunk boundaries', 'boundary_velocity_jump':'no chunk boundaries',
        'goal_recovery_seconds':'requires numeric goal and update time', 'path_drift':'requires explicit reference path'}.items()})
    pos_errors, rot_errors = [], []
    for constraint in scene.get('constraints', []):
        frame = int(round(constraint['time_seconds'] * fps))
        name = constraint['joint']
        if name not in index or not 0 <= frame < len(xyz):
            raise ValueError(f'constraint {name} at {frame} not represented in motion')
        if 'position_m' in constraint:
            pos_errors.append(np.linalg.norm(xyz[frame,index[name]] - constraint['position_m']))
        if 'rotation_matrix' in constraint and rot is not None:
            rot_errors.append(so3_degrees(rot[frame,index[name]],constraint['rotation_matrix']))
    if pos_errors:
        metrics['hand_position_error'] = measured(np.mean(pos_errors),'m',len(pos_errors))
        metrics['position_success_rate'] = measured(np.mean(np.asarray(pos_errors)<=thresholds['position_m']),'fraction',len(pos_errors),threshold_m=thresholds['position_m'])
    if rot_errors:
        metrics['hand_orientation_error'] = measured(np.mean(rot_errors),'deg',len(rot_errors))
        metrics['orientation_success_rate'] = measured(np.mean(np.asarray(rot_errors)<=thresholds['orientation_deg']),'fraction',len(rot_errors),threshold_deg=thresholds['orientation_deg'])
    elif any('rotation_matrix' in c for c in scene.get('constraints',[])):
        metrics['hand_orientation_error'] = missing('generator provides no true global wrist rotations')
    metrics['jerk_rms'] = measured(np.sqrt(np.mean((np.diff(xyz,n=3,axis=0)*fps**3)**2)),'m/s^3',3*(len(xyz)-3)*len(names),denominator='scalar XYZ components',definition='component RMS, including clip boundaries') if len(xyz)>3 else missing('fewer than four frames')
    bounds = np.asarray(motion.get('chunk_boundaries',[]),dtype=int)
    valid = bounds[(bounds>=2)&(bounds<len(xyz)-1)]
    if len(valid):
        metrics['boundary_position_jump'] = measured(np.linalg.norm(xyz[valid]-xyz[valid-1],axis=-1).mean(),'m',len(valid)*len(names))
        before = (xyz[valid-1]-xyz[valid-2])*fps
        after = (xyz[valid+1]-xyz[valid])*fps
        metrics['boundary_velocity_jump'] = measured(np.linalg.norm(after-before,axis=-1).mean(),'m/s',len(valid)*len(names))
        jerk = np.diff(xyz,n=3,axis=0)*fps**3
        metrics['boundary_jerk_rms'] = measured(np.sqrt(np.mean(jerk[valid-2]**2)),'m/s^3',3*len(valid)*len(names),denominator='scalar XYZ components')
    if 'overlap_errors_m' in motion and motion['overlap_errors_m'].size:
        metrics['history_consistency'] = measured(motion['overlap_errors_m'].mean(),'m',motion['overlap_errors_m'].size)
    up = np.array([0.,0.,1.])
    if 'pelvis' in index and 'neck' in index:
        trunk = xyz[:,index['neck']]-xyz[:,index['pelvis']]
        angles, mask = angle_degrees(trunk,np.broadcast_to(up,trunk.shape))
        if mask.any():
            # Magnitude of inclination, not a signed flexion angle without sagittal body axis.
            metrics['trunk_inclination'] = measured(angles[mask].mean(),'deg',mask.sum(),proxy=True)
            metrics['trunk_risk_fraction'] = measured((angles[mask]>thresholds['trunk_deg']).mean(),'fraction',mask.sum(),proxy=True,threshold_deg=thresholds['trunk_deg'])
            if 'left_hip' in index and 'right_hip' in index:
                right=xyz[:,index['right_hip']]-xyz[:,index['left_hip']]
                right[:,2]=0
                right_norm=np.linalg.norm(right,axis=-1)
                forward=np.cross(np.broadcast_to(up,right.shape),right/np.maximum(right_norm[:,None],1e-12))
                flexion=np.degrees(np.arctan2(np.sum(trunk*forward,axis=-1),trunk[:,2]))
                valid=mask & (right_norm>1e-9)
                if valid.any():
                    metrics['trunk_flexion']=measured(flexion[valid].mean(),'deg',valid.sum(),proxy=True,basis='hip lateral axis crossed with world up')
    for side in ('left','right'):
        a,b = side+'_shoulder',side+'_elbow'
        if a in index and b in index:
            arm = xyz[:,index[b]]-xyz[:,index[a]]
            angles, mask = angle_degrees(arm,np.broadcast_to(-up,arm.shape))
            if mask.any():
                metrics[side+'_upper_arm_elevation'] = measured(angles[mask].mean(),'deg',mask.sum(),proxy=True)
                metrics[side+'_upper_arm_risk_fraction'] = measured((angles[mask]>thresholds['upper_arm_deg']).mean(),'fraction',mask.sum(),proxy=True,threshold_deg=thresholds['upper_arm_deg'])
        a,b,c=side+'_shoulder',side+'_elbow',side+'_wrist'
        if all(x in index for x in (a,b,c)):
            angles,mask=angle_degrees(xyz[:,index[a]]-xyz[:,index[b]],xyz[:,index[c]]-xyz[:,index[b]])
            if mask.any():
                flexion=180-angles[mask]
                metrics[side+'_elbow_rom']=measured(np.ptp(flexion),'deg',mask.sum(),proxy=True)
    if 'hand_object_distances_m' in motion:
        dist=motion['hand_object_distances_m']
        metrics['contact_distance']=measured(dist.mean(),'m',dist.size)
        if 'contact_phase' in motion:
            mask=np.asarray(motion['contact_phase'],dtype=bool)
            if mask.any():
                metrics['contact_retention']=measured((dist[mask]<=thresholds['contact_m']).mean(),'fraction',dist[mask].size,threshold_m=thresholds['contact_m'])
    if 'pred_contact' in motion and 'gt_contact' in motion:
        pred,gt=np.asarray(motion['pred_contact'],bool),np.asarray(motion['gt_contact'],bool)
        if pred.shape!=gt.shape:
            raise ValueError('Contact label shapes differ')
        tp=np.logical_and(pred,gt).sum(); fp=np.logical_and(pred,~gt).sum(); fn=np.logical_and(~pred,gt).sum()
        denom=2*tp+fp+fn
        metrics['contact_f1']=measured(2*tp/denom,'fraction',pred.size,tp=int(tp),fp=int(fp),fn=int(fn),f1_denominator=int(denom)) if denom else missing('no positive contacts; F1 undefined')
    if 'object_positions' in motion and 'object_goal_m' in scene:
        metrics['object_goal_error']=measured(np.linalg.norm(motion['object_positions'][-1]-scene['object_goal_m']),'m',1)
    if 'signed_surface_distances_m' in motion:
        dist=motion['signed_surface_distances_m']
        metrics['penetration']=measured(np.maximum(-dist,0).mean(),'m',dist.size)
    if 'root_reference_m' in motion:
        if motion['root_reference_m'].shape != xyz[:,index['pelvis']].shape:
            raise ValueError('Reference root path shape mismatch')
        metrics['path_drift']=measured(np.linalg.norm(xyz[:,index['pelvis']]-motion['root_reference_m'],axis=-1).mean(),'m',len(xyz))
    recovery_rows=[]
    updates=sorted(scene.get('updates',[]),key=lambda x:x['time_seconds'])
    for ordinal,update in enumerate(updates):
        if 'position_m' not in update or update.get('joint') not in index:
            continue
        start=int(round(update['time_seconds']*fps)); hold=max(1,int(np.ceil(thresholds['recovery_hold_seconds']*fps)))
        if not 0<=start<len(xyz):
            raise ValueError('Numeric goal update lies outside the generated motion')
        end=next((int(round(u['time_seconds']*fps)) for u in updates[ordinal+1:] if u.get('joint')==update['joint'] and 'position_m' in u),len(xyz))
        hit=np.linalg.norm(xyz[start:end,index[update['joint']]]-update['position_m'],axis=-1)<=thresholds['position_m']
        recovery=None
        for k in range(max(0,len(hit)-hold+1)):
            if hit[k:k+hold].all():
                recovery=k/fps; break
        recovery_rows.append({'joint':update['joint'],'update_seconds':update['time_seconds'],'recovery_seconds':recovery,'observation_end_seconds':min(end,len(xyz))/fps})
    if recovery_rows:
        recovered=[r['recovery_seconds'] for r in recovery_rows if r['recovery_seconds'] is not None]
        metrics['goal_recovery_seconds']=measured(np.mean(recovered),'s',len(recovered),clock='motion time',scope='recovered updates only; unrecovered updates are censored') if recovered else missing('numeric goals not recovered during observation windows')
        metrics['goal_recovery_rate']=measured(len(recovered)/len(recovery_rows),'fraction',len(recovery_rows),updates=recovery_rows)
    return {'schema_version':1,'scenario':scene['name'],'fps':fps,'n_frames':len(xyz),'metrics':metrics,'threshold_basis':policy['threshold_basis']}
