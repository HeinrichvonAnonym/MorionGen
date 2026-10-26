"""CPU-only independent object metrics from saved predictions and source windows.

Does not instantiate CHOIS, run generation, or reconstruct ground-truth bodies.
Consequently it does not claim independent penetration or contact-F1 validation.
"""
import json
import sys
from pathlib import Path
import joblib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.runtime import checksum,json_write

base=ROOT/'develop_logs/test_runs'
analysis=json.loads((base/'20261024_native_matrix_analysis/analysis.json').read_text())
plan=json.loads((base/'20261024_native_followups/plan.json').read_text())
data=Path(load_config()['data_root'])/'outputs/assets/chois/processed_data'
source=data/'cano_test_diffusion_manip_window_120_joints24.p'
identity=checksum(source)
if identity!=plan['source_windows_sha256']:raise ValueError('Registered CHOIS source windows changed')
windows=joblib.load(source)
# Match the official text / full-first-window filters, preserving insertion order.
filtered=[v for v in windows.values() if (data/'omomo_text_anno_json_data'/(v['seq_name']+'.json')).exists()
          and len(v['motion'])>=120 and v['start_t_idx']==0]
checks=[];persistence_checks=[]
for row in analysis['rows']:
    if row['method']!='chois':continue
    run=ROOT/row['run'];config=json.loads((run/'upstream_config.json').read_text())
    if config['use_object_split']:raise ValueError('Audit filter does not represent object-split evaluation')
    window=filtered[row['task_spec'].get('native_window_index',0)]
    metas=list((run/'raw/chois_results/res_npz_files').rglob('*.npz'))
    if len(metas)!=1:raise ValueError('Expected one native CHOIS sequence')
    with np.load(metas[0]) as meta:
        if str(meta['seq_name'])!=window['seq_name']:raise ValueError('Prediction/source window identity mismatch')
    with np.load(run/'raw/native_prediction.npz') as pred:
        position=pred['object_positions'];rotation=pred['object_rotations']
        gt_position=np.asarray(window['window_obj_com_pos'],dtype=np.float32)
        gt_rotation=np.asarray(window['obj_rot_mat'],dtype=np.float32)
        if position.shape!=gt_position.shape or rotation.shape!=gt_rotation.shape or len(position)!=120:
            raise ValueError('Native prediction and ground truth must have matching 120-frame shape')
        distances=np.linalg.norm(position-gt_position,axis=1)
        # Rotation residual is Frobenius matrix error, NOT SO(3) angle.
        residual=np.eye(3)-rotation@np.linalg.inv(gt_rotation)
        expected={
            'native_object_mean_obj_com_pos_err':float(distances.mean()*1000),
            'native_object_mean_start_obj_com_pos_err':float(distances[0]*1000),
            'native_object_mean_end_obj_com_pos_err':float(distances[-1]*1000),
            'native_object_mean_waypoints_xy_pos_err':float(np.linalg.norm(position[[29,59,89],:2]-gt_position[[29,59,89],:2],axis=1).mean()*1000),
            'native_object_mean_obj_rot_dist':float(np.linalg.norm(residual,axis=(-2,-1)).mean()),
            'native_object_mean_contact_percent':float((pred['hand_object_distances_m']<.05).any(axis=1).mean()),
        }
        # Independent literal per-hand scan; do not call the analysis helper.
        contact=pred['hand_object_distances_m']<.05
        eligible=retained=longest=0
        for hand in range(2):
            consecutive=0
            for frame in range(len(contact)):
                active=bool(contact[frame,hand])
                consecutive=consecutive+1 if active else 0
                longest=max(longest,consecutive)
                if frame<len(contact)-1 and active:
                    eligible+=1;retained+=int(contact[frame+1,hand])
        p_expected={'geometric_hand_contact_eligible_transitions':eligible,
                    'geometric_hand_contact_retention_fraction':retained/eligible if eligible else None,
                    'geometric_hand_contact_longest_span_seconds':max(0,longest-1)/row['fps']}
        p_valid=all(k not in row['values'] if v is None else k in row['values'] and np.isclose(row['values'][k],v,rtol=0,atol=1e-12) for k,v in p_expected.items())
        persistence_checks.append({'run':row['run'],'task':row['task'],'seed':row['seed'],
            'recomputed':p_expected,'retained_hand_transitions':retained,'passed':bool(p_valid)})
        # Anatomical wrists are raw joints 20/21, not the appended middle-finger
        # anchors 22/23 used by the official contact metric.
        xyz=np.asarray(pred['joints'],dtype=np.float64)
        for side,shoulder,elbow,wrist in [('left',16,18,20),('right',17,19,21)]:
            a=xyz[:,shoulder]-xyz[:,elbow];b=xyz[:,wrist]-xyz[:,elbow]
            a_length=np.sqrt(np.einsum('ij,ij->i',a,a));b_length=np.sqrt(np.einsum('ij,ij->i',b,b))
            valid=a_length*b_length>1e-9
            cosine=np.einsum('ij,ij->i',a[valid]/a_length[valid,None],b[valid]/b_length[valid,None])
            flexion=180-np.arccos(np.clip(cosine,-1,1))*180/np.pi
            expected[side+'_elbow_rom']=float(flexion.max()-flexion.min())
    metrics=[]
    for metric,value in expected.items():
        stored=row['values'][metric]
        unit='deg' if metric.endswith('elbow_rom') else 'fraction' if metric.endswith('contact_percent') else 'dimensionless' if metric.endswith('obj_rot_dist') else 'mm'
        # One micron for positional roundoff, 2e-6 for raw matrix roundoff;
        # native GT passes through float32 normalization/relative rotations.
        atol=2e-7 if unit=='deg' else 0. if unit=='fraction' else 2e-6 if unit=='dimensionless' else 1e-3
        passed=bool(np.isfinite([value,stored]).all() and np.isclose(value,stored,rtol=0 if unit=='fraction' else 1e-6,atol=atol))
        metrics.append({'metric':metric,'unit':unit,'recomputed':value,'stored':stored,
            'absolute_difference':abs(value-stored),'atol':atol,'passed':passed})
    checks.append({'run':row['run'],'task':row['task'],'seed':row['seed'],'sequence':window['seq_name'],
        'metrics':metrics,'passed':all(m['passed'] for m in metrics)})
summary=[{'metric':metric,'unit':unit,'checked_runs':len(checks),
          'passed_runs':sum(next(m['passed'] for m in c['metrics'] if m['metric']==metric) for c in checks),
          'max_absolute_difference':max(m['absolute_difference'] for c in checks for m in c['metrics'] if m['metric']==metric)}
         for metric,unit in [(m['metric'],m['unit']) for m in checks[0]['metrics']]] if checks else []
receipt={'analysis_cutoff':analysis['updated_at'],'source':str(source),'source_sha256':identity,
         'checks':checks,'summary':summary,'persistence_checks':persistence_checks,
         'persistence_scope':'Post-hoc sampled same-hand geometric continuity, strict 5 cm at native middle-finger anchors. No contact denominator means missing retention; longest observed span is window-censored, not force or continuous grasp.',
         'passed':bool(checks) and all(c['passed'] for c in checks) and all(c['passed'] for c in persistence_checks),
         'scope':'Independent NumPy object-position/rotation, predicted contact-frame fraction and anatomical elbow-ROM checks. Elbows use original SMPL body wrists 20/21, not the appended middle-finger contact anchors 22/23 (trainer_chois.py:143-149). Raw GT vs native float32 normalization/relative-rotation roundtrip uses explicit numerical tolerances. Does not reconstruct human surfaces, GT contact labels, F1 or SDF penetration.'}
json_write(base/'20261024_chois_native_metrics/audit.json',receipt)
print(json.dumps({'passed':receipt['passed'],'runs':len(checks),'summary':summary,
    'failures':[c for c in checks if not c['passed']]}))
if not receipt['passed']:raise SystemExit(1)
