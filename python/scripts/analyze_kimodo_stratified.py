#!/usr/bin/env python
"""Paired generated/GT E proxies and timing summaries for bounded Kimodo evaluation."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from kimodo.constraints import load_constraints_lst
from kimodo.skeleton import SOMASkeleton77

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.metrics import evaluate,so3_degrees
from motiogen.motion import SOMA77_NAMES,canonicalize
from motiogen.runtime import json_write

E_METRICS=['jerk_rms','trunk_inclination','trunk_risk_fraction','left_upper_arm_elevation',
           'left_upper_arm_risk_fraction','right_upper_arm_elevation','right_upper_arm_risk_fraction',
           'left_elbow_rom','right_elbow_rom']


def metric_row(path):
    with np.load(path,allow_pickle=False) as data:
        joints=np.asarray(data['posed_joints'])
    if joints.ndim==4: joints=joints[0]
    joints,_=canonicalize(joints,'Y-up',units='m')
    motion={'joints':joints,'joint_names':np.asarray(SOMA77_NAMES),'fps':np.array(30.),
            'metadata':{'units':'m','axes':'right-handed Z-up'}}
    result=evaluate(motion,{'name':'kimodo_benchmark','kind':'benchmark','duration_seconds':len(joints)/30,'prompt':'','constraints':[]})
    return {name:result['metrics'][name].get('value') if result['metrics'].get(name,{}).get('status')=='measured' else None for name in E_METRICS}


def rotation_row(motion_path):
    """Measure only true rotation constraints, retaining unavailable as missing."""
    constraint_path=Path(motion_path).parent/'constraints.json'
    if not constraint_path.is_file():
        return {'status':'unavailable','reason':'constraints.json missing','n':0}
    with np.load(motion_path,allow_pickle=False) as data:
        if 'global_rot_mats' not in data:
            return {'status':'unavailable','reason':'motion has no global_rot_mats','n':0}
        rotations=np.asarray(data['global_rot_mats'])
    if rotations.ndim==5: rotations=rotations[0]
    constraints=load_constraints_lst(str(constraint_path),SOMASkeleton77(),device='cpu')
    errors=[]
    for constraint in constraints:
        if not hasattr(constraint,'global_joints_rots'):
            continue
        frames=constraint.frame_indices.detach().cpu().numpy().astype(int)
        if hasattr(constraint,'rot_indices'):
            joints=constraint.rot_indices.detach().cpu().numpy().astype(int)
        else:
            joints=np.arange(rotations.shape[1],dtype=int)
        if not len(frames) or not len(joints):
            continue
        target=constraint.global_joints_rots.detach().cpu().numpy()[:,joints]
        actual=rotations[frames[:,None],joints[None,:]]
        errors.extend(np.asarray(so3_degrees(actual,target)).reshape(-1).tolist())
    if not errors:
        return {'status':'unavailable','reason':'case contains no rotation constraint','n':0}
    values=np.asarray(errors,float)
    return {'status':'measured','unit':'deg','n':int(values.size),'mean':float(values.mean()),
            'p50':float(np.percentile(values,50)),'p95':float(np.percentile(values,95)),
            'max':float(values.max())}


def bootstrap(delta,seed=20261008,n_boot=1000):
    arr=np.asarray(delta,float); rng=np.random.default_rng(seed)
    means=np.mean(rng.choice(arr,(n_boot,len(arr)),replace=True),axis=1)
    return {'n':len(arr),'mean':float(arr.mean()),'median':float(np.median(arr)),
            'ci95':[float(x) for x in np.percentile(means,[2.5,97.5])]}


def timing_summary(path):
    rows=[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    ok=[x for x in rows if x.get('status')=='success']
    def q(key):
        vals=[x[key] for x in ok if key in x]
        return {'n':len(vals),'p50':float(np.percentile(vals,50)),'p95':float(np.percentile(vals,95)),
                'mean':float(np.mean(vals))} if vals else None
    return {'attempt_rows':len(rows),'success':len(ok),'failed_attempts':sum(x.get('status')=='failed_attempt' for x in rows),
            'rtf':q('rtf'),'seconds':q('seconds'),'cuda_peak_allocated_bytes':q('cuda_peak_allocated_bytes'),
            'rss_bytes':q('rss_bytes')}


def official_metrics(case_dir):
    path=Path(case_dir)/'metrics.json'
    if not path.is_file(): return None
    payload=json.loads(path.read_text())
    return {k:float(v) for k,v in payload.get('per_motion_mean_gen',{}).items()
            if isinstance(v,(int,float)) and np.isfinite(v)}


def official_payload(case_dir):
    path=Path(case_dir)/'metrics.json'
    if not path.is_file(): return None
    payload=json.loads(path.read_text())
    def finite(values):
        return {k:float(v) for k,v in values.items() if isinstance(v,(int,float)) and np.isfinite(v)}
    return {'generated':finite(payload.get('per_motion_mean_gen',{})),
            'ground_truth':finite(payload.get('per_motion_mean_gt',{}))}


def official_aggregate(root,cases):
    """Absolute native metrics; missing/not-applicable cases remain explicit."""
    constraint_keys=('constraint_root2d_err','constraint_end_effector','constraint_fullbody_keyframe')
    quality_keys=('foot_skate_from_height','foot_skate_from_pred_contacts','foot_skate_max_vel',
                  'foot_skate_ratio','foot_contact_consistency')
    values={side:{key:[] for key in constraint_keys+quality_keys} for side in ('generated','ground_truth')}
    missing=[]
    for case in cases:
        payload=official_payload(Path(root)/case)
        if payload is None:
            missing.append(case); continue
        for side in values:
            for key in values[side]:
                if key in payload[side]: values[side][key].append(payload[side][key])
    output={'expected_cases':len(cases),'missing_metrics':missing,'success_threshold_m':0.05}
    for side,metrics in values.items():
        output[side]={}
        for key,items in metrics.items():
            if not items: continue
            arr=np.asarray(items,float); row={'n':len(arr),'mean':float(arr.mean()),
                'p50':float(np.percentile(arr,50)),'p95':float(np.percentile(arr,95))}
            if key in constraint_keys:
                row['success_le_5cm']=float(np.mean(arr<=0.05)); row['success_count_le_5cm']=int(np.sum(arr<=0.05))
            output[side][key]=row
    output['paired_quality_delta_gen_minus_gt']={}
    for key in quality_keys:
        deltas=[]
        for case in cases:
            payload=official_payload(Path(root)/case)
            if payload and key in payload['generated'] and key in payload['ground_truth']:
                deltas.append(payload['generated'][key]-payload['ground_truth'][key])
        if deltas: output['paired_quality_delta_gen_minus_gt'][key]=bootstrap(deltas)
    return output


def paired_postprocess(raw,post,case_names):
    rows=[]; missing=[]
    for case in case_names:
        raw_dir=Path(raw)/case; post_dir=Path(post)/case
        raw_metrics=official_metrics(raw_dir); post_metrics=official_metrics(post_dir)
        if raw_metrics is None or post_metrics is None:
            missing.append({'case':case,'raw_metrics':raw_metrics is not None,'postprocess_metrics':post_metrics is not None})
            continue
        keys=sorted(set(raw_metrics)&set(post_metrics))
        row={'case':case,'raw':raw_metrics,'postprocess':post_metrics,
             'delta_post_minus_raw':{key:post_metrics[key]-raw_metrics[key] for key in keys}}
        try:
            row['raw_rotation']=rotation_row(raw_dir/'motion.npz')
            row['postprocess_rotation']=rotation_row(post_dir/'motion.npz')
        except Exception as exc:
            row['rotation_error']=f'{type(exc).__name__}: {exc}'
        rows.append(row)
    aggregates={}
    keys=sorted({key for row in rows for key in row['delta_post_minus_raw']})
    for key in keys:
        values=[row['delta_post_minus_raw'][key] for row in rows if key in row['delta_post_minus_raw']]
        if values: aggregates[key]=bootstrap(values)
    rotation_delta=[]
    for row in rows:
        before=row.get('raw_rotation',{}); after=row.get('postprocess_rotation',{})
        if before.get('status')=='measured' and after.get('status')=='measured':
            rotation_delta.append(after['mean']-before['mean'])
    if rotation_delta: aggregates['rotation_geodesic_mean_deg']=bootstrap(rotation_delta)
    rotation_absolute={}
    for label in ('raw_rotation','postprocess_rotation'):
        vals=[row[label]['mean'] for row in rows if row.get(label,{}).get('status')=='measured']
        if vals:
            rotation_absolute[label]={'n':len(vals),'mean':float(np.mean(vals)),
                'p50':float(np.percentile(vals,50)),'p95':float(np.percentile(vals,95))}
    return rows,{'expected':len(case_names),'paired':len(rows),'missing':missing,
                 'absolute_rotation_geodesic_deg':rotation_absolute,'delta_post_minus_raw':aggregates}


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--raw',required=True); p.add_argument('--timing',required=True)
    p.add_argument('--postprocess'); p.add_argument('--postprocess-timing'); p.add_argument('--manifest'); p.add_argument('--output',required=True)
    args=p.parse_args(argv); raw=Path(args.raw); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    expected=[str(path.parent.relative_to(raw)) for path in sorted(raw.rglob('meta.json'))]
    if args.manifest:
        expected=[row['case'] for row in json.loads(Path(args.manifest).read_text())['cases']]
    rows=[]; failures=[]; rotation_status={'measured_cases':0,'unavailable_cases':0,'samples':0,'errors':[]}
    for case in expected:
        rel=Path(case); generated=raw/rel/'motion.npz'; gt=raw/rel/'gt_motion.npz'
        if not generated.is_file(): failures.append({'case':str(rel),'reason':'missing generated motion.npz'}); continue
        if not gt.is_file(): failures.append({'case':str(rel),'reason':'missing gt_motion.npz'}); continue
        try:
            gen=metric_row(generated); truth=metric_row(gt)
            rotation=rotation_row(generated)
            if rotation['status']=='measured':
                rotation_status['measured_cases']+=1; rotation_status['samples']+=rotation['n']
            else: rotation_status['unavailable_cases']+=1
            rows.append({'case':str(rel),'generated':gen,'ground_truth':truth,'rotation_geodesic':rotation,
                         'delta':{k:gen[k]-truth[k] if gen[k] is not None and truth[k] is not None else None for k in E_METRICS}})
        except Exception as exc:
            failures.append({'case':str(rel),'reason':f'{type(exc).__name__}: {exc}'})
            rotation_status['errors'].append({'case':str(rel),'reason':f'{type(exc).__name__}: {exc}'})
    with (output/'paired_e_metrics.jsonl').open('w') as handle:
        for row in rows: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
    aggregate={}
    for name in E_METRICS:
        deltas=[r['delta'][name] for r in rows if r['delta'][name] is not None]
        if deltas: aggregate[name]=bootstrap(deltas)
    measured_rot=[r['rotation_geodesic']['mean'] for r in rows if r['rotation_geodesic']['status']=='measured']
    if measured_rot:
        rotation_status['case_mean_deg']={'mean':float(np.mean(measured_rot)),'p50':float(np.percentile(measured_rot,50)),
                                          'p95':float(np.percentile(measured_rot,95))}
    summary={'schema_version':2,'status':'success' if not failures else 'partial','scope':'stratified subset; paired GT, not full official benchmark',
             'expected_cases':len(expected),'paired_cases':len(rows),'failures':failures,'paired_delta':aggregate,
             'rotation_geodesic':rotation_status,'raw_timing':timing_summary(args.timing),
             'raw_official':official_aggregate(raw,expected)}
    if args.postprocess_timing and Path(args.postprocess_timing).is_file(): summary['postprocess_timing']=timing_summary(args.postprocess_timing)
    if args.postprocess:
        post_cases=[str(path.parent.relative_to(Path(args.postprocess))) for path in sorted(Path(args.postprocess).rglob('meta.json'))]
        post_rows,post_summary=paired_postprocess(raw,args.postprocess,post_cases)
        with (output/'raw_postprocess_pairs.jsonl').open('w') as handle:
            for row in post_rows: handle.write(json.dumps(row,ensure_ascii=False)+'\n')
        summary['raw_postprocess']=post_summary
        summary['raw_postprocess']['raw_paired_official']=official_aggregate(raw,post_cases)
        summary['postprocess_official']=official_aggregate(args.postprocess,post_cases)
    json_write(output/'summary.json',summary); print(json.dumps({'status':summary['status'],'paired_cases':len(rows),'failures':len(failures)},ensure_ascii=False))
    return 0 if not failures else 2


if __name__=='__main__': raise SystemExit(main())
