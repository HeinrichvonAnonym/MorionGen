"""Recheck saved native correction pairs; bootstrap by source BVH, not frames/crops."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT,load_config
from motiogen.runtime import checksum,json_write
from motiogen.native_analysis import clustered_interval
from motiogen.metrics import evaluate
from motiogen.motion import SOMA77_NAMES,canonicalize

base=Path(load_config()['data_root'])/'benchmarks/kimodo-stratified-20261008'
previous=ROOT/'develop_logs/test_runs/20261008_064957_kimodo_evaluate-native_aa2faed4/supplemental_analysis/raw_postprocess_pairs.jsonl'
out=ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit'
rows=[]; failures=[]
for original in map(json.loads,previous.read_text().splitlines()):
    case=original['case']; raw=base/'generated_raw'/case; post=base/'generated_postprocess'/case
    try:
        source=json.loads((base/'testsuite'/case/'seed_motion.json').read_text())
        hashes={}
        for name in ['meta.json','constraints.json','gt_motion.npz']:
            hashes[name]=checksum(raw/name)
            if hashes[name]!=checksum(post/name): raise ValueError(f'Paired input mismatch: {name}')
        meta=json.loads((raw/'meta.json').read_text())
        if meta.get('seed') is None: raise ValueError('Missing sampling seed')
        metrics={}; outputs={}
        for label,folder in [('raw',raw),('postprocess',post)]:
            with np.load(folder/'motion.npz',allow_pickle=False) as data:
                for key in data.files:
                    value=data[key]
                    if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():
                        raise ValueError(f'Nonfinite {label}/{key}')
                joints=np.asarray(data['posed_joints'])
            if joints.ndim==4: joints=joints[0]
            xyz,_=canonicalize(joints,'Y-up')
            e=evaluate({'joints':xyz,'joint_names':SOMA77_NAMES,'fps':30}, {'name':'kimodo_native_correction'})['metrics']
            native=json.loads((folder/'metrics.json').read_text())['per_motion_mean_gen']
            if native!=original[label]: raise ValueError(f'Previous metric extraction differs: {label}')
            metrics[label]={**native,**{k:e[k]['value'] for k in ['jerk_rms','trunk_inclination']}}
            rotation=original[label+'_rotation']
            if rotation.get('status')=='measured': metrics[label]['rotation_geodesic_deg']=rotation['mean']
            outputs[label]={'sha256':checksum(folder/'motion.npz'),'frames':len(joints)}
        if outputs['raw']['frames']!=outputs['postprocess']['frames']: raise ValueError('Paired duration mismatch')
        row={'case':case,'family':'/'.join(Path(case).parts[1:-1]),'cluster':source['bvh_path'],
             'seed':meta['seed'],'inputs_sha256':hashes,'outputs':outputs,'metrics':metrics,
             'delta':{k:metrics['postprocess'][k]-v for k,v in metrics['raw'].items() if k in metrics['postprocess']}}
        rows.append(row)
    except Exception as exc:
        failures.append({'case':case,'reason':f'{type(exc).__name__}: {exc}'})
    if (len(rows)+len(failures))%32==0:
        print(json.dumps({'checked':len(rows)+len(failures),'accepted':len(rows),'failed':len(failures)}),flush=True)
summary=[]
for family in ['all']+sorted({r['family'] for r in rows}):
    selected=[r for r in rows if family=='all' or r['family']==family]
    for metric in sorted({k for r in selected for k in r['delta']}):
        available=[r for r in selected if metric in r['delta']]
        stats=clustered_interval([r['delta'][metric] for r in available],[r['cluster'] for r in available])
        if metric in ['foot_contact_consistency','constraint_root2d_acc']:
            stats['improved_fraction']=float(np.mean([r['delta'][metric]>0 for r in available]))
        summary.append({'family':family,'metric':metric,**stats})
json_write(out/'audit.json',{'expected_pairs':416,'accepted_pairs':len(rows),'failures':failures,
    'cluster_definition':'source seed_motion.json bvh_path; all overlapping crops/families from a source grouped together',
    'scope':'Existing fixed-checkpoint correction contrast; conditional on selected native benchmark clips, not new random subjects',
    'recomputed':['jerk_rms','trunk_inclination'],
    'reused':'Native per-case metrics checked against original extraction; existing SO(3) evaluator values retained',
    'rows':rows,'summary':summary})
print(json.dumps({'accepted':len(rows),'failures':failures,'source_bvh_groups':len({r['cluster'] for r in rows})}),flush=True)
