#!/usr/bin/env python3
"""K reference development/held-out audit. Whole registered records; bounded per-record memory."""
import argparse
import hashlib
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.geometry import distance,valid_rotation,tiers,recover_short_gaps,frame,rotation_errors,coordinate_precision_valid
from motiogen.evalkit.k.reference import andy,marker_frames,template_fit,archive
from motiogen.evalkit.k.config import implementation_key,digest
from motion_retarget.bvh import parse_bvh
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.validation import profile_checks,segment_fixture


def reference_audit(row,roots):
    if row['dataset']=='collaborate':
        src=row['source'];bvh=parse_bvh(archive(str(Path(roots['collaborate'])/src['archive'])).read(src['member']))
        p,r=bvh.forward_kinematics(True);names=[j.name for j in bvh.joints];t=np.arange(len(p))/bvh.fps
        lookup={n.lower():i for i,n in enumerate(names)}
        def index(*keys):return next(lookup[k.lower()] for k in keys if k.lower() in lookup)
        h=index('Head');c=index('Spine3','Chest','Spine2');neck=index('Neck','Neck1')
        head,thorax=r[:,h].astype(float),r[:,c].astype(float)
        # Independent positional proxy, compared with published segment rotation.
        right=p[:,index('RightShoulder','RightArm')]-p[:,index('LeftShoulder','LeftArm')]
        proxy=frame(p[:,neck]-p[:,c],right)
        precision=coordinate_precision_valid(p*.01)
        proxy[~precision]=np.nan
        # Native BVH segment axes and geometric axes have different rest bases.
        # Rebuild the rest skeleton from local translations, with identity joint rotations.
        local=np.median(bvh.local_positions(True),axis=0);rest=np.zeros_like(local)
        for j,bone in enumerate(bvh.joints):rest[j]=local[j] if bone.parent<0 else rest[bone.parent]+local[j]
        basis=frame(rest[neck]-rest[c],rest[index('RightShoulder','RightArm')]-rest[index('LeftShoulder','LeftArm')])
        thorax_proxy_reference=thorax@basis
        relative=thorax.swapaxes(-1,-2)@head
        valid=valid_rotation(relative)
        result={'reference_origin':'published_bvh_segment_frames', 'frames':len(t),
                'head_observed':int(valid_rotation(head).sum()),'thorax_observed':int(valid_rotation(thorax).sum()),
                'position_precision_invalid_frames':int((~precision).sum()),
                'paired':int(valid.sum()),'position_thorax_proxy':rotation_errors(thorax_proxy_reference,proxy),
                'proxy_basis':'BVH_rest_from_median_local_translations',
                'anatomical_accuracy':'unverified_published_segment_axes'}
        result['rotation_codec']={}
        reference=Motion(p*.01,names,t,'reference',r,names,{'segment_basis':basis.tolist(),'scope':'full_body_published_bvh_cm'})
        result['profiles']=profile_checks(reference,names[h],names[c],roots['scratch'])
        for name,encode,decode in [
            ('matrix',lambda r:r,lambda r:r),
            ('quaternion_xyzw',lambda r:Rotation.from_matrix(r).as_quat(),lambda q:Rotation.from_quat(q).as_matrix()),
            ('axis_angle',lambda r:Rotation.from_matrix(r).as_rotvec(),lambda v:Rotation.from_rotvec(v).as_matrix())]:
            result['rotation_codec'][name]=tiers(distance(relative[valid],decode(encode(relative[valid]))))
        truth=relative
    else:
        if row['dataset']=='andy':t,names,p=andy(row,roots['andy'])
        else:
            f=Path(roots['han_markers'])/(hashlib.sha256(row['id'].encode()).hexdigest()[:20]+'.npz')
            with np.load(f,allow_pickle=False) as z:t,names,p=z['times'],list(z['names']),z['points']
        ref=marker_frames(p,names,row['dataset']);result={'reference_origin':'four_marker_segment_frame','frames':len(t),'regions':{}}
        truths={};recovered={};calibration=set()
        for region,(truth,points) in ref.items():
            fitted,res,ids=template_fit(points,truth,max(3,int(round(1/np.median(np.diff(t))))))
            calibration.update(ids.tolist());truths[region]=truth;recovered[region]=fitted
            valid=valid_rotation(truth)&valid_rotation(fitted);valid[ids]=False
            result['regions'][region]={'calibration_frames':len(ids),'reference_frames':int(valid_rotation(truth).sum()),
                'reconstructed_frames':int(valid_rotation(fitted).sum()),'comparison':rotation_errors(truth[valid],fitted[valid],len(t)-len(ids)),
                'residual_gates':{str(g):rotation_errors(truth[valid&(res<=g)],fitted[valid&(res<=g)],len(t)-len(ids)) for g in (.005,.01,.02)}}
        truth=truths['thorax'].swapaxes(-1,-2)@truths['head']
        fitted=recovered['thorax'].swapaxes(-1,-2)@recovered['head']
        valid=valid_rotation(truth)&valid_rotation(fitted);valid[list(calibration)]=False
        result.update(paired=int(valid.sum()),head_observed=int(valid_rotation(truths['head']).sum()),
                      thorax_observed=int(valid_rotation(truths['thorax']).sum()),neck_template_comparison=rotation_errors(truth[valid],fitted[valid],len(t)-len(calibration)))
        head=truths['head'].copy();chest=truths['thorax'].copy()
        head[list(calibration)]=np.nan;chest[list(calibration)]=np.nan
        reference=segment_fixture(t,head,chest)
        result['profiles']=profile_checks(reference,'head','spine3',roots['scratch'])
    # Deterministic centre gaps inside valid reference windows. Invalid windows remain unscored.
    result['gap_recovery']={}
    good=valid_rotation(truth)
    for seconds in (.1,.5,1.,2.):
        width=max(1,int(round(seconds/np.median(np.diff(t)))))
        candidates=np.flatnonzero(np.convolve(good.astype(int),np.ones(width+2,dtype=int),'valid')==width+2) if len(t)>width+2 else []
        if not len(candidates):result['gap_recovery'][str(seconds)]={'status':'no_complete_window'};continue
        start=int(candidates[len(candidates)//2])+1
        observed=good.copy();observed[start:start+width]=False
        missing=truth.copy();missing[~observed]=np.nan
        pred,mask=recover_short_gaps(t,missing,observed,seconds)
        result['gap_recovery'][str(seconds)]={**rotation_errors(truth[mask],pred[mask],width),'status':'offline_prediction_only'}
    return result


def audit_one(row,roots):
    start=time.monotonic()
    try:result={'status':'computed',**reference_audit(row,roots)}
    except Exception as e:result={'status':'failed','reason':f'{type(e).__name__}: {e}'}
    result.update(id=row['id'],dataset=row['dataset'],group_id=row['group_id'],recording_id=row['recording_id'],seconds=time.monotonic()-start)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--partition',choices=['development','final_holdout'],required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--dataset',choices=['andy','collaborate','han']);ap.add_argument('--limit',type=int)
    ap.add_argument('--freeze',type=Path);ap.add_argument('--workers',type=int,default=1);args=ap.parse_args()
    if not 1<=args.workers<=8:raise ValueError('Use 1–8 bounded CPU workers')
    reg=ROOT/'config/gold_k_20261115';ev=json.loads((reg/'evidence.json').read_text())
    ev['local_roots']['han_markers']=str(Path(ev['local_roots']['han']).parents[2]/'outputs/evalkit_k_20261115/han_markers')
    ev['local_roots']['scratch']=str(args.out/'scratch')
    key={**implementation_key(), 'validation_script':digest(__file__),'split':digest(reg/'split.json'),'manifest':digest(reg/'trials.jsonl')}
    if args.partition=='final_holdout':
        if args.limit:raise ValueError('Final run must cover the registered dataset')
        if args.freeze is None or json.loads(args.freeze.read_text())['implementation']!=key:raise ValueError('Final requires matching development freeze')
    rows=[json.loads(l) for l in (reg/'trials.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['partition']==args.partition and r['status']=='included' and (not args.dataset or r['dataset']==args.dataset)]
    if args.limit:rows=rows[:args.limit]
    args.out.mkdir(parents=True,exist_ok=True);receipt=args.out/'receipt.json'
    if receipt.exists() and json.loads(receipt.read_text())['implementation']!=key:raise ValueError('Output belongs to another implementation')
    receipt.write_text(json.dumps({'implementation':key,'partition':args.partition,'dataset':args.dataset,'limit':args.limit,'records':len(rows),'workers':args.workers},indent=2))
    target=args.out/'records.jsonl';done={json.loads(l)['id'] for l in target.read_text().splitlines()} if target.exists() else set()
    pending=[row for row in rows if row['id'] not in done]
    with target.open('a') as out, ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(partial(audit_one,roots=ev['local_roots']),pending)):
            out.write(json.dumps(result,allow_nan=False)+'\n');out.flush()
            print(f'{len(done)+i+1}/{len(rows)} {result["group_id"]} {result["status"]}',flush=True)


if __name__=='__main__':main()
