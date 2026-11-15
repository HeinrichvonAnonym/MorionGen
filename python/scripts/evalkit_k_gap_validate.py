#!/usr/bin/env python3
"""Registered head/thorax/combined gaps at the decoded segment interface, with no alternate pose inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.reference import andy,marker_frames,archive
from motiogen.evalkit.k.geometry import valid_rotation,recover_short_gaps,rotation_errors
from motiogen.evalkit.k.config import implementation_key,digest
from motion_retarget.bvh import parse_bvh


def check(row,roots):
    try:
        if row['dataset']=='collaborate':
            src=row['source'];b=parse_bvh(archive(str(Path(roots['collaborate'])/src['archive'])).read(src['member']))
            _,r=b.forward_kinematics(True);names=[j.name for j in b.joints];t=np.arange(len(r))/b.fps
            head=r[:,names.index('Head')].astype(float)
            chest=r[:,next(names.index(n) for n in ('Spine3','Chest','Spine2') if n in names)].astype(float)
        else:
            if row['dataset']=='andy':t,names,p=andy(row,roots['andy'])
            else:
                f=Path(roots['han']).parents[2]/'outputs/evalkit_k_20261115/han_markers'/(hashlib.sha256(row['id'].encode()).hexdigest()[:20]+'.npz')
                with np.load(f) as z:t,names,p=z['times'],list(z['names']),z['points']
            ref=marker_frames(p,names,row['dataset']);head,chest=ref['head'][0],ref['thorax'][0]
        both=valid_rotation(head)&valid_rotation(chest)&(t>=t[0]+1.)
        cases={}
        for region in ('head','thorax','head_and_thorax'):
            cases[region]={}
            for seconds in (.1,.5,1.,2.):
                width=max(1,int(round(seconds/np.median(np.diff(t)))))
                starts=np.flatnonzero(np.convolve(both.astype(int),np.ones(width+2,int),'valid')==width+2) if len(t)>width+2 else []
                if not len(starts):cases[region][str(seconds)]={'status':'no_complete_window'};continue
                seed=f'gold_k_20261115_v1|{row["recording_id"]}|{region}|{seconds}|0'
                start=int(starts[int(hashlib.sha256(seed.encode()).hexdigest(),16)%len(starts)])+1
                end=start+width;pred={}
                # Prediction receives only these two masked matrices and their clock.
                # Original markers, positions, local rotations and features stay outside this API.
                for name,truth in [('head',head),('thorax',chest)]:
                    observed=valid_rotation(truth)
                    if region in (name,'head_and_thorax'):observed[start:end]=False
                    masked=truth.copy();masked[~observed]=np.nan
                    pred[name],_=recover_short_gaps(t,masked,observed,seconds)
                truth_neck=chest[start:end].swapaxes(-1,-2)@head[start:end]
                predicted_neck=pred['thorax'][start:end].swapaxes(-1,-2)@pred['head'][start:end]
                cases[region][str(seconds)]={'status':'computed','seed':seed,'interval_s':[float(t[start]),float(t[end-1])],
                    'runtime_policy_supported':seconds<=.1,'neck':rotation_errors(truth_neck,predicted_neck,width),
                    'head':rotation_errors(head[start:end],pred['head'][start:end],width),
                    'thorax':rotation_errors(chest[start:end],pred['thorax'][start:end],width)}
        return {'status':'computed','id':row['id'],'group_id':row['group_id'],'recording_id':row['recording_id'],'dataset':row['dataset'],'cases':cases}
    except Exception as e:return {'status':'failed','id':row['id'],'group_id':row['group_id'],'recording_id':row['recording_id'],'dataset':row['dataset'],'reason':str(e)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--partition',choices=['development','final_holdout'],required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--freeze',type=Path)
    a=ap.parse_args()
    if not 1<=a.workers<=8:raise ValueError('Use 1–8 bounded CPU workers')
    key={**implementation_key(),'gap_script':digest(__file__)}
    if a.partition=='final_holdout':
        if not a.freeze or json.loads(a.freeze.read_text())['gap_implementation']!=key:raise ValueError('Matching freeze required')
    reg=ROOT/'config/gold_k_20261115';roots=json.loads((reg/'evidence.json').read_text())['local_roots']
    rows=[json.loads(s) for s in (reg/'trials.jsonl').read_text().splitlines()]
    rows=[r for r in rows if r['partition']==a.partition and r['status']=='included']
    a.out.mkdir(parents=True,exist_ok=True);receipt=a.out/'receipt.json';target=a.out/'records.jsonl'
    if receipt.exists() and json.loads(receipt.read_text())['implementation']!=key:raise ValueError('Different implementation')
    receipt.write_text(json.dumps({'implementation':key,'records':len(rows),'partition':a.partition,
        'scope':'decoded independent segment observations; source rotations are removed before interpolation; no full reference pose initialization',
        'mode':'offline bidirectional; 0.5/1/2 second cases are stress tests beyond runtime recovery policy'},indent=2))
    done={json.loads(s)['id'] for s in target.read_text().splitlines()} if target.exists() else set()
    with target.open('a') as f,ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,result in enumerate(pool.map(partial(check,roots=roots),[r for r in rows if r['id'] not in done])):
            f.write(json.dumps(result,allow_nan=False)+'\n');f.flush()
            print(f'{len(done)+i+1}/{len(rows)} {result["status"]}',flush=True)


if __name__=='__main__':main()
