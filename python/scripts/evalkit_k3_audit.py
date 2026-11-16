#!/usr/bin/env python3
"""Development-only source audit, deterministic geometric sampling and eight-format exports."""
import argparse, hashlib, json, sys, warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.config import digest
from motiogen.evalkit.k.reference import archive,andy,marker_frames
from motiogen.evalkit.k.geometry import valid_rotation,frame,unit
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.body import evaluate_body
from motiogen.evalkit.k.export import ALIASES,retarget,serialize
from motiogen.evalkit.k.bones import common,lengths
from motion_retarget.bvh import parse_bvh
warnings.filterwarnings('ignore',category=RuntimeWarning)
REG=ROOT/'config/gold_k_20261115'
EV=json.loads((REG/'evidence.json').read_text());ROOTS=EV['local_roots']
METHODS=list(json.loads((ROOT/'config/evalkit_k_v3.json').read_text())['profiles'])


def native(row,window=None,light=False):
    src=row['source'];b=parse_bvh(archive(str(Path(ROOTS['collaborate'])/src['archive'])).read(src['member']))
    total=len(b.frames);fps=b.fps
    if window is None:window=[0.,(total-1)/fps]
    start,end=[int(round(t*fps)) for t in window]
    b.frames=b.frames[start:end+1]
    stride=max(1,int(np.ceil(len(b.frames)/240))) if light else 1
    b.frames=b.frames[::stride]; b.fps=fps/stride
    p,r=b.forward_kinematics(True);names=[j.name for j in b.joints]
    # Use native rest hierarchy; this geometric basis is declared, not called publisher anatomy.
    local=np.median(b.local_positions(True),0);rest=np.zeros_like(local)
    for j,bone in enumerate(b.joints):rest[j]=local[j] if bone.parent<0 else rest[bone.parent]+local[j]
    lower={n.lower():i for i,n in enumerate(names)}
    ix=lambda *n:next(lower[x.lower()] for x in n if x.lower() in lower)
    c=ix('Spine3','Chest','Spine2');neck=ix('Neck','Neck1')
    basis=frame(rest[neck]-rest[c],rest[ix('RightShoulder','RightArm')]-rest[ix('LeftShoulder','LeftArm')])
    return Motion(p*.01,names,np.arange(len(p))/b.fps,'reference',r,names,
                  {'segment_basis':basis.tolist(),'scope':'full_body_published_bvh_cm','axes_evidence':'BVH channels + geometry inference; anatomy unverified'}),stride


def one(row):
    result={k:row[k] for k in ['id','dataset','group_id','recording_id']}
    try:
        if row['dataset']=='collaborate':
            motion,stride=native(row,light=True)
            canonical=Motion(motion.positions,[ALIASES.get(n,n) for n in motion.names],motion.times,'reference')
            # Duplicate Neck/Neck1 aliases are uncommon; report rather than silently select.
            cp=common(canonical.positions,canonical.names);ls=lengths(cp)
            body=evaluate_body(canonical)
            feats=[]
            for n in ['trunk_incline','left_knee_flexion','right_knee_flexion']:
                a=body[n]['values_deg'];feats.append(float(np.nanpercentile(a,90)))
            for side in ['left','right']:
                i={n:j for j,n in enumerate(canonical.names)}
                vec=unit(canonical.positions[:,i[side+'_elbow']]-canonical.positions[:,i[side+'_shoulder']])
                feats.append(float(np.nanpercentile(np.degrees(np.arccos(np.clip(-vec[:,1],-1,1))),90)))
            feats.append(float(np.nanpercentile(np.linalg.norm(np.diff(cp,axis=0),axis=-1)/np.diff(motion.times)[:,None],90)))
            if not np.isfinite(feats).all():raise ValueError('unobservable_or_precision_limited_geometry')
            conversions={}
            # Three sampled poses on an explicit probe clock: topology/rotation validation only.
            probe=Motion(motion.positions[:3],motion.names,np.arange(3)/20.,'reference',motion.rotations[:3],motion.rotation_names,motion.provenance)
            for method in METHODS:
                try:
                    target,_=retarget(probe,method)
                    conversions[method]={'status':'computed','frames':len(target.times),'prior_joints':target.provenance['prior_joints'],
                                         'valid_rotation_share':float(valid_rotation(target.rotations).mean())}
                except Exception as e:conversions[method]={'status':'failed','reason':str(e)}
            result.update(status='computed',stride=stride,sampled_frames=len(motion.times),features=feats,
                          segment_lengths_m=[None if not np.isfinite(x) else float(x) for x in ls],
                          position_extent_m=(np.nanmax(cp.reshape(-1,3),axis=0)-np.nanmin(cp.reshape(-1,3),axis=0)).tolist(),conversion_scope='three sampled poses on probe clock; no temporal precision claim',conversions=conversions)
        else:
            if row['dataset']=='andy':t,names,p=andy(row,ROOTS['andy'])
            else:
                f=Path(ROOTS['han']).parents[2]/'outputs/evalkit_k_20261115/han_markers'/(hashlib.sha256(row['id'].encode()).hexdigest()[:20]+'.npz')
                with np.load(f) as z:t,names,p=z['times'],list(z['names']),z['points']
            frames=marker_frames(p,names,row['dataset'])
            result.update(status='computed',frames=len(t),regions={k:{'valid_frames':int(valid_rotation(v[0]).sum()),'total_frames':len(t)} for k,v in frames.items()},
                          definition='project geometric marker frames; head/thorax only; no template body scored')
    except Exception as e:result.update(status='failed',reason=type(e).__name__+': '+str(e))
    return result


def select(out,rows):
    audit=[json.loads(l) for l in (out/'light.jsonl').read_text().splitlines()]
    source={r['id']:r for r in rows}
    eligible=[r for r in audit if r['dataset']=='collaborate' and r['status']=='computed' and np.isfinite(r['features']).all() and source[r['id']]['frames']*source[r['id']]['dt']>=1]
    features=np.array([r['features'] for r in eligible]);scale=np.std(features,axis=0);scale[scale<1e-6]=1
    selected=[]
    for group in sorted({r['group_id'] for r in eligible}):
        candidates=sorted((r for r in eligible if r['group_id']==group),key=lambda r:hashlib.sha256(('k3-20261116|'+r['id']).encode()).hexdigest())
        picks=[candidates.pop(0)]
        while candidates and len(picks)<4:
            scores=[min(np.linalg.norm((np.array(r['features'])-p['features'])/scale) for p in picks) for r in candidates]
            picks.append(candidates.pop(int(np.argmax(scores))))
        for p in picks:
            row=source[p['id']];duration=(row['frames']-1)*row['dt'];start=max(0.,(duration-5)/2)
            selected.append({**row,'source_window_s':[start,min(duration,start+5)],'selection':'hash_first_then_geometric_farthest','historical':False})
    history=Path(ROOTS['andy']).parents[2]/'outputs/evalkit_k_arms_20261116'
    for cohort in ['development','development_first']:
        for old in json.loads((history/cohort/'manifest.json').read_text())['records']:
            if old['method']!='motionlcm':continue
            # Retain historical windows exactly; a different selected window stays separate.
            selected=[r for r in selected if r['id']!=old['id']]
            selected.append({**source[old['id']],'source_window_s':old['source_window_s'],'selection':'historical_22','historical':True})
    selected.sort(key=lambda r:r['id'])
    for i,r in enumerate(selected):r['window_key']=f'{i:03d}'
    receipt={'selection':'4 per group: hash first then farthest standardized original geometry; historical 22 override same recording windows',
             'features':['trunk_p90','left_knee_p90','right_knee_p90','left_arm_elevation_p90','right_arm_elevation_p90','joint_speed_p90'],
             'standardization_std':scale.tolist(),'light_sha256':digest(out/'light.jsonl'),'records':selected}
    (out/'selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (ROOT/'config/evalkit_k3_20261116/selection.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('selected',len(selected),'groups',len({r['group_id'] for r in selected}),flush=True)


def export_one(task):
    row,out=task;motion,_=native(row,row['source_window_s']);folder=out/'references'/row['window_key'];folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'original.npz',points=motion.positions,names=motion.names,times=motion.times,rotations=motion.rotations,segment_basis=np.asarray(motion.provenance['segment_basis']))
    results=[]
    for method in METHODS:
        rec={k:row[k] for k in ['id','group_id','subset','source_window_s','window_key','historical']};rec.update(method=method,partition='development',reference=str(folder/'original.npz'))
        try:
            target,parents=retarget(motion,method);src=serialize(target,parents,out/'inputs'/row['window_key']/method)
            rec.update(status='exported',source=str(src),retarget_provenance=target.provenance)
        except Exception as e:rec.update(status='failed',reason=str(e))
        results.append(rec)
    return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['light','prepare']);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(l) for l in (REG/'trials.jsonl').read_text().splitlines()];rows=[r for r in rows if r['partition']=='development' and r['status']=='included']
    if a.phase=='light':
        done={json.loads(l)['id'] for l in (a.out/'light.jsonl').read_text().splitlines()} if (a.out/'light.jsonl').exists() else set()
        with ProcessPoolExecutor(max_workers=a.workers) as pool,(a.out/'light.jsonl').open('a') as f:
            for i,r in enumerate(pool.map(one,[r for r in rows if r['id'] not in done])):
                f.write(json.dumps(r,allow_nan=False)+'\n');f.flush()
                if i%25==0:print(i+len(done),len(rows),r['status'],flush=True)
        select(a.out,rows)
    else:
        selected=json.loads((a.out/'selection.json').read_text())['records']
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            results=[]
            for i,items in enumerate(pool.map(export_one,[(r,a.out) for r in selected])):
                results.extend(items);print(i+1,len(selected),'exports',len(results),flush=True)
        (a.out/'exports.json').write_text(json.dumps(results,indent=2)+'\n')

if __name__=='__main__':main()
