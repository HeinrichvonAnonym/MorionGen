#!/usr/bin/env python3
"""Reproducible development-only arm fitting comparison against a frozen checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]


def prepare(out, cohort='tasks'):
    sys.path[:0] = [str(ROOT/'utils'), str(ROOT/'utils/motiogen')]
    import numpy as np
    from motion_retarget.bvh import parse_bvh
    from motiogen.evalkit.k.reference import archive
    from motiogen.evalkit.k.motion import Motion
    from motiogen.evalkit.k.export import retarget, serialize
    from motiogen.evalkit.k.geometry import frame
    rows = [json.loads(s) for s in (ROOT/'config/gold_k_20261115/trials.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['partition']=='development' and r['status']=='included'
            and r.get('subset') in ('LAB','AirplaneRiveting','Glassblowing_Process') and r['frames']*r['dt']>=5.1]
    selected = []
    groups = sorted(set(r['group_id'] for r in rows if r['subset']=='LAB'))
    # Deterministic selection independent of any K fitting result, one record per LAB person.
    for group in groups:
        candidates = [r for r in rows if r['group_id']==group]
        selected.append(min(candidates, key=(lambda r:r['id']) if cohort=='first_records'
                            else (lambda r:hashlib.sha256(r['id'].encode()).hexdigest())))
    for subset in (() if cohort=='first_records' else ('AirplaneRiveting','Glassblowing_Process')):
        selected.append(min((r for r in rows if r['subset']==subset), key=lambda r:hashlib.sha256(r['id'].encode()).hexdigest()))
    root = Path(json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['local_roots']['collaborate'])
    records=[]
    for i,row in enumerate(selected):
        b=parse_bvh(archive(str(root/row['source']['archive'])).read(row['source']['member']))
        p,r=b.forward_kinematics(True); names=[j.name for j in b.joints]
        local=np.median(b.local_positions(True),0);rest=np.zeros_like(local)
        for k,j in enumerate(b.joints):rest[k]=local[k] if j.parent<0 else rest[j.parent]+local[k]
        c,neck=(names.index(n) for n in ('Spine3','Neck'))
        basis=frame(rest[neck]-rest[c],rest[names.index('RightShoulder')]-rest[names.index('LeftShoulder')])
        n=int(round(5*b.fps))+1;start=0 if cohort=='first_records' else (len(p)-n)//2
        reference=Motion(p[start:start+n]*.01,names,np.arange(n)/b.fps,'reference',r[start:start+n],names,
                         {'segment_basis':basis.tolist(),'scope':'full_body_published_bvh_cm'})
        for method in ('motionlcm','omnicontrol','kimodo'):
            target,parents=retarget(reference,method)
            source=serialize(target,parents,out/'inputs'/f'{i:02d}'/method)
            records.append({'id':row['id'],'partition':'development','group_id':row['group_id'],'subset':row['subset'],
                            'source_bvh':row['source'],'source_window_s':[start/b.fps,(start+n-1)/b.fps],
                            'method':method,'source':str(source),'key':f'{i:02d}/{method}',
                            'retarget_provenance':target.provenance})
    selection = ('Lexicographically first registered record per LAB person; first five seconds; supplementary calibration coverage cohort'
                 if cohort=='first_records' else 'SHA256-min per LAB person plus per Airplane/Glass subset; centred five seconds; selected without fitted errors')
    (out/'manifest.json').write_text(json.dumps({'selection':selection,
                                              'scope':'controlled format exports of development BVH; source-fidelity rather than anatomical accuracy',
                                              'records':records},indent=2))


def run_one(args):
    repo,out,row=args
    sys.path[:0]=[str(repo/'utils'),str(repo/'utils/motiogen'),str(repo/'python/scripts')]
    from evalkit_k_regression import check
    result=check((row['method'],row['source'],out/row['key'],True))
    return {**row,**result}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase',choices=['prepare','fit'],required=True)
    ap.add_argument('--cohort',choices=['tasks','first_records'],default='tasks')
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--repo',type=Path,default=ROOT)
    ap.add_argument('--version',default='k2')
    ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if a.phase=='prepare':prepare(a.out,a.cohort);return
    rows=json.loads((a.out/'manifest.json').read_text())['records']
    if any(r['partition']!='development' for r in rows):raise ValueError('Arm audit accepts development records only')
    with ProcessPoolExecutor(max_workers=a.workers,max_tasks_per_child=1) as pool:
        results=list(pool.map(run_one,[(a.repo.resolve(),a.out/a.version,r) for r in rows]))
    (a.out/f'{a.version}_summary.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    print(a.version,len(results),'computed',sum(r['status']=='computed' for r in results))


if __name__=='__main__':main()
