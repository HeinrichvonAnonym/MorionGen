#!/usr/bin/env python3
"""Eight-format four-cell development ablation. Original references enter scoring only."""
import argparse,copy,json,sys,time,shutil,contextlib,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.config import digest
VARIANTS=['legacy_raw','legacy_projected','optimized_raw','optimized_projected']


def interp(t,p,q):
    out=np.stack([np.interp(q,t,v) for v in p.reshape(len(t),-1).T],1).reshape(len(q),*p.shape[1:])
    out[(q<t[0]-1e-8)|(q>t[-1]+1e-8)]=np.nan
    return out


def head_score(z,motion,query,config):
    from scipy.spatial.transform import Rotation,Slerp
    from motiogen.evalkit.k.export import target_template
    from motiogen.evalkit.k.calibration import UPRIGHT
    from motiogen.evalkit.k.geometry import frame,rotation_errors
    from motiogen.evalkit.k.posture import evaluate
    names=list(z['names']);lookup={n.lower():i for i,n in enumerate(names)}
    ix=lambda *keys:next(lookup[k.lower()] for k in keys if k.lower() in lookup)
    t=z['times'];basis=z['segment_basis']
    head=Slerp(t,Rotation.from_matrix(z['rotations'][:,ix('Head')]))(query).as_matrix()@basis
    chest=Slerp(t,Rotation.from_matrix(z['rotations'][:,ix('Spine3','Chest','Spine2')]))(query).as_matrix()@basis
    tn,rest,_=target_template(motion.method);ni={n:i for i,n in enumerate(tn)}
    b=frame(np.array([0.,1,0]),rest[ni['right_hip']]-rest[ni['left_hip']]);hb=UPRIGHT@b if motion.method=='closd' else b;cb=hb
    if motion.method in ('motionlcm','omnicontrol'):
        cb=frame(rest[ni['neck']]-rest[ni['spine3']],rest[ni['right_collar']]-rest[ni['left_collar']]).T@b
    result=evaluate(motion,{'head':hb,'thorax':cb},config=config);out={}
    for region,reference in [('head',head),('thorax',chest),('neck',chest.swapaxes(-1,-2)@head)]:
        item=result['metrics'][region];valid=item.get('anatomical_valid',item['valid']);valid &= np.isfinite(item['rotations']).all((1,2))
        if valid.sum()<2:out[region]={'status':'unreliable','valid':0,'total':len(query),'reason':item.get('reason')};continue
        inside=(query>=motion.times[valid][0])&(query<=motion.times[valid][-1])
        observed=np.interp(query,motion.times,valid.astype(float))>=.99999;inside &= observed
        if inside.any():
            measured=Slerp(motion.times[valid],Rotation.from_matrix(item['rotations'][valid]))(query[inside]).as_matrix()
            out[region]={'status':'computed','end_to_end':rotation_errors(reference[inside],measured,len(query)),
                         'scope':'native K orientation path; explicit controlled segment convention; no HYOID fit'}
        else:out[region]={'status':'unreliable','valid':0,'total':len(query)}
    return out


def one(task):
    row,out=task
    if shutil.disk_usage(out).free<80*1024**3:return {**row,'status':'disk_reserve_stop'}
    key=row['window_key']+'/'+row['method'];work=out/'fits'/key;work.mkdir(parents=True,exist_ok=True)
    target=work/'result.json';config=json.loads((ROOT/'config/evalkit_k_v3.json').read_text())
    # Resume only against exactly the same implementation and input receipts.
    from motiogen.evalkit.k.config import implementation_key
    identity={'implementation':implementation_key(),'config':config,'script':digest(__file__),'reference_sha256':digest(row['reference'])}
    if target.exists():
        old=json.loads(target.read_text())
        if old.get('identity')==identity and old.get('status')=='computed':return old
    start=time.monotonic()
    with (work/'run.log').open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        try:
            import opensim as osim
            osim.Logger.setLevelString('error')
            from motiogen.evalkit.k import load
            from motiogen.evalkit.k.fitting import fit
            from motiogen.evalkit.k.export import ALIASES
            from motiogen.evalkit.k.bones import common
            from motiogen.evalkit.k.scoring import compare
            motion=load(row['source'],row['method'],config=config);fitted={};failures={};receipts={}
            for variant in VARIANTS:
                c=copy.deepcopy(config);c['fitting'].update(strategy='direction_refinement' if variant.startswith('optimized') else 'legacy',projection=variant.endswith('projected'),
                    base_cache=str(work/('base_projected' if variant.endswith('projected') else 'base_raw')))
                try:
                    result=fit(row['source'],work/variant,75.,0.,config=c)
                    with np.load(work/variant/'kinematics.npz') as a: fitted[variant]={k:a[k] for k in a.files}
                    receipts[variant]=result
                except Exception as e:failures[variant]=type(e).__name__+': '+str(e)
            if not fitted:raise RuntimeError(str(failures))
            with np.load(row['reference']) as f:z={k:f[k] for k in f.files}
            first=next(iter(fitted.values()));clock=first['times'];offset=motion.times[0]
            mask=(clock+offset>=z['times'][0]-1e-8)&(clock+offset<=z['times'][-1]+1e-8)&(clock<=motion.times[-1]-offset+1e-8)
            for a in fitted.values():mask &= (clock>=a['times'][0]-1e-8)&(clock<=a['times'][-1]+1e-8)
            clock=clock[mask];query=np.clip(clock+offset,z['times'][0],z['times'][-1])
            reference=common(interp(z['times'],z['points'],query),[ALIASES.get(n,n) for n in z['names']])
            source=common(interp(first['times'],first['source'],clock),list(first['names']))
            Fs={v:common(interp(a['times'],a['fitted'],clock),list(a['names'])) for v,a in fitted.items()}
            Ps={v:common(interp(a['times'],a['projected'],clock),list(a['names'])) for v,a in fitted.items()}
            masks={v:{k:np.interp(clock,receipts[v]['times'],m.astype(float))>=.99999 for k,m in receipts[v]['region_angle_valid'].items()} for v in fitted}
            floor=float(np.nanpercentile(source[:,[5,9],1],5))
            scores,arrays=compare(reference,source,Ps,Fs,clock,masks,floor)
            # Position support remains separate from angle priority and controls downstream eligibility.
            for v in fitted:
                scores[v]['position_coverage']={k:float(np.mean(m)) for k,m in receipts[v]['region_position_valid'].items()}
                scores[v]['registered_angle_coverage']={k:float(np.mean(m)) for k,m in receipts[v]['region_registered_valid'].items()}
                scores[v]['calibration']=receipts[v]['receipt']['calibration']
            np.savez_compressed(work/'comparison.npz',**arrays)
            original={k:row[k] for k in ['id','method','group_id','subset','window_key','historical','source_window_s','reference','source']}
            rec={**original,'status':'computed','identity':identity,'scores':scores,'failures':failures,
                 'head_thorax':head_score(z,motion,query,config),'frames':len(clock),'source_clock_offset_s':float(offset),
                 'arrays':str(work/'comparison.npz'),'seconds':time.monotonic()-start}
        except Exception as e:
            import traceback;traceback.print_exc()
            rec={**row,'status':'failed','reason':type(e).__name__+': '+str(e),'identity':identity,'seconds':time.monotonic()-start}
    target.write_text(json.dumps(rec,allow_nan=False)+'\n');return rec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--limit',type=int);a=ap.parse_args()
    rows=json.loads((a.out/'exports.json').read_text());failed=[r for r in rows if r['status']!='exported'];rows=[r for r in rows if r['status']=='exported']
    assert all(r['partition']=='development' for r in rows)
    if a.limit:rows=rows[:a.limit]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(a.out/('experiment_pilot.jsonl' if a.limit else 'experiment.jsonl')).open('w') as f:
        for i,r in enumerate(pool.map(one,[(r,a.out) for r in rows])):
            f.write(json.dumps(r,allow_nan=False)+'\n');f.flush();print(i+1,len(rows),r['method'],r['status'],round(r['seconds'],1),r.get('reason',''),flush=True)
    print('export_failures',len(failed),flush=True)
if __name__=='__main__':main()
