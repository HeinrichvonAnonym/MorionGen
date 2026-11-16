"""Source-chain correction: full fixed development inventory and paired fit probes."""
import argparse,json,sys,hashlib,shutil,os,time,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
from motiogen.evalkit.k.export import retarget,serialize,ALIASES,target_template
from motiogen.evalkit.k.motion import Motion,load
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.scoring import angles,stats,compare
from motiogen.evalkit.k.config import digest,implementation_key
from evalkit_k3_experiment import interp,source_hashes
from evalkit_k31_full import read,atomic


def reference(row):
    with np.load(row['reference']) as z:return Motion(z['points'],list(z['names']),z['times'],'reference',z['rotations'],list(z['names']),{'segment_basis':z['segment_basis']})


def source_shape(target,ref,parents):
    """Diagnostic source-matched offsets; not a valid fixed-target-model export."""
    sn=[ALIASES.get(n,n) for n in ref.names];ix={n:i for i,n in enumerate(sn)}
    alias={'Spine':'spine1','Spine3':'spine3','RightShoulder':'right_collar','LeftShoulder':'left_collar'}
    mapped=[target.provenance['spine_mapping'].get('mapping',{}).get(n,alias.get(n,n)) for n in target.names]
    rest=target_template(target.method)[1];p=np.zeros_like(target.positions);r=target.rotations
    xyz=interp(ref.times,ref.positions,target.times)
    for j,par in enumerate(parents):
        if par<0:p[:,j]=target.positions[:,j];continue
        offset=rest[j]-rest[par]
        if mapped[j] in ix and mapped[par] in ix:
            local=np.einsum('tji,tj->ti',r[:,par],xyz[:,ix[mapped[j]]]-xyz[:,ix[mapped[par]]]);offset=np.median(local,axis=0)
        p[:,j]=p[:,par]+r[:,par]@offset
    return p


def light(rows,out,selected):
    results=[];exports=[]
    for k,row in enumerate(rows):
        if shutil.disk_usage(out).free<80*1024**3:raise RuntimeError('80 GiB reserve')
        ref=reference(row)
        with np.load(row['arrays']) as z:t=z['times'];R=z['R']
        # PRIMAL fit time starts after its two native history frames.
        old_motion=load(row['source'],row['method'],config='config/evalkit_k_v3_1.json')
        query=t+old_motion.times[0];ar=angles(R);entry={key:row[key] for key in ['id','method','window_key','group_id','subset']};entry['variants']={}
        for policy in ['legacy','topology_v2']:
            target,parents=retarget(ref,row['method'],policy)
            p=common(interp(target.times,target.positions,query),[ALIASES.get(n,n) for n in target.names]);a=angles(p)
            entry['variants'][policy]={m:stats(a[m]-ar[m]) for m in ar}
            if policy=='topology_v2':
                diagnostic=source_shape(target,ref,parents)
                angle=angles(common(interp(target.times,diagnostic,query),[ALIASES.get(n,n) for n in target.names]))['trunk_incline']
                entry['source_shape_diagnostic']=stats(angle-ar['trunk_incline']);entry['mapping']=target.provenance['spine_mapping']
                if (row['method'],row['window_key']) in selected:
                    src=serialize(target,parents,out/'inputs'/row['window_key']/row['method'])
                    exports.append({**row,'old_source':row['source'],'source':str(src)})
        results.append(entry)
        if (k+1)%40==0:print('light',k+1,len(rows),flush=True)
    atomic(out/'light.json',results);atomic(out/'exports.json',exports)


def fit_one(task):
    row,out=task;work=out/'fits'/row['window_key']/row['method'];work.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    config=read(ROOT/'config/evalkit_k_v3_1.json')
    identity={'implementation':implementation_key(),'input':source_hashes(row['source']),'reference':digest(row['reference']),'config':config}
    if (work/'result.json').exists():
        prior=read(work/'result.json')
        if prior.get('identity')==identity and prior['status']=='computed':return prior
    try:
        if shutil.disk_usage(out).free<80*1024**3:raise RuntimeError('80 GiB reserve')
        import opensim as osim
        osim.Logger.setLevelString('error')
        from motiogen.evalkit.k.fitting import fit
        motion=load(row['source'],row['method'],config=config)
        result=fit(motion,work/'fit',75.,0.,config=config)
        with np.load(work/'fit/kinematics.npz') as z:ft=z['times'];p=z['fitted'];dn=z['source'];names=list(z['names'])
        with np.load(row['arrays']) as z:t=z['times'];R=z['R'];lengths=z['reference_lengths']
        assert t[0]>=ft[0]-1e-8 and t[-1]<=ft[-1]+1e-8
        F=common(interp(ft,p,t),names);D=common(interp(ft,dn,t),names)
        masks={k:np.interp(t,result['times'],m.astype(float))>=.99999 for k,m in result['region_angle_valid'].items()}
        scores,arrays=compare(R,D,{'new':D},{'new':F},t,{'new':masks},float(np.nanpercentile(D[:,[5,9],1],5)),reference_lengths=lengths)
        np.savez_compressed(work/'comparison.npz',**arrays)
        item={**row,'status':'computed','identity':identity,'scores':scores['new'],'coverage':{k:float(v.mean()) for k,v in masks.items()},'output':str(work),'seconds':time.monotonic()-start}
    except Exception as exc:
        (work/'error.log').write_text(traceback.format_exc());item={**row,'status':'failed','identity':identity,'reason':str(exc),'seconds':time.monotonic()-start}
    atomic(work/'result.json',item);return item


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--light',action='store_true');ap.add_argument('--fit',action='store_true');ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    assert 1<=a.workers<=3
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:assert os.environ.get(key)=='1'
    a.out.mkdir(parents=True,exist_ok=True)
    rows=[read(p) for p in read(ROOT/'config/evalkit_k3_20261116/results_summary.json')['result_paths']]
    assert len(rows)==656
    selected={(r['method'],r['window_key']) for r in read(ROOT/'config/evalkit_k32_development_20261116/torso.json')}
    for c in read(ROOT/'develop_logs/notes/20261116_k3_figures/index.json')['figures']:
        if c['reason']=='典型基准误差':
            r=next(r for r in rows if r['method']==c['method'] and r['id']==c['id']);selected.add((r['method'],r['window_key']))
    freeze={'implementation':implementation_key(),'runner':digest(__file__),'baseline_commit':'26395af','selection':sorted(selected),
      'records':[{k:r[k] for k in ['method','id','window_key','reference']} for r in rows],
      'reference_hashes':{r['reference']:digest(r['reference']) for r in rows},
      'protected':{str(p.relative_to(ROOT)):digest(p) for d in ['config','plans'] for p in (ROOT/d).rglob('*') if p.is_file()},
      'scope':'82 fixed development windows x eight formats; 16 fixed fit probes; source-shape variant diagnostic only; no new holdout'}
    if (a.out/'freeze.json').exists():assert read(a.out/'freeze.json')==freeze,'Use a new output directory for changed implementation'
    else:atomic(a.out/'freeze.json',freeze)
    if a.light:light(rows,a.out,selected)
    if a.fit:
        exports=read(a.out/'exports.json')
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for r in pool.map(fit_one,[(r,a.out) for r in exports]):print('fit',r['method'],r['window_key'],r['status'],r.get('reason',''),flush=True)
if __name__=='__main__':main()
