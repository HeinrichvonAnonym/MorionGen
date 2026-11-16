"""Frozen K3.1 regression on all 82 x 8 registered development windows."""
import argparse,json,sys,shutil,time,hashlib,traceback,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
from motiogen.evalkit.k import load
from motiogen.evalkit.k.refinement import refine
from motiogen.evalkit.k.fitting3 import positions,support
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.scoring import compare,angles,region
from motiogen.evalkit.k.config import digest,implementation_key
from evalkit_k3_experiment import interp,source_hashes
from evalkit_k3_rescue_accept import score as temporal_score
REGIONS=['pelvis','torso','left_arm','right_arm','left_leg','right_leg']

def read(p):return json.loads(Path(p).read_text())
def atomic(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,allow_nan=False));tmp.replace(p)

def one(task):
    row,out,config,freeze=task;work=out/row['window_key']/row['method'];work.mkdir(parents=True,exist_ok=True);result=work/'result.json'
    if result.exists():
        previous=read(result)
        if previous.get('freeze')==freeze and previous['status']=='computed':return previous
    started=time.monotonic()
    try:
        if shutil.disk_usage(out).free<80*1024**3:raise RuntimeError('80 GiB disk reserve')
        import opensim as osim
        osim.Logger.setLevelString('error')
        folder=Path(row['arrays']).parent;oldfolder=folder/'optimized_raw'
        identity={'reference':digest(row['reference']),'comparison':digest(row['arrays']),'baseline_kinematics':digest(oldfolder/'kinematics.npz'),
            'baseline_model':digest(oldfolder/'model.osim'),'inputs':source_hashes(row['source'])}
        motion=load(row['source'],row['method'],config=config)
        settings=dict(config['fitting']);settings.update(settings['per_method'][row['method']])
        with np.load(oldfolder/'kinematics.npz') as z:
            t=z['times'];labels=list(z['labels']);initial=z['coordinates'];source=z['source'];names=list(z['names']);frozen_fit=z['fitted']
        model=work/'model.osim';shutil.copy2(oldfolder/'model.osim',model)
        if settings.get('rescue_after_baseline'):
            # First stage is the immutable, already validated K3 computation.
            # Production composition was checked exactly on the full Kimodo window.
            settings['rescue_after_baseline']=False
            coordinates,diagnostics=refine(model,motion,t,labels,initial,settings)
            fitted,fitted_names=positions(model,t,labels,coordinates);assert fitted_names==names
            reuse='frozen K3 first stage; newly computed K3.1 second stage'
        else:
            coordinates=initial;fitted=frozen_fit;diagnostics={'strategy':'unchanged K3'};reuse='unchanged CLoSD policy; exact frozen output reuse'
        np.savez_compressed(work/'kinematics.npz',times=t,labels=labels,coordinates=coordinates,names=names,source=source,fitted=fitted)
        np.savez_compressed(work/'probe.npz',times=t,coordinates=coordinates,fitted=common(fitted,names))
        q=support(source,fitted,names,settings['quality_angle_deg'],settings['quality_position_m'],motion.legacy_prior_joints())
        with np.load(row['arrays']) as z:clock=z['times'];R=z['R'];D=z['D'];lengths=z['reference_lengths']
        assert clock[0]>=t[0]-1e-8 and clock[-1]<=t[-1]+1e-8
        F=common(interp(t,fitted,clock),names)
        masks={k:np.interp(clock,t,v['angle_valid'].astype(float))>=.99999 for k,v in q.items()}
        floor=float(np.nanpercentile(D[:,[5,9],1],5))
        scores,arrays=compare(R,D,{'k31':D},{'k31':F},clock,{'k31':masks},floor,reference_lengths=lengths)
        np.savez_compressed(work/'comparison.npz',**arrays)
        old=read(oldfolder/'k3_receipt.json');calibration=bool(old['calibration']['qualified'])
        coverage={k:{'angle':float(v['angle_valid'].mean()),'registered_angle':float((v['angle_valid']&calibration).mean()),'registered_position':float((v['position_valid']&calibration).mean())} for k,v in q.items()}
        temporal=temporal_score(row,work/'probe.npz')
        record={k:row[k] for k in ['id','method','group_id','subset','window_key','reference','source','arrays']}
        record.update(status='computed',freeze=freeze,identity=identity,reuse=reuse,scores=scores['k31'],coverage=coverage,temporal=temporal,
            output=str(work),diagnostics=diagnostics,frames=len(clock),seconds=time.monotonic()-started)
    except Exception as e:
        (work/'error.log').write_text(traceback.format_exc());record={'status':'failed','id':row['id'],'method':row['method'],'freeze':freeze,'reason':str(e),'seconds':time.monotonic()-started}
    atomic(result,record);return record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--limit',type=int);a=ap.parse_args();assert 1<=a.workers<=3
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:assert os.environ.get(key)=='1',key
    a.out.mkdir(parents=True,exist_ok=True);config=read(ROOT/'config/evalkit_k_v3_1.json')
    old=read(ROOT/'config/evalkit_k3_20261116/results_summary.json');rows=[read(p) for p in old['result_paths']]
    assert len(rows)==656 and len({r['window_key'] for r in rows})==82
    evidence={'runtime_commit':'c0018cf','config':config,'config_sha256':digest(ROOT/'config/evalkit_k_v3_1.json'),
        'implementation':implementation_key(),'runner_sha256':digest(__file__),
        'selection_sha256':digest(ROOT/'config/evalkit_k3_20261116/selection.json'),'baseline_summary_sha256':digest(ROOT/'config/evalkit_k3_20261116/results_summary.json'),
        'scope':'all fixed development windows; same raw and bone-normalized reference; no new holdout',
        'identifiers':[{'id':r['id'],'method':r['method'],'window_key':r['window_key'],'reference':r['reference']} for r in rows]}
    freeze=hashlib.sha256(json.dumps(evidence,sort_keys=True).encode()).hexdigest();evidence['freeze']=freeze
    path=a.out/'freeze.json'
    if path.exists():assert read(path)==evidence,'Immutable run identity mismatch; use a new output directory'
    else:atomic(path,evidence)
    if a.limit:rows=rows[:a.limit]
    with ProcessPoolExecutor(max_workers=a.workers) as pool,(a.out/'progress.jsonl').open('a') as log:
        futures=[pool.submit(one,(r,a.out,config,freeze)) for r in rows]
        for i,f in enumerate(as_completed(futures)):
            r=f.result();line={k:r[k] for k in ['id','method','status','seconds']};log.write(json.dumps(line)+'\n');log.flush();print(i+1,len(rows),line,flush=True)
    print('completed',len(rows),flush=True)
if __name__=='__main__':main()
