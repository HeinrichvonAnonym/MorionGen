"""Full fixed-window acceptance and typical-record regression against original R."""
import json,sys,shutil,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k3_rescue_20261116')
from motiogen.evalkit.k import load
from motiogen.evalkit.k.refinement import refine
from motiogen.evalkit.k.fitting3 import positions,support
from motiogen.evalkit.k.bones import common,PARENTS
from motiogen.evalkit.k.scoring import angles,region

def typical(task):
    import opensim as osim
    osim.Logger.setLevelString('error')
    row=task;folder=Path(row['arrays']).parent;work=BASE/'typical'/row['method'];work.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'config/evalkit_k3_20261116/experiment_config.json').read_text());motion=load(row['source'],row['method'],config=config)
    with np.load(folder/'optimized_raw/kinematics.npz') as z:t=z['times'];labels=list(z['labels']);coordinates=z['coordinates']
    model=work/'model.osim';shutil.copy2(folder/'optimized_raw/model.osim',model)
    settings=dict(config['fitting']);settings.update(interior_rescue=True,max_nfev=100)
    output,diagnostics=refine(model,motion,t,labels,coordinates,settings)
    p,names=positions(model,t,labels,output)
    np.savez_compressed(work/'probe.npz',times=t,coordinates=output,fitted=common(p,names))
    return {'method':row['method'],'id':row['id'],'arrays':row['arrays'],'probe':str(work/'probe.npz')}

def score(row,path):
    with np.load(row['arrays']) as z:t=z['times'];R=z['R'];D=z['D'];old=z['F:optimized_raw']
    with np.load(path) as z:
        p=z['fitted'];F=np.stack([np.interp(t,z['times'],v) for v in p.reshape(len(p),-1).T],1).reshape(R.shape)
    A={k:angles(v) for k,v in [('R',R),('D',D),('K3',old),('new',F)]};out={}
    for region_name in ['pelvis','torso','left_arm','right_arm','left_leg','right_leg']:
        keys=[m for m in A['R'] if region(m)==region_name];s={}
        for name in ['D','K3','new']:
            errors=np.stack([abs(A[name][m]-A['R'][m]) for m in keys],1)
            velocity=np.stack([np.gradient(A[name][m],t)-np.gradient(A['R'][m],t) for m in keys],1)
            steps=np.stack([np.diff(A[name][m]) for m in keys],1)
            s[name]={'mean_metric_p95_deg':float(np.nanpercentile(errors,95,axis=0).mean()),'within10':float(np.mean(errors<=10)),
                'velocity_error_p95_deg_s':float(np.nanpercentile(abs(velocity),95)),'maximum_step_deg':float(np.nanmax(abs(steps)))}
        edges={'pelvis':[2,6],'torso':[1,10,13],'left_arm':[11,12],'right_arm':[14,15],'left_leg':[3,4,5],'right_leg':[7,8,9]}[region_name]
        for name,points in [('K3',old),('new',F)]:
            ds=np.stack([D[:,j]-D[:,PARENTS[j]] for j in edges],1);df=np.stack([points[:,j]-points[:,PARENTS[j]] for j in edges],1)
            dot=np.sum(ds*df,axis=-1)/(np.linalg.norm(ds,axis=-1)*np.linalg.norm(df,axis=-1))
            s[name]['input_direction_coverage']=float(np.mean(np.all(np.degrees(np.arccos(np.clip(dot,-1,1)))<=10,axis=1)))
        s['fit_deviation_p95_deg']=float(np.mean([np.percentile(abs(A['new'][m]-A['D'][m]),95) for m in keys]));out[region_name]=s
    position={name:float(np.percentile(np.linalg.norm(p-R,axis=-1),95)) for name,p in [('K3',old),('new',F)]}
    return {'method':row['method'],'id':row['id'],'frames':len(t),'regions':out,'world_position_p95_m':position,'reference':row['reference'],'reference_sha256':hashlib.sha256(Path(row['reference']).read_bytes()).hexdigest(),'probe':str(path)}

def main():
    cfg=ROOT/'config/evalkit_k3_20261116';summary=json.loads((cfg/'results_summary.json').read_text());rows=[json.loads(Path(p).read_text()) for p in summary['result_paths']]
    if '--typical' in sys.argv:
        index=json.loads((ROOT/'develop_logs/notes/20261116_k3_figures/index.json').read_text())
        cases=[r for r in index['figures'] if r['reason']=='典型基准误差'];assert len(cases)==8
        chosen=[next(r for r in rows if r['id']==c['id'] and r['method']==c['method']) for c in cases]
        (BASE/'typical_selection.json').write_text(json.dumps([{'id':r['id'],'method':r['method'],'arrays':r['arrays']} for r in chosen],indent=2))
        with ProcessPoolExecutor(max_workers=2) as pool: results=list(pool.map(typical,chosen))
        (BASE/'typical_results.json').write_text(json.dumps(results,indent=2))
    else:
        tails=json.loads((cfg/'tail_diagnostics.json').read_text())['methods'];result={'tail':[],'typical':[]}
        for method,entry in tails.items():
            focus=entry['worst_region_constraints']['region'];case=entry['regions'][focus]['highest_remaining_error'];row=next(r for r in rows if r['id']==case['id'] and r['method']==method)
            for variant in ['original_seed','interior_seed','interior_wide_chart']:
                r=score(row,BASE/'full'/method/variant/'probe.npz');r.update(variant=variant,focus=focus);result['tail'].append(r)
        for c in json.loads((BASE/'typical_results.json').read_text()):
            row=next(r for r in rows if r['id']==c['id'] and r['method']==c['method']);result['typical'].append(score(row,Path(c['probe'])))
        (BASE/'acceptance.json').write_text(json.dumps(result,indent=2))
        print('scored',len(result['tail']),len(result['typical']))
if __name__=='__main__':main()
