"""Fixed failure-frame ablation; source input alone drives fitting, R scores afterwards."""
import json,sys,shutil,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k import load
from motiogen.evalkit.k.refinement import refine
from motiogen.evalkit.k.fitting3 import positions
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.scoring import angles,region
FULL='--full' in sys.argv
cfg=ROOT/'config/evalkit_k3_20261116';tails=json.loads((cfg/'tail_diagnostics.json').read_text())['methods']
summary=json.loads((cfg/'results_summary.json').read_text());rows=[json.loads(Path(p).read_text()) for p in summary['result_paths']]
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k3_rescue_20261116');base=base/('full' if FULL else 'peak');base.mkdir(parents=True,exist_ok=True)
settings=json.loads((cfg/'experiment_config.json').read_text());results=[]
for method,entry in tails.items():
    focus=entry['worst_region_constraints']['region'];case=entry['regions'][focus]['highest_remaining_error'];r=next(r for r in rows if r['method']==method and r['id']==case['id']);folder=Path(r['arrays']).parent
    with np.load(folder/'optimized_raw/kinematics.npz') as z: times=z['times'];labels=list(z['labels']);coords=z['coordinates']
    peak=entry['worst_region_constraints']['peak_time_s'];idx=np.unique(np.clip(np.searchsorted(times,peak)+np.array([-5,0,5]),0,len(times)-1));idx=np.arange(len(times)) if FULL else idx;t=times[idx]
    motion=load(r['source'],method,config=settings)
    with np.load(r['arrays']) as z: reference=np.stack([np.interp(t,z['times'],v) for v in z['R'].reshape(len(z['times']),-1).T],1).reshape(len(t),-1,3);degraded=np.stack([np.interp(t,z['times'],v) for v in z['D'].reshape(len(z['times']),-1).T],1).reshape(len(t),-1,3)
    ar=angles(reference);ad=angles(degraded);metrics=[m for m in ar if region(m)==focus]
    result={'method':method,'id':r['id'],'region':focus,'times':t.tolist(),'scope':'full frozen window' if FULL else 'three frames around frozen failure peak; diagnostic only','degradation_deg':float(np.mean([abs(ad[m]-ar[m]).mean() for m in metrics])),'variants':{}}
    for variant in ['original_seed','interior_seed','interior_wide_chart']:
        work=base/method/variant;work.mkdir(parents=True,exist_ok=True);model=work/'model.osim';shutil.copy2(folder/'optimized_raw/model.osim',model)
        if variant.endswith('wide_chart'):
            import opensim as osim
            mm=osim.Model(str(model));cs=mm.getCoordinateSet()
            prefixes=['arm_flex','arm_add','arm_rot'] if focus.endswith('arm') else ['hip_flexion','hip_adduction','hip_rotation']
            for prefix in prefixes:
                c=cs.get(prefix+('_l' if focus.startswith('left') else '_r'));c.setRangeMin(-np.pi);c.setRangeMax(np.pi)
            mm.printToXML(str(model))
        s=dict(settings['fitting']);s.update(interior_rescue=variant!='original_seed',max_nfev=100)
        output,diag=refine(model,motion,t,labels,coords[idx],s);p,names=positions(model,t,labels,output);af=angles(common(p,names))
        result['variants'][variant]={'end_to_end_deg':float(np.mean([abs(af[m]-ar[m]).mean() for m in metrics])),'fit_deviation_deg':float(np.mean([abs(af[m]-ad[m]).mean() for m in metrics])),'diagnostics':diag}
        np.savez_compressed(work/'probe.npz',times=t,coordinates=output,reference=reference,degraded=degraded,fitted=common(p,names))
    results.append(result);(base/'results.json').write_text(json.dumps(results,indent=2));print(method,result['degradation_deg'],{k:v['end_to_end_deg'] for k,v in result['variants'].items()},flush=True)
