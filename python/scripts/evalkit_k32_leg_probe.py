"""Isolated coordinate-range ablation, diagnostic only, original-reference scoring."""
import json,sys,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
from motiogen.evalkit.k import load
from motiogen.evalkit.k.refinement import refine
from motiogen.evalkit.k.fitting3 import positions
from motiogen.evalkit.k.bones import common
from evalkit_k3_rescue_accept import score
from evalkit_k31_full import read
import opensim as osim
osim.Logger.setLevelString('error')
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k32_leg_20261116')
rows=[read(Path(p)) for p in read(ROOT/'config/evalkit_k3_20261116/results_summary.json')['result_paths']]
row=next(r for r in rows if r['method']=='kimodo' and r['window_key']=='074')
cfg=read(ROOT/'config/evalkit_k_v3_1.json');motion=load(row['source'],'kimodo',config=cfg)
settings=dict(cfg['fitting']);settings.update(settings['per_method']['kimodo']);settings['rescue_after_baseline']=False
folder=Path(row['arrays']).parent/'optimized_raw'
with np.load(folder/'kinematics.npz') as z:t=z['times'];labels=list(z['labels']);initial=z['coordinates']
variants={'knee175':{'knee_angle_l':(0,175)},'knee_hip_ankle':{'knee_angle_l':(0,175),'hip_adduction_l':(-90,90),'hip_flexion_l':(-90,175),'hip_rotation_l':(-90,90),'ankle_angle_l':(-90,90)}}
results=[]
for name,ranges in variants.items():
    work=BASE/name;work.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(BASE).free<80*1024**3:raise RuntimeError('disk reserve')
    model=work/'model.osim';shutil.copy2(folder/'model.osim',model)
    mm=osim.Model(str(model));cs=mm.getCoordinateSet()
    for key,(lo,hi) in ranges.items():cs.get(key).setRangeMin(np.deg2rad(lo));cs.get(key).setRangeMax(np.deg2rad(hi))
    mm.printToXML(str(model))
    coordinates,diag=refine(model,motion,t,labels,initial,settings)
    p,names=positions(model,t,labels,coordinates);np.savez_compressed(work/'probe.npz',times=t,coordinates=coordinates,fitted=common(p,names))
    result=score(row,work/'probe.npz');result.update(variant=name,ranges=ranges,diagnostics=diag,scope='engineering range ablation only; no muscle validity claim; subtalar lock unchanged')
    results.append(result);(BASE/'results.json').write_text(json.dumps(results,indent=2));print(name,result['regions']['left_leg'],flush=True)
