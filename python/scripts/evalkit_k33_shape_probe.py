"""Separate length-only from full local-offset replacement; diagnostics, not exports."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'python/scripts'))
from evalkit_k33_conversion import reference,source_shape,retarget,ALIASES,target_template,common,angles,stats,interp,read,load
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs')
rows=[read(p) for p in read(ROOT/'config/evalkit_k3_20261116/results_summary.json')['result_paths']];result=[]
for c in read(ROOT/'config/evalkit_k32_development_20261116/torso.json'):
    row=next(r for r in rows if r['method']==c['method'] and r['window_key']==c['window_key']);ref=reference(row);target,parents=retarget(ref,row['method'],'topology_v2')
    with np.load(row['arrays']) as z:t=z['times'];R=z['R']
    query=t+load(row['source'],row['method'],config='config/evalkit_k_v3_1.json').times[0]
    offset_shape=source_shape(target,ref,parents);rest=target_template(row['method'])[1];p=np.zeros_like(target.positions)
    for j,par in enumerate(parents):
        if par<0:p[:,j]=target.positions[:,j];continue
        a=rest[j]-rest[par];length=np.median(np.linalg.norm(offset_shape[:,j]-offset_shape[:,par],axis=-1));p[:,j]=p[:,par]+target.rotations[:,par]@(a/np.linalg.norm(a)*length)
    angle=lambda p:angles(common(interp(target.times,p,query),[ALIASES.get(n,n) for n in target.names]))['trunk_incline']
    ar=angles(R)['trunk_incline'];result.append({**{k:row[k] for k in ['id','method','window_key']},'metrics':{n:stats(angle(a)-ar) for n,a in [('target_shape',target.positions),('source_lengths_only',p),('source_local_vectors',offset_shape)]}})
    print(row['method'],{k:round(v['p95_abs'],3) for k,v in result[-1]['metrics'].items()})
(ROOT/'develop_logs/test_runs/20261117_k_conversion/shape_ablation.json').write_text(json.dumps(result,indent=2))
