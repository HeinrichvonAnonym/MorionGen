"""Isolate resampling, rig conversion, and readback in worst trunk development cases."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.export import ALIASES,retarget,resample,CONFIG
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.scoring import angles,stats
from evalkit_k3_experiment import interp
from evalkit_k31_full import read
rows=[read(p) for p in read(ROOT/'config/evalkit_k3_20261116/results_summary.json')['result_paths']]
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k31_full_20261116');out=[]
for method in sorted({r['method'] for r in rows}):
    candidates=[(r,read(base/r['window_key']/method/'result.json')) for r in rows if r['method']==method]
    row,record=max(candidates,key=lambda pair:pair[1]['scores']['raw']['angles']['trunk_incline']['degradation']['p95_abs'])
    with np.load(row['reference']) as z:
        motion=Motion(z['points'],list(z['names']),z['times'],'reference',z['rotations'],list(z['names']),{'segment_basis':z['segment_basis']})
    sampled=resample(motion,CONFIG['profiles'][method]['fps']);target,_=retarget(motion,method)
    with np.load(row['arrays']) as z:t=z['times'];R=z['R'];D=z['D']
    geom=lambda m:common(interp(m.times,m.positions,t),[ALIASES.get(n,n) for n in m.names])
    S=geom(sampled);T=geom(target)
    a={k:angles(v)['trunk_incline'] for k,v in [('R',R),('S',S),('T',T),('D',D)]}
    out.append({'method':method,'id':row['id'],'window_key':row['window_key'],'selection':'highest trunk D-R P95 in existing 82 development windows; diagnostic only',
        'metrics':{x+'-'+y:stats(a[x]-a[y]) for x,y in [('S','R'),('T','S'),('T','R'),('D','T'),('D','R')]}})
    print(method,row['window_key'],{k:round(v['p95_abs'],3) for k,v in out[-1]['metrics'].items()},flush=True)
(ROOT/'develop_logs/test_runs/20261116_k32/torso.json').write_text(json.dumps(out,indent=2))
