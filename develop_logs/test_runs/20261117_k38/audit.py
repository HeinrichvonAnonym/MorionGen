import sys,json
from pathlib import Path
sys.path[:0]=['utils','utils/motiogen','python/scripts']
import numpy as np
from evalkit_k35_legs import BASE,ROOT,read,atomic
from evalkit_k33_conversion import reference,interp
from motiogen.evalkit.k.export import retarget,ALIASES,resample
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.config import digest
row=read(BASE/'evalkit_k31_full_20261116/074/kimodo/result.json');ref=reference(row);target,_=retarget(ref,'kimodo');motion=load(row['source'],'kimodo');q=resample(ref,motion.fps)
n=[ALIASES.get(n,n) for n in q.names];a=q.positions[:,n.index('pelvis')];b=.5*(q.positions[:,n.index('left_hip')]+q.positions[:,n.index('right_hip')]);c=target.positions[:,target.names.index('pelvis')];d=.5*(target.positions[:,target.names.index('left_hip')]+target.positions[:,target.names.index('right_hip')])
stats=lambda v:{'p95_m':float(np.percentile(np.linalg.norm(v,axis=-1),95)),'vector_bias_m':v.mean(0).tolist()}
result={'retarget_reload_max_m':float(abs(target.positions-motion.positions).max()),'source_root_vs_target_root_max_m':float(abs(a-c).max()),'reference_mid_minus_root':stats(b-a),'target_mid_minus_root':stats(d-c),'mid_error':stats(d-b),'root_contract':'legacy preserves named root, despite different root-to-hip-midpoint geometry'}
atomic(Path('develop_logs/test_runs/20261117_k38/audit.json'),result);print(json.dumps(result,indent=2))
