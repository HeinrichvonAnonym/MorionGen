import sys,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import biomech,opensim as osim
from motiogen.evalkit.k import load
from motiogen.evalkit.k.refinement import refine
from motiogen.evalkit.k.fitting3 import positions
from motiogen.evalkit.k.bones import common
osim.Logger.setLevelString('error')
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k3_rescue_20261116')
summary=json.loads((ROOT/'config/evalkit_k3_20261116/results_summary.json').read_text());row=next(json.loads(Path(p).read_text()) for p in summary['result_paths'] if '/069/kimodo/' in p)
folder=Path(row['arrays']).parent;work=base/'runtime';work.mkdir(exist_ok=True)
cfg=json.loads((ROOT/'config/evalkit_k3_20261116/experiment_config.json').read_text());motion=load(row['source'],'kimodo',config=cfg)
t,labels,data=biomech.read_storage(folder/'base_raw/coordinates.mot');mask=(t>=0)&(t<=motion.times[-1]-motion.times[0]+1e-8);t=t[mask];data=data[mask]
model=work/'model.osim';shutil.copy2(folder/'base_raw/model.osim',model)
s=dict(cfg['fitting']);s.update(rescue_after_baseline=True,baseline_max_nfev=35,interior_rescue=True,max_nfev=100)
out,diag=refine(model,motion,t,labels,data,s);p,n=positions(model,t,labels,out)
with np.load(base/'full/kimodo/interior_seed/probe.npz') as z:
 diff=float(np.max(abs(out-z['coordinates'])));pd=float(np.max(abs(common(p,n)-z['fitted'])))
assert pd<1e-5,(diff,pd)
result={'status':'passed','coordinate_max_difference':diff,'position_max_difference_m':pd,'scope':'two-pass production composition versus frozen-baseline rescue, complete Kimodo failure window'}
(base/'runtime_check.json').write_text(json.dumps(result,indent=2));print(result)
