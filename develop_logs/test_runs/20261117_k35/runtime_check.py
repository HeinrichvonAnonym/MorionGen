import sys,json
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path.cwd()/'utils'),str(Path.cwd()/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
from motiogen.evalkit.k.fitting import fit
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.config import digest,implementation_key
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs');row=next(r for r in json.loads((base/'evalkit_k34_20261117/exports.json').read_text()) if r['method']=='omnicontrol' and r['window_key']=='038');cfg=json.loads(Path('config/evalkit_k35_20261117/experimental_config.json').read_text());work=base/'evalkit_k35_legs_20261117/runtime/038/omnicontrol'
r=fit(load(row['source'],'omnicontrol',config=cfg),work,75.,0.,config=cfg);z=np.load(work/'kinematics.npz');expected=np.load(base/'evalkit_k35_legs_20261117/accepted/038/omnicontrol/comparison.npz');p=common(z['fitted'],list(z['names']));error=float(abs(p-expected['F:selected']).max());assert error<1e-6
result={'maximum_endpoint_discrepancy_m':error,'leg_receipt':r['receipt']['optimization']['leg_refinement'],'config_hash':digest('config/evalkit_k35_20261117/experimental_config.json'),'implementation':implementation_key()};Path('develop_logs/test_runs/20261117_k35/runtime_check.json').write_text(json.dumps(result,indent=2));print('PASS endpoint difference',error)
