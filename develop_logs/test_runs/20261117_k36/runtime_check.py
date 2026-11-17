import sys,json
from pathlib import Path
import numpy as np
ROOT=Path.cwd();sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
from motiogen.evalkit.k.domain_fitting import fit_kinematic_domain
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.config import implementation_key,digest
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs');row=next(r for r in json.loads((base/'evalkit_k34_20261117/exports.json').read_text()) if r['method']=='kimodo' and r['window_key']=='010');work=base/'evalkit_k36_runtime_20261117/010/kimodo'
r=fit_kinematic_domain(row['source'],work,config='config/evalkit_k36_20261117/kinematic_config.json',method='kimodo');expected=np.load(base/'evalkit_k36_panel_20261117/combined/010/comparison.npz')['F:selected'];error=float(abs(common(r['fitted'],r['names'])-expected).max());assert error<1e-6
assert not r['force_eligible'] and r['output_kind']=='kinematic_only'
assert not (work/'model.osim').exists() and not (work/'coordinates.mot').exists()
r2=fit_kinematic_domain(row['source'],work,config='config/evalkit_k36_20261117/kinematic_config.json',method='kimodo');np.testing.assert_array_equal(r['coords'],r2['coords'])
report={'endpoint_error_m':error,'cache_replay_equal':True,'force_eligible':False,'distinct_output_names':True,'config_hash':digest('config/evalkit_k36_20261117/kinematic_config.json'),'implementation':implementation_key(),'coverage':{k:float(v.mean()) for k,v in r['region_angle_valid'].items()}}
(ROOT/'develop_logs/test_runs/20261117_k36/runtime_check.json').write_text(json.dumps(report,indent=2));print('PASS',error)
