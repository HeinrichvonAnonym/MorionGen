import sys,json
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path.cwd()/'utils'),str(Path.cwd()/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
from motiogen.evalkit.k.fitting import fit
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.config import digest,implementation_key
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k34_20261117');row=next(r for r in json.loads((base/'exports.json').read_text()) if r['method']=='kimodo' and r['window_key']=='010')
c=json.loads(Path('config/evalkit_k34_20261117/runtime_config.json').read_text());work=base/'selected_runtime/010/kimodo'
r=fit(load(row['source'],'kimodo',config=c),work,75.,0.,config=c)
old=np.load(base/'fits/baseline/010/kimodo/fit/kinematics.npz');new=np.load(work/'kinematics.npz')
report={'max_position_change_m':float(abs(old['fitted']-new['fitted']).max()),'max_coordinate_change':float(abs(old['coordinates']-new['coordinates']).max()),'left_knee_input_observed':float(r['metric_quality']['left_knee_flexion']['observed'].mean()),'left_knee_model_fit_valid':float(r['metric_quality']['left_knee_flexion']['model_fit_valid'].mean()),'left_leg_model_fit_valid':float(r['region_angle_valid']['left_leg'].mean()),'implementation':implementation_key(),'config_hash':digest('config/evalkit_k34_20261117/runtime_config.json')}
assert report['max_position_change_m']<1e-6
Path('develop_logs/test_runs/20261117_k34/runtime_check.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='implementation'})
