from pathlib import Path
import json,hashlib
import numpy as np
import opensim as osim
osim.Logger.setLevelString('error')
from motiogen.evalkit.k.k4_fitting import model_fk
base=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');rows=[]
for directory in ['whole_rotation_20261118_v2','whole_rotation_20261118_stress_v2']:
 protocol=json.loads((base/directory/'protocol.json').read_text())
 for item in protocol['records']:
  for condition in protocol['conditions']:
   path=base/directory/'replays'/item['window']/item['method']/'views'/condition/item['window']/item['method']
   if not (path/'fk.npz').exists():
    rows.append(dict(run=directory,window=item['window'],method=item['method'],condition=condition,status='unavailable_legacy_control'));continue
   with np.load(path/'coordinates.npz') as q,np.load(path/'fk.npz') as f:
    body,names,consistent=model_fk(Path(item['fit'])/'model.osim',list(q['labels']),q['coords'],direct_couplers=True,return_coordinates=True)
    actual=f['positions'][:,[list(f['names']).index(n) for n in names]]
    error=float(abs(body-actual).max());coupling=float(abs(consistent-q['coords']).max())
    assert error<1e-8 and coupling<1e-8,(path,error,coupling)
    rows.append(dict(run=directory,window=item['window'],method=item['method'],condition=condition,status='passed',frames=len(body),body_fk_max_abs_m=error,dependent_coordinate_error=coupling))
result=dict(rows=rows,passed=sum(r['status']=='passed' for r in rows),expected=len(rows),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Every saved frame/coordinate, independent OpenSim FK re-evaluation and coupler consistency; head-chain correctness covered by separate known-FK tests')
(base/'whole_rotation_replayed_fk_verification_20261118.json').write_text(json.dumps(result,indent=2)+'\n');print(result['passed'],'/',result['expected'])
