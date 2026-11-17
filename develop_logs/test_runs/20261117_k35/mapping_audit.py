import sys,json
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path.cwd()/'utils'),str(Path.cwd()/'utils/motiogen')]
import opensim as osim
import biomech
from motiogen.evalkit.k.geometry import frame,unit
from motiogen.evalkit.k.config import digest
osim.Logger.setLevelString('error')
root=Path.cwd();cpath=root/'config/evalkit_k_calibration/kimodo.json';c=json.loads(cpath.read_text());rest=np.asarray(c['neutral_joints']);names=c['names']
p=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k34_20261117/fits/baseline/010/kimodo/fit/model.osim');model=osim.Model(str(p));state=model.initSystem();cs=model.getCoordinateSet()
for i in range(cs.getSize()):cs.get(i).setValue(state,0.,False)
model.realizePosition(state)
source_basis=frame(np.array([0.,1.,0.]),rest[names.index('right_hip')]-rest[names.index('left_hip')])
model_basis=frame(np.array([0.,1.,0.]),biomech._child_origin(model,state,'hip_r')-biomech._child_origin(model,state,'hip_l'))
rotation=model_basis@source_basis.T;records={}
for side,s in [('left','l'),('right','r')]:
 ids=[names.index(side+'_'+n) for n in ['hip','knee','ankle','foot']];source=rest[ids]@rotation.T;target=np.array([biomech._child_origin(model,state,j+'_'+s) for j in ['hip','walker_knee','ankle','mtp']]);v=unit(np.diff(source,axis=0));w=unit(np.diff(target,axis=0))
 records[side]={'source_native_nodes':[c['native_names'][i] for i in ids],'source_indices':ids,'source_parents':[c['parents'][i] for i in ids],'model_joints':[j+'_'+s for j in ['hip','walker_knee','ankle','mtp']], 'neutral_direction_difference_deg':np.degrees(np.arccos(np.clip((v*w).sum(-1),-1,1))).tolist(),'source_directions_in_model_frame':v.tolist(),'model_directions':w.tolist()}
report={'records':records,'calibration_hash':digest(cpath),'model_hash':digest(p),'source_asset_record':c['source_path'],'registration':'pelvis lateral axis + declared Y-up; neutral frame diagnostic only, not subject anatomical calibration','finding':'ToeBase mapped to mtp child origin; ToeEnd excluded; both sides preserve proximal-distal order'}
asset_parts=Path(c['source_path']).parts
asset=next(parent for parent in p.parents if parent.name=='MorionGen')/Path(*asset_parts[asset_parts.index('baselines'):])
assert digest(asset)==c['sha256']
report.update(resolved_source_asset=str(asset),resolved_source_sha256=digest(asset),asset_matches_registered_hash=True)
(root/'develop_logs/test_runs/20261117_k35/mapping_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
