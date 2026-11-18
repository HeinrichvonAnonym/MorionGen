import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
ROOT=Path('/home/heinrich/MorionGen');sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.output import write_output
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
out=B/'output_refactor_equivalence_20261118_v1';out.mkdir(exist_ok=False)
rows=[]
for run,condition,w,m in [('closure_position_foot_chain_20261118_v1','chain','002','motionlcm'),('closure_trajectory_chart_20261118_v1','interior_chart','077','motionlcm'),('closure_torso_20261118_v1','registered_centres','043','kimodo')]:
 source=B/run/'fits'/condition/w/m;folder=out/(w+'_'+m);folder.mkdir()
 receipt=json.loads((source/'receipt.json').read_text());job=json.loads((B/'jobs'/f'{w}_{m}.json').read_text())
 motion=load(job['input'],m,config=ROOT/'config/evalkit_k4_20261117/input.json')
 fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
 shutil.copyfile(source/'model.osim',folder/'model.osim')
 times,labels,q=biomech.read_storage(source/'coordinates_filtered_unassembled.mot')
 values,head,_=write_output(motion,folder,labels,q,times,receipt['key']['config'])
 errors={}
 with np.load(source/'fit.npz') as z:
  assert set(z.files)==set(values)
  for name in z.files:
   if z[name].dtype.kind in 'biufc':
    np.testing.assert_allclose(values[name],z[name],atol=1e-7,rtol=0,equal_nan=True)
    errors[name]=float(np.nanmax(np.abs(np.asarray(values[name],float)-np.asarray(z[name],float))))
   else:np.testing.assert_array_equal(values[name],z[name])
 assert fingerprints=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()}
 rows.append(dict(window=w,method=m,source_fit=str(source),max_abs_error=errors,source_sha256=fingerprints,passed=True))
 print(w,m,'all fields agree',flush=True)
(out/'audit.json').write_text(json.dumps(dict(records=rows,scope='Saved-model independent FK output regression; no new native solve or R access',new_holdout_uses=0),indent=2)+'\n')
