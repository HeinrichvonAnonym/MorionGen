"""Fixed development-only K3.4 transfer and solver ablations (no holdout)."""
import os,sys,json,time,shutil,argparse,traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
from evalkit_k33_conversion import reference,interp,source_hashes
from evalkit_k31_full import read,atomic
from motiogen.evalkit.k.export import retarget,serialize,ALIASES
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.scoring import angles,stats,compare
from motiogen.evalkit.k.config import digest,implementation_key
from motiogen.evalkit.k.motion import load

BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs')
OUT=BASE/'evalkit_k34_20261117'

def candidate():
 c=read(ROOT/'config/evalkit_k33_20261117/runtime_config.json')
 c['version']='k-3.4-input-geometry-seed';c['fitting'].update(geometry_knee_seed=True,independent_coordinates_only=True)
 return c

def light(rows):
 results=[];exports=[]
 selected={(r['method'],r['window_key']) for r in read(ROOT/'config/evalkit_k33_20261117/summary.json')['fits']}
 for i,row in enumerate(rows):
  ref=reference(row)
  with np.load(row['arrays']) as z:t=z['times'];R=z['R']
  old=load(row['source'],row['method'],config='config/evalkit_k_v3_1.json');query=t+old.times[0]
  target,parents=retarget(ref,row['method'],'direction_v3')
  p=common(interp(target.times,target.positions,query),[ALIASES.get(n,n) for n in target.names]);a,b=angles(p),angles(R)
  results.append({**{k:row[k] for k in ['method','window_key','id','group_id']},'angles':{k:stats(a[k]-b[k]) for k in b},'transfer':target.provenance['trunk_transfer']})
  if (row['method'],row['window_key']) in selected:
   src=serialize(target,parents,OUT/'inputs'/row['window_key']/row['method']);exports.append({**row,'source':str(src)})
  if (i+1)%40==0:print('light',i+1,flush=True)
 atomic(OUT/'light.json',results);atomic(OUT/'exports.json',exports)

def fit_one(task):
 row,variant=task;work=OUT/'fits'/variant/row['window_key']/row['method'];work.mkdir(parents=True,exist_ok=True)
 config=candidate() if variant=='solver' else read(ROOT/'config/evalkit_k33_20261117/runtime_config.json')
 try:
  assert shutil.disk_usage(OUT).free>80*1024**3
  import opensim as osim
  osim.Logger.setLevelString('error')
  from motiogen.evalkit.k.fitting import fit
  motion=load(row['source'],row['method'],config=config);result=fit(motion,work/'fit',75.,0.,config=config)
  with np.load(work/'fit/kinematics.npz') as z:ft=z['times'];p=z['fitted'];dn=z['source'];names=list(z['names'])
  with np.load(row['arrays']) as z:t=z['times'];R=z['R'];lengths=z['reference_lengths']
  assert t[0]>=ft[0]-1e-8 and t[-1]<=ft[-1]+1e-8
  F=common(interp(ft,p,t),names);D=common(interp(ft,dn,t),names)
  masks={k:np.interp(t,result['times'],m.astype(float))>=.99999 for k,m in result['region_angle_valid'].items()}
  scores,arrays=compare(R,D,{'new':D},{'new':F},t,{'new':masks},float(np.nanpercentile(D[:,[5,9],1],5)),reference_lengths=lengths)
  np.savez_compressed(work/'comparison.npz',**arrays)
  item={**row,'variant':variant,'status':'computed','config':config,'scores':scores['new'],'output':str(work)}
 except Exception as e:
  (work/'error.log').write_text(traceback.format_exc());item={**row,'variant':variant,'status':'failed','reason':str(e)}
 atomic(work/'result.json',item);return item

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--light',action='store_true');ap.add_argument('--fit',action='store_true');a=ap.parse_args()
 for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(key)=='1'
 OUT.mkdir(exist_ok=True)
 rows=[read(p) for p in read(ROOT/'config/evalkit_k3_20261116/results_summary.json')['result_paths']];assert len(rows)==656
 freeze={'implementation':implementation_key(),'runner':digest(__file__),'references':{r['reference']:digest(r['reference']) for r in rows},'protected':{str(p.relative_to(ROOT)):digest(p) for d in ['config','plans'] for p in (ROOT/d).rglob('*') if p.is_file()},'scope':'fixed development only; no holdout','candidate':candidate()}
 if (OUT/'freeze.json').exists():assert read(OUT/'freeze.json')==freeze,'implementation changed: use separate output directory'
 else:atomic(OUT/'freeze.json',freeze)
 if a.light:light(rows)
 if a.fit:
  exports=read(OUT/'exports.json');tasks=[(r,'baseline') for r in exports]
  # Same corrected inputs for baseline/candidate on the fixed probes; include
  # the known leg failures using their frozen older inputs as separate tests.
  tasks += [(r,'solver') for r in exports]
  tasks += [(r,'solver') for r in rows if r['method']=='kimodo' and r['window_key'] in ['074','077']]
  with ProcessPoolExecutor(max_workers=3) as pool:
   for r in pool.map(fit_one,tasks):print('fit',r['method'],r['window_key'],r['variant'],r['status'],r.get('reason',''),flush=True)
if __name__=='__main__':main()
