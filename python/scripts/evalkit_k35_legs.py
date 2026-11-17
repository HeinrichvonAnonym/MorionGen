"""Same-input, same-model full-window leg objective ablation; development only."""
import sys,os,json,shutil,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
from evalkit_k31_full import read,atomic
from motiogen.evalkit.k.config import digest,implementation_key
from motiogen.evalkit.k.legs import refine_legs
from motiogen.evalkit.k.fitting3 import positions
from motiogen.evalkit.k.bones import common,NAMES
from motiogen.evalkit.k.scoring import compare,angles,region
from motiogen.evalkit.k.quality import metric_fidelity,apply_metric_gates
from motiogen.evalkit.k.fitting3 import support
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs');OUT=BASE/'evalkit_k35_legs_20261117'
SETTINGS={'hz':20,'max_nfev':70,'position_weight':.05,'continuity_weight':.002,'minimum_improvement_deg':1.,'knee_tolerance_deg':1.,'foot_tolerance_deg':5.,'foot_position_tolerance_m':.02,'step_tolerance_deg':5.,'velocity_tolerance_deg_s':10.}

def inventory():
 old=read(ROOT/'config/evalkit_k34_20261117/acceptance.json')['selected'];rows=[]
 # Two diagnosed failures plus a predeclared cross-format representative panel.
 for m,w in [('kimodo','010'),('kimodo','008'),('motionlcm','031'),('ardy','051'),('primal','046'),('chois','000'),('dart','026'),('closd','006'),('omnicontrol','038')]:
  r=next(r for r in old if r['method']==m and r['window_key']==w);rows.append({'method':m,'window':w,'kinematics':str(Path(r['output'])/'fit/kinematics.npz'),'model':str(Path(r['output'])/'fit/model.osim'),'comparison':str(Path(r['output'])/'comparison.npz'),'fit_key':'F:new','version':'K3.4 corrected conversion, baseline solver'})
 for w in ['074','077']:
  folder=BASE/'evalkit_k31_full_20261116'/w/'kimodo';rows.append({'method':'kimodo','window':w,'kinematics':str(folder/'kinematics.npz'),'model':str(folder/'model.osim'),'comparison':str(folder/'comparison.npz'),'fit_key':'F:k31','version':'frozen K3.1 input, original ranges'})
 return rows

def run(task):
 row,weight=task;work=OUT/f'foot_{weight}'/row['window']/row['method'];work.mkdir(parents=True,exist_ok=True)
 assert shutil.disk_usage(OUT).free>80*1024**3
 import opensim as osim
 osim.Logger.setLevelString('error')
 settings={**SETTINGS,'foot_weight':weight};z=np.load(row['kinematics']);t=z['times'];labels=list(z['labels']);names=list(z['names']);initial=z['coordinates'];source=z['source'];model=work/'model.osim';shutil.copy2(row['model'],model)
 selected,raw,receipt=refine_legs(model,t,labels,initial,source,names,settings)
 old=np.load(row['comparison']);assert np.allclose(t,old['times']);D=old['D'];R=old['R'];F=old[row['fit_key']]
 fits={};masks={}
 for key,q in [('baseline',initial),('candidate',raw),('selected',selected)]:
  p,ns=positions(model,t,labels,q);fits[key]=common(p,ns)
  quality=apply_metric_gates(support(D,fits[key],NAMES),metric_fidelity(D,fits[key],NAMES,10))
  masks[key]={k:v['angle_valid'] for k,v in quality.items()}
 np.testing.assert_allclose(fits['baseline'],F,atol=1e-6)
 scores,arrays=compare(R,D,{k:D for k in fits},fits,t,masks,float(np.nanpercentile(D[:,[5,9],1],5)),reference_lengths=old['reference_lengths'])
 np.savez_compressed(work/'comparison.npz',**arrays);np.savez_compressed(work/'coordinates.npz',times=t,labels=labels,baseline=initial,candidate=raw,selected=selected)
 report={'row':row,'settings':settings,'receipt':receipt,'scores':scores,'work':str(work)};atomic(work/'result.json',report)
 print(row['method'],row['window'],weight,{s:r['accepted'] for s,r in receipt['legs'].items()},flush=True)
 return {'method':row['method'],'window':row['window'],'foot_weight':weight,'path':str(work/'result.json')}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',action='store_true');a=ap.parse_args()
 for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:assert os.environ.get(k)=='1'
 OUT.mkdir(exist_ok=True);rows=inventory()
 freeze={'rows':rows,'inputs':{r[k]:digest(r[k]) for r in rows for k in ['kinematics','model','comparison']},'implementation':implementation_key(),'runner':digest(__file__),'settings':SETTINGS,'protected':{str(p.relative_to(ROOT)):digest(p) for d in ['config','plans'] for p in (ROOT/d).rglob('*') if p.is_file()},'scope':'11 fixed development windows; 8 formats; no holdout; same input and model; no new fields'}
 path=OUT/'freeze.json'
 if path.exists():assert read(path)==freeze
 else:atomic(path,freeze)
 tasks=[(r,w) for r in rows if a.panel or r['window'] in ['010','074'] for w in ([1.,.5,.25] if not a.panel else [.5])]
 with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,tasks))
 atomic(OUT/('panel.json' if a.panel else 'diagnostic.json'),results)
if __name__=='__main__':main()
