import sys,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path[:0]=['utils','utils/motiogen','python/scripts']
ROOT=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k_arms_20261116/ablation_v2')
def run(task):
 method,source,variant=task
 import biomech,opensim as osim
 from motiogen.evalkit.k import fitting
 from evalkit_k_regression import check
 original=biomech.scaled_model
 def scaled(*args,**kwargs):
  result=original(*args,**kwargs)
  if variant in ('range','both'):
   p=args[3];m=osim.Model(str(p))
   for side in 'rl':
    c=m.updCoordinateSet().get('arm_flex_'+side);c.setRangeMin(-3.141592653589793);c.setRangeMax(3.141592653589793)
   m.printToXML(str(p))
  return result
 biomech.scaled_model=scaled
 if variant in ('centers','both'):
  fitting.C={**fitting.C,'registered_limb_markers':['hip','knee']}
 return check((method,source,ROOT/variant/method,True))
if __name__=='__main__':
 rows=json.loads(Path('config/evalkit_k_validation_20261116/regression_summary.json').read_text())
 tasks=[(r['method'],r['source'],v) for v in ['range','centers','both'] for r in rows if r['method'] in ('motionlcm','omnicontrol','kimodo')]
 with ProcessPoolExecutor(max_workers=3, max_tasks_per_child=1) as p:results=list(p.map(run,tasks))
 ROOT.mkdir(exist_ok=True,parents=True);(ROOT/'ablation.json').write_text(json.dumps([{'variant':t[2],**r} for t,r in zip(tasks,results)],indent=2))
 for t,r in zip(tasks,results):print(t[0],t[2],r['status'],r.get('reason',''),{k:round(v['p95_deg'],2) if v['p95_deg'] else None for k,v in r.get('geometric_fit_errors',{}).items()})
