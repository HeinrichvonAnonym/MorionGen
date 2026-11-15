import json,sys
from pathlib import Path
import numpy as np
sys.path[:0]=['utils','utils/motiogen']
import opensim as osim
import biomech
from motiogen.evalkit.k import load,Motion
from motiogen.evalkit.k.body import evaluate_body
root=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k_20261116/regression_candidate')
rows=json.loads(Path('config/evalkit_k_validation_20261116/regression_summary.json').read_text());out={}
for row in rows:
 if row['method'] not in ('motionlcm','omnicontrol','kimodo'):continue
 method=row['method'];motion=load(row['source']);points,names,fps=motion.legacy();ref=evaluate_body(motion)
 model=osim.Model(str(root/method/'work/model.osim'));state=model.initSystem();result={}
 for phase,file in [('raw','ik/ik.mot'),('filtered','coordinates.mot')]:
  times,labels,data=biomech.read_storage(root/method/'work'/file);idx=np.clip(np.rint(times*fps).astype(int),0,len(points)-1)
  centers=[];markers=[]
  for frame in data:
   biomech._set(model,state,labels,frame)
   centers.append([biomech._child_origin(model,state,j) for j,_,_ in biomech.MARKERS.values()])
   markers.append([[model.getMarkerSet().get(n).getLocationInGround(state).get(k) for k in range(3)] for n in biomech.MARKERS])
  cnames=[v[2] for v in biomech.MARKERS.values()];centers=np.array(centers);markers=np.array(markers)
  metrics={}
  for variant,arr in [('centers',centers),('markers',markers)]:
   pred=evaluate_body(Motion(arr,cnames,times,'fitted'))
   metrics[variant]={}
   for side in ('left','right'):
    key=side+'_elbow_flexion';err=pred[key]['values_deg']-np.interp(times,motion.times,ref[key]['values_deg']);metrics[variant][side]={'median_abs':float(np.median(abs(err))),'p95':float(np.percentile(abs(err),95)),'bias':float(np.mean(err))}
  limits={}
  for label in labels:
   if label.startswith(('arm_','elbow_','pro_sup')):
    c=model.getCoordinateSet().get(label);x=data[:,labels.index(label)];lo,hi=np.rad2deg([c.getRangeMin(),c.getRangeMax()]);limits[label]={'range':[float(x.min()),float(x.max())],'limit':[lo,hi],'near_limit_share':float(np.mean((x<lo+.5)|(x>hi-.5)))}
  result[phase]={'angles':metrics,'limits':limits}
 result['length_variation']={}
 for side in ('left','right'):
  for a,b in [('shoulder','elbow'),('elbow','wrist')]:
   length=np.linalg.norm(points[:,names.index(side+'_'+a)]-points[:,names.index(side+'_'+b)],axis=-1)
   result['length_variation'][side+'_'+a+'_'+b]={'p5':float(np.percentile(length,5)),'p95':float(np.percentile(length,95))}
 out[method]=result
Path('develop_logs/test_runs/20261116_k_arms/diagnosis.json').write_text(json.dumps(out,indent=2))
for m,r in out.items():print(m,{k:v['angles'] for k,v in r.items() if k in ('raw','filtered')})
