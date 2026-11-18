"""D-only fixed MTP-translation feasibility diagnostic; not a promoted calibration."""
import sys,json,pathlib,hashlib
import numpy as np
from scipy.optimize import minimize
r=pathlib.Path('/home/heinrich/MorionGen');sys.path[:0]=[str(r/'utils'),str(r/'utils/motiogen')]
import opensim as osim,biomech
from motiogen.evalkit.k.leg_seed import LegDirectionSeeder
from motiogen.evalkit.k.foot_domain import polar_range,model_curve
from motiogen.evalkit.k.resampling.model import _matrix
from motiogen.evalkit.k.constraints import synchronize
from motiogen.evalkit.k.motion import load
from motiogen.degradation.metrics import points
osim.Logger.setLevelString('error');b=pathlib.Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');out=b/'foot_fixed_geometry_probe_002_v1';out.mkdir(exist_ok=False)
for method in ('motionlcm','omnicontrol'):
 src=b/f'closure_formats_20261118_v1/fits/geometry_domain/002/{method}';mp=b/f'closure_position_foot_full_domain_20261118_v1/{method}/domain/model.osim'
 motion=load(b/f'runtime_inputs/002/{method}/candidate/motion.npz',method,config=r/'config/evalkit_k4_20261117/input.json');d=points(motion);t,labels,q=biomech.read_storage(src/'coordinates_refined_native.mot');seeder=LegDirectionSeeder(mp,labels,'left');m,s,cs=seeder.model,seeder.state,seeder.cs;ankle=m.getJointSet().get('ankle_l');frame=osim.PhysicalOffsetFrame.safeDownCast(m.getJointSet().get('mtp_l').getParentFrame());old=np.array([frame.get_translation().get(i) for i in range(3)])
 angles=[];rows=[]
 for i in range(len(q)):
  row,rec=seeder.seed(q[i],np.array([d['left_'+n][i] for n in ('hip','knee','ankle')]))
  if row is None:continue
  cs.get('ankle_angle_l').setValue(s,0,False);cs.get('subtalar_angle_l').setValue(s,0,False);synchronize(s,seeder.bindings);m.realizePosition(s)
  a=_matrix(ankle.getParentFrame(),s)[:,2];foot=d['left_foot'][i]-d['left_ankle'][i];angles.append(np.arccos(np.clip(a@foot/np.linalg.norm(foot),-1,1)));rows.append((i,row))
 i,row=rows[len(rows)//2];curve=model_curve(mp,labels,row,'left');c,u,v,axis=[np.array(curve[n]) for n in ('center','cosine','sine','axis')]
 rot=[]
 for angle in (0,np.pi,np.pi/2):
  biomech._set(m,s,labels,row);cs.get('ankle_angle_l').setValue(s,0,False);cs.get('subtalar_angle_l').setValue(s,float(angle),False);synchronize(s,seeder.bindings);m.realizePosition(s);rot.append(_matrix(frame.getParentFrame(),s))
 C=(rot[0]+rot[1])/2;U=(rot[0]-rot[1])/2;V=rot[2]-C
 footlength=np.linalg.norm(d['left_foot']-d['left_ankle'],axis=-1);length=float(np.median(footlength));lo,hi=min(angles),max(angles)
 def describe(delta):
  cc,uu,vv=c+C@delta,u+U@delta,v+V@delta
  domain=polar_range(cc,uu,vv,axis);g=cc@cc+uu@uu;amp=np.hypot(2*cc@uu,2*cc@vv)
  return domain,float(np.sqrt(g-amp)),float(np.sqrt(g+amp))
 def cons(delta):
  try:dom,mn,mx=describe(delta)
  except ValueError:return np.ones(4)*-10
  l,h=dom['polar_interval_rad'];return np.array([lo-l,h-hi,mn-(length-.02),(length+.02)-mx])
 attempts=[]
 for start in (np.zeros(3),np.array([0,-.05,0]),np.array([0,.05,0]),np.array([0,0,.05]),np.array([0,0,-.05])):
  fit=minimize(lambda x:float(x@x)/length**2,start,method='SLSQP',bounds=[(-length,length)]*3,constraints=[dict(type='ineq',fun=cons)],options=dict(maxiter=200,ftol=1e-12))
  attempts.append(dict(delta=fit.x.tolist(),success=bool(fit.success),message=fit.message,objective=float(fit.fun),constraints=cons(fit.x).tolist()))
 valid=[x for x in attempts if min(x['constraints'])>=-1e-7];selected=min(valid,key=lambda x:x['objective']) if valid else None
 result=dict(method=method,supported=len(rows),total=len(q),target_polar_range_deg=np.rad2deg([lo,hi]).tolist(),input_foot_length_m=length,input_length_range_m=[float(footlength.min()),float(footlength.max())],asset=describe(np.zeros(3)),attempts=attempts,selected=selected,source_model_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),scope='D-only full-window direction-derived proximal poses; fixed endpoint geometry search, all-angle length bounded to observed median +/-2cm. Not anatomical neutral calibration or full fit acceptance. No R, no input changes, no scoring masks altered.')
 if selected:
  delta=np.array(selected['delta']);frame.set_translation(osim.Vec3(*(old+delta)));m.finalizeConnections();m.initSystem();dest=out/(method+'_candidate.osim');m.printToXML(str(dest));actual=model_curve(dest,labels,row,'left');expected=describe(delta)[0]
  np.testing.assert_allclose(actual['polar_interval_rad'],expected['polar_interval_rad'],atol=1e-8);result['candidate_geometry']=describe(delta);result['saved_model']=str(dest)
 (out/(method+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(method,json.dumps(result),flush=True)
