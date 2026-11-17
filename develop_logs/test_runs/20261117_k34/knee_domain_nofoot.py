exec(open('develop_logs/test_runs/20261117_k34/knee_direction.py').read().split('results=[]')[0]+'results=[]')
from scipy.optimize import least_squares
coord_names=['hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l'];columns=[labels.index(n) for n in coord_names];coords=[cs.get(n) for n in coord_names]
target4=z['source'][i,[names.index('left_'+n) for n in ['hip','knee','ankle','foot']]];wanted=unit(np.diff(target4,axis=0));records=[]
for policy in ['official','hip_rotation','signed_knee','both']:
 lo=np.array([c.getRangeMin() for c in coords])+1e-7;hi=np.array([c.getRangeMax() for c in coords])-1e-7
 if policy in ['hip_rotation','both']:lo[2]=-np.pi;hi[2]=np.pi
 if policy in ['signed_knee','both']:lo[3]=np.deg2rad(-40)
 biomech._set(model,state,labels,z['coordinates'][i])
 def residual(x):
  for c,v in zip(coords,x):c.setValue(state,float(v),False)
  model.realizePosition(state);actual=np.array([biomech._child_origin(model,state,j) for j in ['hip_l','walker_knee_l','ankle_l','mtp_l']]);return (unit(np.diff(actual,axis=0))[:2]-wanted[:2]).ravel()
 seeds=[np.clip(np.deg2rad(z['coordinates'][i,columns]),lo,hi)]+list(np.random.default_rng(1729).uniform(lo,hi,(12,len(lo))))
 sols=[least_squares(residual,s,bounds=(lo,hi),max_nfev=200,gtol=1e-8,ftol=1e-8,xtol=1e-8) for s in seeds];best=min(sols,key=lambda s:float(s.fun@s.fun))
 records.append({'policy':policy,'cost':float(best.fun@best.fun),'coordinates_deg':dict(zip(coord_names,np.rad2deg(best.x).tolist())),'direction_error_deg':np.degrees(2*np.arcsin(np.clip(np.linalg.norm(best.fun.reshape(2,3),axis=1)/2,0,1))).tolist()})
report={'frame':i,'time':float(z['times'][i]),'variants':records,'scope':'input-only single-frame domain diagnostic; outside-domain variants excluded from production'}
Path('develop_logs/test_runs/20261117_k34/knee_domain_nofoot.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
