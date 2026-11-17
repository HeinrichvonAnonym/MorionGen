exec(open('develop_logs/test_runs/20261117_k34/knee_direction.py').read().split('results=[]')[0]+'results=[]')
from scipy.optimize import least_squares
records=[]
for policy in ['locked','unlocked']:
 cs.get('subtalar_angle_l').setLocked(state,policy=='locked')
 coord_names=['hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l']+(['subtalar_angle_l'] if policy=='unlocked' else [])
 coords=[cs.get(n) for n in coord_names];columns=[labels.index(n) for n in coord_names]
 lo=np.array([c.getRangeMin() for c in coords])+1e-7;hi=np.array([c.getRangeMax() for c in coords])-1e-7
 target4=z['source'][i,[names.index('left_'+n) for n in ['hip','knee','ankle','foot']]];wanted=unit(np.diff(target4,axis=0));initial=z['coordinates'][i].copy();initial[labels.index('subtalar_angle_l')]=0;biomech._set(model,state,labels,initial)
 def residual(x):
  for c,v in zip(coords,x):c.setValue(state,float(v),False)
  model.realizePosition(state);actual=np.array([biomech._child_origin(model,state,j) for j in ['hip_l','walker_knee_l','ankle_l','mtp_l']]);return (unit(np.diff(actual,axis=0))-wanted).ravel()
 seeds=[np.clip(np.deg2rad(initial[columns]),lo,hi)]+list(np.random.default_rng(1729).uniform(lo,hi,(12,len(lo))))
 sols=[least_squares(residual,s,bounds=(lo,hi),max_nfev=200) for s in seeds];best=min(sols,key=lambda s:float(s.fun@s.fun));records.append({'policy':policy,'coordinates_deg':dict(zip(coord_names,np.rad2deg(best.x).tolist())),'direction_errors':np.degrees(2*np.arcsin(np.clip(np.linalg.norm(best.fun.reshape(3,3),axis=1)/2,0,1))).tolist()})
Path('develop_logs/test_runs/20261117_k35/subtalar_probe.json').write_text(json.dumps(records,indent=2));print(json.dumps(records,indent=2))
