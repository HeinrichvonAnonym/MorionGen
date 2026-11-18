exec(open('develop_logs/test_runs/20261118_k4_next/seed_clip_audit.py').read().split('results=[]')[0])
from motiogen.evalkit.k.candidate_quality import DIRECTIONS
rows=[]
for win in ('073','077'):
 for geom in ('prior','chain'):
  fit=b/f'closure_trajectory_geometry_20261118_v1/fits/balanced_{geom}/{win}/motionlcm';rc=json.loads((fit/'receipt.json').read_text());cfg=rc['key']['config']['trajectory_refinement']
  motion=load(b/f'runtime_inputs/{win}/motionlcm/candidate/motion.npz','motionlcm',config=r/'config/evalkit_k4_20261117/input.json');obs,names,_=motion.legacy();obs=obs.copy()
  for n in motion.legacy_prior_names():
   if n in names:obs[:,names.index(n)]=np.nan
  t,labels,q=biomech.read_storage(fit/'coordinates_before_trajectory.mot');obj,x,*_=problem(fit/'model.osim',labels,q,obs,names,t,cfg)
  res=obj.residual(x);space=res[:obj.n*obj.k].reshape(obj.n,obj.k);rate=res[obj.n*obj.k:obj.n*obj.k+(obj.n-1)*len(obj.indices)].reshape(obj.n-1,-1,3);pen=res[obj.n*obj.k+(obj.n-1)*len(obj.indices):]
  values=obj.values(x);goals=obj.target;edges=[n for n in DIRECTIONS if n!='head_neck_axis'];cost=(rate**2).sum(-1);big=[]
  for idx in np.argsort(cost.ravel())[-8:][::-1]:
   i,j=np.unravel_index(idx,cost.shape);sel=obj.indices[j*3:j*3+3];u,v=goals[i,sel],goals[i+1,sel];fu,fv=values[i,sel],values[i+1,sel]
   angle=lambda a,b:float(np.rad2deg(np.arccos(np.clip(a@b,-1,1))))
   big.append(dict(frame=int(i),edge=edges[j],cost=float(cost[i,j]),D_step_deg=angle(u,v),F_step_deg=angle(fu,fv),err0=angle(u,fu),err1=angle(v,fv)))
  rows.append(dict(window=win,geometry=geom,space=float((space**2).sum()),rate=float((rate**2).sum()),penalty=float((pen**2).sum()),top=big))
(b/'trajectory_seed_cost_parts_v1.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
