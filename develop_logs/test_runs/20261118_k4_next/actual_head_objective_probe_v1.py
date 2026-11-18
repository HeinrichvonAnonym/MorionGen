"""Conditional local body-pose solves with/without fixed head-node position goals."""
import hashlib,json,sys
from pathlib import Path
root=Path.cwd();b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');code=b/'closure_budget_v3_checkout'
sys.path[:0]=[str(code/'utils'),str(code/'utils/motiogen')]
import numpy as np
import opensim as osim
from scipy.optimize import least_squares
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.candidate_quality import exact_indices,DIRECTIONS
from motiogen.evalkit.k.constraints import couplers,synchronize
from motiogen.evalkit.k.head_neck import forward,swing
from motiogen.evalkit.k.trajectory import problem
from motiogen.degradation.metrics import points,angle
osim.Logger.setLevelString('error')
record=json.loads((root/'config/evalkit_k4_20261118/closure_solver_trust_v2/protocol.json').read_text())['records'][-1];fit=Path(record['source_fit'])
motion=load(record['input'],'omnicontrol',config=code/'config/evalkit_k4_20261117/input.json');observed,names,_=motion.legacy();D=points(motion)
for n in motion.legacy_prior_names():
 if n in names:observed[:,names.index(n)]=np.nan
with np.load(fit/'fit.npz') as z:
 indices=exact_indices(motion.times,z['times']);q=z['coords'][indices];labels=z['labels'].tolist();saved={n:z['positions'][indices,k] for k,n in enumerate(z['names'])}
receipt=json.loads((fit/'receipt.json').read_text());settings=receipt['key']['config']['trajectory_refinement']
base,x0,seed,cols,active=problem(fit/'model.osim',labels,q,observed,names,motion.times,settings);x0=x0.reshape(len(q),-1)
model=osim.Model(str(fit/'model.osim'));state=model.initSystem();bindings=couplers(model,state);cs=model.getCoordinateSet();handles=[cs.get(n) for n in active];body=model.getBodySet().get('torso')
head=json.loads((fit/'head_neck_model.json').read_text());assert head['attachment_policy']=='input_fixed_offset_position_profile'
offsets=np.asarray(head['offsets_m']);attachment=np.asarray(head['attachment_local_m']);target_head=np.stack([D[n]-(D['left_hip']+D['right_hip'])/2 for n in ('neck','head')],1).reshape(len(q),6)
body_names=[n for _,_,n in biomech.MARKERS.values()]
def extra(frame,x):
 biomech._set(model,state,labels,q[frame])
 for h,v in zip(handles,x):h.setValue(state,float(v),False)
 synchronize(state,bindings);model.realizePosition(state)
 transform=body.getTransformInGround(state);origin=np.array([transform.p().get(j) for j in range(3)]);rotation=np.array([[transform.R().get(j,k) for k in range(3)] for j in range(3)])
 pelvis=np.mean([biomech._child_origin(model,state,n) for n in ('hip_l','hip_r')],axis=0)
 neck_rotation=swing(rotation,offsets[1],D['head'][frame]-D['neck'][frame]);local=np.stack([rotation.T@neck_rotation,np.eye(3)])[None]
 p,_=forward((origin+rotation@attachment)[None],rotation[None],offsets,local)
 return (p[0,1:]-pelvis).ravel()
# Independent stored FK must match the diagnostic extension before any solve.
reconstructed=np.stack([extra(i,x) for i,x in enumerate(x0)])
stored=np.stack([saved[n]-(saved['left_hip']+saved['right_hip'])/2 for n in ('neck','head')],1).reshape(len(q),6)
np.testing.assert_allclose(reconstructed,stored,atol=1e-7,rtol=0)
error=np.linalg.norm(saved['neck']-D['neck'],axis=-1)
frames=sorted(set(np.linspace(0,len(q)-1,3,dtype=int).tolist()+np.argsort(error)[-3:].tolist()))
files=[fit/x for x in ('model.osim','fit.npz','head_neck_model.json','receipt.json')]+[Path(record['input']),Path(__file__)]
out=b/'actual_head_objective_probe_v1';out.mkdir(exist_ok=False)
protocol=dict(scope='Actual D-only conditional local reachability/target-conflict probe, not whole-window or temporal acceptance. Body model/domain/couplings and existing head attachment/lengths held fixed. No R. Both objectives share seed, bounds and numerical solver. Head targets add existing observable FK nodes; no structural change.',frames=frames,frame_selection='Three uniform indices plus three largest saved input-relative neck errors; fixed before local fits',conditions=['body_control','body_plus_fixed_head_nodes'],max_nfev=200,starts=['saved','neutral'],source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},head_extension_vs_saved_fk_max_abs_m=float(np.max(abs(reconstructed-stored))))
(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
def measurements(frame,x):
 p=base.feature(frame,x)[:len(body_names)*3].reshape(-1,3);v={n:p[k] for k,n in enumerate(body_names)};v.update(zip(('neck','head'),extra(frame,x).reshape(2,3)))
 v['pelvis_mid']=(v['left_hip']+v['right_hip'])/2;v['shoulder_mid']=(v['left_shoulder']+v['right_shoulder'])/2
 d={n:p[frame] for n,p in D.items()};d['pelvis_mid']=(d['left_hip']+d['right_hip'])/2;d['shoulder_mid']=(d['left_shoulder']+d['right_shoulder'])/2
 pos={n:float(100*np.linalg.norm(p-(d[n]-d['pelvis_mid']))) for n,p in v.items() if n in d}
 directions={n:float(angle((v[t]-v[s])[None],(d[t]-d[s])[None])[0]) for n,(s,t) in DIRECTIONS.items()}
 return dict(positions_cm=pos,directions_deg=directions)
rows=[]
for frame in frames:
 for condition in protocol['conditions']:
  include=condition=='body_plus_fixed_head_nodes';goal=np.r_[base.target[frame],target_head[frame]] if include else base.target[frame];scales=np.r_[base.scales,np.full(6,.05)] if include else base.scales
  groups=list(base.groups)+([np.arange(base.k,base.k+3),np.arange(base.k+3,base.k+6)] if include else [])
  def residual(x):
   features=np.r_[base.feature(frame,x),extra(frame,x)] if include else base.feature(frame,x);mask=np.isfinite(goal);e=np.where(mask,(features-goal)/scales,0)
   violation=[base.violation_weight*max(np.linalg.norm(e[g])-base.violation_margin,0)*mask[g].all() for g in groups]
   return np.r_[e,violation]
  for start in protocol['starts']:
   initial=x0[frame] if start=='saved' else np.clip(np.zeros(len(active)),base.lower+1e-8,base.upper-1e-8)
   solution=least_squares(residual,initial,bounds=(base.lower,base.upper),x_scale='jac',max_nfev=200)
   metrics=measurements(frame,solution.x);row=dict(frame=frame,time_s=float(motion.times[frame]),condition=condition,start=start,success=bool(solution.success),nfev=solution.nfev,objective=float(2*solution.cost),optimality=float(solution.optimality),metrics=metrics)
   rows.append(row);(out/'progress.json').write_text(json.dumps(dict(expected=len(frames)*4,results=rows),indent=2)+'\n')
   print(frame,condition,start,solution.nfev,round(metrics['positions_cm']['neck'],3),round(max(metrics['positions_cm'].values()),3),round(max(metrics['directions_deg'].values()),3),flush=True)
(out/'completion.json').write_text(json.dumps(dict(expected=len(frames)*4,results=rows,promotion='none',scope=protocol['scope']),indent=2)+'\n')
