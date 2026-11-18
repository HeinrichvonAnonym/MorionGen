"""D-only neck attachment / head-length decomposition on an unchanged saved fit."""
import hashlib,json,sys
from pathlib import Path
root=Path.cwd();b=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117');code=b/'closure_budget_v3_checkout'
sys.path[:0]=[str(code/'utils'),str(code/'utils/motiogen')]
import numpy as np
import opensim as osim
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.candidate_quality import exact_indices
from motiogen.evalkit.k.head_neck import body_frames
from motiogen.degradation.metrics import points,stats
osim.Logger.setLevelString('error')
record=json.loads((root/'config/evalkit_k4_20261118/closure_solver_trust_v2/protocol.json').read_text())['records'][-1]
fit=Path(record['source_fit']);motion=load(record['input'],'omnicontrol',config=code/'config/evalkit_k4_20261117/input.json');D=points(motion)
with np.load(fit/'fit.npz') as z:
 i=exact_indices(motion.times,z['times']);coordinates=z['coords'][i];labels=z['labels'].tolist();F={n:z['positions'][i,k] for k,n in enumerate(z['names'])}
origin,rot,_=body_frames(fit/'model.osim',labels,coordinates);head=json.loads((fit/'head_neck_model.json').read_text());offsets=np.asarray(head['offsets_m']);current=np.asarray(head['attachment_local_m'])+offsets[0]
local=np.einsum('tij,tj->ti',rot.swapaxes(-1,-2),D['neck']-origin)
head_vector=D['head']-D['neck'];length=np.linalg.norm(head_vector,axis=-1);head_length=np.linalg.norm(offsets[1]);length_term=(head_length/length[:,None]-1)*head_vector
neck_error=F['neck']-D['neck'];head_error=F['head']-D['head']
assert np.max(abs(head_error-neck_error-length_term))<1e-8
rows={}
for name,centre in [('current_separate_medians',current),('direct_neck_median',np.median(local,axis=0)),('direct_neck_mean',np.mean(local,axis=0))]:
 pred=origin+np.einsum('tij,j->ti',rot,centre);error=pred-D['neck']
 rows[name]=dict(effective_neck_offset_m=centre.tolist(),fixed_attachment_local_m=(centre-offsets[0]).tolist(),neck_error_cm=stats(100*np.linalg.norm(error,axis=-1)),head_error_cm=stats(100*np.linalg.norm(error+length_term,axis=-1)),offset_change_cm=float(100*np.linalg.norm(centre-current)))
# Pairwise distance is a radius lower bound for covering EVERY observed frame;
# it is not a lower bound on a P95 acceptance threshold.
diameter=float(np.max(np.linalg.norm(local[:,None]-local[None,:],axis=-1)))
files=[fit/x for x in ('model.osim','fit.npz','head_neck_model.json')]+[Path(record['input']),Path(__file__),code/'utils/motiogen/evalkit/k/head_neck.py']
result=dict(scope='Actual Omni input-only conditional attachment diagnostic. Body poses fixed. No R, no solver changes, no new joints. Geometry options are constant per clip. Does not establish whole-pipeline accuracy or temporal qualification.',frames=len(motion.times),neck_to_head_length_projection_cm=stats(100*np.linalg.norm(length_term,axis=-1)),head_error_vector_decomposition_max_abs_m=float(np.max(abs(head_error-neck_error-length_term))),effective_neck_local_diameter_cm=100*diameter,all_frame_covering_radius_lower_bound_cm=50*diameter,lower_bound_scope='All-frame max only, not P95',candidates=rows,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
# Each disjoint pair farther apart than 2r forces at least one residual >r
# for every constant neck offset. Match enough pairs to exceed the allowed
# tail of numpy's linear P95 order statistic (not merely a 95% count rule).
radius=.05;distance=np.linalg.norm(local[:,None]-local[None,:],axis=-1)
unused=set(range(len(local)));pairs=[]
while len(unused)>=2:
    ids=sorted(unused);block=distance[np.ix_(ids,ids)];left,right=np.unravel_index(np.argmax(block),block.shape)
    i,j=ids[left],ids[right]
    if distance[i,j]<=2*radius+1e-10:break
    pairs.append(dict(frame_a=i,frame_b=j,distance_cm=float(100*distance[i,j])))
    unused.remove(i);unused.remove(j)
required_outliers=len(local)-int(np.floor(.95*(len(local)-1)))
result['conditional_p95_proof']=dict(radius_cm=5.,disjoint_pairs=pairs,forced_outside_frames=len(pairs),required_outside_frames=required_outliers,
    proves_every_fixed_offset_exceeds_p95_5cm=len(pairs)>=required_outliers,
    scope='Fixed saved body origins/rotations only. Triangle inequality plus disjoint-pair counting under linear percentile. Does not constrain a refitted torso or prove the joint model globally inadequate.')
print('conditionalP95proof',len(pairs),required_outliers,len(pairs)>=required_outliers)
out=b/'actual_head_attachment_diagnostic_v2.json'
with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
for name,row in rows.items():print(name,'neck',row['neck_error_cm']['p95'],'head',row['head_error_cm']['p95'],'offset',row['offset_change_cm'])
print('head_length_term_cm',result['neck_to_head_length_projection_cm']['p95'],'all_frame_radius_bound_cm',50*diameter)
