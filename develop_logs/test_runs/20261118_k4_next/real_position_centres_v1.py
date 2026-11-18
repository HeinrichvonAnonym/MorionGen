"""D-only effective centres diagnostic. Geometric frames are explicit priors."""
import argparse,json,hashlib,sys
from pathlib import Path
ap=argparse.ArgumentParser()
ap.add_argument('--code-root',type=Path,required=True)
ap.add_argument('--input',type=Path,required=True)
ap.add_argument('--fit',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();sys.path[:0]=[str(a.code_root/'utils'),str(a.code_root/'utils/motiogen')]
import numpy as np
import opensim as osim
import biomech
from motiogen.evalkit.k.motion import load
osim.Logger.setLevelString('error')
m=load(a.input,'omnicontrol',config=a.code_root/'config/evalkit_k4_20261117/input.json')
ix={n:i for i,n in enumerate(m.names)}
required=['left_hip','right_hip','left_shoulder','right_shoulder','spine1','spine3']
assert all(n in ix for n in required)
prior=set(m.provenance.get('degradation',{}).get('prior_joints',[]))
assert not prior.intersection(required)
p=lambda n:m.positions[:,ix[n]]
hip=(p('left_hip')+p('right_hip'))/2;shoulder=(p('left_shoulder')+p('right_shoulder'))/2
def frame(left,up):
    ln=np.linalg.norm(left,axis=1);left=left/np.maximum(ln[:,None],1e-12)
    up=up-left*np.sum(left*up,axis=1)[:,None];un=np.linalg.norm(up,axis=1);up=up/np.maximum(un[:,None],1e-12)
    return np.stack([np.cross(left,up),up,-left],axis=-1),(ln>1e-6)&(un>1e-6)
pelvis,okp=frame(p('left_hip')-p('right_hip'),p('spine1')-hip)
torso,okt=frame(p('left_shoulder')-p('right_shoulder'),p('spine3')-p('spine1'))
prox=np.einsum('tji,tj->ti',pelvis,p('spine1')-hip)
dist=np.einsum('tji,tj->ti',torso,shoulder-p('spine1'))
valid=okp&okt&np.isfinite(prox).all(1)&np.isfinite(dist).all(1)
assert valid.sum()>=2
vectors={name:np.median(x[valid],axis=0) for name,x in [('proximal',prox),('distal',dist)]}
model=osim.Model(str(a.fit/'model.osim'));state=model.initSystem()
hip0=(biomech._child_origin(model,state,'hip_l')+biomech._child_origin(model,state,'hip_r'))/2
width=np.linalg.norm(biomech._child_origin(model,state,'acromial_l')-biomech._child_origin(model,state,'acromial_r'))
def transform(frame,state):
    tf=frame.getTransformInGround(state)
    return np.array([[tf.R().get(i,j) for j in range(3)] for i in range(3)]),np.array([tf.p().get(i) for i in range(3)])
rp,_=transform(model.getBodySet().get('pelvis'),state)
rt,_=transform(model.getBodySet().get('torso'),state)
back=hip0+rp@vectors['proximal'];mid=back+rt@vectors['distal']
targets={'back':back,'acromial_l':mid+rt@np.array([0,0,-width/2]),'acromial_r':mid+rt@np.array([0,0,width/2])}
changes={}
for joint,target in targets.items():
    state=model.initSystem();f=osim.PhysicalOffsetFrame.safeDownCast(model.getJointSet().get(joint).getParentFrame())
    assert f is not None
    r,o=transform(f.getParentFrame(),state);local=r.T@(target-o)
    before=[f.get_translation().get(i) for i in range(3)]
    f.set_translation(osim.Vec3(*local));model.finalizeConnections();state=model.initSystem()
    actual=biomech._child_origin(model,state,joint);np.testing.assert_allclose(actual,target,atol=1e-9,rtol=0)
    changes[joint]=dict(before=before,after=local.tolist(),world_target=target.tolist(),world_FK=actual.tolist())
a.output.mkdir(parents=True,exist_ok=False)
model.printToXML(str(a.output/'model.osim'))
# Saved native coordinates are a common initialization, not a newly fitted result.
import shutil
shutil.copyfile(a.fit/'coordinates_refined_native.mot',a.output/'coordinates_refined_native.mot')
receipt=dict(scope='Diagnostic effective fixed centre prior from D geometric frames; no observed segment rotation or neutral/anatomical calibration claim. Input hip line/spine1 defines pelvis frame; shoulder line/spine1-to-spine3 defines torso frame. Median local centre vectors fixed for clip. Original axes, DOFs, coupling and limits retained.',
    policy='observed_position_centres_diagnostic_v1',source_frames=int(valid.sum()),total_frames=len(valid),neutral_geometry_identified=False,
    local_vectors_m={k:v.tolist() for k,v in vectors.items()},
    local_vector_quantiles_m={k:np.percentile(v[valid],[5,50,95],axis=0).tolist() for k,v in [('proximal',prox),('distal',dist)]},
    uncertainty='Input-derived geometric frames are a reduced-model assumption; variation remains in receipt, no threshold relaxation or accuracy grant.',
    changes=changes,source_hashes={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [a.input,a.fit/'model.osim',Path(__file__)]})
(a.output/'registration.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
