"""One-worker, development-only transport/FK/scoring boundary checks."""
import json,sys,subprocess,os
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen');sys.path.insert(0,str(ROOT/'python/scripts'))
import evalkit_k4_andy as a
transport=a.setup()
import numpy as np
from scipy.spatial.transform import Rotation
from motiogen.evalkit.k.motion import load
from motiogen.degradation.positions import SOURCE
from motiogen.evalkit.k.head_neck import interpolate_rotations
from motiogen.evalkit.k.native_frames import read as native
out=a.BASE/'andy_adapter_development_20261120_v1';out.mkdir(exist_ok=True)
# Known orientation independent of the exporter: head superior axis is H*[0,1,0].
t=np.arange(121)/60;c=Rotation.from_euler('xyz',np.tile([20.,-15.,10.],(len(t),1)),degrees=True).as_matrix()
h=Rotation.from_euler('xyz',np.column_stack([np.linspace(0,35,len(t)),np.linspace(0,70,len(t)),np.full(len(t),12)]),degrees=True).as_matrix()
r,prior=a.fixture(t,h,c);nh=r.names.index('Head');nn=r.names.index('Neck')
v=r.positions[:,nh]-r.positions[:,nn];v/=np.linalg.norm(v,axis=-1,keepdims=True)
assert np.max(abs(v-h[:,:,1]))<1e-12
assert np.max(abs(r.rotations[:,r.names.index('Spine3')]-c))==0
ref=out/'known_optical.npz';np.savez_compressed(ref,times=t,head=h,thorax=c,head_markers=np.zeros((len(t),4,3)),thorax_markers=np.zeros((len(t),4,3)))
checks=[]
for method in a.METHODS:
 path,d,loss,consistency=transport.export_input(r,method,out/'known_inputs'/method)
 # Compare model scoring to an ideal FK, without using the transport score.
 ff=out/(method+'_ideal.npz');hh=interpolate_rotations(t,h,d.times);cc=interpolate_rotations(t,c,d.times)
 np.savez_compressed(ff,positions=d.positions,names=d.names,times=d.times,head_relative_thorax=cc.swapaxes(-1,-2)@hh,head_world_rotation=hh)
 job=dict(reference=str(ref),input=str(path),method=method,fk=str(ff),output=str(out/(method+'_ideal_score.json')))
 a.score(job);sc=a.read(job['output']);assert all(x['acceptance']['status']=='passed' for x in sc['clocks'].values()),method
 req=sc['clocks']['D_clock']['acceptance']['required'];rel=next(x for x in req if x['metric']=='head_relative_thorax_deg')
 assert (rel['status']=='unobservable')==(method in ('motionlcm','omnicontrol'))
 # Bad head direction must fail even if all body and relative rotations are perfect.
 changed=d.positions.copy();j=d.names.index('head');n=d.names.index('neck');vec=changed[:,j]-changed[:,n]
 changed[:,j]=changed[:,n]+np.einsum('ij,tj->ti',Rotation.from_euler('z',90,degrees=True).as_matrix(),vec)
 bad=out/(method+'_bad.npz');np.savez_compressed(bad,positions=changed,names=d.names,times=d.times,head_relative_thorax=cc.swapaxes(-1,-2)@hh,head_world_rotation=hh)
 job.update(fk=str(bad),output=str(out/(method+'_bad_score.json')));a.score(job)
 assert a.read(job['output'])['clocks']['D_clock']['acceptance']['status']=='failed'
 checks.append(dict(method=method,ideal_passed=True,bad_direction_rejected=True,rotation_observable=rel['status']!='unobservable'))
# Actual optical development record (never one of the 75 held-out subjects).
registry=[json.loads(s) for s in (ROOT/'config/gold_k_20261115/trials.jsonl').read_text().splitlines()]
row=next(r for r in registry if r['dataset']=='andy' and r['partition']=='development' and r['subject']=='2193')
history=[json.loads(s) for s in (a.BASE.parent/'evalkit_k_20261116/development_candidate/records.jsonl').read_text().splitlines()]
old=next(r for r in history if r['id']==row['id'])['profiles']['window_s'];mid=sum(old)/2
row=dict(row,window_key='DEV2193',historical_window_s=old,source_window_s=[mid-1,mid+1]);prepared=a.build(row,out)
fits=[]
for method in ['ardy','motionlcm']:
 item=prepared['records'][method];folder=out/'fits'/method;folder.mkdir(parents=True,exist_ok=True)
 job=dict(input=item['input'],method=method,output=str(folder),mass_kg=75.,floor_m=item['floor_m'],fit_config=a.read(a.TV/'protocol.json')['fit_config'])
 req=out/(method+'_runtime.json');a.save(req,job)
 if not (folder/'receipt.json').exists():
  with (out/(method+'_fit.log')).open('a') as log:
   subprocess.run([sys.executable,str(a.CODE/'python/scripts/evalkit_k4_run.py'),'--code-root',str(a.CODE),'--job',str(req)],env=a.ENV,stdout=log,stderr=subprocess.STDOUT,check=True)
 views=out/'views'/method
 if not (views/'receipt.json').exists():
  subprocess.run([sys.executable,str(a.CODE/'python/scripts/evalkit_k4_native_export_views.py'),'--code-root',str(a.CODE),'--fit',str(folder),'--input',item['input'],'--method',method,'--window','DEV2193','--output',str(views)],env=a.ENV,check=True)
 for view in ['native','export']:
  job=dict(reference=prepared['reference'],input=item['input'],method=method,fk=str(views/view/'DEV2193'/method/'fk.npz'),output=str(out/(method+'_'+view+'_optical_score.json')));a.score(job)
  result=a.read(job['output']);fits.append(dict(method=method,view=view,acceptance={k:v['acceptance']['status'] for k,v in result['clocks'].items()}))
# Pipeline validity, not arbitrary success on the real optical clip, gates registration.
a.save(ROOT/'develop_logs/test_runs/20261120_k4_andy/development_verification.json',dict(known_orientation=checks,actual_development_fit=fits,reference_quality=prepared['quality'],test_records_used=0,adapter_sha256=a.sha(Path(a.__file__)),scope='Transport and scoring gates; optical development acceptance retained without method tuning'))
print(json.dumps({'transport_passed':len(checks),'real_development_views':fits}),flush=True)
