"""Development-only selector and candidate-cache checks, no optical R input."""
import json,sys,subprocess,os,time,importlib.util
from pathlib import Path
ROOT=Path('/home/heinrich/MorionGen')
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
CODE=B/'closure_budget_v3_checkout';DEV=B/'andy_adapter_development_20261120_v1'
OUT=B/'andy_head_selector_development_20261120_v2';OUT.mkdir(exist_ok=True)
sys.path[:0]=[str(CODE/'utils'),str(CODE/'utils/motiogen')]
import numpy as np
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.candidate_budget import accepts
from motiogen.evalkit.k.candidate_quality import exact_indices
name='motiogen.evalkit.k.head_task_selection'
spec=importlib.util.spec_from_file_location(name,ROOT/'utils/motiogen/evalkit/k/head_task_selection.py')
sel=importlib.util.module_from_spec(spec);sys.modules[name]=sel;spec.loader.exec_module(sel)
config=ROOT/'config/evalkit_k4_andy_head_20261120/pool.json';settings=json.loads(config.read_text());checks=[]
for method in ('ardy','motionlcm'):
    old=json.loads((DEV/(method+'_runtime.json')).read_text())
    motion=load(old['input'],method,config=CODE/'config/evalkit_k4_20261117/input.json')
    source=DEV/'fits'/method/'candidates/pelvis_continuity'
    with np.load(source/'fit.npz') as z:fit=dict(z)
    with np.load(source/'head_neck.npz') as z:head=z['head_relative_thorax']
    kwargs=dict(head_rotation=head,thresholds=settings['selection_thresholds'],include_export_path=True,ranking_policy=settings['selection_policy'])
    q=sel.assess(motion,fit,**kwargs)
    # A body-prior perturbation must not affect head-task stopping or ranking.
    body=dict(fit);body['positions']=fit['positions'].copy()
    for j,n in enumerate(fit['names']):
        if n not in ('neck','head'):body['positions'][:,j]+=np.array([.5,.3,-.2])
    qb=sel.assess(motion,body,**kwargs);assert q['metrics']==qb['metrics'] and q['rank']==qb['rank']
    bad=dict(fit);bad['positions']=fit['positions'].copy();j=list(fit['names']).index('head');n=list(fit['names']).index('neck')
    bad['positions'][:,j]=bad['positions'][:,n]+np.roll(bad['positions'][:,j]-bad['positions'][:,n],1,axis=-1)
    assert not accepts(sel.assess(motion,bad,**kwargs),settings['candidate_budget'])
    missing=dict(fit);missing['positions']=fit['positions'].copy();missing['positions'][0,j]=np.nan
    assert not accepts(sel.assess(motion,missing,**kwargs),settings['candidate_budget'])
    index=exact_indices(motion.times,fit['times']);inserted=next(i for i in range(len(fit['times'])) if i not in index)
    jump=dict(fit);jump['positions']=fit['positions'].copy();jump['positions'][inserted,j]+=np.array([1.,0.,0.])
    assert not accepts(sel.assess(motion,jump,**kwargs),settings['candidate_budget'])
    if method=='ardy':
        from scipy.spatial.transform import Rotation
        twist=head.copy();twist[1:]=twist[1:]@Rotation.from_euler('y',20,degrees=True).as_matrix()
        assert not accepts(sel.assess(motion,fit,**dict(kwargs,head_rotation=twist)),settings['candidate_budget'])
    else:assert not any('relative_thorax' in k for k in q['metrics'])
    request=dict(old,output=str(OUT/method),fit_config=str(config),reuse_fit=old['output'],registered_identity=str(B/'andy_head_test_20261120_v1/protocol.json'),selector_module=str(ROOT/'utils/motiogen/evalkit/k/head_task_selection.py'))
    job=OUT/(method+'.json');job.write_text(json.dumps(request,indent=2)+'\n');started=time.time()
    with (OUT/(method+'.log')).open('a') as log:
        subprocess.run([sys.executable,str(ROOT/'python/scripts/evalkit_k4_head_run.py'),'--code-root',str(CODE),'--job',str(job)],stdout=log,stderr=subprocess.STDOUT,check=True)
    chosen=json.loads((OUT/method/'candidate_selection.json').read_text())
    checks.append(dict(method=method,selection=chosen['selected'],visited=list(chosen['qualities']),early=chosen['budget']['early_accepted'],wall_s=time.time()-started,body_prior_invariance=True,bad_direction_rejected=True,missing_rejected=True,export_jump_rejected=True,rotation_observable=method=='ardy'))
(ROOT/'develop_logs/test_runs/20261120_k4_andy_head/verification.json').write_text(json.dumps(dict(checks=checks,test_R_reads=0,scope='Fixed thresholds and metric contract, development D/F only'),indent=2)+'\n')
print(json.dumps(checks),flush=True)
