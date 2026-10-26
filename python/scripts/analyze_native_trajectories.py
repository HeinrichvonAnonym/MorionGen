"""Separate supplementary OmniControl trajectory domain; reuse shared estimators.

CPU-only actual input-packing capture stops before sampling. Completed inference
comes exclusively from registered supervised workers, never from this check.
"""
import hashlib
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.config import load_config
from motiogen.adapters import OmniControl,require
from motiogen.motion import load_motion,HUMANML_NAMES
from motiogen.metrics import evaluate
from motiogen.native_analysis import factorial_block,clustered_interval,foot_sensitivity,task_sensitivity
from motiogen.native_receipt_checks import intervention_checks
from motiogen.weight_identity import snapshot as weight_snapshot
from motiogen.preprocessing_identity import snapshot as preprocessing_snapshot
from motiogen.runtime import json_write,checksum

base=ROOT/'develop_logs/test_runs/20261024_omnicontrol_trajectory'
cfg=load_config();plan=json.loads((base/'pilot_plan.json').read_text())
contract=json.loads((ROOT/'config/native_metric_contract.json').read_text())
weights=weight_snapshot(cfg,'omnicontrol');preprocessing=preprocessing_snapshot(cfg,'omnicontrol')
adapter=OmniControl.__new__(OmniControl);adapter.cfg=cfg;adapter.name='omnicontrol'
adapter.spec=cfg['baselines']['omnicontrol'];adapter.names=HUMANML_NAMES;adapter.fps=20
adapter.raw_mean=np.load(require(cfg,'omnicontrol','Mean_raw.npy')).reshape(22,3)
adapter.raw_std=np.load(require(cfg,'omnicontrol','Std_raw.npy')).reshape(22,3)
adapter.model=SimpleNamespace(njoints=263,nfeats=1)
adapter.timed=lambda *a,**k:nullcontext()
class PackingCaptured(Exception):pass
captured={}
def capture(_model,_shape,**kwargs):
    captured['hint']=kwargs['model_kwargs']['y']['hint'].numpy().copy()
    raise PackingCaptured()
adapter.diffusion=SimpleNamespace(p_sample_loop=capture)
original_tensor=torch.tensor
def cpu_tensor(*args,**kwargs):
    kwargs.pop('device',None)
    return original_tensor(*args,**kwargs)
packing=[]
for task in plan['tasks']:
    adapter.scene=task['scene']
    with patch.object(torch.Tensor,'cuda',lambda self,*a,**k:self),patch.object(torch,'tensor',cpu_tensor):
        try:adapter.generate()
        except PackingCaptured:pass
        else:raise ValueError('Diagnostic unexpectedly reached inference')
    expected=np.zeros((1,196,22,3),np.float32)
    for c in adapter.scene['constraints']:
        frame=round(c['time_seconds']*20);j=HUMANML_NAMES.index(c['joint']);x,y,z=c['position_m']
        expected[0,frame,j]=(np.array([x,z,-y])-adapter.raw_mean[j])/adapter.raw_std[j]
    actual=captured['hint'].reshape(expected.shape)
    count=int(np.any(actual!=0,axis=-1).sum())
    packing.append({'task':task['id'],'anchors':count,'hint_exact':bool(np.array_equal(actual,expected))})
if not all(r['anchors']==9 and r['hint_exact'] for r in packing):raise ValueError('Native trajectory hint packing mismatch')
rows=[];seen=set();failures=[]
for prefix in ['pilot','batch']:
    plan_path=base/(prefix+'_plan.json');state_path=base/(prefix+'_queue/state.json')
    if not plan_path.exists() or not state_path.exists():continue
    current_plan=json.loads(plan_path.read_text());state=json.loads(state_path.read_text())
    for job in current_plan['jobs']:
        state_row=state['jobs'][job['id']]
        if state_row['status']!='success':
            if state_row['status'] not in ('running','pending'):failures.append({'job':job['id'],**state_row})
            continue
        run=Path(state_row['run_dir']);relative=str(run.relative_to(ROOT))
        if relative in seen:continue
        seen.add(relative)
        manifest=json.loads((run/'manifest.json').read_text());scene=manifest['scenario']
        assert scene==job['scene'] and manifest['seed']==job['seed']
        binding=json.loads((run/'weight_binding.json').read_text())
        assert binding['after_run_unchanged'] and binding['identity']==weights['identity']
        events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
        started=datetime.fromisoformat(next(e['wall_time'] for e in events if e['event']=='process_started')).timestamp()*1e9
        assert all(max(f['stat']['mtime_ns'],f['stat']['ctime_ns'])<started for f in preprocessing['files'])
        motion=load_motion(run/'motion.npz');receipt=json.loads((run/'intervention.json').read_text())
        stored=json.loads((run/'metrics.json').read_text())['metrics']
        policy=json.loads((run/'contracts.json').read_text())['metrics.json']['content']
        recomputed=evaluate(motion,scene,policy)['metrics']
        for key,value in recomputed.items():
            assert value['status']==stored[key]['status'], (relative,key)
            if value['status']=='measured':
                assert value['n']==stored[key]['n'] and value['unit']==stored[key]['unit']
                assert np.isclose(value['value'],stored[key]['value'],rtol=1e-6,atol=1e-8), (relative,key)
        names=list(motion['joint_names']);errors=[]
        for c in scene['constraints']:
            frame=round(c['time_seconds']*motion['fps']);j=names.index(c['joint'])
            errors.append(float(np.sqrt(np.square(motion['joints'][frame,j]-c['position_m']).sum())))
        assert len(errors)==9 and stored['hand_position_error']['n']==9
        assert np.isclose(np.mean(errors),stored['hand_position_error']['value'],rtol=0,atol=1e-12)
        values={k:v['value'] for k,v in stored.items() if v.get('status')=='measured'}
        units={k:v.get('unit','') for k,v in stored.items()}
        values['all_anchors_success_5cm']=int(max(errors)<=.05);units['all_anchors_success_5cm']='fraction'
        values['worst_anchor_error_m']=max(errors);units['worst_anchor_error_m']='m'
        worker=json.loads((run/'worker_result.json').read_text())
        values['model_rtf']=worker['timing']['generation']['mean_seconds']/(196/20);units['model_rtf']='s/s'
        values['peak_torch_allocated_gib']=worker['peak_cuda_allocated_bytes']/2**30;units['peak_torch_allocated_gib']='GiB'
        feet=foot_sensitivity(motion,contract)
        if feet['status']=='measured':
            values['foot_floor_penetration_fraction']=feet['floor_penetration_fraction'];units['foot_floor_penetration_fraction']='fraction'
            f=next(f for f in feet['rows'] if f['height_m']==.05 and f['speed_m_s']==.1)
            if f['sliding_fraction_given_candidate'] is not None:
                values['foot_sliding_given_5cm_contact_10cms']=f['sliding_fraction_given_candidate'];units['foot_sliding_given_5cm_contact_10cms']='fraction'
        row={'method':'omnicontrol','domain':'multi-anchor wrist trajectory','task':job['task_id'],'seed':job['seed'],
             'clock':'paused','condition':receipt['condition'],'calls':receipt['calls'],'frames':len(motion['joints']),
             'fps':float(motion['fps']),'finite':bool(np.isfinite(motion['joints']).all()),'run':relative,
             'task_spec':scene,'values':values,'units':units,'weight_identity':binding['identity'],
             'preprocessing_identity':preprocessing['identity'],'foot_sensitivity':feet,'anchor_errors_m':errors,
             'input_signature':hashlib.sha256(json.dumps(scene,sort_keys=True).encode()).hexdigest(),
             'trajectory_sha256':hashlib.sha256(np.asarray(motion['joints'],dtype='<f8').tobytes()).hexdigest()}
        row['intervention_checks']=intervention_checks(row,ROOT)
        assert row['intervention_checks']['passed'],row['intervention_checks']
        rows.append(row)
groups=defaultdict(list)
for row in rows:groups[(row['task'],row['seed'])].append(row)
blocks=[];effects=defaultdict(list);simple=defaultdict(list)
for (task,seed),group in groups.items():
    assert all(len({r[k] for r in group})==1 for k in ['input_signature','weight_identity','preprocessing_identity'])
    result=factorial_block(group,['realism','spatial']);result.update(task=task,seed=seed)
    if result['complete']:
        for c in result['contrasts']:effects[(tuple(c['factors']),c['metric'])].append((task,c['delta']))
        for c in result['simple_contrasts']:simple[(c['factor'],tuple(sorted(c['fixed'].items())),c['metric'])].append((task,c['delta']))
    blocks.append(result)
def summarize(metric,values):
    result=clustered_interval([v for _,v in values],[t for t,_ in values]);result.pop('improved_fraction',None)
    result['sensitivity']=task_sensitivity([v for _,v in values],[t for t,_ in values])
    result['unit']=next(r['units'][metric] for r in rows if metric in r['values'])
    return result
contrasts=[{'method':'omnicontrol','clock':'paused','factors':list(f),'metric':m,**summarize(m,v)} for (f,m),v in sorted(effects.items())]
simple_effects=[{'method':'omnicontrol','clock':'paused','factor':f,'fixed':dict(fixed),'metric':m,**summarize(m,v)} for (f,fixed,m),v in sorted(simple.items())]
negative=defaultdict(set);negative_members=defaultdict(list)
for r in rows:
    if r['condition']=={'realism':0,'spatial':0}:
        key=(r['seed'],r['task_spec']['prompt'])
        negative[key].add(r['trajectory_sha256']);negative_members[key].append(r['task'])
assert all(len(v)==1 for v in negative.values()),'All control paths off must ignore trajectory geometry'
pilot=next((b for b in blocks if b['task']==0 and b['seed']==20),{})
output={'updated_at':datetime.now(timezone.utc).isoformat(),'rows':rows,'blocks':blocks,'contrasts':contrasts,
    'simple_effects':simple_effects,'packing':packing,'pilot_passed':bool(pilot.get('complete')),
    'accepted_cells':len(rows),'expected_cells':240,'complete_blocks':sum(b['complete'] for b in blocks),
    'complete_tasks':sum(sum(b['complete'] for b in blocks if b['task']==t)==5 for t in range(12)),
    'failures':failures,'negative_control_groups':len(negative),
    'negative_control_compared_groups':sum(len(v)>1 for v in negative_members.values()),
    'negative_control_members':[{'seed':seed,'prompt':prompt,'tasks':tasks} for (seed,prompt),tasks in negative_members.items()],
    'negative_controls_passed':True,
    'weight_identity':weights,'preprocessing_identity':preprocessing,
    'source_sha256':{str(p.relative_to(ROOT)):checksum(p) for p in [Path(__file__).resolve(),ROOT/'utils/motiogen/native_analysis.py',ROOT/'utils/motiogen/native_receipt_checks.py',ROOT/'config/native_trajectory_experiments.json']},
    'scope':plan['scope']+' Independently re-counted all nine anchor distances; shared metric reconstruction and scene-level factorial estimators. CPU capture executes native input packing but deliberately stops before sampling.'}
json_write(base/'analysis.json',output)
print(json.dumps({k:output[k] for k in ['accepted_cells','expected_cells','complete_blocks','complete_tasks','pilot_passed','negative_control_groups']}))
