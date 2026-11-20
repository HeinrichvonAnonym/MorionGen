"""Andy optical head/thorax test, with explicitly synthetic body context.

No fitting changes. Optical frames are independent scoring references. The body,
neck attachment, segment lengths and all joint-centre positions are template
priors. No position or whole-body accuracy is claimed from these fixtures.
"""
import argparse
import hashlib
import html
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path('/home/heinrich/MorionGen')
BASE=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
TV=BASE/'test_budget_20261120_v1'
CODE=BASE/'closure_budget_v3_checkout'
METHODS=['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']
ENV={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def setup():
    spec=importlib.util.spec_from_file_location('tv_transport',TV/'control/evalkit_k4_test_evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.setup(CODE)
    return module


def fixture(times,head,thorax):
    """Virtual Y-directed head link carries observed head-up, not joint centres.

    Rotations on the synthetic chest and head carry measured marker-frame
    orientations. Neck rotation is a virtual articulation; it is not measured.
    """
    import numpy as np
    from motiogen.evalkit.k.export import target_template
    from motiogen.evalkit.k.motion import Motion
    from motiogen.degradation.positions import SOURCE
    names,rest,parents=target_template('primal');rest=rest.copy()
    neck,headid=names.index('neck'),names.index('head')
    length=float(np.linalg.norm(rest[headid]-rest[neck]))
    rest[headid]=rest[neck]+np.array([0.,length,0.])
    rot=np.broadcast_to(np.eye(3),(len(times),len(names),3,3)).copy()
    # Same fixed template-body context as the historical segment fixture,
    # plus explicit virtual neck articulation to preserve head-up in positions.
    for j in (3,6,9,12,13,14,16,17,18,19,20,21):rot[:,j]=thorax
    rot[:,neck]=head;rot[:,headid]=head
    points=np.zeros((len(times),len(names),3))
    for j,parent in enumerate(parents):
        points[:,j]=rest[j] if parent<0 else points[:,parent]+np.einsum('tij,j->ti',rot[:,parent],rest[j]-rest[parent])
    canonical=[SOURCE[n] for n in names]
    motion=Motion(points,canonical,times,'reference',rot,canonical,
                  {'rest_positions':rest.tolist(),'parents':parents,'scope':'optical head/chest rotations on synthetic body; positions and neck joint are priors'})
    return motion,{'prior_asset':str(CODE/'config/evalkit_k_calibration/primal.json'),
                   'prior_asset_sha256':sha(CODE/'config/evalkit_k_calibration/primal.json'),
                   'head_link_length_m':length,'head_link_axis':[0.,1.,0.],
                   'measured':['head marker-frame orientation','thorax marker-frame orientation'],
                   'derived_observed':['head superior-axis direction','head relative thorax rotation'],
                   'prior':['all joint-centre positions','body geometry','neck location and rotation','segment lengths','root translation'],
                   'claim':'angular reconstruction of measured marker frames under declared virtual-body context; no anatomical positions, full-body accuracy or force validity'}


def measurement_metadata():
    return dict(version='andy-optical-fixture-1',positions='virtual template FK, not measured joint centres',
                segment_rotations='Head/Spine3 carry optical marker-frame orientation; neck is virtual articulation',
                allowed_physical_targets=['optical head-up direction','head relative optical thorax (rotation profiles)'],
                body_context='fixed template prior, not subject-specific anthropometry')


def build(row,out):
    import numpy as np
    from motiogen.evalkit.k.reference import andy,marker_frames
    from motiogen.evalkit.k.geometry import valid_rotation
    transport=setup();folder=out/'prepared'/row['window_key'];receipt=folder/'receipt.json'
    if receipt.exists():
        result=read(receipt)
        if result['row']!=row or any(sha(p)!=h for p,h in result['hashes'].items()):raise ValueError('Andy input/reference identity changed')
        return result
    roots=read(CODE/'config/gold_k_20261115/evidence.json')['local_roots']
    times,names,points=andy(row,roots['andy'])
    frames=marker_frames(points,names,'andy');head=frames['head'][0];thorax=frames['thorax'][0]
    hvalid,tvalid=valid_rotation(head),valid_rotation(thorax);valid=hvalid&tvalid
    start,end=row['source_window_s'];mask=(times>=start)&(times<=end)
    selected=np.flatnonzero(mask)
    quality=dict(full_record_frames=len(times),head_valid=int(hvalid.sum()),thorax_valid=int(tvalid.sum()),
                 paired_valid=int(valid.sum()),selected_frames=len(selected),selected_paired_valid=int(valid[mask].sum()),
                 selected_duration_s=float(times[selected[-1]]-times[selected[0]]) if len(selected)>1 else 0.,
                 selection_rule='centre <=5s within historically registered K1 optical-valid window; no current fit error selection',
                 reference_method='direct four-marker frames; no template imputation, redundant-marker recovery or temporal filling')
    folder.mkdir(parents=True,exist_ok=True);save(folder/'reference_quality.json',quality)
    if len(selected)<2 or times[selected[-1]]-times[selected[0]]<1 or not valid[mask].all():
        raise ValueError('Registered optical window lacks complete >=1s head/thorax reference; no replacement window')
    t=times[mask];h=head[mask];c=thorax[mask]
    reference=folder/'optical_reference.npz'
    np.savez_compressed(reference,times=t,head=h,thorax=c,
                        head_markers=frames['head'][1][mask],thorax_markers=frames['thorax'][1][mask],
                        all_times=times,all_head_valid=hvalid,all_thorax_valid=tvalid)
    synthetic,provenance=fixture(t,h,c)
    np.savez_compressed(folder/'virtual_body.npz',positions=synthetic.positions,rotations=synthetic.rotations,
                        names=synthetic.names,times=t)
    save(folder/'template_prior.json',provenance)
    records={}
    for method in METHODS:
        target=out/'runtime_inputs'/row['window_key']/method
        path,d,error,consistency=transport.export_input(synthetic,method,target)
        provpath=(path.parent if method=='primal' else path)/'k_provenance.json'
        provenance_d=read(provpath);provenance_d['measurement_scope']=measurement_metadata();save(provpath,provenance_d)
        # The old observed_nodes flags denote nodes available in the input rig,
        # not independent physical measurements. Explicitly qualify their source.
        rigpath=provpath.parent/'input_rig.json'
        if rigpath.exists():
            rig=read(rigpath);rig['measurement_scope']=measurement_metadata();save(rigpath,rig)
            provenance_d['rig_sha256']=sha(rigpath);save(provpath,provenance_d)
        feet=[d.names.index(s+'_foot') for s in ('left','right')]
        records[method]=dict(input=str(path),mass_kg=75.,floor_m=float(min(np.percentile(d.positions[:,j,1],5) for j in feet)),
                             scaffold_degradation_diagnostic=error,codec_consistency=consistency,
                             scope='Positions describe the declared virtual body, not Andy joint centres')
    hashes={str(f):sha(f) for f in folder.iterdir() if f.is_file()}
    hashes.update({str(f):sha(f) for f in (out/'runtime_inputs'/row['window_key']).rglob('*') if f.is_file()})
    value=dict(row=row,reference=str(reference),quality=quality,records=records,hashes=hashes,reference_scope=measurement_metadata())
    save(receipt,value);return value


def directional_scores(times,r,d,f,tolerance):
    import numpy as np
    from motiogen.evalkit.k.scoring import summary,unit
    from motiogen.degradation.metrics import angle
    observed=np.isfinite(r).all(-1)&np.isfinite(d).all(-1)
    frames={k:np.where(observed[:,None],v,np.nan) for k,v in [('R',r),('D',d),('F',f)]}
    comparisons={}
    for label,x,y in [('D-R','R','D'),('F-D','D','F'),('F-R','R','F')]:
        a,b=frames[x],frames[y];error=angle(a,b)
        sa,sb=angle(a[:-1],a[1:]),angle(b[:-1],b[1:])
        rate=np.linalg.norm((np.diff(unit(b),axis=0)-np.diff(unit(a),axis=0))/np.diff(times)[:,None],axis=-1)*180/np.pi
        strata={}
        for name,mask in [('full',np.ones(len(times),bool)),('continuous',~tolerance),('source_jump_tolerance',tolerance)]:
            steps=mask[:-1]&mask[1:]
            strata[name]=dict(directions_deg={'head_superior_axis':summary(error[mask],[5,10,20])},joint_angles_deg={},
                             positions_cm={k:{} for k in ['head_neck','shoulders','hands','knees','feet']},
                             temporal={'head_superior_axis':dict(reference_step_deg=summary(sa[steps]),output_step_deg=summary(sb[steps]),
                                 direction_rate_vector_error_deg_s=summary(rate[steps]))})
        comparisons[label]=strata
    return dict(comparisons=comparisons,clock={'frames':len(times)},fixed_observation_support=True)


def score(job):
    import numpy as np
    from scipy.spatial.transform import Rotation
    from motiogen.evalkit.k.motion import Motion,load
    from motiogen.evalkit.k.head_neck import interpolate_rotations
    from motiogen.evalkit.k.head_scoring import compare,distance
    from motiogen.evalkit.k.strict_acceptance import assess
    from motiogen.evalkit.k.scoring import summary,unit
    import evalkit_k4_ablation_score as helpers
    with np.load(job['reference']) as z:
        rt=z['times'];rh=z['head'];rc=z['thorax'];hm=z['head_markers'];cm=z['thorax_markers'][:,:4]
    d=load(job['input'],job['method'],config=CODE/'config/evalkit_k4_20261117/input.json')
    with np.load(job['fk']) as z:fdata={k:z[k] for k in z.files}
    f=Motion(fdata['positions'],list(fdata['names']),fdata['times'],'fitted_FK')
    # Reference for the frozen relative-rotation scorer: only real optical frames.
    ref=Motion(np.zeros((len(rt),2,3)),['Spine3','Head'],rt,'optical_reference',np.stack([rc,rh],1),['Spine3','Head'])
    angles=np.maximum(distance(rh[:-1],rh[1:]),distance(rc[:-1],rc[1:]))
    shifts=np.maximum(np.linalg.norm(np.diff(hm.mean(1),axis=0),axis=-1),np.linalg.norm(np.diff(cm.mean(1),axis=0),axis=-1))
    events=[{'start_s':float(rt[i]),'end_s':float(rt[i+1])} for i in np.flatnonzero((angles>45)|(shifts>.25))]
    outputs={};policy=read(CODE/'config/evalkit_k4_20261117/strict_v1/acceptance.json')
    for clock,tt in [('D_clock',d.times),('R_clock',rt)]:
        truth=interpolate_rotations(rt,rh,tt)[:,:,1]
        dd,ff=helpers.at(d,tt),helpers.at(f,tt)
        def direction(m):
            v=m.positions[:,m.names.index('head')]-m.positions[:,m.names.index('neck')]
            return unit(v)
        mask=helpers.jump_mask(tt,d,events)
        block=directional_scores(tt,truth,direction(dd),direction(ff),mask)
        block.update(compare(ref,d,fdata,tt,mask))
        outputs[clock]=dict(score=block,acceptance=assess(block,None,policy))
    result=dict(scope=measurement_metadata(),source_jump_events=events,clocks=outputs,
                positions='Not physically scored: virtual template joint centres',
                reference=job['reference'],input=job['input'],fitted_fk=job['fk'])
    save(job['output'],helpers.base.serializable(result))


def register(a):
    if a.run.exists():raise ValueError('Use a new Andy run directory')
    gate=ROOT/'develop_logs/test_runs/20261120_k4_andy/development_verification.json'
    verified=read(gate)
    if verified['test_records_used']!=0 or len(verified['known_orientation'])!=8 or len(verified['actual_development_fit'])!=4:
        raise ValueError('Development adapter checks are incomplete')
    if verified.get('adapter_sha256')!=sha(Path(__file__)):
        raise ValueError('Verify the final adapter before test registration')
    old=ROOT/'config/evalkit_k_validation_20261116/final_usage.json'
    history=Path(read(old)['format_output'])/'records.jsonl'
    windows={r['id']:r['profiles']['window_s'] for r in map(json.loads,history.read_text().splitlines()) if r['dataset']=='andy'}
    trials=[r for r in map(json.loads,(ROOT/'config/gold_k_20261115/trials.jsonl').read_text().splitlines()) if r['partition']=='final_holdout' and r['dataset']=='andy' and r['status']=='included']
    trials=sorted(trials,key=lambda r:r['id']);assert len(trials)==75
    records=[]
    for i,r in enumerate(trials):
        left,right=windows[r['id']];width=min(5.,right-left);mid=(left+right)/2
        records.append(dict(r,window_key=f'A{i:03}',historical_window_s=[left,right],source_window_s=[mid-width/2,mid+width/2]))
    a.run.mkdir(parents=True);control=a.run/'control';control.mkdir()
    shutil.copyfile(Path(__file__),control/Path(__file__).name)
    shutil.copyfile(ROOT/'python/scripts/evalkit_k4_slot_lease.py',control/'evalkit_k4_slot_lease.py')
    tv=read(TV/'protocol.json')
    identity=dict(tv['numerical_identity']);identity.update({str(f):sha(f) for f in control.glob('*.py')})
    identity.update({str(history):sha(history),str(old):sha(old),str(gate):sha(gate)})
    p=dict(version='andy-head-optical-k4-1',run=str(a.run),records=records,expected_routes=600,workers=1,
           code_root=str(CODE),fit_config=tv['fit_config'],fit_commit=tv['fit_commit'],methods=METHODS,
           identity=identity,protected_at_start=tv['protected_at_test_start'],known_historical_differences=tv['known_historical_differences'],
           measured_targets=measurement_metadata(),historical_window_source=str(history),
           window_policy='central <=5s within previously registered complete optical interval; no new window replacement',
           optical_reference='direct complete four-marker head/chest frames; no imputation; full-trial visibility reported separately',
           scoring='Measured head-up direction for eight formats; measured head relative thorax for six rotational formats; no physical position or whole-body acceptance',
           selector='Frozen D-only early budget on the declared virtual-body input; existing priors remain explicit',
           resource='one guest numerical worker reserved from running TV; total active numerical slots <=14',
           html=str(ROOT/'docs/20261120_k4_andy_test.html'),minimum_free_GiB=80)
    save(a.run/'protocol.json',p);save(a.run/'usage.json',dict(status='registered',prior_K1_use_retained=True,numerical_started=False,expected_records=75,expected_routes=600))
    print(a.run/'protocol.json')


def report(p,results,state):
    rows=list(results.values());done=[r for r in rows if r['status']=='completed']
    d=dict(state=state,expected=600,evaluated=len(done),passed=sum(r['passed'] for r in done),
           metric_failed=sum(not r['passed'] for r in done),execution_failed=sum(r['status']=='failed' for r in rows),
           reference_insufficient=sum(r['status']=='reference_insufficient' for r in rows),pending=600-len(rows),
           updated_unix_s=time.time(),rows=rows)
    d['per_format']={m:dict(expected=75,evaluated=sum(r['method']==m for r in done),passed=sum(r['method']==m and r['passed'] for r in done)) for m in METHODS}
    d['per_selector']={s:dict(evaluated=sum(r.get('selector_version','whole_body_v1')==s for r in done),passed=sum(r.get('selector_version','whole_body_v1')==s and r['passed'] for r in done)) for s in sorted({r.get('selector_version','whole_body_v1') for r in done})}
    d['per_person']={s:dict(expected=120,evaluated=sum(r['subject']==s for r in done),passed=sum(r['subject']==s and r['passed'] for r in done)) for s in sorted({r['subject'] for r in p['records']})}
    save(Path(p['run'])/'progress.json',d)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Andy头胸test</title><style>body{font:16px/1.7 system-ui;background:#f2f5f7;color:#193143}main{max-width:1000px;margin:auto;padding:28px}section{padding:22px;background:white;margin:20px 0;border-radius:10px}td,th{padding:8px;border-bottom:1px solid #dce3e7;text-align:left}table{width:100%;border-collapse:collapse}</style><main><h1>Andy：光学头胸参考 × 冻结K4</h1><p>75条、5人、八格式，共600路；与电视4256路全身测试分开报告。</p>'
    page+=f'<section><h2>{len(done)}/600已验 · {d["passed"]}通过</h2><p>{d["metric_failed"]}指标未通过，{d["execution_failed"]}执行失败，{d["reference_insufficient"]}参考不足。状态：{state}。</p><p>更新时间：{time.strftime("%Y-%m-%d %H:%M:%S")}；在线每30秒刷新。待运行项保留完整分母。</p></section>'
    if p.get('resume_without_tree_rehash'):
        page+='<section><h2>头胸D-only加速续跑</h2><p>当前规则仅以D支持的头部方向、头胸相对旋转、覆盖及原生/导出时序停止；模板位置、肩和骨盆仅作诊断。阈值及0.8余量保持原值，数值求解器和候选顺序保持原值。</p><p>切换前已完成结果保留旧规则，切换后结果使用独立新版本；600路总表是进度，不是同一规则的完整验证。</p><pre>'+html.escape(json.dumps(d['per_selector'],ensure_ascii=False,indent=2))+'</pre></section>'
    page+='<section><h2>参考与先验</h2><p>实测：原始头部四标记、胸部四标记构造的段坐标架。评价八格式的头部上方向，以及六旋转格式的头相对胸旋转；两位置格式的轴向旋转不适用。</p><p>模板先验：全身位置、体型、骨长、颈点、附着和根位置。虚拟头颈连杆将实测头部方向传入位置格式，其关节位置不是实测人体关节中心。模板身体只提供输入上下文，任何身体位置拟合不计入实测通过率。</p><p>窗口：K1已登记光学有效区间的中央最多5秒。整条试次的参考可见率另存reference_quality，不能用窗口内完整覆盖代替全试次覆盖。直接标记参考，无缺测补齐；若旧窗口不满足参考要求，保留不足状态，不另选好窗口。</p></section>'
    page+='<section><h2>按格式</h2><table><tr><th>格式</th><th>已验</th><th>通过</th></tr>'+''.join(f'<tr><td>{m}</td><td>{v["evaluated"]}/75</td><td>{v["passed"]}</td></tr>' for m,v in d['per_format'].items())+'</table></section>'
    page+='<section><h2>逐路结果</h2><p>每项展开显示选中候选的独立评分。角度均为度；完整窗口和连续区间分别保留，时序门槛采用冻结规则。</p>'
    for r in sorted(rows,key=lambda r:(r['window'],r['method'])):
        status=('通过' if r['passed'] else '指标未通过') if r['status']=='completed' else r['status']
        title=f"{r['window']} · {r['subject']} · {r['method']} · {status}"
        detail={k:r[k] for k in ('selector_version','origin','selected','visited','visited_passed','quality','optical_metrics','failed_metrics','error') if k in r}
        page+=f'<details><summary>{html.escape(title)}</summary><pre style="overflow:auto;font-size:12px">{html.escape(json.dumps(detail,ensure_ascii=False,indent=2))}</pre></details>'
    page+='</section>'
    page+='<section><h2>实验边界</h2><p>D-only提前接受，选中后以独立光学R评分。所有实际访问且成功输出的候选单独留证；未访问候选未知。D−R、F−D、F−R，原生/导出与D/R时钟分别保存。5°角度、全观测覆盖和原冻结时序门槛；本测试不授予5cm物理位置或全身精度资格。来源跳变分层依据原始光学帧与标记质心。</p><p>模板条件下头胸角度重构的证据不等于生理解剖角验证；标记软组织与坐标架定义误差仍属于参考自身限制。K1使用历史保留，本次为K4重新评估。GPU继续冻结。</p></section></main><script>if(location.protocol!=="file:")setTimeout(()=>location.reload(),30000)</script></html>'
    target=Path(p['html']);tmp=target.with_suffix('.tmp');tmp.write_text(page);tmp.replace(target)


def run(a):
    import fcntl
    p=read(a.run/'protocol.json');lock=(a.run/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not p.get('resume_without_tree_rehash'):
        for f,h in {**p['identity'],**p['protected_at_start']}.items():
            if sha(f)!=h:raise ValueError('Registered file changed: '+f)
    # The external lease is mandatory while TV is still active.
    lease=read(a.lease)
    for _ in range(30):
        if lease.get('state')=='guest_running':break
        time.sleep(.1);lease=read(a.lease)
    if lease.get('state')!='guest_running' or lease.get('guest_pid')!=os.getpid():raise ValueError('Missing live one-slot reservation')
    script=a.run/'control/evalkit_k4_andy.py';results={f.parent.name:read(f) for f in (a.run/'tasks').glob('*/result.json')}
    save(a.run/'usage.json',dict(status='running',numerical_started=True,prior_K1_use_retained=True,started_unix_s=time.time(),expected_routes=600))
    def call(file,args,log):
        if shutil.disk_usage(a.run).free<80*1024**3:raise RuntimeError('80 GiB disk reserve')
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        while available<8*1024**3:
            report(p,results,'waiting_for_8GiB_available_memory');time.sleep(30)
            available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        with log.open('a') as stream:subprocess.run([sys.executable,str(file),*map(str,args)],stdout=stream,stderr=subprocess.STDOUT,env=ENV,check=True)
    report(p,results,'running')
    for row in p['records']:
        w=row['window_key'];prep=a.run/'preparation'/w;prep.mkdir(parents=True,exist_ok=True)
        if all(w+'_'+method in results for method in METHODS):continue
        request=prep/'request.json';save(request,dict(row=row,run=str(a.run)))
        try:
            if not (a.run/'prepared'/w/'receipt.json').exists():call(script,['prepare','--request',request],prep/'prepare.log')
            prepared=read(a.run/'prepared'/w/'receipt.json')
            if not p.get('resume_without_tree_rehash') and any(sha(f)!=h for f,h in prepared['hashes'].items()):raise ValueError('Prepared reference/input changed')
        except Exception as error:
            # Signal/reference issues are separate from numerical K failures.
            quality=a.run/'prepared'/w/'reference_quality.json'
            q=read(quality) if quality.exists() else {}
            insufficient=q and (q['selected_paired_valid']<q['selected_frames'] or q['selected_frames']<2 or q.get('selected_duration_s',0)<1)
            status='reference_insufficient' if insufficient else 'failed'
            for method in METHODS:
                key=w+'_'+method;value=dict(window=w,subject=row['subject'],method=method,status=status,error=str(error),phase='reference_preparation')
                save(a.run/'tasks'/key/'result.json',value);results[key]=value
            report(p,results,'running');continue
        for method in METHODS:
            key=w+'_'+method
            if key in results:continue
            task=a.run/'tasks'/key;task.mkdir(parents=True,exist_ok=True)
            value=dict(window=w,subject=row['subject'],method=method,status='running',started_unix_s=time.time(),quality=prepared['quality'],selector_version=p.get('selection_scope','whole_body_v1'))
            save(task/'state.json',value)
            report(p,results,'running '+key)
            try:
                item=prepared['records'][method];fit=a.run/'fits'/w/method
                if fit.exists() and not (fit/'receipt.json').exists() and not p.get('fit_runner'):fit.rename(fit.with_name(method+'_interrupted_'+str(time.time_ns())))
                job=dict(input=item['input'],method=method,output=str(fit),mass_kg=item['mass_kg'],floor_m=item['floor_m'],fit_config=p['fit_config'])
                if p.get('fit_runner'):
                    job.update(registered_identity=p['registered_identity'],selector_module=p['selector_module'],reuse_fit=str(Path(p['previous_run'])/'fits'/w/method))
                save(task/'runtime_job.json',job)
                if not (fit/'receipt.json').exists():call(Path(p.get('fit_runner',CODE/'python/scripts/evalkit_k4_run.py')),['--code-root',CODE,'--job',task/'runtime_job.json'],task/'fit.log')
                selection=read(fit/'candidate_selection.json');selection_hash=sha(fit/'candidate_selection.json');save(task/'selection_before_R.json',dict(hash=selection_hash,selected=selection['selected']))
                reviewed={}
                for candidate in selection['qualities']:
                    dest=task/'visited'/candidate;dest.mkdir(parents=True,exist_ok=True);folder=fit/'candidates'/candidate
                    if (dest/'review.json').exists():reviewed[candidate]=read(dest/'review.json');continue
                    views=dest/'views'
                    if not (views/'receipt.json').exists():call(CODE/'python/scripts/evalkit_k4_native_export_views.py',['--code-root',CODE,'--fit',folder,'--input',item['input'],'--method',method,'--window',w,'--output',views],dest/'views.log')
                    accept={};score_hashes={};metrics={}
                    for view in ('native','export'):
                        output=dest/(view+'_optical_score.json');req=dest/(view+'_score_request.json')
                        if not output.exists():
                            save(req,dict(reference=prepared['reference'],input=item['input'],method=method,fk=str(views/view/w/method/'fk.npz'),output=str(output)))
                            call(script,['score','--request',req],dest/(view+'_score.log'))
                        scored=read(output);accept[view]={k:v['acceptance'] for k,v in scored['clocks'].items()};score_hashes[str(output)]=sha(output)
                        metrics[view]={clock:{label:{stratum:dict(head_up_deg=sc['comparisons'][label][stratum]['directions_deg']['head_superior_axis'],head_relative_thorax_deg=sc['head_neck_rotation_comparisons'][label][stratum]) for stratum in ('full','continuous')} for label in ('D-R','F-D','F-R')} for clock,entry in scored['clocks'].items() for sc in [entry['score']]}
                    audit=dest/'roundtrip/audit.json'
                    if not audit.exists():
                        if audit.parent.exists():audit.parent.rename(dest/('roundtrip_interrupted_'+str(time.time_ns())))
                        call(CODE/'python/scripts/evalkit_k4_export_audit.py',['--fit',folder,'--output',audit.parent],dest/'roundtrip.log')
                    failed=[dict(view=v,clock=c,**x) for v,clocks in accept.items() for c,s in clocks.items() for x in s['required']+s['temporal']+s['full_window_coverage'] if x['status'] not in ('passed','unobservable')]
                    review=dict(passed=not failed and read(audit)['passed'],failed_metrics=failed,optical_metrics=metrics,score_hashes=score_hashes,roundtrip_sha256=sha(audit),scope='Optical head-direction/relative-rotation only')
                    save(dest/'review.json',review);reviewed[candidate]=review
                assert sha(fit/'candidate_selection.json')==selection_hash
                selected=reviewed[selection['selected']]
                value.update(status='completed',passed=selected['passed'],selected=selection['selected'],failed_metrics=selected['failed_metrics'],optical_metrics=selected['optical_metrics'],visited=len(reviewed),visited_passed=sum(r['passed'] for r in reviewed.values()),runtime_failures=selection['failures'])
            except Exception as error:value.update(status='failed',error=str(error),traceback=traceback.format_exc())
            value['elapsed_s']=time.time()-value['started_unix_s'];save(task/'result.json',value);results[key]=value
            print(key,value['status'],value.get('passed'),flush=True);report(p,results,'running')
    changed=[] if p.get('resume_without_tree_rehash') else [f for f,h in {**p['identity'],**p['protected_at_start']}.items() if sha(f)!=h]
    report(p,results,'completed' if not changed else 'integrity_failed')
    save(a.run/'completion.json',dict(routes=len(results),expected=600,changed=changed,complete=not changed and len(results)==600,integrity='registered evidence retained; tree rehash explicitly skipped' if p.get('resume_without_tree_rehash') else 'hash verified'))
    save(a.run/'usage.json',dict(status='completed',prior_K1_use_retained=True,expected_routes=600,finished_routes=len(results),finished_unix_s=time.time()))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=['register','prepare','score','run'])
    ap.add_argument('--run',type=Path);ap.add_argument('--request',type=Path);ap.add_argument('--lease',type=Path)
    a=ap.parse_args()
    if a.phase=='register':register(a)
    elif a.phase=='run':run(a)
    else:
        setup();job=read(a.request)
        if a.phase=='prepare':build(job['row'],Path(job['run']))
        else:score(job)


if __name__=='__main__':main()
