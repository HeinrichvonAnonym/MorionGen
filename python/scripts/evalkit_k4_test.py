"""Frozen K4 test orchestration: one D-only budget decision, R-only evaluation.

Each thread owns one record and starts at most one numerical child at a time.
No unused candidates are fitted or scored. Existing dev/holdout data are read-only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
METHODS=['motionlcm','omnicontrol','ardy','chois','closd','dart','kimodo','primal']
ENV={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)


def register(a):
    out=a.run.resolve()
    if out.exists():raise ValueError('Use a new test output directory')
    parent=read(B/'full_regression_20261118_budget_v3/protocol.json')
    code=Path(parent['fit_code_root'])
    for path,h in parent['implementation_sha256'].items():
        if sha(path)!=h:raise ValueError('Frozen CPU implementation changed: '+path)
    if sha(parent['fit_config'])!=parent['fit_config_sha256']:raise ValueError('Frozen selector changed')
    registry=ROOT/'config/gold_k_20261115/trials.jsonl'
    rows=[json.loads(s) for s in registry.read_text().splitlines()]
    trials=sorted([r for r in rows if r['partition']=='final_holdout' and r['status']=='included' and r['dataset']=='collaborate'],key=lambda r:r['id'])
    assert len(trials)==532
    windows=[]
    for i,row in enumerate(trials):
        duration=(row['frames']-1)*row['dt']
        if duration<1:raise ValueError('Registered test record shorter than one second')
        start=max(0.,(duration-5.)/2)
        windows.append(dict(row,window_key=f'T{i:03}',source_window_s=[start,min(duration,start+5.)]))
    out.mkdir(parents=True);control=out/'control';control.mkdir()
    for name in ('evalkit_k4_test.py','evalkit_k4_test_evaluate.py'):
        shutil.copyfile(ROOT/'python/scripts'/name,control/name)
    # New baseline explicitly retains the known pre-existing plan mismatch.
    old_manifest=ROOT/'config/evalkit_k4_20261117/preservation.json'
    original=read(old_manifest)['files'];protected={str(ROOT/n):sha(ROOT/n) for n in original}
    known=[dict(path=n,original=h,current=protected[str(ROOT/n)]) for n,h in original.items() if protected[str(ROOT/n)]!=h]
    if [x['path'] for x in known] != ['plans/20261112_gold_anchor_and_own_method_plan.md']:
        raise ValueError('Unexpected protected-file changes: '+str(known))
    model=Path.home()/'.local/share/opensimcreator/resources/OpenSimCreator/models/RajagopalModel/RajagopalLaiUhlrich2023.osim'
    files={str(p):sha(p) for folder in ('utils','config','python/scripts') for p in (code/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
    files.update({str(p):sha(p) for p in control.glob('*.py')})
    files.update({str(model):sha(model),str(registry):sha(registry),str(old_manifest):sha(old_manifest)})
    roots=read(ROOT/'config/gold_k_20261115/evidence.json')['local_roots']
    upstream=Path(roots['andy']).parents[2]/'baselines/OmniControl/data_loaders/humanml'
    files.update({str(f):sha(f) for folder in ('common','scripts','utils') for f in (upstream/folder).glob('*.py')})
    protocol=dict(version='k4-test-budget-1',created_unix_s=time.time(),output=str(out),code_root=str(code),
        fit_commit=parent['fit_code_commit'],fit_config=parent['fit_config'],fit_config_sha256=parent['fit_config_sha256'],
        numerical_identity=files,protected_at_test_start=protected,known_historical_differences=known,
        parent_development_protocol=str(B/'full_regression_20261118_budget_v3/protocol.json'),
        prior_holdout_usage=str(ROOT/'config/evalkit_k_validation_20261116/final_usage.json'),
        window_rule='Each included TV recording: central 5 physical seconds, shorter recordings full length; original source samples retained.',
        records=windows,methods=METHODS,expected_routes=532*8,workers=a.workers,
        source_root=roots['collaborate'],minimum_free_GiB=80,numerical_threads=1,
        evaluation='Actual visited candidates only; selected output fixed before independent R scoring; unvisited unknown; no oracle/missed-selection claims.',
        test_scope='K4 frozen re-evaluation of historically used holdout; TV full-body, Andy head/thorax separate pending adapter',
        andy_manifest=[r for r in rows if r['partition']=='final_holdout' and r['dataset']=='andy' and r['status']=='included'],
        html=str(ROOT/'docs/20261120_k4_test.html'),workspace_root=str(ROOT),
        measurement_policy='Frozen strict_v1 native/export x D/R clocks. expected_full_paths and holdout strings in old scoring policy are historical metadata; new test denominator and authorization are declared here.')
    atomic(out/'protocol.json',protocol)
    atomic(out/'usage.json',dict(status='registered',prior_use_retained=True,television_records=532,expected_routes=4256,numerical_started=False,andy_status='head_thorax_adapter_pending'))
    print(out/'protocol.json')


def render(p,results,active,phase):
    rows=list(results.values());complete=[r for r in rows if r['status']=='completed']
    passed=sum(r['passed'] for r in complete)
    progress=dict(state=phase,expected=p['expected_routes'],finished=len(rows),evaluated=len(complete),passed=passed,
                  metric_failed=len(complete)-passed,execution_failed=sum(r['status']=='failed' for r in rows),
                  pending=p['expected_routes']-len(rows),active=active,updated_unix_s=time.time(),
                  visited_evaluated=sum(r.get('visited_evaluated',0) for r in rows),
                  skipped_candidates='unvisited unknown; no full-pool coverage estimate',
                  per_format={m:dict(expected=532,evaluated=sum(r['method']==m for r in complete),passed=sum(r['method']==m and r['passed'] for r in complete)) for m in p['methods']},rows=rows)
    def breakdown(field):
        groups={}
        for record in p['records']:
            group=record['group_id'] if field=='group' else ('Packaging' if '/Packaging/' in record['source']['member'] else 'Board_Connection' if '/Board_Connection/' in record['source']['member'] else 'Drilling')
            bucket=groups.setdefault(group,dict(expected=0,evaluated=0,passed=0,execution_failed=0))
            bucket['expected']+=8
            for method in p['methods']:
                r=results.get(record['window_key']+'_'+method)
                if r:
                    bucket['evaluated']+=r['status']=='completed'
                    bucket['passed']+=r.get('passed') is True
                    bucket['execution_failed']+=r['status']=='failed'
        return groups
    progress['per_group']=breakdown('group');progress['per_task']=breakdown('task')
    atomic(Path(p['output'])/'progress.json',progress)
    cells=''.join(f"<tr><td>{html.escape(m)}</td><td>{v['evaluated']}/532</td><td>{v['passed']}</td><td>{v['evaluated']-v['passed']}</td></tr>" for m,v in progress['per_format'].items())
    failures=''.join(f"<tr><td>{html.escape(r['window']+' '+r['method'])}</td><td>{html.escape(r.get('selected','—'))}</td><td>{html.escape(r.get('error') or '; '.join(sorted({x['metric'] for x in r.get('failed_metrics',[])})))}</td></tr>" for r in rows if r.get('passed') is False or r['status']=='failed')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K4 test：提前接受</title><style>body{{font:16px/1.7 system-ui;background:#f0f4f6;color:#193143;margin:0}}main{{max-width:1100px;margin:auto;padding:30px}}section{{padding:22px;background:white;margin:18px 0;border-radius:10px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #dce3e7}}.scroll{{overflow:auto}}h1{{font-size:30px}}.number{{font-size:26px;color:#167364}}small{{color:#607080}}</style><main><h1>K4既有留出集：D-only提前接受</h1><p>532条电视轨迹 × 八格式；每条中央5秒，较短用全段。固定CPU版本dfe8239，选择仅读取D，选中后相对R独立验收。</p><section><p class="number">{len(complete)}/4256 已验 · {passed} 通过 · {len(complete)-passed} 指标未通过</p><p>{progress['execution_failed']} 执行失败 · {progress['pending']} 待完成 · {active} 条记录正在处理</p><p>已独立验收的实际访问候选：{progress['visited_evaluated']}。未访问候选保持未知，本实验不报告候选并集或漏选率。</p><small>状态：{phase}；更新：{time.strftime('%Y-%m-%d %H:%M:%S')}。在线每30秒刷新；本文件亦可离线阅读当前快照。</small></section><section><h2>八格式</h2><table><tr><th>格式</th><th>已验</th><th>通过</th><th>指标未通过</th></tr>{cells}</table></section><section><h2>尚未通过项</h2><div class="scroll"><table><tr><th>路线</th><th>选中候选</th><th>超标指标／执行原因</th></tr>{failures}</table></div></section><section><h2>解释边界</h2><p>所有候选及选择规则沿用开发冻结版本。D−R、F−D、F−R、原始／统一骨长、原生／导出和D／R时钟均在独立评分文件中保存；本页计数要求四个视图及实际文件回读全部通过。全窗口保留，来源跳变按原冻结规则单列。</p><p>电视来源：板件连接322条、钻孔157条、包装53条；S01/S02/S03为353/162/17条。最终另报按人和任务汇总。名单曾用于K1评估，本次为K4冻结重新评估。Andy75条头胸测试单列，当前尚未启动，不进入4256分母。</p><p>历史保护清单存在1份plan文档哈希差异，已在本次运行前登记。原清单与旧回执保留；本次新增修改另行检查。GPU保持冻结。</p></section></main><script>if(location.protocol!=='file:')setTimeout(()=>location.reload(),30000)</script></html>'''
    target=Path(p['html']);tmp=target.with_suffix('.html.tmp');tmp.write_text(page);tmp.replace(target)
    return progress


def run(a):
    out=a.run.resolve();p=read(out/'protocol.json');code=Path(p['code_root']);control=out/'control'
    if not 1<=p['workers']<=14:raise ValueError('Invalid worker cap')
    lock=(out/'.controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for path,h in {**p['numerical_identity'],**p['protected_at_test_start']}.items():
        if sha(path)!=h:raise ValueError('Registered identity changed: '+path)
    results={};state_lock=threading.Lock()
    for result in (out/'tasks').glob('*/result.json'):
        r=read(result);results[result.parent.name]=r
    usage=read(out/'usage.json');usage.update(status='running',numerical_started=True,last_started_unix_s=time.time())
    atomic(out/'usage.json',usage)
    stop=threading.Event()
    def call(script,args,log):
        # Low-cost resource gate: only /proc metadata, no experiment rehash loop.
        while not stop.is_set():
            available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
            if available>=8*1024**3:break
            time.sleep(15)
        if stop.is_set():raise RuntimeError('New numerical batches paused')
        if shutil.disk_usage(out).free<p['minimum_free_GiB']*1024**3:
            stop.set();raise RuntimeError('Disk reserve reached: no more tasks launched')
        with log.open('a') as f:
            subprocess.run([sys.executable,str(script),*map(str,args)],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    def execute(row):
        if stop.is_set():return
        w=row['window_key'];prep_task=out/'preparation'/w;prep_task.mkdir(parents=True,exist_ok=True)
        try:
            request=prep_task/'request.json';atomic(request,dict(row=row,root=p['source_root'],output=str(out)))
            if not (out/'prepared'/w/'receipt.json').exists():
                call(control/'evalkit_k4_test_evaluate.py',['prepare','--code-root',code,'--request',request],prep_task/'prepare.log')
            prepared=read(out/'prepared'/w/'receipt.json')
            if prepared['row']!=row or any(sha(path)!=h for path,h in prepared['artifact_hashes'].items()):raise ValueError('Prepared artifact changed')
        except Exception as e:
            atomic(prep_task/'failure.json',dict(error=str(e),traceback=traceback.format_exc()))
            for m in p['methods']:
                key=w+'_'+m;value=dict(window=w,method=m,status='failed',phase='input_preparation',error=str(e))
                atomic(out/'tasks'/key/'result.json',value)
                with state_lock:results[key]=value
            return
        for m in p['methods']:
            if stop.is_set():return
            key=w+'_'+m;task=out/'tasks'/key;task.mkdir(parents=True,exist_ok=True)
            with state_lock:saved=results.get(key)
            if saved:continue
            start=time.time();result=dict(window=w,method=m,status='running',group_id=row['group_id'],recording_id=row['recording_id'],source_window_s=row['source_window_s'])
            atomic(task/'state.json',result)
            try:
                item=prepared['records'][m];fit=out/'fits'/w/m
                if fit.exists() and not (fit/'receipt.json').exists():
                    fit.rename(fit.with_name(m+'_interrupted_'+str(time.time_ns())))
                job=dict(input=item['input'],method=m,output=str(fit),mass_kg=item['mass_kg'],floor_m=item['floor_m'],fit_config=p['fit_config'])
                atomic(task/'runtime_job.json',job)
                if not (fit/'receipt.json').exists():call(code/'python/scripts/evalkit_k4_run.py',['--code-root',code,'--job',task/'runtime_job.json'],task/'fit.log')
                receipt=read(fit/'receipt.json')
                for path,h in receipt['key']['inputs'].items():
                    if sha(path)!=h:raise ValueError('Runtime D identity changed')
                selection=read(fit/'candidate_selection.json')
                atomic(task/'selection_frozen_before_R.json',dict(selection_sha256=sha(fit/'candidate_selection.json'),selected=selection['selected'],at_unix_s=time.time()))
                evaluated={}
                # Read only actually computed successful candidates. Runtime failures
                # remain separately in selection['failures']; never run extra choices.
                for name in selection['qualities']:
                    candidate=fit/'candidates'/name;dest=task/'visited'/name;dest.mkdir(parents=True,exist_ok=True)
                    review=dest/'review.json'
                    if review.exists():evaluated[name]=read(review);continue
                    views=dest/'views'
                    if not (views/'receipt.json').exists():call(code/'python/scripts/evalkit_k4_native_export_views.py',['--code-root',code,'--fit',candidate,'--input',item['input'],'--method',m,'--window',w,'--output',views],dest/'views.log')
                    accept={};scores={}
                    for view in ('native','export'):
                        scorefile=dest/(view+'_score.json');request=dest/(view+'_request.json')
                        if not Path(str(scorefile)+'.acceptance.json').exists():
                            atomic(request,dict(reference=prepared['reference'],input=item['input'],method=m,artifact={k:item[k] for k in ('source_jump_events','head_neck_relative_deg')},fk=str(views/view/w/m/'fk.npz'),output=str(scorefile)))
                            call(control/'evalkit_k4_test_evaluate.py',['score','--code-root',code,'--request',request],dest/(view+'_score.log'))
                        accept[view]=read(str(scorefile)+'.acceptance.json');scores[str(scorefile)]=sha(scorefile)
                    audit=dest/'roundtrip/audit.json'
                    if not audit.exists():
                        if audit.parent.exists():audit.parent.rename(dest/('roundtrip_interrupted_'+str(time.time_ns())))
                        call(code/'python/scripts/evalkit_k4_export_audit.py',['--fit',candidate,'--output',audit.parent],dest/'roundtrip.log')
                    failures=[dict(view=v,clock=c,**x) for v,b in accept.items() for c,t in b.items() for x in t['required']+t['temporal']+t['full_window_coverage'] if x['status'] not in ('passed','unobservable')]
                    value=dict(passed=all(t['status']=='passed' for b in accept.values() for t in b.values()) and read(audit)['passed'],failed_metrics=failures,score_hashes=scores,export_audit_sha256=sha(audit),acceptance_files={v:str(dest/(v+'_score.json.acceptance.json')) for v in accept})
                    atomic(review,value);evaluated[name]=value
                if sha(fit/'candidate_selection.json')!=read(task/'selection_frozen_before_R.json')['selection_sha256']:raise ValueError('Selection changed during scoring')
                selected=evaluated[selection['selected']]
                max_dr=max(v['world_cm']['max'] or 0 for v in item['degradation']['positions'].values())
                result.update(status='completed',passed=selected['passed'],selected=selection['selected'],failed_metrics=selected['failed_metrics'],visited_evaluated=len(evaluated),visited_passed=sum(v['passed'] for v in evaluated.values()),runtime_candidate_failures=selection['failures'],early_accepted=selection['budget']['early_accepted'],input_alignment_max_cm=max_dr,input_alignment_exceeds_dev_numeric_bound=max_dr>.01,fit=str(fit))
            except Exception as e:
                result.update(status='failed',error=str(e),traceback=traceback.format_exc())
            result['elapsed_s']=time.time()-start
            atomic(task/'result.json',result)
            with state_lock:results[key]=result
            print(key,result['status'],result.get('passed'),flush=True)
    render(p,results,0,'starting')
    with ThreadPoolExecutor(max_workers=p['workers']) as pool:
        pending={pool.submit(execute,r) for r in p['records'] if any(r['window_key']+'_'+m not in results for m in p['methods'])}
        while pending:
            done,pending=wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for f in done:f.result()
            with state_lock:snapshot=dict(results)
            render(p,snapshot,min(len(pending),p['workers']),'running' if not stop.is_set() else 'resource_paused')
    changed=[n for n,h in p['protected_at_test_start'].items() if sha(n)!=h]
    numerical_changed=[n for n,h in p['numerical_identity'].items() if sha(n)!=h]
    final=render(p,results,0,'completed' if len(results)==p['expected_routes'] and not changed and not numerical_changed else 'incomplete')
    atomic(out/'integrity_after.json',dict(protected_changed=changed,numerical_changed=numerical_changed,known_historical_differences=p['known_historical_differences']))
    usage.update(status=final['state'],finished_routes=len(results),completed_unix_s=time.time());atomic(out/'usage.json',usage)
    if final['state']=='completed':atomic(out/'completion.json',{k:v for k,v in final.items() if k!='rows'})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['register','run'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=14)
    a=parser.parse_args()
    if a.phase=='register':register(a)
    else:run(a)


if __name__=='__main__':main()
