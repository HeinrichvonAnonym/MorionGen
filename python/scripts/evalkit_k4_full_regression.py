"""Resumeable frozen 656-path regression; D-only fitting, independent R scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def execution_workers(protocol, protocol_hash, original_identity, amendment):
    """A scheduling amendment cannot replace any scientific protocol field."""
    if amendment is None:
        if protocol.get('version')=='k4-full-frozen-regression-2':
            budget=protocol.get('resource_budget',{})
            if (set(budget)!={'max_workers','numerical_threads','minimum_free_GiB','authorization'}
                    or not isinstance(budget['max_workers'],int) or not 1<=budget['max_workers']<=8
                    or budget['numerical_threads']!=1 or budget['minimum_free_GiB']<80
                    or not budget['authorization']):
                raise ValueError('New protocol must declare the authorized resource budget')
            if not isinstance(protocol['workers'],int) or not 1<=protocol['workers']<=budget['max_workers']:
                raise ValueError('Worker count exceeds declared budget')
        elif not 1<=protocol['workers']<=3:raise ValueError('Original protocol allows at most three workers')
        return protocol['workers']
    allowed={'version','protocol_sha256','original_runner_sha256','output','max_workers','authorization'}
    expanded=amendment.get('version')=='k4-execution-amendment-2'
    if expanded:allowed|={'total_worker_limit','reserved_workers'}
    if set(amendment)!=allowed or amendment['version'] not in ('k4-execution-amendment-1','k4-execution-amendment-2'):
        raise ValueError('Invalid execution-only amendment')
    if not original_identity or amendment['protocol_sha256']!=protocol_hash or original_identity['protocol_sha256']!=protocol_hash:
        raise ValueError('Execution amendment protocol mismatch')
    if amendment['original_runner_sha256']!=original_identity['runner_sha256'] or amendment['output']!=protocol['output']:
        raise ValueError('Execution amendment run identity mismatch')
    limit=8
    if expanded:
        total=amendment['total_worker_limit'];reserved=amendment['reserved_workers']
        if (type(total) is not int or not 1<=total<=12 or type(reserved) is not int
                or not 0<=reserved<total or not amendment['authorization']):
            raise ValueError('Explicit total budget and reserved workers required; maximum twelve')
        limit=total-reserved
    if type(amendment['max_workers']) is not int or not 1<=amendment['max_workers']<=limit:
        raise ValueError('Worker count exceeds authorized execution budget')
    return amendment['max_workers']


def validation_batches(protocol):
    """Scheduling subsets preserve the registered full denominator and method."""
    stages=protocol.get('validation_stages')
    if stages is None:return None
    if protocol.get('version')!='k4-full-frozen-regression-2' or protocol.get('preflight_paths'):
        raise ValueError('Staged validation requires explicit v2 without legacy preflight paths')
    if not isinstance(stages,list) or not stages:raise ValueError('Declare validation stages')
    records={r['window']+'_'+r['method']:r for r in protocol['records']}
    used=set();names=set();batches=[]
    for stage in stages:
        if set(stage)!={'name','paths'} or not stage['name'] or stage['name'] in names:
            raise ValueError('Invalid or repeated validation stage')
        paths=stage['paths']
        if (not isinstance(paths,list) or not paths or len(paths)!=len(set(paths))
                or set(paths)-set(records) or set(paths)&used):
            raise ValueError('Validation paths must be unique registered development records')
        batches.append((stage['name'],[records[k] for k in paths],True))
        used.update(paths);names.add(stage['name'])
    if 'full_remaining' in names:raise ValueError('Reserved stage name')
    remaining=[r for k,r in records.items() if k not in used]
    if remaining:batches.append(('full_remaining',remaining,False))
    return batches


def acceptance_readiness(protocol,policy_sha256):
    """Independent development acceptance gates scheduling, never fitting inputs."""
    waiting=[];evidence={}
    for prerequisite in protocol.get('acceptance_prerequisites',[]):
        run=Path(prerequisite['run'])
        files={name:run/(name+'.json') for name in ('identity','completion','review')}
        if any(not path.exists() for path in files.values()):
            waiting.append(str(run));continue
        try:
            contents={name:path.read_bytes() for name,path in files.items()}
            identity,completion,review=[json.loads(contents[name]) for name in files]
        except (FileNotFoundError,json.JSONDecodeError):
            waiting.append(str(run));continue
        if identity['code_commit']!=prerequisite['code_commit']:
            raise ValueError('Prerequisite implementation changed: '+str(run))
        expected=prerequisite['expected']
        if review['expected']!=expected or completion['expected']!=expected:
            raise ValueError('Prerequisite denominator changed: '+str(run))
        if review['completed']+review['execution_failed']<expected:
            waiting.append(str(run));continue
        if review['execution_failed'] or (prerequisite['require_all_passed'] and review['passed']!=expected):
            raise RuntimeError('Prerequisite acceptance failed; preserve results and diagnose: '+str(run))
        if review['policy_sha256']!=policy_sha256:
            raise ValueError('Prerequisite acceptance policy changed: '+str(run))
        evidence[str(run)]={'expected':expected,'passed':review['passed'],
                            'require_all_passed':prerequisite['require_all_passed'],
                            'sha256':{name:hashlib.sha256(data).hexdigest() for name,data in contents.items()}}
    return waiting,evidence


def archive_execution_failure(out,key):
    """Explicit retry preserves all failed artifacts; metric failures never retry."""
    task=out/'tasks'/key;result=task/'result.json'
    if result.exists():
        if json.loads(result.read_text()).get('status')!='failed':return False
    elif not (task/'state.json').exists() and not (task/'runtime_job.json').exists():return False
    destination=out/'attempt_history'/key/str(time.time_ns());destination.mkdir(parents=True)
    task.rename(destination/'task')
    window,method=key.split('_',1);fit=out/'fits'/window/method
    if fit.exists():fit.rename(destination/'fit')
    for condition in ('native','export'):
        score=out/'scores'/condition/(key+'.json')
        if score.exists():score.rename(destination/(condition+'_score.json'))
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--execution-config',type=Path)
    parser.add_argument('--retry-execution-failures',action='store_true')
    a=parser.parse_args();p=json.loads(a.protocol.read_text())
    code=Path(p['fit_code_root']);control=Path(p['control_code_root']);out=Path(p['output']);base=Path(p['baseline_root'])
    records=p['records'];batches=validation_batches(p)
    if len(records)!=656 or len({(r['window'],r['method']) for r in records})!=656:
        raise ValueError('Require all 656 unique registered paths')
    windows={r['window'] for r in records};methods={r['method'] for r in records}
    if len(windows)!=82 or len(methods)!=8 or {(r['window'],r['method']) for r in records}!={(w,m) for w in windows for m in methods}:
        raise ValueError('Require the complete 82 by 8 Cartesian product')
    for path,expected in p['implementation_sha256'].items():
        if sha(path)!=expected:raise ValueError('Frozen implementation changed: '+path)
    config=Path(p['fit_config']);settings=json.loads(config.read_text())
    if sha(config)!=p['fit_config_sha256']:raise ValueError('Frozen configuration changed')
    out.mkdir(parents=True,exist_ok=True)
    controller_lock=(out/'.controller.lock').open('a')
    fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'protocol_sha256':sha(a.protocol),'runner_sha256':sha(__file__)}
    previous=json.loads((out/'identity.json').read_text()) if (out/'identity.json').exists() else None
    amendment=json.loads(a.execution_config.read_text()) if a.execution_config else None
    workers=execution_workers(p,identity['protocol_sha256'],previous,amendment)
    interrupted=[str(task.parent) for task in (out/'tasks').glob('*/state.json') if not (task.parent/'result.json').exists()]
    if interrupted and not a.retry_execution_failures:
        raise ValueError('Interrupted paths require explicit --retry-execution-failures; partial artifacts stay intact: '+str(interrupted))
    if previous and previous!=identity and amendment is None:
        raise ValueError('Full regression identity changed')
    if previous is None:atomic(out/'identity.json',identity)
    if amendment is not None:
        epochs=out/'execution_epochs';epochs.mkdir(exist_ok=True)
        atomic(epochs/(str(time.time_ns())+'.json'),dict(original_identity=previous,execution_identity=identity,
            effective_workers=workers,amendment=amendment,amendment_sha256=sha(a.execution_config)))
    if not (out/'protocol.json').exists():shutil.copyfile(a.protocol,out/'protocol.json')
    preservation=json.loads(Path(p['preservation_manifest']).read_text());workspace=Path(p['workspace_root'])
    changed=[name for name,value in preservation['files'].items()
             if not (workspace/name).exists() or sha(workspace/name)!=value]
    if changed:raise ValueError('Protected files changed: '+str(changed))
    atomic(out/'preservation_before.json',{'checked':len(preservation['files']),'changed':changed,'new_holdout_uses':0})
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
             PYTHONPATH=str(code/'utils')+':'+str(code/'utils/motiogen'))
    expected_impl={str(path.relative_to(code/'utils/motiogen/evalkit/k')):sha(path)
                   for path in sorted((code/'utils/motiogen/evalkit/k').rglob('*.py'))}
    def valid_fit(folder,job):
        receipt=json.loads((folder/'receipt.json').read_text());key=receipt['key']
        common=json.loads((base/'common_v1'/job['window']/job['method']/'receipt.json').read_text())['key']
        expected_config=dict(settings,method=job['method'])
        if (key['config']!=expected_config or key['implementation']!=expected_impl
                or key['inputs']!=common['inputs'] or key['model_sha256']!=common['model_sha256']
                or key['mass_kg']!=job['mass_kg'] or key['floor_m']!=job['floor_m']):
            raise ValueError('Cached fit differs from frozen input/config/model/implementation')
        for path,value in key['inputs'].items():
            if sha(path)!=value:raise ValueError('Input content changed: '+path)
        return receipt
    def execute(row):
        w,m=row['window'],row['method'];task=out/'tasks'/f'{w}_{m}'
        if a.retry_execution_failures:archive_execution_failure(out,f'{w}_{m}')
        task.mkdir(parents=True,exist_ok=True)
        result_file=task/'result.json'
        if result_file.exists():
            saved=json.loads(result_file.read_text())
            if saved.get('status') in ('completed','failed'):return saved
        started=time.time();result={'window':w,'method':m,'status':'running','started_unix_s':started}
        atomic(task/'state.json',result)
        def call(script,args,label):
            with (task/(label+'.log')).open('w') as log:
                subprocess.run([sys.executable,str(script),*map(str,args)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        try:
            if shutil.disk_usage(out).free<80*1024**3:raise RuntimeError('80 GiB reserve reached')
            path=Path(row['job_file'])
            if sha(path)!=row['job_sha256']:raise ValueError('Registered job changed')
            source=json.loads(path.read_text());job={k:source[k] for k in ('input','method','mass_kg','floor_m')}
            checked=dict(job,window=w)
            fit=None
            for previous in p['reuse_fit_roots']:
                candidate=Path(previous)/w/m
                if (candidate/'receipt.json').exists():
                    valid_fit(candidate,checked);fit=candidate;break
            if fit is None:
                fit=out/'fits'/w/m
                job.update(output=str(fit),fit_config=str(config))
                atomic(task/'runtime_job.json',job)
                if not (fit/'receipt.json').exists():
                    call(code/'python/scripts/evalkit_k4_run.py',['--code-root',code,'--job',task/'runtime_job.json'],'fit')
                valid_fit(fit,checked);result['reused']=False
            else:result['reused']=True
            result['fit']=str(fit)
            views=task/'views'
            if not (views/'receipt.json').exists():
                call(control/'python/scripts/evalkit_k4_native_export_views.py',
                     ['--code-root',code,'--fit',fit,'--input',source['input'],'--method',m,'--window',w,'--output',views],'views')
            for condition in ('native','export'):
                score=out/'scores'/condition/f'{w}_{m}.json'
                if not score.exists():
                    call(code/'python/scripts/evalkit_k4_ablation_score.py',
                         ['--fits',views/condition,'--baseline',base/'common_v1','--output',score.parent,'--window',w,'--method',m,'--strict'],condition+'_score')
            audit=task/'actual_export/audit.json'
            if not audit.exists():
                target=audit.parent
                if target.exists():target.rename(task/('incomplete_export_audit_'+str(int(time.time()))))
                call(code/'python/scripts/evalkit_k4_export_audit.py',['--fit',fit,'--output',target],'export_audit')
            call(control/'python/scripts/evalkit_k4_full_report.py',
                 ['--run',out,'--record',w+'_'+m,'--code-root',code],'record_summary')
            summary=json.loads((task/'summary.json').read_text())
            result.update(status='completed',passed=summary['passed'],selected=summary['selected'],
                          failed_metrics=summary['failed_metrics'],summary_file=str(task/'summary.json'))
        except Exception as error:
            import traceback
            result.update(status='failed',error=str(error),traceback=traceback.format_exc(),passed=False)
        result['elapsed_s']=time.time()-started
        atomic(result_file,result);return result
    results=[]
    def report():
        subprocess.run([sys.executable,str(control/'python/scripts/evalkit_k4_full_report.py'),
                        '--run',str(out),'--code-root',str(code)],check=True,env=env)
    def save_progress(state,stage=None):
        progress={'state':state,'stage':stage,'expected':656,'finished':len(results),
                  'completed':sum(r['status']=='completed' for r in results),
                  'failed':sum(r['status']=='failed' for r in results),
                  'passed':sum(r.get('passed',False) for r in results),'results':results}
        atomic(out/'progress.json',progress);return progress
    report()
    wait_for=p.get('wait_for',[])
    if isinstance(wait_for,str):wait_for=[wait_for]
    policy_sha256=sha(code/'config/evalkit_k4_20261117/strict_v1/acceptance.json')
    while True:
        waiting=[path for path in wait_for if not Path(path).exists()]
        pending,evidence=acceptance_readiness(p,policy_sha256);waiting.extend(pending)
        if not waiting:
            atomic(out/'readiness.json',{'prerequisites':evidence,'scope':'Development validation gate; no R fields passed to fitting children'})
            break
        progress=save_progress('waiting_for_registered_runs');progress['waiting_for']=waiting
        atomic(out/'progress.json',progress);time.sleep(10)
    if batches is not None:
        for stage,rows,gate in batches:
            stage_results=[];save_progress('running',stage)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for future in as_completed([pool.submit(execute,row) for row in rows]):
                    item=future.result();results.append(item);stage_results.append(item)
                    progress=save_progress('running',stage)
                    print(stage,item['window'],item['method'],item['status'],item.get('passed'),flush=True)
                    if len(stage_results)%4==0:report()
            stage_record={'stage':stage,'expected':len(rows),'results':stage_results,
                          'require_all_passed':gate,'passed':all(r['status']=='completed' and r['passed'] for r in stage_results)}
            (out/'stages').mkdir(exist_ok=True);atomic(out/'stages'/(stage+'.json'),stage_record);report()
            if gate and not stage_record['passed']:
                atomic(out/'stage_failure.json',stage_record);save_progress('validation_stage_failed',stage)
                raise RuntimeError('Development stage failed; preserve this version and diagnose before later stages')
    else:
        preflight=set(p.get('preflight_paths',[]))
        for row in records:
            if row['window']+'_'+row['method'] not in preflight:continue
            result=execute(row);results.append(result)
            if result['status']!='completed' or not result['passed']:
                atomic(out/'preflight_failure.json',result)
                raise RuntimeError('Bulk scoring preflight failed; no new full-regression fits started')
        atomic(out/'preflight.json',{'expected':len(preflight),'results':results});report()
        remaining=[r for r in records if r['window']+'_'+r['method'] not in preflight]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for future in as_completed([pool.submit(execute,row) for row in remaining]):
                item=future.result();results.append(item);progress=save_progress('running')
                print(item['window'],item['method'],item['status'],item.get('passed'),flush=True)
                if len(results)%16==0:report()
    progress=save_progress('completed')
    changed=[name for name,value in preservation['files'].items()
             if not (workspace/name).exists() or sha(workspace/name)!=value]
    atomic(out/'preservation_after.json',{'checked':len(preservation['files']),'changed':changed,'new_holdout_uses':0})
    atomic(out/'completion.json',dict(progress,state='completed',preservation_unchanged=not changed))
    report()


if __name__=='__main__':main()
