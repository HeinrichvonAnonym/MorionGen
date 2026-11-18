"""Frozen development candidates: input-only fit children, then independent R scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('run','code-root','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--config-root',type=Path)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--wait-for',type=Path);ap.add_argument('--wait-pid',type=int)
    a=ap.parse_args();config=a.config_root or a.code_root/'config/evalkit_k4_20261117/optimization_v1'
    if not 1<=a.workers<=3:raise ValueError('At most three numerical subprocesses allowed')
    deadline=time.monotonic()+4*3600
    while a.wait_for and not a.wait_for.exists():
        if time.monotonic()>deadline:raise TimeoutError('Preceding numerical workers did not finish')
        if a.wait_pid:
            try:os.kill(a.wait_pid,0)
            except ProcessLookupError:raise RuntimeError('Preceding run exited without completion')
        time.sleep(10)
    protocol=json.loads((config/'protocol.json').read_text());panel=json.loads((config/'sentinels.json').read_text())
    a.output.mkdir(parents=True,exist_ok=a.resume)
    lock=(a.output/'.controller.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    identity={'code_commit':subprocess.check_output(['git','-C',str(a.code_root),'rev-parse','HEAD']).decode().strip(),
        'config_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in config.glob('*.json')},
        'workers':a.workers,'new_holdout_uses':0}
    if a.resume:
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
        from motiogen.evalkit.experiment_recovery import prepare
        results,jobs=prepare(a.output,a.code_root,config,a.run,identity,protocol,panel)
    else:
        shutil.copyfile(config/'protocol.json',a.output/'protocol.json');shutil.copyfile(config/'sentinels.json',a.output/'panel.json')
        (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        results=[];jobs=[(row,condition) for row in panel['records'] for condition in protocol['conditions']]
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'PYTHONPATH':str(a.code_root/'utils')+':'+str(a.code_root/'utils/motiogen')}
    def execute(row,condition):
        key,method=row['window'],row['method']
        original=json.loads((a.run/'jobs'/f'{key}_{method}.json').read_text())
        if shutil.disk_usage(a.output).free<80*1024**3:raise RuntimeError('80 GiB reserve required')
        folder=a.output/'tasks'/condition/f'{key}_{method}';folder.mkdir(parents=True)
        fit=a.output/'fits'/condition/key/method
        payload={k:original[k] for k in ('input','method','mass_kg','floor_m')}
        payload.update(output=str(fit),fit_config=str(config/(condition+'.json')))
        job=folder/'runtime_job.json';job.write_text(json.dumps(payload,indent=2)+'\n')
        def call(arguments,label):
            with (folder/(label+'.log')).open('w') as log:
                subprocess.run([sys.executable,*map(str,arguments)],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
        result={'window':key,'method':method,'condition':condition,'status':'failed'}
        try:
            native=protocol.get('frozen_native_bank')
            if native:
                source_run=Path(native['run']);source_task=source_run/'tasks'/native['condition']/f'{key}_{method}'/'status.json'
                producer=json.loads((source_run/'identity.json').read_text())
                if producer['code_commit']!=native['code_commit']:raise ValueError('Native producer implementation changed')
                timeout=time.monotonic()+4*3600
                while not source_task.exists():
                    if (source_run/'completion.json').exists():raise RuntimeError('Completed native bank has no task record')
                    if time.monotonic()>timeout:raise TimeoutError('Native producer has not completed this input')
                    time.sleep(10)
                if json.loads(source_task.read_text())['status']!='completed':
                    raise RuntimeError('Native producer failed; preserve its failure history before recovery')
                source_fit=source_run/'fits'/native['condition']/key/method
                payload.update(native_bank=str(source_fit),native_contract=str(config/native['contract']))
                job.write_text(json.dumps(payload,indent=2)+'\n')
                call([a.code_root/'python/scripts/evalkit_k4_compose_bank.py','--input',original['input'],
                      '--method',method,'--mass',original['mass_kg'],'--floor',original['floor_m'],
                      '--native-bank',source_fit,'--native-contract',config/native['contract'],
                      '--config',config/(condition+'.json'),'--output',fit],'fit')
            else:
                call([a.code_root/'python/scripts/evalkit_k4_run.py','--code-root',a.code_root,'--job',job],'fit')
            saved=json.loads((fit/'receipt.json').read_text())['key']
            common=json.loads((a.run/'common_v1'/key/method/'receipt.json').read_text())['key']
            if saved['inputs']!=common['inputs'] or saved['model_sha256']!=common['model_sha256']:
                raise ValueError('Frozen input or model asset changed')
            views=a.output/'views'/condition
            if protocol.get('native_export_scoring'):
                views=views/key/method
                call([a.code_root/'python/scripts/evalkit_k4_native_export_views.py','--code-root',a.code_root,
                      '--fit',fit,'--input',original['input'],'--method',method,'--window',key,'--output',views],'views')
                conditions=('native','export')
            else:
                call([a.code_root/'python/scripts/evalkit_k4_views.py','--fit',fit,'--input',original['input'],
                      '--method',method,'--window',key,'--output',views],'views')
                conditions=('before','after')
            for root in conditions:
                call([a.code_root/'python/scripts/evalkit_k4_ablation_score.py','--fits',views/root,
                    '--baseline',a.run/'common_v1','--output',a.output/'scores'/condition/root,
                    '--window',key,'--method',method,*(['--strict'] if protocol.get('strict_scoring') else [])],'score_'+root)
            if protocol.get('native_export_scoring'):
                call([a.code_root/'python/scripts/evalkit_k4_export_audit.py','--fit',fit,
                      '--output',folder/'actual_export'],'export_audit')
            result['status']='completed'
        except Exception as error:
            import traceback
            result.update(error=str(error),traceback=traceback.format_exc())
        (folder/'status.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
    expected=len(panel['records'])*len(protocol['conditions'])
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(execute,row,condition) for row,condition in jobs]
        for future in as_completed(futures):
            result=future.result();results.append(result);print(result,flush=True)
            temporary=a.output/'progress.tmp'
            temporary.write_text(json.dumps({'results':results,'expected':expected},indent=2)+'\n')
            temporary.replace(a.output/'progress.json')
    (a.output/'completion.json').write_text(json.dumps({'results':results,'expected':len(panel['records'])*len(protocol['conditions']),'promotion':'none'},indent=2)+'\n')


if __name__=='__main__':main()
