"""Frozen-panel geometry x solver ablation, queued after complete baseline review."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('run','output','code-root','panel','report'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--workers',type=int,default=3);ap.add_argument('--wait-pid',type=int)
    a=ap.parse_args()
    if not 1<=a.workers<=3:raise ValueError('At most three numerical workers')
    a.output.mkdir(parents=True,exist_ok=True)
    protocol=a.code_root/'config/evalkit_k4_20261117/ablation_v1/protocol.json'
    version={'commit':subprocess.check_output(['git','-C',str(a.code_root),'rev-parse','HEAD']).decode().strip(),
             'protocol_sha256':sha(protocol),'runner_sha256':sha(Path(__file__)),'new_holdout_uses':0}
    identity=a.output/'identity.json'
    if identity.exists() and json.loads(identity.read_text())!=version:raise ValueError('Ablation identity changed')
    identity.write_text(json.dumps(version,indent=2)+'\n')
    deadline=time.monotonic()+8*3600
    while not (a.run/'baseline_review.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Baseline review still unavailable after 8h')
        if a.wait_pid:
            try:os.kill(a.wait_pid,0)
            except ProcessLookupError:raise RuntimeError('Baseline continuation exited without review')
        time.sleep(10)
    review=json.loads((a.run/'baseline_review.json').read_text())
    if review['protected_changed'] or any(x['expected_paths']!=656 for x in review['baselines'].values()):
        raise ValueError('Complete baseline accounting and protected files required for this ablation')
    panel=json.loads(a.panel.read_text());snapshot=a.output/'fixed_panel.json'
    if snapshot.exists() and snapshot.read_bytes()!=a.panel.read_bytes():raise ValueError('Panel changed')
    shutil.copyfile(a.panel,snapshot)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    conditions=json.loads(protocol.read_text())['conditions']
    tasks=[(condition,row) for condition in ['legacy_common',*conditions] for row in panel['records']]
    def execute(task):
        condition,row=task;key,method=row['window'],row['method']
        taskdir=a.output/'tasks'/condition/f'{key}_{method}';taskdir.mkdir(parents=True,exist_ok=True)
        if shutil.disk_usage(a.output).free<80*1024**3:raise RuntimeError('80 GiB reserve required')
        original=json.loads((a.run/'jobs'/f'{key}_{method}.json').read_text())
        def call(arguments,label):
            with (taskdir/(label+'.log')).open('w') as log:
                subprocess.run([sys.executable,*map(str,arguments)],stdout=log,stderr=subprocess.STDOUT,check=True,env=env)
        try:
            if condition=='legacy_common':fit=a.run/'common_v1'/key/method
            else:
                fit=a.output/'fits'/condition/key/method
                payload={k:original[k] for k in ('input','method','mass_kg','floor_m')}
                payload.update({'output':str(fit),'fit_config':str(a.code_root/f'config/evalkit_k4_20261117/ablation_v1/{condition}.json')})
                job=taskdir/'runtime_job.json'
                if job.exists() and json.loads(job.read_text())!=payload:raise ValueError('Runtime job changed')
                job.write_text(json.dumps(payload,indent=2)+'\n')
                call([a.code_root/'python/scripts/evalkit_k4_run.py','--code-root',a.code_root,'--job',job],'fit')
                saved=json.loads((fit/'receipt.json').read_text())['key']
                common_key=json.loads((a.run/'common_v1'/key/method/'receipt.json').read_text())['key']
                if saved['inputs']!=common_key['inputs'] or saved['model_sha256']!=common_key['model_sha256']:
                    raise ValueError('Frozen D or source model asset changed')
            views=a.output/'views'/condition
            command=[a.code_root/'python/scripts/evalkit_k4_views.py','--fit',fit,'--input',original['input'],
                     '--method',method,'--window',key,'--output',views]
            if condition=='legacy_common':command+=['--control']
            call(command,'derive')
            labels=['before','after']+(['filtered_direct_before','filtered_direct_after'] if condition=='legacy_common' else [])
            for label in labels:
                call([a.code_root/'python/scripts/evalkit_k4_ablation_score.py','--fits',views/label,
                    '--baseline',a.run/'common_v1','--output',a.output/'scores'/condition/label,
                    '--window',key,'--method',method],'score_'+label)
            result={'condition':condition,'window':key,'method':method,'status':'completed'}
        except Exception as error:
            import traceback
            result={'condition':condition,'window':key,'method':method,'status':'failed','error':str(error),'traceback':traceback.format_exc()}
        (taskdir/'status.json').write_text(json.dumps(result,indent=2)+'\n');return result
    results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for result in pool.map(execute,tasks):
            results.append(result);print(result['condition'],result['window'],result['method'],result['status'],flush=True)
    (a.output/'completion.json').write_text(json.dumps({'results':results,'expected':len(tasks),'panel_sha256':sha(snapshot)},indent=2)+'\n')
    sys.path[:0]=[str(a.code_root/'utils'),str(a.code_root/'utils/motiogen')]
    from motiogen.evalkit.k.acceptance import compare
    comparisons=[]
    for row in panel['records']:
        key,method=row['window'],row['method'];name=f'{key}_{method}.json'
        reference=a.output/'scores/legacy_common/before'/name
        if not reference.exists():continue
        base=json.loads(reference.read_text())
        for condition in conditions:
            for root in ('before','after'):
                file=a.output/'scores'/condition/root/name
                if not file.exists():continue
                candidate=json.loads(file.read_text())
                comparisons.append({'window':key,'method':method,'condition':condition,'root':root,
                    'full':compare(base,candidate),'paired':compare(base['paired_baseline_clock'],candidate['paired_baseline_clock'])})
    (a.output/'acceptance.json').write_text(json.dumps({'comparisons':comparisons,'promotion':'none; causal review required'},indent=2)+'\n')
    subprocess.run([sys.executable,str(a.code_root/'python/scripts/evalkit_k4_ablation_review.py'),
                    '--run',str(a.output),'--report',str(a.report)],check=True,env=env)
    print('Ablation complete; no configuration promoted.',flush=True)


if __name__=='__main__':main()
