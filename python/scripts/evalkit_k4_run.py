"""Run explicit K4 fits in input-only child processes; score in separate children."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def worker(args):
    sys.path[:0]=[str(args.code_root/'utils'),str(args.code_root/'utils/motiogen')]
    import opensim as osim
    osim.Logger.setLevelString('error')
    from motiogen.evalkit.k.fitting import fit
    job=json.loads(args.job.read_text())
    if set(job)!={'input','method','output','mass_kg','floor_m','fit_config'}:raise ValueError('Invalid runtime job fields')
    out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
    try:
        config=json.loads(Path(job['fit_config']).read_text());config['method']=job['method']
        result=fit(job['input'],out,job['mass_kg'],job['floor_m'],config=config)
        status={'status':'completed','frames':len(result['times']),'kinematic_only':True,'force_eligible':False}
    except Exception as error:
        import traceback
        status={'status':'failed','error':str(error),'traceback':traceback.format_exc()}
    (out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status),flush=True)
    return 0 if status['status']=='completed' else 1


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code-root',type=Path,required=True);ap.add_argument('--job',type=Path)
    ap.add_argument('--baseline-root',type=Path);ap.add_argument('--output',type=Path)
    ap.add_argument('--fit-config',default='config/evalkit_k4_20261117/common.json')
    ap.add_argument('--workers',type=int,default=1);ap.add_argument('--limit',type=int)
    ap.add_argument('--selection',type=Path)
    args=ap.parse_args()
    if args.job:return worker(args)
    if not 1<=args.workers<=3:raise ValueError('At most three numerical workers')
    args.output.mkdir(parents=True,exist_ok=True)
    config=args.code_root/args.fit_config
    identity={'code_commit':subprocess.check_output(['git','-C',str(args.code_root),'rev-parse','HEAD']).decode().strip(),
              'runner_sha256':sha(Path(__file__)),'fit_config':str(config),'fit_config_sha256':sha(config),
              'baseline_root':str(args.baseline_root),'new_holdout_uses':0}
    manifest=args.output/'identity.json'
    if manifest.exists() and json.loads(manifest.read_text())!=identity:raise ValueError('Changed run identity')
    manifest.write_text(json.dumps(identity,indent=2)+'\n')
    paths=sorted((args.baseline_root/'jobs').glob('*.json'))
    if args.selection:
        selected={(r['window'],r['method']) for r in json.loads(args.selection.read_text())['records']}
        paths=[p for p in paths if tuple(p.stem.split('_',1)) in selected]
    paths=paths[:args.limit];jobs=[]
    for p in paths:
        source=json.loads(p.read_text());key,method=p.stem.split('_',1)
        job={k:source[k] for k in ('input','method','mass_kg','floor_m')}
        job.update({'output':str((args.output/key/method).resolve()),'fit_config':str(config.resolve())})
        file=args.output/'jobs'/p.name;file.parent.mkdir(exist_ok=True)
        if file.exists() and json.loads(file.read_text())!=job:raise ValueError('Changed input job')
        file.write_text(json.dumps(job,indent=2)+'\n');jobs.append((file,key,method))
    def execute(item):
        file,key,method=item
        if shutil.disk_usage(args.output).free<80*1024**3:raise RuntimeError('80 GiB reserve required')
        out=args.output/key/method;out.mkdir(parents=True,exist_ok=True)
        status=out/'status.json'
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        if not status.exists():
            with (out/'worker.log').open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__)), '--code-root',str(args.code_root),'--job',str(file)],
                               stdout=log,stderr=subprocess.STDOUT,env=env)
        value=json.loads(status.read_text())['status'] if status.exists() else 'worker_failed'
        # Scoring runs only after the fitting child has exited. Neither R nor
        # scores is present in the fitting job or passed to that process.
        if value=='completed':
            with (out/'score.log').open('w') as log:
                code=subprocess.run([sys.executable,str(args.code_root/'python/scripts/evalkit_k4_score.py'),
                    '--fits',str(args.output),'--artifact','fit.npz','--output',str(args.output/'scores'),
                    '--window',key,'--method',method],stdout=log,stderr=subprocess.STDOUT,env=env).returncode
            if code:value='score_failed'
        return key,method,value
    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute,jobs):print(*result,flush=True);results.append(result)
    (args.output/'completion.json').write_text(json.dumps({'results':results,'expected':len(jobs)},indent=2)+'\n')
    return int(any(r[2]!='completed' for r in results))


if __name__=='__main__':sys.exit(main())
