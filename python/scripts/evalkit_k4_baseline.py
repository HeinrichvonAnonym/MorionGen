"""Prepare input-only copies and run the historical K implementation on development D."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.evalkit.k.motion import load


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--code-root',type=Path,required=True);ap.add_argument('--limit',type=int)
    ap.add_argument('--workers',type=int,default=3);args=ap.parse_args()
    assert 1<=args.workers<=3
    args.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(args.output).free<80*1024**3:raise RuntimeError('80 GiB reserve required')
    worker=ROOT/'python/scripts/evalkit_k4_worker.py';snapshot=args.output/'worker_frozen.py'
    if snapshot.exists():assert snapshot.read_bytes()==worker.read_bytes(),'Use a new output for changed worker'
    else:shutil.copyfile(worker,snapshot)
    identity={'worker_sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),
              'code_root_commit':subprocess.check_output(['git','-C',str(args.code_root),'rev-parse','HEAD']).decode().strip(),
              'mass_kg':75.,'floor_policy':'input toe-height P05 proxy','new_holdout_uses':0}
    manifest=args.output/'historical_baseline_identity.json'
    if manifest.exists():assert json.loads(manifest.read_text())==identity
    else:manifest.write_text(json.dumps(identity,indent=2)+'\n')
    freeze=json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())
    run=Path(freeze['run']);records=json.loads((run/'artifact_verification.json').read_text())['records']
    jobs=[]
    for record in records[:args.limit]:
        key,method=record['window'],record['method'];original=run/'inputs'/key/method/'candidate'
        target=args.output/'runtime_inputs'/key/method/'candidate'
        for name,h in record['input_hashes'].items():
            assert hashlib.sha256((original/name).read_bytes()).hexdigest()==h
            file=target/name;file.parent.mkdir(parents=True,exist_ok=True)
            if not file.exists():shutil.copyfile(original/name,file)
            assert hashlib.sha256(file.read_bytes()).hexdigest()==h
        source=target/'factorial/reference.npz' if method=='primal' else target
        motion=load(source,method)
        # Fixed across baseline and later candidates: an input-derived landmark
        # height proxy, not a claim about measured ground or support geometry.
        feet=[motion.names.index(side+'_foot') for side in ('left','right')]
        floor=float(min(np.percentile(motion.positions[:,j,1],5) for j in feet))
        out=args.output/'historical_k2'/key/method
        payload={'input':str(source.resolve()),'method':method,'output':str(out.resolve()),
                 'mass_kg':75.,'floor_m':floor,'floor_policy':'minimum left/right toe-point height P05; input-only proxy',
                 'phase':'historical_k2_on_frozen_D'}
        job=args.output/'jobs'/f'{key}_{method}.json';job.parent.mkdir(exist_ok=True)
        if job.exists():assert json.loads(job.read_text())==payload
        else:job.write_text(json.dumps(payload,indent=2)+'\n')
        jobs.append((job,out,key,method))
    def execute(item):
        job,out,key,method=item;out.mkdir(parents=True,exist_ok=True)
        status=out/'status.json'
        if status.exists():return key,method,json.loads(status.read_text())['status']
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with (out/'worker.log').open('w') as log:
            subprocess.run([sys.executable,str(snapshot),
                            '--job',str(job),'--code-root',str(args.code_root)],stdout=log,stderr=subprocess.STDOUT,env=env)
        return key,method,json.loads(status.read_text())['status'] if status.exists() else 'worker_failed'
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(execute,jobs):print(*result,flush=True)


if __name__=='__main__':main()
