"""One-worker fixed-panel initialization regression after supplementary ablation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('run','code-root','panel','output'):ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--wait-pid',type=int,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+4*3600
    while not (a.run/'supplement_022_ablation_v1/review.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Supplementary ablation not finished')
        try:os.kill(a.wait_pid,0)
        except ProcessLookupError:raise RuntimeError('Supplementary ablation exited without review')
        time.sleep(5)
    panel=json.loads(a.panel.read_text());shutil.copyfile(a.panel,a.output/'fixed_panel.json')
    identity={'code_commit':subprocess.check_output(['git','-C',str(a.code_root),'rev-parse','HEAD']).decode().strip(),
        'panel_sha256':hashlib.sha256(a.panel.read_bytes()).hexdigest(),
        'workers':1,'new_holdout_uses':0,'scope':'Four initializations; frozen common geometry/weights/domain. R scored only after fitting process exits. No production promotion.'}
    (a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'PYTHONPATH':str(a.code_root/'utils')+':'+str(a.code_root/'utils/motiogen')}
    sys.path[:0]=[str(a.code_root/'utils'),str(a.code_root/'utils/motiogen')]
    from motiogen.evalkit.k.acceptance import compare
    results=[]
    for row in panel['records']:
        if shutil.disk_usage(a.output).free<80*1024**3:raise RuntimeError('80 GiB reserve required')
        key,method=row['window'],row['method'];dest=a.output/f'{key}_{method}'
        original=json.loads((a.run/'jobs'/f'{key}_{method}.json').read_text())
        result={'window':key,'method':method,'status':'failed'}
        try:
            with (a.output/f'{key}_{method}_fit.log').open('w') as log:
                subprocess.run([sys.executable,str(a.code_root/'python/scripts/evalkit_k4_knee_seed_probe.py'),
                    '--fit',str(a.run/'common_v1'/key/method),'--input',original['input'],'--method',method,
                    '--window',key,'--output',str(dest)],check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            for condition in ('default_replay','knee_only','heading_only','heading_knee'):
                with (a.output/f'{key}_{method}_{condition}_score.log').open('w') as log:
                    subprocess.run([sys.executable,str(a.code_root/'python/scripts/evalkit_k4_ablation_score.py'),
                        '--fits',str(dest/'views'/condition),'--baseline',str(a.run/'common_v1'),
                        '--output',str(dest/'scores'/condition),'--window',key,'--method',method],
                        check=True,env=env,stdout=log,stderr=subprocess.STDOUT)
            base=json.loads((dest/'scores/default_replay'/f'{key}_{method}.json').read_text())
            gates={name:compare(base,json.loads((dest/'scores'/name/f'{key}_{method}.json').read_text()))
                   for name in ('knee_only','heading_only','heading_knee')}
            (dest/'acceptance.json').write_text(json.dumps(gates,indent=2)+'\n')
            result.update(status='completed',gates={name:g['status'] for name,g in gates.items()})
        except Exception as e:result['error']=str(e)
        results.append(result);print(result,flush=True)
        (a.output/'progress.json').write_text(json.dumps({'results':results,'expected':len(panel['records'])},indent=2)+'\n')
    (a.output/'completion.json').write_text(json.dumps({'results':results,'expected':len(panel['records']),'promotion':'none'},indent=2)+'\n')


if __name__=='__main__':main()
