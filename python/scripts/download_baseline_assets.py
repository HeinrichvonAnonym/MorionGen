"""Download the five selected official baselines and shared text adapters."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import threading
import time

from motiogen.config import ROOT
from motiogen.runtime import json_write


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--jobs',type=int,default=6,choices=[1,2,3,4,5,6])
    args=parser.parse_args()
    stamp=time.strftime('%Y%m%d_%H%M%S')
    from uuid import uuid4
    directory=ROOT/'develop_logs/test_runs'/f'{stamp}_official_asset_downloads_{uuid4().hex[:8]}'
    directory.mkdir(parents=True)
    tasks=[('ardy',['--asset','model']),('kimodo',['--asset','model']),('motionlcm',[]),
           ('omnicontrol',[]),('chois',[]),('ardy-text',['--asset','local-text'])]
    state={'status':'running','jobs':args.jobs,'transport':'current inherited proxy; official sources; no mirrors',
           'tasks':{name:{'status':'queued'} for name,_ in tasks},
           'restricted_resources':[
               {'name':'SMPL-X body models','url':'https://smpl-x.is.tue.mpg.de/','reason':'Requires upstream account and model licence; no anonymous official weights URL.'},
               {'name':'Full HumanML3D training motions','url':'https://github.com/EricGuo5513/HumanML3D#how-to-obtain-the-data','reason':'Must prepare from licensed AMASS/SMPL+H/DMPL assets; tiny official test set is downloaded separately.'}]}
    json_write(directory/'status.json',state)
    lock=threading.Lock()
    print(str(directory),flush=True)
    def launch(name,extra):
        command=['bash',str(ROOT/'scripts/run_baseline.sh'),'prepare','--baseline',name.split('-')[0],*extra]
        with (directory/(name+'.stdout.log')).open('w') as out, (directory/(name+'.stderr.log')).open('w') as err:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err)
            with lock:
                state['tasks'][name]={'status':'running','pid':proc.pid}
                json_write(directory/'status.json',state)
            # Worker progress remains under its unique framework run and asset *.download.json.
            returncode=proc.wait()
        return {'status':'success' if returncode==0 else 'incomplete','returncode':returncode,'stdout':str(directory/(name+'.stdout.log'))}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures={pool.submit(launch,name,extra):name for name,extra in tasks}
        for future in as_completed(futures):
            name=futures[future]
            try:
                result=future.result()
            except Exception as exc:
                result={'status':'failed','reason':type(exc).__name__}
            with lock:
                state['tasks'][name]=result
                json_write(directory/'status.json',state)
            print(name,state['tasks'][name]['status'],flush=True)
    state['status']='success' if all(t['status']=='success' for t in state['tasks'].values()) else 'incomplete'
    json_write(directory/'status.json',state)
    return 0 if state['status']=='success' else 1


if __name__=='__main__':
    raise SystemExit(main())
