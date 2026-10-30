"""One supervised lease owns producer, generator, and optional load process."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--events',type=Path,required=True);p.add_argument('--method',choices=['dart','ardy'],required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--ignore',action='store_true');p.add_argument('--load',action='store_true');p.add_argument('--clock',choices=['realtime','paused'],default='realtime');a=p.parse_args()
root=Path(__file__).resolve().parents[2];a.run_dir.mkdir(parents=True,exist_ok=True)
# Linux sockaddr_un has a 108-byte limit; use a short unique local IPC path.
address=f'/tmp/mg-event-{os.getpid()}.sock'
worker=[sys.executable,str(root/f'python/scripts/{a.method}_runtime_worker.py'),'--run-dir',str(a.run_dir),'--socket',address,'--seed',str(a.seed),'--clock',a.clock]
if a.ignore:worker.append('--ignore')
producer=[sys.executable,str(root/'python/scripts/reveal_runtime_event.py'),'--socket',address,'--ready',str(a.run_dir/'ready.json'),'--events',str(a.events),'--output',str(a.run_dir/'producer.json')]
children=[]
try:
    for command in [producer,worker]:children.append(subprocess.Popen(command))
    load=None
    if a.load:
        load=[sys.executable,str(root/'python/scripts/runtime_compute_load.py'),'--ready',str(a.run_dir/'ready.json'),'--output',str(a.run_dir/'compute_load.json')]
        children.append(subprocess.Popen(load))
    (a.run_dir/'processes.json').write_text(json.dumps({'generator':{'pid':children[1].pid,'argv':worker},'producer':{'pid':children[0].pid,'argv':producer},'load':None if load is None else {'pid':children[2].pid,'argv':load},'lease_gpu':os.getenv('CUDA_VISIBLE_DEVICES')}))
    code=children[1].wait()
    if code:raise RuntimeError(f'Generator exit {code}')
    if children[0].wait(timeout=10):raise RuntimeError('Producer failed')
    if load and children[2].wait(timeout=30):raise RuntimeError('Controlled load failed')
finally:
    for child in children:
        if child.poll() is None:child.terminate()
    for child in children:child.wait()
    Path(address).unlink(missing_ok=True)
