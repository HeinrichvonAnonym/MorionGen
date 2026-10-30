"""Registered, bounded GPU contention within the parent experiment lease."""
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--ready',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
import torch
torch.set_num_threads(1)
x=torch.ones((1024,1024),device='cuda');y=torch.ones_like(x);torch.cuda.synchronize()
while not a.ready.exists():time.sleep(.05)
start=json.loads(a.ready.read_text())['start_ns'];rows=[]
while time.monotonic_ns()<start+20_000_000_000:
    tick=time.monotonic_ns();active=0
    while time.monotonic_ns()-tick<20_000_000:
        torch.mm(x,y);torch.cuda.synchronize();active+=1
    finish=time.monotonic_ns();rows.append({'start_ns':tick,'end_ns':finish,'matmuls':active})
    time.sleep(.02)
a.output.write_text(json.dumps({'pid':os.getpid(),'device_visible':os.getenv('CUDA_VISIBLE_DEVICES'),'matrix':1024,
    'requested_busy_seconds':.02,'requested_rest_seconds':.02,'allocated_bytes':torch.cuda.max_memory_allocated(),'cycles':rows,
    'scope':'Measured active wall intervals and matrix operations; resource monitor provides observed GPU utilization, not an assumed exact 50% load.'}))
