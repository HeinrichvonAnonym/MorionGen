"""Finish development checks, freeze Andy protocol, then launch one reserved slot."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home/heinrich/MorionGen')
LOG=ROOT/'develop_logs/test_runs/20261120_k4_andy'
B=Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs/evalkit_k4_20261117')
TV=B/'test_budget_20261120_v1'
RUN=B/'andy_head_test_20261120_v1'
PY='/home/heinrich/.local/share/opensim-api/bin/python'
ENV={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
def guard(command,receipt,script):
    return [PY,str(script),'--tv-controller','585745','--tv-run',str(TV),'--command',str(command),'--receipt',str(receipt)]

previous=LOG/'development_lease.json';stamp=None;value={}
while 'released_unix_s' not in value:
    new=previous.stat().st_mtime_ns
    if new!=stamp:stamp=new;value=json.loads(previous.read_text())
    if 'released_unix_s' not in value:time.sleep(5)
assert value.get('returncode')==0,value
shutil.copyfile(LOG/'development_verification.json',LOG/'development_verification_first.json')
# Recheck the final adapter, reusing the exact completed D and fitted models.
subprocess.run(guard(LOG/'development_command.json',LOG/'development_final_lease.json',ROOT/'python/scripts/evalkit_k4_slot_lease.py'),env=ENV,check=True)
subprocess.run([PY,str(ROOT/'python/scripts/evalkit_k4_andy.py'),'register','--run',str(RUN)],env=ENV,check=True)
command=[PY,str(RUN/'control/evalkit_k4_andy.py'),'run','--run',str(RUN),'--lease',str(RUN/'slot_lease.json')]
(RUN/'launch_command.json').write_text(json.dumps(command,indent=2)+'\n')
with (LOG/'run.log').open('a') as out:
    process=subprocess.Popen(guard(RUN/'launch_command.json',RUN/'slot_lease.json',RUN/'control/evalkit_k4_slot_lease.py'),env=ENV,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
(LOG/'launch.json').write_text(json.dumps(dict(guard_pid=process.pid,run=str(RUN),launched_unix_s=time.time()),indent=2)+'\n')
print('Andy launched',process.pid,str(RUN),flush=True)
