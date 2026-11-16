#!/usr/bin/env python3
"""Sequential post-processing after the bounded development batch has released its workers."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--wait-pid',type=int)
    ap.add_argument('--experiment-log',type=Path,required=True);a=ap.parse_args()
    logs=ROOT/'develop_logs/test_runs/20261116_k3';status=a.out/'postprocess_status.json'
    while a.wait_pid:
        with a.experiment_log.open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-4096));tail=f.read().decode(errors='replace')
        if 'export_failures 0' in tail:break
        try:os.kill(a.wait_pid,0)
        except ProcessLookupError:break
        status.write_text(json.dumps({'status':'waiting_for_experiment','updated':time.time()}))
        time.sleep(20)
    commands=[
        ('rescore',['evalkit_k3_rescore.py','--out',str(a.out),'--workers','3']),
        ('summary',['evalkit_k3_report.py','--out',str(a.out)]),
        ('selection',['evalkit_k3_select.py']),
        ('normalization',['evalkit_k3_normalized_outputs.py','--out',str(a.out),'--workers','3']),
        ('spatial_effects',['evalkit_k3_spatial_effects.py','--out',str(a.out),'--workers','3']),
        ('parity_corrected',['evalkit_k3_regression.py','parity','--out',str(a.out),'--core-root',str(a.out/'k2_checkout'),'--workers','3']),
        ('head_reuse_corrected',['evalkit_k3_head_reuse.py','--out',str(a.out)]),
        ('downstream',['evalkit_k3_regression.py','downstream','--out',str(a.out),'--workers','3']),
        ('figures',['evalkit_k3_visualize.py','--out',str(a.out),'--dest',str(ROOT/'develop_logs/notes/20261116_k3_figures')]),
        ('report',['evalkit_k3_write_report.py','--out',str(a.out)]),
    ]
    results=[]
    for stage,args in commands:
        status.write_text(json.dumps({'status':'running','stage':stage,'completed':results,'updated':time.time()},indent=2))
        with (logs/(stage+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,str(ROOT/'python/scripts'/args[0]),*args[1:]],cwd=logs,stdout=log,stderr=subprocess.STDOUT,
                             env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        results.append({'stage':stage,'returncode':p.returncode});print(stage,p.returncode,flush=True)
        if p.returncode:
            status.write_text(json.dumps({'status':'failed','stage':stage,'completed':results},indent=2));raise SystemExit(p.returncode)
    status.write_text(json.dumps({'status':'computed_awaiting_review','completed':results,'updated':time.time()},indent=2))

if __name__=='__main__':main()
