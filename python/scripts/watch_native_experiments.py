"""Visible monitoring session; reuses queues and analysis, never launches GPU jobs."""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
STATUS=ROOT/'develop_logs/test_runs/20261024_native_execution/status.json'
def record_failure(kind,error,tb):
    receipt={'time_unix':time.time(),'error_type':kind.__name__,'reason':str(error),
             'traceback':''.join(traceback.format_exception(kind,error,tb)),
             'scope':'Monitoring/analysis failure; this process does not own or stop GPU queues.'}
    with (STATUS.parent/'watch_failures.jsonl').open('a') as handle:
        handle.write(json.dumps(receipt,ensure_ascii=False)+'\n')
    sys.__excepthook__(kind,error,tb)
sys.excepthook=record_failure
p=argparse.ArgumentParser()
p.add_argument('--interval',type=float,default=50.)
p.add_argument('--refresh-seconds',type=float,default=900.)
p.add_argument('--once',action='store_true')
args=p.parse_args()
if not 1<=args.interval<=60:raise ValueError('Monitoring interval must be at most 60 seconds')
tracked=None;previous={};last_refresh=time.monotonic()
while True:
    subprocess.run([sys.executable,str(ROOT/'python/scripts/native_execution_status.py')],check=True,
                   cwd=ROOT,stdout=subprocess.DEVNULL)
    status=json.loads(STATUS.read_text())
    if tracked is None:
        tracked={q['queue'] for q in status['queues'] if q['process_alive'] and q['status']=='running'}
    # Supplemental registered queues can join after monitoring starts. Retain
    # earlier queues for terminal checks, and include new live queues immediately.
    tracked.update(q['queue'] for q in status['queues'] if q['process_alive'] and q['status']=='running')
    active=[q for q in status['queues'] if q['queue'] in tracked]
    bad=[q['queue'] for q in active if not q['process_alive'] and q['status']=='running']
    bad+=[q['queue'] for q in active if q['process_alive'] and q['heartbeat_age_seconds']>30]
    bad+=[q['queue'] for q in active if any(r.get('worker_alive') is False and (r.get('resource_sample_age_seconds') or 0)>10 for r in q['active'])]
    changed=[]
    for q in active:
        if previous.get(q['queue'])!=q['counts']:
            changed.append({'queue':q['queue'],'counts':q['counts']})
        previous[q['queue']]=q['counts']
    receipt={'timestamp':status['timestamp'],'wall_remaining_hours':round(status['wall_remaining_hours'],2),
             'live_queues':sum(q['process_alive'] for q in active),'changes':changed,
             'running_jobs':[{'job':r['job'],'device':r['device'],'pid':r.get('worker_pid'),
                              'worker_alive':r.get('worker_alive'),'resource_age_s':r.get('resource_sample_age_seconds')}
                              for q in active for r in q['active']],
             'projected_gpu_lease_hours':round(status['workload_projection']['estimated_gpu_lease_hours'],2),
             'alerts':bad}
    print(json.dumps(receipt,ensure_ascii=False),flush=True)
    with (STATUS.parent/'watch.jsonl').open('a') as log:
        log.write(json.dumps(receipt,ensure_ascii=False)+'\n')
    if bad:raise RuntimeError('Live execution audit requires investigation: '+str(bad))
    drained=not any(q['process_alive'] and q['status']=='running' for q in active)
    if not args.once and (drained or time.monotonic()-last_refresh>=args.refresh_seconds):
        for script in ['analyze_native_matrix.py','audit_native_weights.py','analyze_native_matrix.py',
                       'audit_native_invariants.py','audit_native_execution.py','audit_native_preprocessing.py','audit_native_metrics.py']:
            subprocess.run([sys.executable,str(ROOT/'python/scripts'/script)],check=True,cwd=ROOT)
        for script in ['audit_native_clock_observations.py','audit_ardy_history.py','audit_chois_native_metrics.py']:
            subprocess.run([sys.executable,str(ROOT/'python/scripts'/script)],check=True,cwd=ROOT)
        sys.path.insert(0,str(ROOT/'utils'))
        from motiogen.config import load_config,environment_root
        cfg=load_config()
        subprocess.run([str(environment_root(cfg,'motionlcm')/'bin/python'),
                        str(ROOT/'python/scripts/audit_spatial_coordinates.py')],check=True,cwd=ROOT)
        if (STATUS.parent.parent/'20261024_omnicontrol_trajectory/pilot_plan.json').exists():
            subprocess.run([str(environment_root(cfg,'omnicontrol')/'bin/python'),
                            str(ROOT/'python/scripts/analyze_native_trajectories.py')],check=True,cwd=ROOT)
        if (STATUS.parent.parent/'20261025_source_replays/queue/state.json').exists():
            for script in ['audit_native_source_cohorts.py','audit_native_execution_epochs.py','analyze_closd_marker_probe.py']:
                subprocess.run([sys.executable,str(ROOT/'python/scripts'/script)],check=True,cwd=ROOT)
        for relative,key in [('20261024_native_invariants/audit.json','status'),
                             ('20261024_native_execution/audit.json','passed_observed_checks')]:
            audit=json.loads((STATUS.parent.parent/relative).read_text())
            if audit[key] not in (True,'passed'):raise RuntimeError('Scientific audit failed: '+relative)
        weights=json.loads((STATUS.parent.parent/'20261024_native_weight_audit/audit.json').read_text())
        if not all(r['passed'] for r in weights['rows']):raise RuntimeError('Weight evidence audit failed')
        preprocessing=json.loads((STATUS.parent.parent/'20261024_native_preprocessing/audit.json').read_text())
        if not preprocessing['passed']:raise RuntimeError('Preprocessing evidence audit failed')
        last_refresh=time.monotonic()
        print(json.dumps({'event':'evidence_refreshed','all_tracked_queues_drained':drained}),flush=True)
    if args.once or drained:break
    time.sleep(args.interval)
