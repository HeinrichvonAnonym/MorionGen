"""Read-only live queue/lease status and a fixed-deadline resource ledger."""
import json
import sys
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import psutil
import statistics
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
base=ROOT/'develop_logs/test_runs';budget=json.loads((ROOT/'config/survey_revision.json').read_text())
start=datetime.fromisoformat(budget['experiment_started_at'].replace('Z','+00:00'));deadline=datetime.fromisoformat(budget['experiment_deadline_at'].replace('Z','+00:00'));now=datetime.now(timezone.utc)
queues=[]
def log_tail(path):
    if not path.exists():return None
    with path.open('rb') as handle:
        handle.seek(max(0,path.stat().st_size-4096))
        lines=handle.read().decode(errors='replace').splitlines()
    return lines[-1] if lines else None

for path in base.glob('*/**/state.json'):
    state=json.loads(path.read_text())
    if 'jobs' not in state or 'pid' not in state:continue
    # Queue directories retain their creation date; new queues may also join
    # after midnight. Select the shared experiment deadline, not a date prefix.
    if state.get('deadline') is None or datetime.fromisoformat(state['deadline'].replace('Z','+00:00'))!=deadline:continue
    try:
        process=psutil.Process(state['pid']);alive=process.is_running() and process.create_time()==state.get('process_create_time')
    except psutil.NoSuchProcess:alive=False
    active=[{'job':key,**row} for key,row in state['jobs'].items() if row['status']=='running']
    for row in active:
        run=Path(row['run_dir']) if row.get('run_dir') else None
        if run is None:continue
        event_path=run/'events.jsonl'
        if event_path.exists():
            begun=None
            for line in event_path.open():
                entry=json.loads(line)
                if entry['event']=='process_started':begun=entry;break
            if begun:
                row['worker_pid']=begun['pid']
                try:
                    worker=psutil.Process(begun['pid'])
                    row['worker_alive']=worker.is_running() and worker.status()!=psutil.STATUS_ZOMBIE and abs(worker.create_time()-datetime.fromisoformat(begun['wall_time']).timestamp())<2
                except psutil.NoSuchProcess:row['worker_alive']=False
        stdout=run/'stdout.log'
        row['stdout_bytes']=stdout.stat().st_size if stdout.exists() else 0
        row['stdout_age_seconds']=max(0.,now.timestamp()-stdout.stat().st_mtime) if stdout.exists() else None
        row['last_stdout_line']=(log_tail(stdout) or '')[-300:]
        resource=run/'resources.jsonl'
        row['resource_sample_age_seconds']=max(0.,now.timestamp()-resource.stat().st_mtime) if resource.exists() else None
        tail=log_tail(resource)
        row['last_resource_sample']=json.loads(tail) if tail else None
    workload=[]
    if alive:
        # Job IDs are opaque: source-replay IDs share a prefix across methods.
        # Recover the actual registered method from this live queue's plan.
        try:
            command=process.cmdline()
            process_directory=Path(process.cwd())
        except (psutil.NoSuchProcess,psutil.ZombieProcess):
            # A drained queue can exit between its liveness check and this
            # optional projection. Do not crash the monitor during that race.
            command=[]
            process_directory=ROOT
        method_by_job={}
        if '--plan' in command:
            plan_path=Path(command[command.index('--plan')+1])
            if not plan_path.is_absolute():plan_path=process_directory/plan_path
            plan=json.loads(plan_path.read_text())
            method_by_job={job['id']:job['method'] for job in plan['jobs']}
        samples=defaultdict(list);remaining=Counter()
        for job,row in state['jobs'].items():
            method=method_by_job.get(job,'unknown_method')
            if row['status']=='success' and row.get('elapsed_seconds') is not None:
                samples[method].append((row.get('started_at',''),row['elapsed_seconds']))
            elif row['status'] in ('pending','running'):
                remaining[method]+=1
        for method,count in remaining.items():
            recent=[seconds for _,seconds in sorted(samples[method])[-10:]]
            median=statistics.median(recent) if recent else None
            workload.append({'method':method,'remaining_jobs':count,'recent_completed_jobs':len(recent),
                'recent_median_worker_seconds':median,
                'projected_remaining_gpu_lease_hours':count*median/3600 if median is not None else None})
    queues.append({'queue':str(path.parent.relative_to(ROOT)),'pid':state['pid'],'process_alive':alive,
                   'status':state['status'],'heartbeat_age_seconds':max(0.,(now-datetime.fromisoformat(state['heartbeat'])).total_seconds()),
                   'counts':dict(Counter(r['status'] for r in state['jobs'].values())),'active':active,'workload_projection':workload})
seconds=defaultdict(float);observed_seconds=defaultdict(float);records=[]
for path in base.glob('*/manifest.json'):
    event_path=path.parent/'events.jsonl'
    if not event_path.exists():continue
    events=[json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
    begun=next((r for r in events if r['event']=='process_started'),None)
    if begun is None:continue
    stamp=datetime.fromisoformat(begun['wall_time'])
    if not start<=stamp<=now:continue
    manifest=json.loads(path.read_text());device=manifest.get('budgets',{}).get('gpu',{}).get('index')
    if device not in (0,1):continue
    elapsed=manifest.get('elapsed_seconds')
    scope='measured supervisor duration'
    if elapsed is None:
        terminal=next((r for r in reversed(events) if r['event'] in ('queue_drain_reconciled','queue_interruption_reconciled')),None)
        if terminal is not None:
            elapsed=(datetime.fromisoformat(terminal.get('end_upper_bound',terminal['wall_time']))-stamp).total_seconds()
            scope='conservative upper bound through reconciliation; not measured compute or confirmed continuous occupancy'
        elif manifest['status']=='running':
            elapsed=(now-stamp).total_seconds();scope='ongoing lease elapsed'
        else:
            records.append({'run':str(path.parent.relative_to(ROOT)),'device':device,'seconds':None,'status':manifest['status'],'timing_scope':'terminal duration absent; excluded, not extended to now'})
            continue
    lower=manifest.get('interruption_timing',{}).get('observed_elapsed_lower_seconds',elapsed)
    if 'upper bound' in scope and 'interruption_timing' not in manifest:
        resource_tail=log_tail(path.parent/'resources.jsonl')
        lower=json.loads(resource_tail)['elapsed_seconds'] if resource_tail else 0.
    seconds[device]+=elapsed;observed_seconds[device]+=lower
    records.append({'run':str(path.parent.relative_to(ROOT)),'device':device,'seconds':elapsed,
                    'observed_elapsed_lower_seconds':lower,'status':manifest['status'],'timing_scope':scope})
result={'timestamp':now.isoformat(),'deadline':deadline.isoformat(),'wall_elapsed_hours':(now-start).total_seconds()/3600,
        'wall_remaining_hours':(deadline-now).total_seconds()/3600,'report_reserve_hours':4,
        'supervised_gpu_lease_hours':{str(k):v/3600 for k,v in seconds.items()},
        'observed_gpu_lease_hours_lower_bound':{str(k):v/3600 for k,v in observed_seconds.items()},
        'ledger_scope':'Supervisor worker durations plus explicitly labeled conservative upper bounds for interrupted/drained workers; not utilization-integrated CUDA time or proof of activity during unobserved gaps. Includes failed/rejected supervised runs; historical manual runs without process receipts are not inferred.',
        'queues':queues,'runs':records}
projected=[r for q in queues for r in q['workload_projection']]
result['workload_projection']={'remaining_jobs':sum(r['remaining_jobs'] for r in projected),
    'estimated_gpu_lease_hours':sum(r['projected_remaining_gpu_lease_hours'] or 0 for r in projected),
    'jobs_without_timing_basis':sum(r['remaining_jobs'] for r in projected if r['projected_remaining_gpu_lease_hours'] is None),
    'scope':'Latest ten completed worker durations per queue/method, median times remaining jobs, including full duration of active jobs. Conditional throughput projection, not a deadline guarantee; no sample selection based on results.'}
destination=base/'20261024_native_execution/status.json';json_write(destination,result)
print(json.dumps({k:result[k] for k in ['timestamp','deadline','wall_elapsed_hours','wall_remaining_hours','supervised_gpu_lease_hours']}))
for row in queues:
    if row['process_alive']:print(json.dumps(row))
