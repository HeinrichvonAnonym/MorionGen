"""Check actual supervised intervals and live CUDA routing, not queue labels."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write

base=ROOT/'develop_logs/test_runs'
budget=json.loads((ROOT/'config/survey_revision.json').read_text())
start=datetime.fromisoformat(budget['experiment_started_at'].replace('Z','+00:00'))
now=datetime.now(timezone.utc)
intervals=defaultdict(list);unresolved=[];routing=[]
for path in base.glob('*/manifest.json'):
    events_path=path.parent/'events.jsonl'
    if not events_path.exists():continue
    events=[json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    begun=next((e for e in events if e['event']=='process_started'),None)
    if not begun:continue
    begin=datetime.fromisoformat(begun['wall_time'])
    if not start<=begin<=now:continue
    manifest=json.loads(path.read_text())
    device=manifest.get('budgets',{}).get('gpu',{}).get('index')
    if device not in (0,1):continue
    finished=next((e for e in reversed(events) if e['event']=='run_finished'),None)
    timing_scope='original process completion receipt'
    if finished:
        end=datetime.fromisoformat(finished['wall_time'])
    else:
        reconciled=next((e for e in reversed(events) if e['event'] in ('queue_drain_reconciled','queue_interruption_reconciled')),None)
        if reconciled:
            end=datetime.fromisoformat(reconciled.get('end_upper_bound',reconciled['wall_time']))
            timing_scope='conservative termination upper bound from reconciliation; not proof of continuous activity'
            intervals[device].append({'run':str(path.parent.relative_to(ROOT)), 'begin':begin.timestamp(),
                'end':end.timestamp(),'status':manifest['status'],'timing_scope':timing_scope})
            continue
        try:
            proc=psutil.Process(begun['pid'])
            alive=proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE and abs(proc.create_time()-begin.timestamp())<2
        except psutil.NoSuchProcess:alive=False
        if not alive:
            unresolved.append({'run':str(path.parent.relative_to(ROOT)),'reason':'No live matching worker and no original run_finished event; interval not extrapolated'})
            continue
        end=now;timing_scope='live process cut at audit time'
        try:visible=proc.environ().get('CUDA_VISIBLE_DEVICES')
        except psutil.NoSuchProcess:continue
        routing.append({'run':str(path.parent.relative_to(ROOT)),'pid':proc.pid,'physical_device':device,
                        'cuda_visible_devices':visible,'passed':visible==str(device)})
    intervals[device].append({'run':str(path.parent.relative_to(ROOT)), 'begin':begin.timestamp(),
                              'end':end.timestamp(),'status':manifest['status'],'timing_scope':timing_scope})
overlaps=[]
for device,rows in intervals.items():
    rows.sort(key=lambda r:r['begin'])
    for i,left in enumerate(rows):
        for right in rows[i+1:]:
            if right['begin']>=left['end']:break
            overlaps.append({'device':device,'left':left['run'],'right':right['run'],
                             'overlap_seconds':min(left['end'],right['end'])-right['begin']})
matrix_path=base/'20261024_native_matrix_analysis/analysis.json'
matrix=json.loads(matrix_path.read_text()) if matrix_path.exists() else {'rows':[],'updated_at':None}
assignments=defaultdict(lambda:defaultdict(set));device_cache={}
for row in matrix['rows']:
    run=row['run']
    if run not in device_cache:
        device_cache[run]=json.loads((ROOT/run/'manifest.json').read_text())['budgets']['gpu']['index']
    key=(row['method'],row['clock'],json.dumps(row['condition'],sort_keys=True))
    assignments[key][device_cache[run]].add(run)
condition_devices=[{'method':method,'clock':clock,'condition':json.loads(condition),
    'runs_by_device':{str(device):len(runs) for device,runs in devices.items()}}
    for (method,clock,condition),devices in sorted(assignments.items())]
receipt={'as_of':now.isoformat(),'interval_counts':{str(d):len(rows) for d,rows in intervals.items()},
         'assignment_analysis_cutoff':matrix['updated_at'],'condition_devices':condition_devices,
         'assignment_scope':'Accepted primary runs, counted once per condition and physical GPU. Descriptive allocation evidence; similar counts do not establish equal hardware speed or remove host contention. No arbitrary balance threshold gates execution.',
         'overlaps':overlaps,'live_routing':routing,'missing_terminal_intervals':unresolved,
         'intervals':dict(intervals),
         'scope':'Observed supervised process intervals from process_started to run_finished; living workers cut at audit time. Queue-drain and interruption receipts provide conservative end upper bounds, labeled individually; these are not proof of continuous activity. Missing terminal evidence excluded explicitly. Does not infer absent historical manual workers.',
         'passed_observed_checks':not overlaps and not unresolved and all(r['passed'] for r in routing)}
json_write(base/'20261024_native_execution/audit.json',receipt)
print(json.dumps({k:v for k,v in receipt.items() if k not in ['intervals','condition_devices']}))
