"""Preserve interrupted attempts; resume only identical queues, else new plans."""
import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'utils'))
from motiogen.experiment_queue import fingerprint
from motiogen.execution_identity import framework_sources
from motiogen.runtime import checksum,json_write,event

parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
base=ROOT/'develop_logs/test_runs';receipt_path=base/'20261025_interruption_recovery/receipt.json'
if receipt_path.exists():raise ValueError('Recovery already recorded; inspect it, do not repeat reconciliation')
specs=[('20261024_control_matrix','continuation_plan.json','queue_fair'),
       ('20261024_ardy_matrix','plan.json','queue'),
       ('20261024_omnicontrol_trajectory','batch_plan.json','batch_queue'),
       ('20261024_closd_marker_probe','extension_plan.json','extension_queue')]
now=datetime.now(timezone.utc);boot=datetime.fromtimestamp(psutil.boot_time(),timezone.utc);actions=[]
def matching_live(pid,created):
    try:
        p=psutil.Process(pid)
        return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and abs(p.create_time()-created)<2
    except psutil.NoSuchProcess:return False
for folder,plan_name,queue_name in specs:
    directory=base/folder;plan_path=directory/plan_name;state_path=directory/queue_name/'state.json'
    plan=json.loads(plan_path.read_text());state=json.loads(state_path.read_text())
    if matching_live(state['pid'],state['process_create_time']):raise ValueError('Live queue must not be reconciled')
    current=fingerprint(plan);same=current==state['identity']
    latest=max((r for r in state['jobs'].values() if r.get('run_dir')),key=lambda r:r.get('started_at',''))
    implementation=json.loads((Path(latest['run_dir'])/'implementation.json').read_text())
    changes=[str(p.relative_to(ROOT)) for p in framework_sources(plan)
             if implementation.get(str(p.relative_to(ROOT)))!=checksum(p)]
    if changes:raise ValueError('Inference source changed since latest attempt: '+str(changes))
    interruptions=[]
    for job,row in state['jobs'].items():
        if row['status']!='running':continue
        run=Path(row['run_dir']);events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
        begun=next(e for e in events if e['event']=='process_started');start=datetime.fromisoformat(begun['wall_time'])
        if matching_live(begun['pid'],start.timestamp()):raise ValueError('Matching worker still alive')
        if (run/'motion.npz').exists() or (run/'worker_result.json').exists():raise ValueError('Inspect partial/complete outputs before reconciliation')
        samples=[json.loads(line) for line in (run/'resources.jsonl').read_text().splitlines()]
        lower=max(s['elapsed_seconds'] for s in samples) if samples else 0.
        end_upper=boot if start<boot<=now else now
        interruptions.append({'job':job,'run_dir':str(run),'status':'interrupted_infrastructure',
            'reason':'No matching queue/worker remains after interruption and host reboot; required outputs absent',
            'observed_elapsed_lower_seconds':lower,'end_upper_bound':end_upper.isoformat(),
            'elapsed_upper_seconds':(end_upper-start).total_seconds(),'returncode':None})
    pending=[copy.deepcopy(j) for j in plan['jobs'] if state['jobs'][j['id']]['status'] in ('pending','running')]
    action={'original_plan':str(plan_path.relative_to(ROOT)),'original_queue':str(state_path.parent.relative_to(ROOT)),
            'identity_matches':same,'original_identity':state['identity'],'current_identity':current,
            'inference_source_changes_since_latest_attempt':changes,'interruptions':interruptions,'remaining_jobs':len(pending)}
    if args.apply:
        json_write(receipt_path.parent/(folder+'_state_before.json'),state)
        for record in interruptions:
            run=Path(record['run_dir']);manifest=json.loads((run/'manifest.json').read_text())
            manifest.update(status=record['status'],reason=record['reason'],interruption_timing=record)
            json_write(run/'manifest.json',manifest)
            event(run/'events.jsonl','queue_interruption_reconciled',**record)
            row=state['jobs'][record['job']];row.setdefault('attempts',[]).append(record)
            row.update(status='pending' if same else record['status'],reason=record['reason'])
        if same:
            state['status']='reconciled_for_resume'
            action.update(action='resume',plan=str(plan_path.relative_to(ROOT)),queue=str(state_path.parent.relative_to(ROOT)))
        else:
            target=directory/'recovery_plan.json'
            if target.exists():raise ValueError('Recovery plan exists')
            recovery={**plan,'jobs':pending,'reused':[],
                'recovery_parent':str(state_path.parent.relative_to(ROOT)),
                'recovery_scope':'New identity, not an identity override. Original completed records retained; pending task definitions copied without modification.',
                'recovery_source_check':changes}
            json_write(target,recovery);state['status']='continued_under_new_identity'
            action.update(action='run',plan=str(target.relative_to(ROOT)),queue=str((directory/'recovery_queue').relative_to(ROOT)))
        json_write(state_path,state)
    actions.append(action)
receipt={'recorded_at':now.isoformat(),'host_boot':boot.isoformat(),'actions':actions,
         'timing_scope':'Last resource observation is a lower bound; current host boot is a conservative termination upper bound. No exact exit time or exit code is fabricated.'}
if args.apply:json_write(receipt_path,receipt)
print(json.dumps(receipt,ensure_ascii=False))
