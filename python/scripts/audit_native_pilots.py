"""Inspect native pilot receipts without treating process success as scientific success."""
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.motion import load_motion
from motiogen.runtime import json_write, checksum

base=ROOT/'develop_logs/test_runs/20261024_native_pilots'
rows=[]
for queue in ['queue','omni_queue_retry1','chois_queue_retry1','dart_queue','ardy_queue_cfg_corrected_fair']:
    path=base/queue/'state.json'
    if not path.exists():
        continue
    state=json.loads(path.read_text())
    for job,receipt in state['jobs'].items():
        if receipt['status']!='success':
            continue
        run=Path(receipt['run_dir'])
        manifest=json.loads((run/'manifest.json').read_text())
        motion=load_motion(run/'motion.npz')
        intervention=json.loads((run/'intervention.json').read_text())
        metrics=json.loads((run/'metrics.json').read_text())['metrics']
        row={'method':manifest['baseline'],'job':job,'run':str(run.relative_to(ROOT)),
             'seed':manifest['seed'],'scene':manifest['scenario'],'condition':intervention['condition'],
             'calls':intervention['calls'],'frames':len(motion['joints']),'fps':float(motion['fps']),
             'finite':bool(np.isfinite(motion['joints']).all()),
             'motion_sha256':checksum(run/'motion.npz'),'metrics':metrics}
        if (run/'worker_result.json').exists():
            timing=json.loads((run/'worker_result.json').read_text())
            row['rtf']=timing.get('real_time_factor')
            row['peak_cuda_allocated_bytes']=timing.get('peak_cuda_allocated_bytes')
        rows.append(row)
groups={}
for row in rows:
    key=(row['method'],row['seed'],json.dumps(row['scene'],sort_keys=True))
    groups.setdefault(key,[]).append(row)
checks=[]
for (method,seed,_),group in groups.items():
    expected={'motionlcm':2,'omnicontrol':4,'chois':2,'dart':2,'ardy':4}[method]
    checks.append({'method':method,'seed':seed,'conditions':len(group),
        'expected_conditions':expected,'complete_block':len(group)==expected,
        'same_fps_and_frames':len({(r['fps'],r['frames']) for r in group})==1,
        'all_finite':all(r['finite'] for r in group),
        'distinct_motion_files':len({r['motion_sha256'] for r in group}),
        'interpretation':('One complete paired pilot block' if len(group)==expected else 'Incomplete paired pilot block')+'; no scenario-level confidence interval or population claim'})
json_write(base/'audited_pilots.json',{'rows':rows,'checks':checks})
for row in rows:
    picked={key:value.get('value') for key,value in row['metrics'].items()
            if key in ['hand_position_error','jerk_rms','foot_skate_ratio','trunk_inclination','contact_distance']}
    print(json.dumps({'method':row['method'],'condition':row['condition'],'calls':row['calls'],'metrics':picked},ensure_ascii=False))
print(json.dumps(checks,ensure_ascii=False))
