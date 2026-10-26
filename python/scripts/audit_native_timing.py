"""Separate cold/setup and repeated native compute from complete worker time."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'utils'))
from motiogen.config import ROOT
from motiogen.runtime import json_write
base=ROOT/'develop_logs/test_runs/20261024_native_timing';rows=[]
for name in ['queue','queue_missing_omni']:
    state=json.loads((base/name/'state.json').read_text())
    for job,receipt in state['jobs'].items():
        if receipt['status']!='success':continue
        path=Path(receipt['run_dir']);manifest=json.loads((path/'manifest.json').read_text())
        result=json.loads((path/'worker_result.json').read_text());intervention=json.loads((path/'intervention.json').read_text())
        events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
        row={'method':manifest['baseline'],'job':job,'condition':intervention['condition'],'run':str(path.relative_to(ROOT)),
             'worker_seconds':receipt['elapsed_seconds'],'peak_torch_gib':result.get('peak_cuda_allocated_bytes',0)/2**30,
             'model_load_seconds':result.get('timing',{}).get('model_load',{}).get('mean_seconds')}
        sampler=path/'native_sampler_timing.json'
        if sampler.exists():
            raw=json.loads(sampler.read_text());samples=[r['seconds'] for r in raw['repetitions'] if not r['warmup']]
            row.update(cold_seconds=raw['cold_sampler_seconds'],scope=raw['scope'])
        else:
            samples=result['timing']['generation']['samples_seconds']
            cold=[r['seconds'] for r in events if r['event']=='generation_finished' and r.get('cold')]
            row.update(cold_seconds=cold[0] if cold else None,scope='Native generation including nested text call; excludes loading, plotting, video, evaluation and queue wait')
        row.update(repeats=len(samples),samples_seconds=samples,steady_mean_seconds=float(np.mean(samples)),
            steady_p50_seconds=float(np.quantile(samples,.5)),steady_p95_seconds=float(np.quantile(samples,.95)))
        rows.append(row)
primal=[]
primal_base=base.parent/'20261024_primal_timing'
if (primal_base/'queue/state.json').exists():
    plan=json.loads((primal_base/'plan.json').read_text())
    state=json.loads((primal_base/'queue/state.json').read_text())
    analysis=json.loads((base.parent/'20261024_native_matrix_analysis/analysis.json').read_text())
    reference_rows=list(analysis['rows'])
    original=base.parent/'20261025_source_replays/original_analysis.json'
    if original.exists():reference_rows+=json.loads(original.read_text())['rows']
    for job in plan['jobs']:
        receipt=state['jobs'][job['id']]
        if receipt['status']!='success':continue
        run=Path(receipt['run_dir']);reference=ROOT/job['reference_run']
        binding=json.loads((run/'weight_binding.json').read_text())
        reference_identity=next(r['weight_identity'] for r in reference_rows if r['run']==job['reference_run'])
        same_weights=binding['after_run_unchanged'] and binding['identity']==reference_identity
        with np.load(run/'factorial/initial_history.npz') as new,np.load(reference/'factorial/initial_history.npz') as old:
            same_input=set(new.files)==set(old.files) and all(np.array_equal(new[k],old[k]) for k in new.files)
        records=json.loads((run/'factorial/timing.json').read_text())
        if len(records)!=8:raise ValueError('PRIMAL timing replay must cover all eight conditions')
        for record in records:
            variant=record['variant']
            with np.load(run/'factorial'/f'{variant}_canonical.npz') as new,np.load(reference/'factorial'/f'{variant}_canonical.npz') as old:
                same_motion=np.array_equal(new['joints'],old['joints']) and np.array_equal(new['fps'],old['fps'])
            total=record['seconds'];wait=record['paced_wait_seconds'];compute=record['nonwait_generation_seconds']
            valid=np.isfinite([total,wait,compute]).all() and min(total,wait,compute)>=0 and np.isclose(total,wait+compute,rtol=0,atol=1e-9)
            valid=bool(valid and np.isclose(wait,sum(t['paced_wait_seconds'] for t in record['primitive_clock_trace']),rtol=0,atol=1e-9))
            passed=bool(same_weights and same_input and same_motion and valid)
            primal.append({'clock':record['clock'],'variant':variant,'total_seconds':total,
                'paced_wait_seconds':wait,'nonwait_seconds':compute,'nonwait_rtf':compute/(record['frames']/record['fps']),
                'same_weights':same_weights,'same_input':same_input,'exact_same_motion':same_motion,'passed':passed,
                'run':str(run.relative_to(ROOT)),'reference_run':job['reference_run'],'scope':record['nonwait_scope']})
        if not all(r['passed'] for r in primal):raise ValueError('PRIMAL direct timing or output-equivalence audit failed')
analysis=json.loads((base.parent/'20261024_native_matrix_analysis/analysis.json').read_text())
primary=[];timing_cache={}
for row in analysis['rows']:
    if row['method']!='primal':continue
    if row['run'] not in timing_cache:
        timing_cache[row['run']]={r['variant']:r for r in json.loads((ROOT/row['run']/'factorial/timing.json').read_text())}
    record=timing_cache[row['run']][row['variant']]
    if 'nonwait_generation_seconds' not in record:continue
    total=record['seconds'];wait=sum(t['paced_wait_seconds'] for t in record['primitive_clock_trace'])
    nonwait=total-wait;rtf=nonwait/(record['frames']/record['fps'])
    expected={'paced_wait_seconds':wait,'nonwait_generation_seconds':nonwait,'nonwait_generation_rtf':rtf}
    valid=np.isfinite([total,wait,nonwait,rtf]).all() and min(total,wait,nonwait)>=0
    valid=bool(valid and all(k in row['values'] and np.isclose(row['values'][k],v,rtol=0,atol=1e-9) for k,v in expected.items()))
    primary.append({'run':row['run'],'variant':row['variant'],'clock':row['clock'],'task':row['task'],'seed':row['seed'],
                    'passed':valid,'total_seconds':total,**expected})
primary_summary=[{'clock':clock,'checked_conditions':sum(r['clock']==clock for r in primary),
    'passed_conditions':sum(r['clock']==clock and r['passed'] for r in primary),'expected_conditions':480} for clock in ['paused','realtime']]
json_write(base/'analysis.json',{'rows':rows,'primal_clock_decomposition':primal,
    'primary_analysis_cutoff':analysis['updated_at'],'primal_primary_decomposition':primary,'primal_primary_summary':primary_summary,
    'primal_primary_scope':'Current accepted primary rows only. Sum recorded per-primitive pacing intervals independently, subtract from generation wall duration, and check the derived row values. Missing historical timers are absent, not imputed. Full frozen-source replays supply 480 conditions per clock without adding independent motion samples.',
    'primal_scope':'Standing seed20 replay only; sixteen condition-clock observations are timing evidence, not additional independent motion samples. Measured wait is subtracted from complete native generation wall time; nonwait includes CPU/GPU work and scheduling, not pure CUDA kernel time.',
    'interpretation':'One warmup excluded; three within-process repeated calls. Descriptive latency quantiles, not independent scene samples or service-level tail guarantees. Hardware/OS contention remains in measured wall compute.'})
print(json.dumps({'completed_conditions':len(rows),'primal_timing_replays':len(primal),'primal_passed':sum(r['passed'] for r in primal),'primal_primary':primary_summary}))
