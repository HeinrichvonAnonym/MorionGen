#!/usr/bin/env python
"""Premise checks 2 and 3 of the section-14 inference registration on the smoke runs (build_inference_smoke.py).

Per run: the recorded task input (scenario.json / PRIMAL timing records) equals the job input, intervention_checks
(utils/motiogen/native_receipt_checks.py) with the native matrix row conventions, the queue receipt (device, worker
seconds) and the native timing records. No outcome metric is read.
Output: develop_logs/test_runs/20261109_inference_smoke/check.json (--round2 / --chois-retry: plan_<name>.json, queue_<name>, check_<name>.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.native_receipt_checks import intervention_checks  # noqa: E402

OUT = ROOT/'develop_logs/test_runs/20261109_inference_smoke'


def motion_fields(path):
    with np.load(path) as m:
        joints = m['joints']; fps = float(m['fps'])
    return {'frames': len(joints), 'fps': fps, 'finite': bool(np.isfinite(joints).all())}


def check_job(job, receipt):
    run = Path(receipt['run_dir']); rel = str(run.relative_to(ROOT))
    out = {'job': job['id'], 'run': rel, 'device': receipt.get('device'), 'worker_seconds': receipt.get('elapsed_seconds'),
           'status': receipt['status'], 'inference_row': job['inference_row']}
    if receipt['status'] != 'success':
        return out
    method = job['method']
    if method == 'primal':
        records = json.loads((run/'factorial/timing.json').read_text())
        setup = json.loads((run/'factorial/setup_timing.json').read_text()) if (run/'factorial/setup_timing.json').exists() else {}
        rows = []
        for r in records:
            row = {'method': 'primal', 'condition': {k: int(r[k]) for k in ('ground_snap', 'reprojection', 'inertialization')},
                   'calls': r.get('native_branch_calls'), 'run': rel,
                   **motion_fields(run/'factorial'/(r['variant']+'_canonical.npz'))}
            rows.append({'variant': r['variant'], 'initial_seed': r.get('initial_seed'), 'initial_history': r.get('initial_history'),
                         'generation_seconds': r['seconds'], 'intervention_checks': intervention_checks(row, ROOT)})
        out['input_matches_job'] = all(r['initial_seed'] == job['scene']['initial_seed'] and r['initial_history'] == job['scene']['initial_history']
                                       for r in rows)
        out['combinations'] = rows
        out['intervention_checks_passed'] = all(r['intervention_checks']['passed'] for r in rows)
        out['native_timing'] = {'setup': {k: v for k, v in setup.items() if k.endswith('_seconds')},
                                'generation_seconds_per_combination': [r['generation_seconds'] for r in rows]}
        return out
    scenario = json.loads((run/'scenario.json').read_text())
    intervention = json.loads((run/'intervention.json').read_text())
    keys = {'motionlcm': ('prompt', 'constraints'), 'omnicontrol': ('prompt', 'constraints'), 'dart': ('goal_start', 'goal_next'),
            'closd': ('goal_start', 'goal_next'), 'ardy': ('prompt', 'updates', 'native_root_path'),
            'chois': ('native_window_index', 'native_sequence')}[method]
    out['input_matches_job'] = {k: scenario.get(k) == job['scene'].get(k) for k in keys}
    if method == 'chois':
        sequences = list((run/'raw/chois_results/res_npz_files').rglob('*.npz'))
        with np.load(sequences[0], allow_pickle=False) as s:
            out['input_matches_job']['native_sequence_in_output'] = str(s['seq_name']) == job['scene']['native_sequence']
    condition = dict(intervention['condition'])
    if method == 'ardy':
        condition['history_long'] = int(condition.pop('history_frames') == 160)
    if method == 'closd':
        condition['execution_context'] = int(condition.pop('context_source') == 'simulation')
    row = {'method': method, 'condition': condition, 'calls': intervention.get('calls'), 'run': rel, **motion_fields(run/'motion.npz')}
    out['intervention_checks'] = intervention_checks(row, ROOT)
    out['intervention_checks_passed'] = out['intervention_checks']['passed']
    timing = {}
    for name in ('timing.json', 'trace.json'):
        if (run/name).exists():
            t = json.loads((run/name).read_text())
            if isinstance(t, dict):
                timing[name] = {k: v for k, v in t.items() if isinstance(v, (int, float)) and ('second' in k or k.endswith('_s'))}
    out['native_timing'] = timing
    return out


def main():
    suffix = next((s for s in ('round2', 'chois_retry') if '--'+s.replace('_', '-') in sys.argv), None)
    plan = json.loads((OUT/(f'plan_{suffix}.json' if suffix else 'plan.json')).read_text())
    state = json.loads((OUT/(f'queue_{suffix}' if suffix else 'queue')/'state.json').read_text())
    res = [check_job(j, state['jobs'][j['id']]) for j in plan['jobs']]
    lock = sorted({(r['job'].split('_')[1], r['device']) for r in res})
    report = {'queue_status': state.get('status'), 'devices_used': lock, 'runs': res,
              'all_success': all(r['status'] == 'success' for r in res),
              'all_inputs_match': all((all(r['input_matches_job'].values()) if isinstance(r['input_matches_job'], dict) else r['input_matches_job'])
                                      for r in res if r['status'] == 'success'),
              'all_intervention_checks_passed': all(r.get('intervention_checks_passed') for r in res if r['status'] == 'success')}
    (OUT/(f'check_{suffix}.json' if suffix else 'check.json')).write_text(json.dumps(report, indent=1))
    for r in res:
        print(r['job'], r['status'], 'device', r['device'], 'worker_s', round(r['worker_seconds'] or 0, 1), 'input', r.get('input_matches_job'),
              'checks', r.get('intervention_checks_passed'))
        if r.get('intervention_checks') and not r['intervention_checks_passed']:
            print('   ', r['intervention_checks'])
        print('   timing', json.dumps(r.get('native_timing'))[:400])
    print({k: v for k, v in report.items() if k != 'runs'})


if __name__ == '__main__':
    main()
