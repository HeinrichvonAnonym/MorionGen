"""Register a four-cell diagnostic; primary matrix and inference stay unchanged."""
import copy
import argparse
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'utils'))
from motiogen.config import ROOT, load_config, data_root
from motiogen.runtime import json_write

parser = argparse.ArgumentParser()
parser.add_argument('--extension', action='store_true', help='Complete the other five same-planner discrepant cases')
args = parser.parse_args()
base = ROOT / 'develop_logs/test_runs'
destination = base / '20261024_closd_marker_probe' / ('extension_plan.json' if args.extension else 'plan.json')
if destination.exists():
    raise ValueError('Refusing to replace a registered diagnostic plan')
analysis = json.loads((base / '20261024_native_matrix_analysis/analysis.json').read_text())
pairs = []
for pair in analysis['clock_pairs']:
    if pair['method'] != 'closd' or pair['exact_same_coordinates']:
        continue
    replay_frame = json.loads((ROOT / pair['realtime_run'] / 'trace.json').read_text())['goal_applied_frame']
    # DiP replans every 60 frames. Both updates must be after frame120 and
    # before/equal frame180, so their first consuming planner step is identical.
    if not 120 < replay_frame <= 180:
        continue
    pilot = pair['task'] == 5 and pair['seed'] == 22 and json.loads(pair['condition'])['execution_context'] == 1
    if pilot != args.extension:
        pairs.append(pair)
assert len(pairs) == (5 if args.extension else 1)
if args.extension:
    pilot_analysis = json.loads((base / '20261024_closd_marker_probe/analysis.json').read_text())
    assert pilot_analysis['complete_cases'] == 1
    assert pilot_analysis['cases'][0]['native_replays_both_exact'] and pilot_analysis['cases'][0]['isolated_schedules_exact']
template = json.loads((base / '20261024_closd_pilot/plan.json').read_text())
original = template['jobs'][0]
data = data_root(load_config())
prefix = data / 'envs/motiogen-closd'
environment = {
    'MOTIONGEN_CLOSD_ENV_READY': '1', 'LD_LIBRARY_PATH': str(prefix / 'lib') + ':' + os.environ.get('LD_LIBRARY_PATH', ''),
    'PATH': str(prefix / 'bin') + ':' + os.environ['PATH'], 'MAX_JOBS': '2',
    'TORCH_EXTENSIONS_DIR': str(data / 'staged_dependencies/torch_extensions'),
    'HF_HOME': str(data / 'outputs/assets/closd-hf-cache'), 'TOKENIZERS_PARALLELISM': 'false'}
jobs = []
for pair in pairs:
    reference = ROOT / pair['paused_run']
    scene = json.loads((reference / 'scenario.json').read_text())
    replay_frame = json.loads((ROOT / pair['realtime_run'] / 'trace.json').read_text())['goal_applied_frame']
    context = 'simulation' if json.loads(pair['condition'])['execution_context'] else 'prediction'
    for target_frame in [150, replay_frame]:
        # All diagnostic jobs share physical GPU0, including the completed pilot.
        for marker in ['native', 'isolated']:
            job = copy.deepcopy(original)
            job.update(id=f"closd_marker_task{pair['task']}_seed{pair['seed']}_{context}_frame{target_frame}_{marker}", seed=pair['seed'],
                       environment=environment, estimated_seconds=90)
            job['scene'] = {**scene, 'name': 'diagnostic_marker_schedule_replay', 'clock': 'paused',
                            'goal_update_frame': target_frame, 'marker_collision': marker,
                            'reference_paused': pair['paused_run'], 'reference_realtime': pair['realtime_run'],
                            'selection': 'All six cross-clock pairs diverging before the same planner update; post-hoc diagnostic, not a population estimate'}
            job['command'] = [str(prefix / 'bin/python'), str(ROOT / 'python/scripts/closd_marker_probe.py'),
                '--probe-goal-frame', str(target_frame), '--probe-marker', marker,
                '--run-dir', '{run_dir}', '--context-source', context, '--frames', '600',
                '--seed', str(pair['seed']), '--clock', 'paused', '--goal-start', *map(str, scene['goal_start']),
                '--goal-next', *map(str, scene['goal_next'])]
            job['required_outputs'] += ['{run_dir}/marker_probe.json', '{run_dir}/trace.json', '{run_dir}/weight_binding.json']
            jobs.append(job)
random.Random(20261024).shuffle(jobs)
template.update(jobs=jobs, devices=[0],
                scope='Post-hoc diagnostic 2x2: fixed-frame target replay x native/isolated marker collider. Complete all six same-planner discrepant cases; pilot reused. No main matrix replacement or H claims.',
                selection_cutoff=analysis['updated_at'])
template['identity_files'] += [str(ROOT / 'python/scripts/closd_marker_probe.py'),
                             str(ROOT / 'python/scripts/closd_native.py'),
                             str(ROOT / 'utils/motiogen/closd_native.py'),
                             str(data / 'baselines/CLoSD/closd/data/assets/urdf/location_marker.urdf')]
json_write(destination, template)
print(destination)
