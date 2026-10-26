"""Scientific receipt checks, separate from process exit and task success."""
import json
from pathlib import Path


def intervention_checks(row, root):
    method = row['method']; condition = row['condition']; calls = row['calls']
    checks = []

    def expect(name, passed, actual):
        checks.append({'check': name, 'passed': bool(passed), 'actual': actual})

    shapes = {'motionlcm': (100, 20), 'omnicontrol': (196, 20), 'chois': (120, 30),
              'ardy': (400, 20), 'dart': (600, 30), 'primal': (600, 30), 'closd': (600, 30)}
    expect('native frame count and FPS', (row['frames'], row['fps']) == shapes[method], [row['frames'], row['fps']])
    expect('finite joint coordinates', row['finite'], row['finite'])
    if calls is None:
        expect('executed intervention receipt', False, 'No branch counters in this historical run')
        return {'passed': False, 'checks': checks}
    if method == 'motionlcm':
        expect('ControlNet hook executed with hints retained', calls.get('controlnet') == 1, calls)
    elif method == 'omnicontrol':
        expect('both control hook sites executed', calls.get('realism') == 2000 and calls.get('spatial') == 1000, calls)
    elif method == 'chois':
        expect('native sampler and optional geometric guidance', calls.get('native_sampler') == 1 and calls.get('geometry_guidance', 0) == 9*condition['guidance'], calls)
    elif method == 'ardy':
        receipt = json.loads((Path(root)/row['run']/'intervention.json').read_text())
        expect('50 native horizons and 500 denoiser calls', calls.get('native_autoregressive_step') == 50 and calls.get('denoiser_with_text_and_spatial_guidance') == 500, calls)
        expect('observed text and spatial CFG weights', receipt.get('observed_denoiser_guidance') == [2., 2.], receipt.get('observed_denoiser_guidance'))
        expect('correction executed iff enabled', calls.get('native_motion_correction', 0) == 50*condition['correction'], calls)
    elif method == 'primal':
        expected = {'initial_ground_snap': condition['ground_snap'], 'ground_projection': 43*condition['ground_snap'],
                    'keypoint_reprojection': 42*condition['reprojection'], 'inertialization': 43*condition['inertialization']}
        expect('native geometric branch counts', all(calls.get(k, 0) == v for k, v in expected.items()), {'observed': calls, 'expected': expected})
    elif method == 'dart':
        trace = json.loads((Path(root)/row['run']/'trace.json').read_text())
        expect('native state update retained', calls.get('native_step') == 75 and calls.get('native_state_transition') == 75, calls)
        changed = any(step['goal'] != trace['steps'][0]['goal'] for step in trace['steps'])
        expect('goal observation changed iff enabled', changed == bool(condition['goal_update']), {'changed': changed, 'applied_frame': trace['applied_update_frame']})
    elif method == 'closd':
        trace = json.loads((Path(root)/row['run']/'trace.json').read_text())
        expect('600 native physical execution steps', calls.get('native_physics_steps') == 600 and len(trace['steps']) == 600, calls)
        expected_prediction = lambda entry: not condition['execution_context'] and entry['native_frame_idx'] > 0
        expect('planner actually consumed assigned source', all((entry['source'] == 'prediction') == expected_prediction(entry) for entry in trace['planner']),
               {'simulation': sum(p['source'] == 'simulation' for p in trace['planner']), 'prediction': sum(p['source'] == 'prediction' for p in trace['planner'])})
        expect('planner counter agrees with trace', calls.get('planner_context') == len(trace['planner']), calls)
    return {'passed': all(c['passed'] for c in checks), 'checks': checks,
            'scope': 'Executed native hook / branch receipts, not process success or a claim of task success'}
