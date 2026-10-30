"""Separate shared-source retargeting, native tracking and RFC interventions."""
import json
from pathlib import Path
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'utils'))
from motiogen.runtime import checksum, json_write

BASE = ROOT / 'develop_logs/test_runs/20261028_supplement'
retarget_path = BASE / 'phc_common_references/retarget_audit.json'
retarget = {r['source']: r for r in json.loads(retarget_path.read_text())['rows']}
inputs = [retarget_path]
rows = []

def errors(pred, reference):
    if pred.shape != reference.shape or not np.isfinite(pred).all() or not np.isfinite(reference).all():
        raise ValueError('Invalid tracking arrays')
    e = np.linalg.norm(pred - reference, axis=-1)
    return dict(mean_joint_error_m=float(e.mean()), p95_joint_error_m=float(np.percentile(e, 95)),
                root_error_m=float(e[:, 0].mean()), evaluated_frames=len(e))

for name in ['20261028_102935_uhc_tracking_3e35f23d', '20261028_103019_uhc_tracking_96ffc470']:
    run = ROOT / 'develop_logs/test_runs' / name
    inputs.append(run / 'tracking.json')
    for original in json.loads((run / 'tracking.json').read_text())['rows']:
        folder = Path(original['output'])
        result = joblib.load(folder / 'native_result.pkl')
        inputs.extend([folder / 'native_result.pkl', folder / 'execution_trace.npz'])
        trace = np.load(folder / 'execution_trace.npz')
        # eval_seq includes the initial pose; exclude it for post-action comparison.
        pred = result['pred_jpos'].reshape(-1, 24, 3)[1:]
        target = result['gt_jpos'].reshape(-1, 24, 3)[1:]
        assert len(pred) == original['steps']
        rows.append(dict(method='UHC', source=original['source'], rfc=bool(original['rfc_enabled']),
            **errors(pred, target), completion_fraction=original['completion'],
            early_termination=original['completion'] < 1, inference_device='cpu',
            observed_contact_frame_fraction=float(np.mean(trace['contacts'] > 0)),
            residual_root_force_component_rms=original['qfrc_root_force_rms'],
            total_inference_seconds=original['seconds'],
            retarget_mean_joint_error_m=0.,
            retarget_scope='Native UHC subject reference; zero denotes no additional cross-engine conversion, not zero AMASS-to-SMPL reconstruction error.',
            evidence=str(folder)))

for statefile in [BASE/'phc_common_tracking/queue_v2/state.json', BASE/'phc_common_tracking/remaining_queue/state.json']:
    inputs.append(statefile)
    state = json.loads(statefile.read_text())
    for job, status in state['jobs'].items():
        if status['status'] != 'success':
            rows.append(dict(method='PHC', job=job, process_status=status['status']))
            continue
        run = Path(status['run_dir'])
        inputs.extend([run/'tracking.json', run/'execution_trace.npz'])
        record = json.loads((run/'tracking.json').read_text())
        trace = np.load(run/'execution_trace.npz')
        source = record['source']
        force = trace['contact_forces']
        rows.append(dict(method='PHC', source=source, rfc=False,
            **errors(trace['joints'], trace['reference_joints']),
            completion_fraction=record['completion_fraction'],
            early_termination=record['steps'][-1]['native_early_termination'],
            native_termination_distance_m=record['native_termination_distance_m'],
            inference_device='cuda',
            observed_contact_frame_fraction=float(np.mean(np.any(np.linalg.norm(force, axis=-1)>0, axis=-1))),
            total_inference_seconds=None,
            step_instrumented_seconds=sum(r['seconds'] for r in record['steps']),
            timing_scope='PHC action computation is outside step timer; not equivalent to UHC full inference time.',
            height_correction_applied=bool(record['native_height_corrections']),
            retarget_mean_joint_error_m=retarget[source]['retarget_mean_joint_error_m'],
            retarget_scope='UHC subject geometry to PHC neutral geometry; separate from execution tracking error.',
            evidence=str(run)))

assert len(rows) == 24 and len({r['source'] for r in rows}) == 8
paired = []
for source in sorted(retarget):
    off = next(r for r in rows if r['source']==source and r['method']=='UHC' and not r['rfc'])
    on = next(r for r in rows if r['source']==source and r['method']=='UHC' and r['rfc'])
    # Full observed means have unequal exposure when failure occurs. Keep both;
    # common-prefix error is the meaningful equal-duration paired quantity.
    sequences = []
    for row in [off, on]:
        r = joblib.load(Path(row['evidence'])/'native_result.pkl')
        sequences.append(np.linalg.norm((r['pred_jpos']-r['gt_jpos']).reshape(-1,24,3)[1:], axis=-1))
    n = min(map(len, sequences))
    paired.append(dict(source=source, common_post_action_frames=n,
        rfc_on_minus_off_common_prefix_error_m=float((sequences[1][:n]-sequences[0][:n]).mean()),
        rfc_on_minus_off_completion=on['completion_fraction']-off['completion_fraction'],
        rfc_on_root_force_component_rms=on['residual_root_force_component_rms']))

json_write(BASE/'common_tracking_analysis.json', dict(rows=rows, paired_rfc=paired,
    inputs={str(p): checksum(p) for p in inputs},
    scope=['Eight independent source motions, not eight seeds.',
           'Native failure thresholds differ; early termination is not an independently classified physical fall.',
           'Contact presence is not contact correctness. No task-constraint preservation claim without task labels.',
           'Different body geometry, policy training, inference devices and engines preclude pure architecture causal comparison.',
           'Turning RFC off intervenes on a fixed RFC-trained policy; it does not evaluate a separately trained no-RFC policy.']))
print(json.dumps({'rows':len(rows), 'sources':len(retarget), 'phc_complete':sum(r['method']=='PHC' and r['completion_fraction']==1 for r in rows)}))
