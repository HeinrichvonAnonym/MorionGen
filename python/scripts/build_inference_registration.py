#!/usr/bin/env python
"""Inference registration of the section-14 batch (design notes sections 2, 3, 5, 14; plan item 4).

Collects the registered rules (written here) and the premise-check results (read from their outputs, never retyped):
  config/inference_tasks_20261109.json (build_inference_tasks.py), develop_logs/test_runs/20261109_task_reachability/*,
  develop_logs/test_runs/20261109_inference_premises/hint_zscores.json, develop_logs/test_runs/20261109_inference_smoke/check*.json
Writes config/inference_registration_20261109.json. No outcome metric of any run is read.
"""
import hashlib
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT/'develop_logs/test_runs'
REACH = RUNS/'20261109_task_reachability'
SMOKE = RUNS/'20261109_inference_smoke'
TASKS = ROOT/'config/inference_tasks_20261109.json'
OUT = ROOT/'config/inference_registration_20261109.json'
MATRIX_WORKER_MEDIAN_S = 'develop_logs/test_runs/20261024_*/**/state.json, successful jobs, elapsed_seconds'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).relative_to(ROOT))


def load(path):
    return json.loads(Path(path).read_text())


def matrix_worker_medians():
    seconds = {}
    for f in RUNS.glob('20261024_*/**/state.json'):
        try:
            state = load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for job, receipt in state.get('jobs', {}).items():
            if receipt.get('status') == 'success' and receipt.get('elapsed_seconds'):
                seconds.setdefault(job.split('_')[0], []).append(receipt['elapsed_seconds'])
    return {m: round(statistics.median(v), 1) for m, v in seconds.items() if m != 'timing'}


def smoke_timing():
    out = {}
    for name, label in (('check.json', 'round 1'), ('check_round2.json', 'round 2'), ('check_chois_retry.json', 'CHOIS retry after freeing RAM')):
        if not (SMOKE/name).exists():
            continue
        for r in load(SMOKE/name)['runs']:
            m = r['job'].split('_')[1]
            out.setdefault(m, []).append({'round': label, 'job': r['job'], 'status': r['status'],
                                          'device': r['device'], 'worker_seconds': round(r['worker_seconds'], 1) if r['worker_seconds'] else None,
                                          'input_matches_job': r.get('input_matches_job'),
                                          'intervention_checks_passed': r.get('intervention_checks_passed')})
    return out


def main():
    tasks = load(TASKS); overhead = load(REACH/'overhead_limit.json'); reach = load(REACH/'reachability.json')
    shift = load(REACH/'reachability_origin_shift_+0.012.json'); hints = load(RUNS/'20261109_inference_premises/hint_zscores.json')
    smoke = smoke_timing(); matrix = matrix_worker_medians()
    bodies = {b: {'shoulder_height_m': {s: round(v['shoulder_body_m'][2], 4) for s, v in overhead['bodies'][b]['nominal']['sides'].items()},
                  'arm_length_m': {s: round(v['arm_length_m'], 4) for s, v in overhead['bodies'][b]['nominal']['sides'].items()},
                  'flat_limits_m': {k.split('/', 2)[2]: v['max_reachable_height_m'] for k, v in overhead['limits'].items() if k.startswith(b+'/flat')},
                  'heel_rise_limits_m': {k.split('/', 2)[2]: v['max_reachable_height_m'] for k, v in overhead['limits'].items() if k.startswith(b+'/heel_rise')}}
              for b in ('rajagopal', 'humanml')}
    hm = load(REACH/'humanml_reference_body.json'); joints = hm['standing']['joints']
    bodies['humanml']['dataset_standing_median'] = {
        'shoulder_height_m': {s: joints['shoulder_'+s[0]]['median_body_m'][2] for s in ('left', 'right')},
        'bone_lengths_m': hm['bone_lengths_m'], 'frames': hm['standing']['frames'], 'clips': hm['standing']['clips'],
        'note': 'shoulder_height_m / arm_length_m above are of the scaled Rajagopal model (torso scaled symmetrically); these are the '
                'HumanML3D joints themselves'}
    matrix.pop('kimodo', None)
    counts = tasks['counts']
    warm = {m: max((x['worker_seconds'] for x in v if x['status'] == 'success' and x['worker_seconds']), default=None) for m, v in smoke.items()}
    estimate = {m: {'runs_to_generate': counts[m]['to_generate'], 'seconds_per_run_basis': warm.get(m),
                    'gpu_hours': round(counts[m]['to_generate']*warm[m]/3600, 1) if warm.get(m) else None}
                for m in counts}
    reg = {
        'version': 'inference-registration-draft-1',
        'created': time.strftime('%Y-%m-%d %H:%M'),
        'status': 'draft; frozen together with config/motion_output_contract_20261108.json, config/evalkit_batch_protocol_20261108.json and '
                  'config/analysis_plan_20261108.json after user review (design section 14, plan item 5); no batch generation before the freeze',
        'builds_on': ['develop_logs/notes/20261108_batch_design_decisions.md sections 2, 3, 5, 14', 'config/inference_tasks_20261109.json'],
        'files': {'tasks': {'path': rel(TASKS), 'sha256': sha(TASKS), 'generator': tasks['generator']},
                  'registration_generator': {'path': rel(Path(__file__)), 'sha256': sha(Path(__file__))}},
        'user_decisions': {
            '2026-11-08 23:00 design section 14': 'sections 2, 3, 5 agreed; development / held-out split over target position, paraphrase and seed; '
                                                 'baseline held-out results viewed once; only three premise checks, none about method success',
            '2026-11-09 01:13-01:28 split': 'nested Latin square for the paraphrases (config/evalkit_batch_protocol_20261108.json user_decisions)',
            '2026-11-09 overhead height': 'option A with two changes: the overhead limit is also computed for the HumanML3D reference skeleton '
                                          '(the preprocessing target skeleton, no method output) by the same standard (feet flat, no step, 0.8 x arm '
                                          'length horizontally); overhead height = min of the four limits (Rajagopal and HumanML3D x front and '
                                          'side90) - 5 cm (约定), one height for both directions; both bodies recorded; nominal upper-arm elevation '
                                          'at that height checked to stay above 90 degrees (result: reach_geometry.elevation_check)',
            '2026-11-09 CLoSD entry': 'patch closd_native.py with outputs only (rotations_xyzw, dof_positions in native_physics.npz); before / '
                                      'after runs per execution context with the same seed must give bit-identical joints, also against the '
                                      '10-24 archive; rotations by the bone-vector method; committed separately (ccb50a3; verified, contract '
                                      'method_capability.closd.smoke)',
            '2026-11-09 03:07 CLoSD tasks': 'CLoSD native REACH is a root-goal task with fixed text: it keeps the 8 directions x 3 distances, '
                                            'covers no risk-zone reaching and has no text factor; STRIKE_PUNCH (never run in the native '
                                            'matrix) is a later extension that needs a new entry point, a smoke test and the input-reaches-'
                                            'the-model check; design section 2 revised by the user accordingly'},
        'tasks': {
            'per_method': {m: {'family': v['family'], 'tasks': len(v['tasks']), 'conditions': v['conditions'],
                               'held_out_positions': v['held_out_positions'], **({'no_held_out_position': v['no_held_out_position']} if 'no_held_out_position' in v else {})}
                           for m, v in tasks['methods'].items()},
            'kimodo': tasks['kimodo'],
            'native_matrix': 'tasks 0-11 of every method are the native matrix tasks (reproduction subset, design section 2)',
            'clock': 'paused only (约定，未经推演): the native matrix also ran DART, CLoSD and ARDY with a realtime clock; the batch does not',
            'closd': 'CLoSD runs its native REACH state, a root-goal task: the goal is a ground point and the text is the state-machine constant '
                     '"A person is walking." (baselines/CLoSD closd/env/tasks/closd_task.py lines 34-37); it gets 8 directions x 3 distances like '
                     'DART and no risk-zone hand targets. Its only hand-target state is STRIKE_PUNCH (wrist target 1-1.5 m away, text "A person is '
                     'punching with the * hand.", same file lines 42-45), never run in the native matrix: a later extension with a new entry '
                     'point, a smoke test and the input-reaches-the-model check, not this batch (user 2026-11-09 03:07)',
        },
        'reach_geometry': {
            'frame': tasks['frames']['reach'],
            'risk_zone_heights_m': tasks['risk_zone_heights_m'],
            'horizontal_reach': '0.8 x arm length from the shoulder (约定), nominal Rajagopal body; front = straight ahead at the shoulder, side90 = lateral',
            'overhead': {'rule': overhead['overhead_height']['rule'], 'height_m': overhead['overhead_height']['height_m'],
                         'argmin': overhead['overhead_height']['argmin'], 'design_value_m': 1.95,
                         'design_value_note': '1.95 m (design section 2, 约定) is above every flat-foot limit and was replaced', 'bodies': bodies,
                         'humanml_body': 'HumanML3D test-set reference skeleton (4648 clips; standing-frame selection thresholds 约定) fitted by '
                                         'segment-wise scaling of the Rajagopal model, Rajagopal joint ranges kept (约定): '
                                         'develop_logs/test_runs/20261109_task_reachability/humanml_reference_body.json'},
            'elevation_check': {'at_height_m': overhead['overhead_height']['height_m'], 'rajagopal': {k.split('/', 1)[1]: v for k, v in overhead['checks_at_height'].items() if k.startswith('rajagopal')},
                                'reading': 'minimum over all reaching poses found (multi-start SLSQP, local): upper arm against the vertical '
                                           '>= 91.2 deg in front and 105.3 deg at the side, i.e. above 90; relative to the thorax (the RULA '
                                           'reading) the side stays above 90 (102.7) but in front a backward trunk lean reaches the target with '
                                           '67.8 (right) / 76.1 (left) deg; the left-right asymmetry of a symmetric model shows the minimum is '
                                           'local, so the true thoracic minimum in front is at most 67.8 deg'},
            'reachability': {'source': rel(REACH/'reachability.json'), 'flat_unreachable': [t['id'] for t in reach['tasks'] if not t['flat']['reachable']],
                             'flat_max_residual_m': max(t['flat']['residual_m'] for t in reach['tasks'])},
            'origin_sensitivity': {'source': rel(REACH/'reachability_origin_shift_+0.012.json'), 'shift_m': shift['origin_shift_m'],
                                   'flat_unreachable': [t['id'] for t in shift['tasks'] if not t['flat']['reachable']]},
        },
        'text': {
            'reach_methods': {'templates_convention_not_derived': tasks['text_table']['templates_convention_not_derived'],
                              'levels': 'neutral ("forward", the native matrix sentence) and consistent (the zone phrase) for MotionLCM and OmniControl',
                              'rotation': tasks['split']['paraphrase_rotation'], 'consistent_phrases': tasks['text_table']['consistent_phrases']},
            'other_methods': {'closd': 'state-machine constant, no text factor (tasks.closd)', 'dart': 'fixed "walk" (native matrix)',
                              'ardy': '"A person walks forward." -> "A person walks backward." at 5 s (native matrix)', 'primal': 'no text input',
                              'chois': 'the OMOMO annotation of the window (native)'},
        },
        'split': {**tasks['split'], 'primal': tasks['methods']['primal'].get('held_out_rule'),
                  'enforcement': 'config/analysis_plan_20261108.json split_use: analysis scripts select development rows by this split and refuse '
                                 'held-out rows unless the final-comparison flag is given; the opening is recorded in the notes with time and commit',
                  'fallback': 'pending user (analysis plan pending_user.held_out_fallback)'},
        'seeds_and_sizes': {'seeds': tasks['seeds'], 'per_cell': 'every task x condition x text level with all 5 seeds; one run per row',
                            'rows': {m: {k: v[k] for k in ('rows', 'development', 'held_out')} for m, v in counts.items()},
                            'primal': counts['primal'].get('note')},
        'reuse': {'rule': 'an already-run row is reused only if its input is identical (method, task input, seed, condition, text) and its run '
                          'meets contract layer 4 (scan state ok or legacy); otherwise it is generated again; identical-but-not-reusable '
                          'rows (CLoSD: no rotations_xyzw; DART: no smplx_params) are regenerated and the old run is kept for reproduction',
                  'counts': {m: {k: v[k] for k in ('identical_seen', 'reuse_candidates', 'to_generate')} for m, v in counts.items()},
                  'kimodo': tasks['kimodo']},
        'official_evaluation_batch': {
            'protocols': {'motionlcm': 'HumanML3D test texts, spatial hints from the matching real motion', 'omnicontrol': 'same as MotionLCM',
                          'chois': 'OMOMO', 'dart': 'BABEL', 'kimodo': 'its own evaluator', 'closd': 'HumanML3D'},
            'per_switch_group': 'at least 512 windows per switch group; FID on all samples',
            'evalkit_subsample': '200 per group, stratified (约定，未经推演)',
            'scale': 'pending user, after the timing below (design section 5); OmniControl may run 2 groups only (design section 12)'},
        'premise_checks': {
            '1_task_valid_for_a_human': 'reach_geometry.reachability / overhead / origin_sensitivity: nominal body only, computed once',
            '2_input_reaches_the_model': {
                'hint_normalisation': {'formula': hints['formula'],
                                       'max_abs_z': {m: {'legacy (already run)': v['legacy_max_abs_z'], 'risk': v['risk_max_abs_z']} for m, v in hints['methods'].items()},
                                       'note': 'no clipping in the adapters; the largest |z| is the floor height (0.15 m against a dataset wrist mean '
                                               'of 0.98 m, sd 0.30 m)', 'source': 'develop_logs/test_runs/20261109_inference_premises/hint_zscores.json'},
                'smoke_receipts': 'scenario equals job input and intervention_checks pass (smoke below)'},
            '3_engineering': {
                'gpu_lock': 'per card: utils/motiogen/runtime.py gpu_lock locks run_root/.gpu-<device>.lock; utils/motiogen/experiment_queue.py '
                            'sets gpu_index and CUDA_VISIBLE_DEVICES per lane (lines 61, 70); both cards ran in parallel in the smoke',
                'ram_budget': 'the RAM budget of the resource review (utils/motiogen/runtime.py budgets: min(ram_max, available - reserve)) is '
                              'machine-wide, checked once at job start and not coordinated by the per-card lock; CHOIS (reviewed bound 10 GiB, '
                              'reserve 6 GiB) was refused in both rounds (15.7 GiB available) and ran after the user freed memory (21.0 GiB); '
                              'the queue does not retry a refused job',
                'resource_gate_rule': 'a resource_gate refusal is an admission refusal (no model code ran, no output): the job is put back in '
                                      'the queue until admitted and is not a generation failure; generation failures stay recorded and are '
                                      'never replaced (contract layer 6) (约定)',
                'smoke': smoke, 'smoke_source': 'develop_logs/test_runs/20261109_inference_smoke (build_inference_smoke.py, check_inference_smoke.py)'},
            'not_done': 'no check of whether a method reaches a target, falls, or stretches bones; no task was filtered (design section 14 item 2)'},
        'compute_estimate': {'basis': 'slower of the two smoke runs per method (worker seconds incl. process start, both cards busy); 约定, a '
                                      'planning figure only; CHOIS timed on one card with the other idle',
                             'native_matrix_worker_median_s': matrix, 'native_matrix_source': MATRIX_WORKER_MEDIAN_S, 'per_method': estimate,
                             'total_gpu_hours': round(sum(v['gpu_hours'] or 0 for v in estimate.values()), 1),
                             'two_cards_wall_hours': round(sum(v['gpu_hours'] or 0 for v in estimate.values())/2, 1)},
        'conventions_not_derived': ['risk heights 0.15 / 0.5 m and the 5 cm overhead margin', '0.8 x arm length', 'task origin: hip-centre midpoint '
                                    'of the nominal body for the pelvis joint', 'paraphrase templates and zone phrases', 'held-out positions '
                                    '(interleaved choice)', 'paused clock only', 'HumanML3D standing-frame thresholds and scaling choices',
                                    '200-row evalkit subsample', 'compute estimate'],
        'pending_user': ['CHOIS and ARDY have no held-out position (their added windows / all 8 directions were already run in V2 / M0) and '
                         'are held out only through seed 24',
                         'overhead front: relative to the thorax the upper arm can stay below 90 deg (reach_geometry.elevation_check)',
                         'paused clock only', 'official evaluation batch scale', 'held_out_fallback (analysis plan)'],
    }
    OUT.write_text(json.dumps(reg, indent=1, ensure_ascii=False)+'\n')
    print(json.dumps({k: reg[k] for k in ('compute_estimate',)}, indent=1)[:3000])


if __name__ == '__main__':
    main()
