#!/usr/bin/env python
"""Flat-foot landmark references of the 12 CHOIS windows added by the inference registration (config/inference_tasks_
20261109.json methods.chois), by the procedure of the 12 registered windows of config/evalkit_floor_reference_20261107.json:
window betas, gender and sequence from the CHOIS processed test windows selected as python/scripts/export_chois_posture_
calibration.py selects them; gendered SMPL-X zero pose, one pose per side with that foot's mesh sole on z = 0
(python/scripts/e_v2_floor_poses.py smplx_body() and save(), rule mesh); landmark heights through the evalkit front end
(python/scripts/evalkit_floor_refs.py measure()). world_floor_m 0.0: OMOMO world floor (config/motion_world/chois.json).

  poses   (motiogen-dart python)  poses of the added windows and of two registered windows (positive control)
  measure (OpenSim python)        config/evalkit_floor_reference_chois_added_20261109.json
Checks written with it: the registered windows rebuilt by the same steps reproduce their frozen reference_m; every added
window's sequence equals the task table's native_sequence; lowest foot-joint height of each source window (first 72 motion
columns = 24 joint positions, Z up) for the added windows against the registered ones.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA = Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
ASSETS = DATA/'outputs/assets/chois/processed_data'
WORK = DATA/'outputs/evalkit_20261109/chois_added_floor_refs'
FROZEN = ROOT/'config/evalkit_floor_reference_20261107.json'
OUT = ROOT/'config/evalkit_floor_reference_chois_added_20261109.json'
CONTROL = ['0', '30']
FEET = (10, 11)


def windows():
    import joblib
    d = joblib.load(ASSETS/'cano_test_diffusion_manip_window_120_joints24.p')
    return [v for v in d.values() if (ASSETS/'omomo_text_anno_json_data'/(v['seq_name']+'.json')).exists()
            and len(v['motion']) >= 120 and v['start_t_idx'] == 0]


def poses():
    import e_v2_floor_poses as efp
    efp.OUT = WORK/'poses'
    tasks = json.loads((ROOT/'config/inference_tasks_20261109.json').read_text())['methods']['chois']['tasks']
    frozen = json.loads(FROZEN.read_text())['methods']['chois']['windows']
    added = {str(t['native_window_index']): t['native_sequence'] for t in tasks if str(t['native_window_index']) not in frozen}
    ws = windows()
    records = {}
    for w in sorted(set(added) | set(CONTROL), key=int):
        v = ws[int(w)]
        seq, gender = str(v['seq_name']), str(np.asarray(v['gender']).item())
        if w in added and seq != added[w]:
            raise SystemExit(f'window {w}: processed sequence {seq} != task native_sequence {added[w]}')
        J, V = efp.smplx_body(ASSETS/f'smpl_all_models/smplx/SMPLX_{gender.upper()}.npz', np.asarray(v['betas']).reshape(-1))
        efp.save(f'chois_w{w}', J, efp.HUMANML, {'method': 'chois', 'window': int(w), 'sequence': seq, 'gender': gender,
                                                 'body': 'gendered SMPL-X, window betas, zero pose',
                                                 'convention': 'mesh sole at z=0 (OMOMO world; CHOIS canonicalisation rotates about z and zeroes xy only)'},
                 'mesh', vertices=V)
        foot_z = np.asarray(v['motion'])[:, :72].reshape(-1, 24, 3)[:, FEET, 2]
        records[w] = {'sequence': seq, 'gender': gender, 'added': w in added, 'source_lowest_foot_joint_z_m': float(foot_z.min())}
    for w in frozen:
        if w not in records:
            foot_z = np.asarray(ws[int(w)]['motion'])[:, :72].reshape(-1, 24, 3)[:, FEET, 2]
            records[w] = {'sequence': str(ws[int(w)]['seq_name']), 'added': False, 'source_lowest_foot_joint_z_m': float(foot_z.min())}
    (WORK/'windows.json').write_text(json.dumps(records, indent=1))
    print(json.dumps({w: r['source_lowest_foot_joint_z_m'] for w, r in records.items()}))


def measure():
    import evalkit_floor_refs as efr
    efr.POSES, efr.WORK = WORK/'poses', WORK/'refs'
    records = json.loads((WORK/'windows.json').read_text())
    frozen = json.loads(FROZEN.read_text())['methods']['chois']['windows']
    got = {}
    for w in [w for w, r in records.items() if r['added'] or w in CONTROL]:
        for s in 'lr':
            got[f'chois_w{w}@{s}'] = efr.measure(f'chois_w{w}@{s}')[1]
    meta = {p.stem: json.loads(str(np.load(p)['meta'])) for p in (WORK/'poses').glob('*.npz')}
    entry = lambda w: {'world_floor_m': 0.0, 'pose': {s: f'chois_w{w}@{s}' for s in 'lr'},
                       'body': 'gendered SMPL-X, window betas, zero pose',
                       'convention': 'mesh sole at z=0 (OMOMO world; CHOIS canonicalisation rotates about z and zeroes xy only)',
                       'placement': {s: {k: meta[f'chois_w{w}@{s}'][k] for k in ('placement_rule', 'placement_value_m', 'shift_m')} for s in 'lr'},
                       'sequence': records[w]['sequence'], 'gender': records[w]['gender'],
                       'reference_m': {s: got[f'chois_w{w}@{s}']['reference_m'][s] for s in 'lr'},
                       'fit': {s: {k: got[f'chois_w{w}@{s}'][k] for k in ('feet', 'ik_marker_error')} for s in 'lr'}}
    control = {w: max(abs(entry(w)['reference_m'][s][k]-frozen[w]['reference_m'][s][k]) for s in 'lr' for k in ('heel', 'toe'))
               for w in CONTROL}
    low = lambda added: [r['source_lowest_foot_joint_z_m'] for r in records.values() if r['added'] == added]
    out = {'created': time.strftime('%Y-%m-%d %H:%M'), 'method': 'chois',
           'scope': 'the 12 CHOIS windows of config/inference_tasks_20261109.json that config/evalkit_floor_reference_20261107.json '
                    'does not hold; same layout as its methods.chois.windows entries; the batch manifest gives each such row '
                    'reference = reference_m - world_floor_m and floor_m = world_floor_m (python/scripts/evalkit_batch.py resolve())',
           'procedure': 'python/scripts/chois_added_window_floor_refs.py: poses as e_v2_floor_poses.py (CHOIS, rule mesh), '
                        'reference_m as evalkit_floor_refs.py measure() (3 s hold at 30 fps, mass 75 kg, floor 0, median over scored frames)',
           'positive_control': {'windows': CONTROL, 'max_abs_reference_m_difference_to_frozen': control},
           'source_check': {'definition': 'lowest foot-joint (SMPL toe joints 10, 11) z over each processed source window, metres',
                            'registered_windows': [min(low(False)), max(low(False))], 'added_windows': [min(low(True)), max(low(True))]},
           'windows': {w: entry(w) for w, r in sorted(records.items(), key=lambda x: int(x[0])) if r['added']}}
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    print(json.dumps({k: out[k] for k in ('positive_control', 'source_check')}))


if __name__ == '__main__':
    {'poses': poses, 'measure': measure}[sys.argv[1]]()
