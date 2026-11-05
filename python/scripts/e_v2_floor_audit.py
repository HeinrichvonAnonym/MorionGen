"""E v2 floor references (plan: E 准则修正版, section 2): for each method, the heights of E's four foot landmarks (calcn and
toes origins of the scaled model, left and right) when the method's own body model or skeleton stands flat-footed on the
method's own floor, taken through the same scaling, IK and coordinate filtering as every scored motion.

Writes config/e_v2_floor_reference_20261106.json and develop_logs/test_runs/20261106_e_v2/floor_audit/. OpenSim python;
pose builders that need a method's body model run in that method's environment and hand over a Z-up joints npz.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils/motiogen'), str(ROOT/'python/scripts')]
import biomech  # noqa: E402

OUT = ROOT/'develop_logs/test_runs/20261106_e_v2/floor_audit'
DISK = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_v2_20261106/floor_audit')
CONFIG = ROOT/'config/e_v2_floor_reference_20261106.json'
HOLD_S, FPS, MASS = 2., 30., 75.
LANDMARK_BODY = {'heel': 'calcn', 'toe': 'toes'}


def reference_from_joints(joints_zup, names, tag, mass=MASS):
    """joints_zup: (J, 3) one standing frame in the method's canonical Z-up world (floor z per its convention)."""
    import e_rules
    joints = biomech.to_opensim(np.repeat(np.asarray(joints_zup, float)[None], int(HOLD_S*FPS), 0))
    biomech.load = lambda _run: (joints, list(names), FPS)
    work = DISK/tag
    rec = biomech.prepare(work, work, mass, free_yaw=True)
    biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
    times, _, _, kin = e_rules.kinematics(work)
    sel = e_rules.trim(times)
    ref = {s: {k: float(np.median(kin['origin'][sel, kin['names'].index(f'{b}_{s}'), 1])) for k, b in LANDMARK_BODY.items()}
           for s in 'rl'}
    src = {n: joints[0, names.index(n)] for n in names}
    ankle_minus_foot = {s: float(src[f'{side}_ankle'][1]-src[f'{side}_foot'][1]) for s, side in (('r', 'right'), ('l', 'left'))}
    return {'reference_m': ref, 'source_lowest_joint_m': float(joints[0, :, 1].min()),
            'source_ankle_minus_foot_m': ankle_minus_foot, 'ik_marker_error': rec['ik_marker_error']}


KIN30 = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/addb_val_20261105/work/kin30')
VAL = ROOT/'develop_logs/test_runs/20261105_val_a'
LOADED_BW, FLAT_SPEED_M_S = .2, .1  # 约定，未经推演


def kin30_trials():
    """AddB floor is the force-plate plane y=0. Per trial and side: landmark heights over frames with measured vertical GRF
    > LOADED_BW body weights (valid, not excluded) and both landmarks of that foot moving vertically slower than
    FLAT_SPEED_M_S (vertical, so that treadmill stance counts)."""
    import e_rules
    rows = json.loads((VAL/'selection.json').read_text())['trials']
    out = {}
    for r in rows:
        work = KIN30/r['id']
        if not (work/'coordinates.mot').exists():
            continue
        times, _, _, kin = e_rules.kinematics(work)
        m = np.load(Path(r['sources']['kin30'])/'measured.npz')
        near = np.clip(np.round(times/np.median(np.diff(m['times']))).astype(int), 0, len(m['times'])-1)
        ok = m['valid'][near] & ~m['excluded'][near]
        dt = float(np.median(np.diff(times))); bw = r['mass_kg']*9.80665
        rec = {'subject': r['subject'], 'class': r['class'], 'sides': {}}
        for s in 'rl':
            fy = np.interp(times, m['times'], m[f'grf_{s}'][:, 1])/bw
            h = {k: kin['origin'][:, kin['names'].index(f'{b}_{s}'), 1] for k, b in LANDMARK_BODY.items()}
            v = {k: np.abs(np.gradient(h[k], dt)) for k in h}
            sel = ok & (fy > LOADED_BW) & (v['heel'] < FLAT_SPEED_M_S) & (v['toe'] < FLAT_SPEED_M_S)
            rec['sides'][s] = {'frames': int(sel.sum()), **{k: (float(np.median(h[k][sel])) if sel.any() else None) for k in h}}
        out[r['id']] = rec
        print(r['id'], json.dumps(rec['sides']), flush=True)
    (OUT/'kin30_trials.json').write_text(json.dumps(out, indent=1))


REFERENCE_TIERS = (('stand',), ('walk', 'sts', 'squat'))


def kin30_reference(trial, trials=None, min_frames=10):
    """Leave-one-trial-out subject reference: median over the subject's other trials of the loaded-flat landmark heights,
    from quiet-standing trials if the subject has any, else from walk / sts / squat (flat-foot stance); run, jump and
    whole-body trials load a raised heel (lunges, forefoot landing) and never contribute. Returns (reference, tier)."""
    trials = trials or json.loads((OUT/'kin30_trials.json').read_text())
    subject = trials[trial]['subject']
    for tier, classes in enumerate(REFERENCE_TIERS):
        use = [t for i, t in trials.items() if i != trial and t['subject'] == subject and t['class'] in classes]
        ref = {s: {k: (float(np.median(v)) if (v := [t['sides'][s][k] for t in use if t['sides'][s]['frames'] >= min_frames]) else None)
                   for k in LANDMARK_BODY} for s in 'rl'}
        if all(ref[s][k] is not None for s in 'rl' for k in LANDMARK_BODY):
            return ref, '+'.join(classes)
    return None, None


VARIANTS = {
    'ardy': ('ardy', 'raw output (correction=0)', 'medium: convention from the training data floor; raw toe base in its lowest 10% of frames 0.013-0.018 m vs 0.0152 m bind'),
    'ardy__correction': ('ardy', 'correction=1 (with or without write-back: the emitted motion is the corrected one, ardy_native.py:313-320)', 'high: enforced by the correction'),
    'kimodo': ('kimodo', 'raw output (post_processing=False)', 'high: bind mesh flat on y=0; ground truth toe base 0.013-0.026 m'),
    'kimodo__postprocess': ('kimodo', 'post_processing=True', 'high: enforced; observed toe base 0.0202 m'),
    'primal': ('primal', 'ground_snap=1 (E4 uses only ground1; ground0 has no floor and is not scored)', 'high: enforced by the snap'),
    'dart': ('dart', 'all DART runs', 'high: enforced at reset, kept within 3 cm by the policy reward (env_reach_location_mld.py:365-374)'),
    'closd': ('closd', 'all CLoSD runs', 'high: simulator plane; box-foot bottoms'),
    'humanml3d': ('motionlcm, omnicontrol', 'HumanML3D-trained text/control models', 'MotionLCM high for the convention, medium for outputs (learned, not enforced); '
                  'OmniControl medium (per-run bone lengths deviate from t2m)'),
}


def side_reference(tag):
    rec = {s: json.loads((OUT/f'{tag}@{s}.json').read_text()) for s in 'lr'}
    return ({s: rec[s]['reference_m'][s] for s in 'lr'}, rec)


def summarize():
    tags = sorted({p.name.rsplit('@', 1)[0] for p in OUT.glob('*@*.json')})
    table = {t: side_reference(t) for t in tags}
    for t, (ref, _) in table.items():
        print(f"{t:32s} heel L {ref['l']['heel']:.4f} R {ref['r']['heel']:.4f}  toe L {ref['l']['toe']:.4f} R {ref['r']['toe']:.4f}")

    def rng(keys):
        vals = {f'{s}_{k}': [table[t][0][s][k] for t in keys] for s in 'lr' for k in LANDMARK_BODY}
        return {n: [float(min(v)), float(max(v))] for n, v in vals.items()}

    def entry(tag, extra=None):
        ref, rec = table[tag]
        meta = rec['l']['meta']
        return {'reference_m': ref, 'world_floor_m': 0., 'pose': {s: f'{tag}@{s}' for s in 'lr'}, 'body': meta.get('body'),
                'convention': meta.get('convention'), 'placement': {s: {k: rec[s]['meta'][k] for k in ('placement_rule', 'placement_value_m', 'shift_m')}
                                                                     for s in 'lr'},
                'ik_marker_error': {s: rec[s]['ik_marker_error'] for s in 'lr'}, 'source_ankle_minus_foot_m': rec['l']['source_ankle_minus_foot_m'],
                **(extra or {})}
    methods = {}
    for tag, (method, scope, confidence) in VARIANTS.items():
        methods[tag] = entry(tag, {'method': method, 'scope': scope, 'confidence': confidence})
    methods['dart']['sensitivity_mesh_sole'] = entry('dart__mesh')['reference_m']
    methods['primal']['sensitivity_mesh_sole'] = entry('primal__mesh')['reference_m']
    methods['primal']['sensitivity_shape_beta0_pm2'] = rng(['smplx_neutral_beta0_-2__mesh', 'primal__mesh', 'smplx_neutral_beta0_+2__mesh'])
    methods['humanml3d']['sensitivity_omnicontrol_foot_length'] = rng(['humanml3d__foot0.111', 'humanml3d', 'humanml3d__foot0.146'])
    chois = sorted((t for t in tags if t.startswith('chois_w')), key=lambda t: int(t[7:]))
    methods['chois'] = {'method': 'chois', 'scope': 'per run, by the CHOIS test window (scenario native_window_index; runs without it ran '
                        'the default window 0, sub16_clothesstand_000, per their stdout)', 'confidence': 'high: mesh sole on z=0 in the source windows',
                        'world_floor_m': 0., 'windows': {t[7:]: entry(t, {'sequence': table[t][1]['l']['meta']['sequence'],
                                                                         'gender': table[t][1]['l']['meta']['gender']}) for t in chois},
                        'range_over_windows': rng(chois)}
    spec = {'created': time.strftime('%Y-%m-%d %H:%M'), 'plan': 'E 准则修正版 (E v2), section 2',
            'definition': "reference_m[side][landmark]: height (m, OpenSim Y-up = canonical Z) of E's landmark (heel = calcn origin, toe = "
                          "toes origin of the scaled model) when the method's own body model or skeleton stands flat-footed with that side's "
                          "foot on the method's own floor, through biomech.prepare (scaling, IK) and write_coordinates; median over the 2 s hold "
                          "after E's 0.5 s trim. Each side is read from a pose with that side's foot placed (body models have unequal leg "
                          "and foot heights). world_floor_m: the plane z of the method's floor.",
            'scripts': ['python/scripts/e_v2_floor_poses.py', 'python/scripts/e_v2_floor_audit.py'],
            'methods': methods,
            'run_mapping': {
                'e4': 'ardy: condition.correction 0 -> ardy, 1 -> ardy__correction; kimodo: condition.correction 0 -> kimodo, 1 -> kimodo__postprocess; '
                      'chois: window; primal -> primal (all ground_snap=1); dart, closd; motionlcm and omnicontrol -> humanml3d',
                'x3': 'dart -> dart; kimodo arms C0, T -> kimodo, O, TO -> kimodo__postprocess (run_x3.py:416-437 post flag); omnicontrol -> humanml3d',
                'm1': 'ARDY: run config _intervention.correction 0 -> ardy, 1 -> ardy__correction',
                'e1': 'ARDY: run config _intervention.correction 0 -> ardy, 1 -> ardy__correction (correction_writeback=False still emits the corrected motion)'},
            'kin30': {'world_floor_m': 0., 'evidence': 'AddB force-plate corners at y=0 and loaded centre of pressure at 0.000 m (0.006 m Uhlrich) on 4 trials of 4 studies',
                      'reference': f'per subject, leave-one-trial-out (kin30_reference): median over the subject\'s other kin30 quiet-standing trials, '
                                   f'else walk/sts/squat trials (run, jump, whole-body never: loaded raised heels), of the landmark heights in frames with '
                                   f'measured vertical GRF > {LOADED_BW} BW on that foot (valid, not excluded) and both landmarks of that foot moving vertically '
                                   f'slower than {FLAT_SPEED_M_S} m/s (thresholds 约定，未经推演); trials with < 10 such frames do not contribute; '
                                   'subjects without a reference are scored with v1 contact only',
                      'trials_file': str((OUT/'kin30_trials.json').relative_to(ROOT))},
            'moco': {'reference': 'reference_m = floor_m of the solve (sphere centres sit on the landmarks, the half-space at floor_m - sphere radius)',
                     'world_floor_m': 'floor_m - 0.03'},
            'no_floor_reference': {'methods': [], 'note': 'every scored variant has a determinable floor; PRIMAL ground_snap=0 would not, and is not in any scored set'}}
    CONFIG.write_text(json.dumps(spec, indent=1))
    print('wrote', CONFIG.relative_to(ROOT))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--summarize', action='store_true')
    a.add_argument('--kin30', action='store_true', help='per-trial loaded-flat landmark heights of the kin30 human clips')
    a.add_argument('--pose', nargs=2, action='append', metavar=('TAG', 'NPZ'),
                   help='standing pose npz (joints (J,3) Z-up, joint_names, method, note) to reference')
    args = a.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.kin30:
        kin30_trials()
    if args.summarize:
        summarize()
    for tag, path in args.pose or []:
        with np.load(path, allow_pickle=False) as m:
            joints, names = m['joints'], [str(n) for n in m['joint_names']]
            meta = json.loads(str(m['meta'])) if 'meta' in m.files else {}
        rec = {'tag': tag, 'pose_file': str(path), 'meta': meta, 'created': time.strftime('%Y-%m-%d %H:%M'),
               **reference_from_joints(joints, names, tag)}
        (OUT/f'{tag}.json').write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: rec[k] for k in ('tag', 'reference_m', 'source_ankle_minus_foot_m')}), flush=True)


if __name__ == '__main__':
    main()
