#!/usr/bin/env python
"""AddBiomechanics trials as E-pipeline sources (config/val_a_addb_20261105.json).

Selects trials by the registered classes and eligibility, maps the AddB joint centres to the source joint names of
biomech (Z-up motion.npz, like a generated run) and stores the measured foot GRF with the AddB missing-GRF flags.
Variants: kin30 (KINEMATICS pass, 30 fps; primary), dyn30 (DYNAMICS pass), kin_native (KINEMATICS pass, native rate).
Runs in the motiogen-addb environment (nimblephysics). Writes <DATA>/addb_val_20261105/sources/<variant>/<id>/ and
develop_logs/test_runs/20261105_val_a/selection.json.
"""
import argparse
import collections
import json
import os
import re
from pathlib import Path

import numpy as np
import nimblephysics as nimble

ROOT = Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT/'config/val_a_addb_20261105.json').read_text())
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen')
FILES = DATA/'datasets/addbiomechanics_v1/files'
OUT = DATA/'addb_val_20261105/sources'
LOG = ROOT/'develop_logs/test_runs/20261105_val_a'
ROTATION = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])
JOINTS = {'pelvis': 'ground_pelvis', 'Spine': 'back', **{f'{side}_{k}': f'{v}_{s}' for side, s in (('right', 'r'), ('left', 'l'))
          for k, v in (('hip', 'hip'), ('knee', 'walker_knee'), ('ankle', 'ankle'), ('foot', 'mtp'), ('shoulder', 'acromial'),
                       ('elbow', 'elbow'), ('wrist', 'radius_hand'))}}
NAMES = [*JOINTS, 'Spine3', 'neck']
CLASSES = {
    'stand': {'Uhlrich2023': {'static'}, 'Han2023': {'static', 'idling'}},
    'walk': {'Han2023': {'walk'}, 'Falisse2017': {'gait'}, 'Uhlrich2023': {'walking', 'walkingts'},
             'Fregly2012': lambda t: 'ngait' in t or (t.startswith('sc_') and t.endswith('_og'))},
    'run': {'Hammer2013': {'run'}},
    'squat': {'Uhlrich2023': {'squats', 'squatsasym'}, 'Han2023': {'squat'}},
    'sts': {'Uhlrich2023': {'sts', 'stsweaklegs'}, 'Falisse2017': {'sts'}},
    'jump': {'Han2023': {'balletsmalljump', 'jumpingjack', 'hopping'}, 'Uhlrich2023': {'dj', 'djasym'}},
    'wholebody': {'Han2023': {'worrier', 'tree', 'treearms', 'chair', 'ballethighleg', 'ballet_high_leg', 'taichi', 'soccerkick',
                              'tennisgroundstroke', 'tennisserve', 'step', 'dog', 'lambadadance', 'sidestretch', 'whirl'}}}
EXCLUDED_REASONS = ('unmeasuredExternalForceDetected', 'manualReview')


def task_name(study, trial):
    base = trial.split('_segment')[0].lower()
    if study == 'Han2023':
        base = re.sub(r'^\d+_', '', base); base = re.sub(r'(_\d+)?(_full)?$', '', base)
    return base.rstrip('0123456789_')


def task_class(study, task):
    for cls, studies in CLASSES.items():
        rule = studies.get(study)
        if rule is not None and (rule(task) if callable(rule) else task in rule):
            return cls
    return None


def scan():
    rows = []
    for f in sorted(FILES.glob('**/*.b3d')):
        if os.path.getsize(f) == 0:
            continue
        s = nimble.biomechanics.SubjectOnDisk(str(f))
        study = f.relative_to(FILES).parts[2].split('_Formatted')[0]
        for t in range(s.getNumTrials()):
            name = s.getTrialName(t); task = task_name(study, name); cls = task_class(study, task)
            seconds = s.getTrialLength(t)*s.getTrialTimestep(t)
            if cls and seconds >= 2.0:
                rows.append({'file': str(f.relative_to(FILES)), 'trial_index': t, 'study': study, 'subject': f'{study}/{f.parent.name}',
                             'trial': name, 'task': task, 'class': cls, 'seconds': seconds})
    return rows


def select(rows):
    rng = np.random.default_rng(20261105); chosen = []
    for cls in CLASSES:
        pool = sorted((r for r in rows if r['class'] == cls), key=lambda r: (r['file'], r['trial_index']))
        per = collections.Counter(); take = []
        for k in rng.permutation(len(pool)):
            r = pool[k]
            if per[r['subject']] < 4:
                take.append(r); per[r['subject']] += 1
            if len(take) == 24:
                break
        chosen += take
    for r in chosen:
        r['id'] = re.sub(r'[^A-Za-z0-9]+', '_', f"{r['study']}_{Path(r['file']).parent.name}_{r['trial']}")
    return chosen


def pass_index(s, kind):
    for p in range(s.getNumProcessingPasses()):
        if str(s.getProcessingPassType(p)).endswith(kind):
            return p
    return None


def export(r, variants):
    s = nimble.biomechanics.SubjectOnDisk(str(FILES/r['file']))
    t = r['trial_index']; n = s.getTrialLength(t); dt = s.getTrialTimestep(t)
    frames = s.readFrames(t, 0, n, includeSensorData=False, includeProcessingPasses=True)
    reasons = [str(x).split('.')[-1] for x in s.getMissingGRF(t)]
    kin, dyn = pass_index(s, 'KINEMATICS'), pass_index(s, 'DYNAMICS')
    mass_dyn = float(s.readSkel(dyn, ignoreGeometry=True).getMass()) if dyn is not None else None
    skel = s.readSkel(kin, ignoreGeometry=True)
    jnames = [skel.getJoint(i).getName() for i in range(skel.getNumJoints())]
    bodies = list(s.getGroundForceBodies())
    out = {}
    for variant in variants:
        p = dyn if variant.startswith('dyn') else kin
        if p is None:
            out[variant] = 'no_pass'; continue
        jc = np.array([np.asarray(f.processingPasses[p].jointCenters) for f in frames]).reshape(n, -1, 3)
        pts = {k: jc[:, jnames.index(v)] for k, v in JOINTS.items()}
        mid = .5*(pts['right_shoulder']+pts['left_shoulder'])
        pts['Spine3'] = pts['Spine']+2/3*(mid-pts['Spine']); pts['neck'] = mid
        joints = np.stack([pts[k] for k in NAMES], 1)
        times = np.arange(n)*dt
        fps = 1/dt
        if variant.endswith('30'):
            fps = 30.; new = np.arange(0, times[-1]+1e-9, 1/fps)
            joints = np.stack([np.stack([np.interp(new, times, joints[:, j, a]) for a in range(3)], -1) for j in range(joints.shape[1])], 1)
        grf = np.array([np.asarray(f.processingPasses[p].groundContactForce) for f in frames]).reshape(n, -1, 3)
        d = OUT/variant/r['id']; d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d/'motion.npz', joints=joints @ ROTATION, joint_names=np.array(NAMES), fps=fps)
        np.savez_compressed(d/'measured.npz', times=times, grf_r=grf[:, bodies.index('calcn_r')], grf_l=grf[:, bodies.index('calcn_l')],
                            valid=np.array([x == 'notMissingGRF' for x in reasons]), excluded=np.array([x in EXCLUDED_REASONS for x in reasons]),
                            mass_kg=mass_dyn if mass_dyn else s.getMassKg(), mass_header_kg=s.getMassKg())
        out[variant] = str(d)
    r.update(mass_kg=mass_dyn if mass_dyn else s.getMassKg(), mass_header_kg=s.getMassKg(), mass_source='dynamics' if mass_dyn else 'header',
             native_fps=1/dt, grf_valid_share=float(np.mean([x == 'notMissingGRF' for x in reasons])), sources=out)
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variants', nargs='+', default=['kin30', 'dyn30', 'kin_native'])
    args = p.parse_args()
    rows = scan(); chosen = select(rows)
    LOG.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(chosen):
        export(r, args.variants)
        print(i+1, len(chosen), r['id'], r['class'], flush=True)
    eligible = collections.Counter(r['class'] for r in rows)
    (LOG/'selection.json').write_text(json.dumps({'eligible_per_class': eligible, 'selected_per_class': collections.Counter(r['class'] for r in chosen),
                                                  'trials': chosen}, indent=1))


if __name__ == '__main__':
    main()
