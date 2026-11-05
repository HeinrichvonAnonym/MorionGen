#!/usr/bin/env python
"""Amendment 4 held-out landmarks (config/val_a_addb_20261105_amend4.json): heel (subtalar joint centre) and toe (mtp joint
centre) from the AddB KINEMATICS pass resampled to 30 fps, plus the measured foot GRF; the 6 Hz low-pass and the 100 Hz
spline of the pipeline are applied at evaluation (val_a_amend4_tune.py --evaluate; this environment has no scipy).

  --development   the development walking trials (bridge check against the pipeline landmarks)
  --heldout       scan the held-out studies, select overground level walking by trial name, export

Writes <DATA>/addb_val_20261105/amend4_landmarks/<id>.npz and, for --heldout, develop_logs/test_runs/20261105_val_a/
amend4_heldout_selection.json. Runs in the motiogen-addb environment (nimblephysics).
"""
import argparse
import collections
import json
import re
from pathlib import Path

import numpy as np
import nimblephysics as nimble

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT/'develop_logs/test_runs/20261105_val_a'
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen')
FILES = DATA/'datasets/addbiomechanics_v1/files'
OUT = DATA/'addb_val_20261105/amend4_landmarks'
HELDOUT_STUDIES = ('Camargo2021', 'Santos2017', 'Tan2021', 'Tan2022', 'Wang2023')
LANDMARKS = {'heel': 'subtalar', 'toe': 'mtp'}
EXCLUDED_REASONS = ('unmeasuredExternalForceDetected', 'manualReview')


def pass_index(s, kind):
    for p in range(s.getNumProcessingPasses()):
        if str(s.getProcessingPassType(p)).endswith(kind):
            return p
    return None


def export(file, trial, tid):
    s = nimble.biomechanics.SubjectOnDisk(str(FILES/file))
    n = s.getTrialLength(trial); dt = s.getTrialTimestep(trial)
    frames = s.readFrames(trial, 0, n, includeSensorData=False, includeProcessingPasses=True)
    kin = pass_index(s, 'KINEMATICS'); skel = s.readSkel(kin, ignoreGeometry=True)
    jnames = [skel.getJoint(i).getName() for i in range(skel.getNumJoints())]
    jc = np.array([np.asarray(f.processingPasses[kin].jointCenters) for f in frames]).reshape(n, -1, 3)
    times = np.arange(n)*dt; t30 = np.arange(0, times[-1]+1e-9, 1/30.)
    out = {'t30': t30}
    for side in 'rl':
        for k, joint in LANDMARKS.items():
            out[f'raw_{side}_{k}'] = np.stack([np.interp(t30, times, jc[:, jnames.index(f'{joint}_{side}'), a]) for a in range(3)], 1)
    reasons = [str(x).split('.')[-1] for x in s.getMissingGRF(trial)]
    grf = np.array([np.asarray(f.processingPasses[kin].groundContactForce) for f in frames]).reshape(n, -1, 3)
    bodies = list(s.getGroundForceBodies())
    out.update(m_times=times, grf_r=grf[:, bodies.index('calcn_r')], grf_l=grf[:, bodies.index('calcn_l')],
               valid=np.array([x == 'notMissingGRF' for x in reasons]), excluded=np.array([x in EXCLUDED_REASONS for x in reasons]))
    OUT.mkdir(parents=True, exist_ok=True); np.savez_compressed(OUT/f'{tid}.npz', **out)
    return float(np.mean(out['valid'])), n*dt


def level_walk(study, name):
    t = name.lower()
    if any(w in t for w in ('treadmill', 'ramp', 'stair', 'incline', 'decline', 'tm_', '_tm', 'run', 'jog', 'transition', 'turn')):
        return False
    return any(w in t for w in ('levelground', 'level_ground', 'walk', 'gait', 'ngait'))


def heldout():
    rows = []
    for f in sorted(FILES.glob('*/No_Arm/*/*/*.b3d')):
        study = f.parts[-3].split('_Formatted')[0]
        if study not in HELDOUT_STUDIES:
            continue
        s = nimble.biomechanics.SubjectOnDisk(str(f))
        for t in range(s.getNumTrials()):
            name = s.getTrialName(t); seconds = s.getTrialLength(t)*s.getTrialTimestep(t)
            rows.append({'file': str(f.relative_to(FILES)), 'trial_index': t, 'study': study, 'subject': f'{study}/{f.parent.name.split("_split")[0]}',
                         'trial': name, 'seconds': seconds, 'level_walk': level_walk(study, name)})
    names = collections.Counter((r['study'], re.sub(r'[\d_]+', '_', r['trial'].lower())) for r in rows)
    pool = sorted((r for r in rows if r['level_walk'] and r['seconds'] >= 2.0), key=lambda r: (r['file'], r['trial_index']))
    rng = np.random.default_rng(20261105); per = collections.Counter(); chosen = []
    for k in rng.permutation(len(pool)):
        r = pool[k]
        if per[r['subject']] >= 4:
            continue
        r['id'] = re.sub(r'[^A-Za-z0-9]+', '_', f"{r['study']}_{Path(r['file']).parent.name}_{r['trial']}")
        valid, _ = export(r['file'], r['trial_index'], r['id'])
        r['grf_valid_share'] = valid
        if valid >= .5:
            chosen.append(r); per[r['subject']] += 1
        if len(chosen) == 40:
            break
    (LOG/'amend4_heldout_selection.json').write_text(json.dumps({'trial_name_patterns': {f'{a}:{b}': n for (a, b), n in names.items()},
                                                                 'trials': chosen}, indent=1))
    print(json.dumps({'scanned_trials': len(rows), 'level_walk_pool': len(pool), 'chosen': len(chosen), 'subjects': len(per)}))


def development():
    reg = json.loads((LOG/'selection.json').read_text())['trials']; held = json.loads((LOG/'selection_heldout.json').read_text())['trials']
    rows = [r for r in reg+held if r['class'] == 'walk' and 'tmf' not in r['task']]
    for r in rows:
        export(r['file'], r['trial_index'], r['id'])
    print(json.dumps({'development_walk': len(rows)}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--development', action='store_true')
    p.add_argument('--heldout', action='store_true')
    a = p.parse_args()
    if a.development:
        development()
    if a.heldout:
        heldout()


if __name__ == '__main__':
    main()
