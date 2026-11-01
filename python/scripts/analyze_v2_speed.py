#!/usr/bin/env python
"""Post hoc (not preregistered) V2 follow-up: is the feet-floor jerk drop more than slower motion.

Under a pure time rescaling x(s t), RMS jerk scales with s^3 and mean speed with s, so
log(jerk) - 3 log(mean speed) is invariant to it.  A drop in this quantity is a change
in path shape, not only in tempo.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'utils'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_next_round import test
from analyze_v2 import ARMS, FEET, OUT, load_table, paired
from motiogen.motion import load_motion
from motiogen.native_analysis import common_step_jerk
from motiogen.runtime import json_write


def tempo_row(run):
    motion = load_motion(run/'motion.npz'); joints = np.asarray(motion['joints'], float); fps = float(motion['fps'])
    names = [str(n) for n in motion['joint_names']]
    values = {}
    for label, xyz in (('foot', joints[:, [names.index(n) for n in FEET]]), ('body', joints)):
        jerk = common_step_jerk(xyz, fps); speed = float((np.linalg.norm(np.diff(xyz, axis=0), axis=-1)*fps).mean())
        values.update({f'{label}_log_jerk': float(np.log(jerk)), f'{label}_3log_speed': float(3*np.log(speed)),
                       f'{label}_tempo_invariant_jerk': float(np.log(jerk)-3*np.log(speed))})
    return values


def main():
    import analyze_v2
    analyze_v2.chois_row = tempo_row
    table, _ = load_table()
    metrics = [f'{p}_{m}' for p in ('foot', 'body') for m in ('log_jerk', '3log_speed', 'tempo_invariant_jerk')]
    result = {'scope': 'post hoc, not preregistered; differences are in log units (x100 is about percent)',
              'runs': len(table),
              'contrasts': {f'{a}-N0': {m: test(*paired(table, a, 'N0', m)) for m in metrics} for a in ARMS[1:]}}
    json_write(OUT/'analysis_tempo_posthoc.json', result)
    for name, row in result['contrasts'].items():
        print(name, {m: (round(t['mean'], 4), [round(x, 4) for x in t['ci95']]) for m, t in row.items()})


if __name__ == '__main__':
    main()
