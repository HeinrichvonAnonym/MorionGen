#!/usr/bin/env python
"""E1 post hoc (not registered): per-marker IK errors, split into limb and axial markers.

The registered IK validity rule pools the low-weight axial markers, whose misfit reflects the single rigid
torso of the Rajagopal model against the multi-joint generated spine. Runs in the OpenSim environment and
reads the per-run files kept on the data disk.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import opensim as osim

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
import biomech

OUT = ROOT/'develop_logs/test_runs/20261101_e1_biomech'


def marker_table(work):
    model = osim.Model(str(work/'model.osim')); state = model.initSystem()
    _, labels, q = biomech.read_storage(work/'ik/ik.mot')
    trc = osim.TimeSeriesTableVec3(str(work/'markers.trc')); names = list(trc.getColumnLabels())
    coords, markers = model.getCoordinateSet(), model.getMarkerSet()
    columns = [(k, coords.get(n)) for k, n in enumerate(labels) if n != 'time' and coords.contains(n)]
    errors, limits = np.zeros((q.shape[0], len(names))), collections.Counter()
    for i in range(q.shape[0]):
        for k, coord in columns:
            coord.setValue(state, np.deg2rad(q[i, k]) if coord.getMotionType() == 1 else q[i, k], False)
        model.assemble(state); model.realizePosition(state)
        row = trc.getRowAtIndex(i)
        for j, name in enumerate(names):
            p, e = markers.get(name).getLocationInGround(state), row[j]
            errors[i, j] = np.sqrt(sum((p.get(a)-e.get(a))**2 for a in range(3)))
        for _, coord in columns:
            value = coord.getValue(state)
            if coord.getClamped(state) and min(abs(value-coord.getRangeMin()), abs(value-coord.getRangeMax())) < 1e-3:
                limits[coord.getName()] += 1
    return names, errors, limits


def summarize(names, errors, limits):
    limb = [j for j, n in enumerate(names) if n in biomech.MARKERS]
    axial = [j for j, n in enumerate(names) if n in biomech.AXIAL]
    part = lambda cols: {'rms_mean_m': float(np.sqrt((errors[:, cols]**2).mean(1)).mean()), 'max_m': float(errors[:, cols].max())}
    worst = collections.Counter(names[j] for j in errors.argmax(1)[errors.max(1) > .06])
    return {'limb': part(limb), 'axial': part(axial), 'worst_marker_frames_above_6cm': dict(worst.most_common()),
            'frames_at_coordinate_limit': dict(limits.most_common()), 'frames': int(errors.shape[0])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    osim.Logger.setLevelString('error')
    for path in sorted((args.out/'runs').glob('*/metrics.json')):
        target = path.parent/'markers_posthoc.json'
        if target.exists() and not args.redo:
            continue
        work = Path(json.loads(path.read_text())['work'])
        target.write_text(json.dumps(summarize(*marker_table(work)), indent=1))


if __name__ == '__main__':
    main()
