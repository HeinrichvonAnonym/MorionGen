"""Shared front end of the OpenSim dimensions: scaled Rajagopal model, IK of the sparse joints, 100 Hz low-passed
coordinates and the frame-level IK gate. OpenSim python."""
import json
from pathlib import Path

import numpy as np

from .spec import SPEC
import biomech


def fit(run, work, mass):
    """run: directory with motion.npz (canonical Z-up joints, joint_names, fps). Writes work/model.osim,
    work/coordinates.mot and the IK files (reused when work already holds the fit of the same run and mass); returns
    the 100 Hz coordinates and the per-frame gate."""
    work = Path(work).resolve(); work.mkdir(parents=True, exist_ok=True)
    old = json.loads((work/'receipt.json').read_text()) if (work/'receipt.json').exists() else None
    if old and old['run'] == str(run) and old['scaling']['mass_kg'] == mass and (work/'coordinates.mot').exists():
        receipt = old
        times, labels, coords = biomech.read_storage(work/'coordinates.mot')
    else:
        receipt = biomech.prepare(run, work, mass, free_yaw=True)
        times, labels, coords = biomech.write_coordinates(work/'ik/ik.mot', work/'coordinates.mot')
    _, el, err = biomech.read_storage(next((work/'ik').glob('*_ik_marker_errors.sto')))
    peak = err[:, el.index('marker_error_max')]
    idx = np.clip(np.round((times-times[0])*receipt['fps']).astype(int), 0, len(peak)-1)
    return {'times': times, 'labels': labels, 'coords': coords, 'ik_max_error_m': peak[idx],
            'unmeasurable': peak[idx] > SPEC['ik_gate']['max_marker_error_m'], 'receipt': receipt}


def scored(times):
    """Frames scored by every dimension: the registered trim at both ends (filter and differentiation edges)."""
    dt = float(np.median(np.diff(times))); k = int(round(SPEC['trim_s']/dt))
    sel = np.zeros(len(times), bool); sel[k:len(times)-k] = True
    return sel
