"""Whole-body muscle load: static optimization (activation exponent 2, weak lower-limb reserves, strong pelvis
residual actuators) on the shared coordinates and the fitted GRF. Effort = sum of squared muscle activations per frame.
OpenSim python."""
from pathlib import Path

import numpy as np

from .spec import SPEC
import biomech


def evaluate(work, t0, t1):
    """work holds model.osim, coordinates.mot and grf.xml. Returns per-frame effort and the summary metrics."""
    work = Path(work)
    unlock = [f'subtalar_angle_{s}' for s in 'rl'] if SPEC['muscle'].get('unlock_subtalar') else []
    summary = biomech.static_optimization(work, t0, t1, step=SPEC['muscle']['so_step'], unlock=unlock)
    import opensim as osim
    model = osim.Model(str(work/'model_so.osim')); model.initSystem()
    times, labels, act = biomech.read_storage(work/'so/so_so_activation.sto')
    muscles = [k for k, l in enumerate(labels) if model.getMuscles().contains(l)]
    reserves = [k for k, l in enumerate(labels) if l.startswith('reserve_') and not l.startswith('reserve_pelvis')]
    effort = (act[:, muscles]**2).sum(1)
    reserve = np.abs(act[:, reserves]*biomech.RESERVE_NM).sum(1) if reserves else np.zeros(len(times))
    return {'times': times, 'effort': effort, 'reserve_abs_sum_nm': reserve, 'summary': summary}


UPPER_BODY = {'lumbar': (('lumbar_ext', 'lumbar_bend', 'lumbar_rot'),),
              'shoulder': tuple(tuple(f'shoulder_{d}_{s}' for d in ('flex', 'add', 'rot')) for s in 'rl')}


def model_mass(path):
    import xml.etree.ElementTree as ET
    return float(sum(float(b.findtext('mass')) for b in ET.parse(path).getroot().iter('Body')))


def upper_body(so_dir, model):
    """Descriptive upper-body moments (config/evalkit_upper_body_20261107.json) from an existing static optimization:
    per-frame lumbar and shoulder (max over sides) moment magnitude / model mass, N m/kg, plus clip mean and p95."""
    times, labels, force = biomech.read_storage(Path(so_dir)/'so_so_force.sto')
    mass = model_mass(model)
    out = {'times': times, 'mass_kg': mass}
    for part, groups in UPPER_BODY.items():
        per = np.max([np.linalg.norm(force[:, [labels.index(c) for c in g]], axis=1) for g in groups], 0)/mass
        out[part] = per
        out[f'{part}_moment_nm_kg_mean'], out[f'{part}_moment_nm_kg_p95'] = float(per.mean()), float(np.percentile(per, 95))
    return out
