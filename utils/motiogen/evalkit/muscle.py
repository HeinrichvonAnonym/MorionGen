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
    summary = biomech.static_optimization(work, t0, t1, step=SPEC['muscle']['so_step'])
    import opensim as osim
    model = osim.Model(str(work/'model_so.osim')); model.initSystem()
    times, labels, act = biomech.read_storage(work/'so/so_so_activation.sto')
    muscles = [k for k, l in enumerate(labels) if model.getMuscles().contains(l)]
    reserves = [k for k, l in enumerate(labels) if l.startswith('reserve_') and not l.startswith('reserve_pelvis')]
    effort = (act[:, muscles]**2).sum(1)
    reserve = np.abs(act[:, reserves]*biomech.RESERVE_NM).sum(1) if reserves else np.zeros(len(times))
    return {'times': times, 'effort': effort, 'reserve_abs_sum_nm': reserve, 'summary': summary}
