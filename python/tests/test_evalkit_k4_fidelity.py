from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion
from motiogen.degradation.metrics import NAMES
from motiogen.evalkit.k.fidelity import score


def test_fidelity_is_input_only_and_common_length_separates_scale_from_direction():
    rng=np.random.default_rng(12);p=np.broadcast_to(rng.normal(size=(1,len(NAMES),3)),(5,len(NAMES),3)).copy()
    t=np.arange(5)/30;m=Motion(p,NAMES,t,'positions');f=dict(times=t,names=NAMES,positions=2*p)
    x=score(m,f);assert x['comparison']=='F-D' and 'F-R' not in x
    assert x['positions_cm']['left_wrist']['world']['p95']>5
    assert x['positions_cm']['left_wrist']['common_length_pelvis_relative']['p95']<1e-10
    assert x['directions_deg']['left_forearm']['p95']<1e-5
    assert x['native_frame_support']['status']=='unavailable'


def test_missing_fitted_node_keeps_input_denominator():
    p=np.random.default_rng(3).normal(size=(5,len(NAMES),3));t=np.arange(5)/30;m=Motion(p,NAMES,t,'positions')
    f=p.copy();f[2,NAMES.index('left_wrist')]=np.nan
    x=score(m,dict(times=t,names=NAMES,positions=f))
    metric=x['positions_cm']['left_wrist']['world'];assert metric['n']==4 and metric['denominator']==5
    assert x['input_quality']['metrics']['position/left_wrist']['missing']==1
