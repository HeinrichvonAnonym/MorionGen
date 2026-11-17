import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.candidate_quality import assess, choose, exact_indices
from motiogen.degradation.metrics import NAMES


def example():
    rng=np.random.default_rng(65)
    # All geometric directions are nonzero; same rigid pose at every timestamp.
    positions=np.broadcast_to(rng.normal(size=(len(NAMES),3)),(12,len(NAMES),3)).copy()
    times=np.arange(len(positions))/30
    motion=Motion(positions,NAMES,times,'motionlcm')
    return motion,dict(times=times,positions=positions.copy(),names=np.array(NAMES))


def test_exact_native_clock_rejects_interpolation_and_extrapolation():
    np.testing.assert_array_equal(exact_indices([0,.1,.2],[0,.05,.1,.15,.2]),[0,2,4])
    with pytest.raises(ValueError,match='exact D knots'):
        exact_indices([0,.1,.2],[0,.09,.21])
    with pytest.raises(ValueError):exact_indices([0,.1],[0,0,.1])


def test_missing_candidate_cannot_improve_coverage():
    motion, fitted=example()
    assert assess(motion,fitted)['passed']
    fitted['positions'][0,NAMES.index('left_wrist')]=np.nan
    quality=assess(motion,fitted)
    assert not quality['passed']
    assert quality['metrics']['position/left_wrist']['expected']==12
    assert quality['metrics']['position/left_wrist']['present']==11
    assert quality['rank'][0]>0


def test_fast_observed_turn_is_not_penalized_but_added_jump_is():
    from scipy.spatial.transform import Rotation
    motion,fitted=example()
    rotations=Rotation.from_euler('y',(np.arange(12)*40)[:,None],degrees=True).as_matrix()
    motion.positions=np.einsum('tij,tnj->tni',rotations,motion.positions)
    fitted['positions']=motion.positions.copy()
    assert assess(motion,fitted)['passed']
    # Add a jump on a different step, without allowing a large D step to hide it.
    fitted['positions'][7,NAMES.index('left_wrist')]+=[1,2,3]
    q=assess(motion,fitted)
    assert not q['metrics']['rate/left_forearm']['passed']


def test_fixed_support_and_whole_clip_ranking():
    motion,fitted=example()
    perfect=assess(motion,fitted)
    fitted['positions']+=.1
    biased=assess(motion,fitted)
    assert choose({'arbitrary_a':biased,'arbitrary_b':perfect})=='arbitrary_b'
    assert choose({'different_name':perfect,'other':biased})=='different_name'
    # A declared position prior remains outside support even when finite.
    motion.provenance['degradation']={'prior_joints':['head']}
    q=assess(motion,fitted)
    assert q['metrics']['position/head']['expected']==0
    assert 'rotation/head_relative_thorax' not in q['metrics']


def test_declared_rotation_missing_from_candidate_is_failure(monkeypatch):
    motion,fitted=example()
    rotations=np.broadcast_to(np.eye(3),(12,3,3)).copy()
    monkeypatch.setattr('motiogen.evalkit.k.native_frames.read',lambda m:dict(
        status='declared_native_frames',head_relative_thorax=rotations,head_relative_valid=np.ones(12,bool)))
    assert not assess(motion,fitted)['passed']
    assert assess(motion,fitted,head_rotation=rotations)['passed']
