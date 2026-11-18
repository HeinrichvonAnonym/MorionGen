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


def test_export_path_checks_inserted_frames_without_changing_native_support():
    motion,fitted=example()
    dense_times=np.linspace(motion.times[0],motion.times[-1],23)
    dense=dict(times=dense_times,names=fitted['names'],positions=np.repeat(fitted['positions'][:1],23,axis=0))
    assert assess(motion,dense,include_export_path=True)['passed']
    dense['positions'][7,NAMES.index('left_foot')]+=[1,2,3]
    assert assess(motion,dense)['passed']
    checked=assess(motion,dense,include_export_path=True)
    assert not checked['passed']
    assert not checked['metrics']['interpolation_direction/left_foot_axis']['passed']
    assert checked['metrics']['position/left_foot']['expected']==12
    assert checked['metrics']['interpolation_direction/left_foot_axis']['expected']==11


def test_observed_targets_take_priority_over_path_assumptions():
    motion,fitted=example();times=np.linspace(motion.times[0],motion.times[-1],23)
    clean=dict(times=times,names=fitted['names'],positions=np.repeat(fitted['positions'][:1],23,axis=0))
    bad_path={**clean,'positions':clean['positions'].copy()}
    bad_path['positions'][1::2]+=np.random.default_rng(731).normal(size=bad_path['positions'][1::2].shape)
    bad_native={**clean,'positions':clean['positions']+.1}
    policy=dict(include_export_path=True,ranking_policy='observed_then_path_v1')
    path=assess(motion,bad_path,**policy);native=assess(motion,bad_native,**policy);good=assess(motion,clean,**policy)
    assert path['rank'][1]==0 and path['rank'][3]>0
    assert native['rank'][1]>0 and native['rank'][3]==0
    assert choose({'observed_supported':path,'observed_failed':native})=='observed_supported'
    assert choose({'path_failed':path,'all_supported':good})=='all_supported'
    assert assess(motion,clean)['ranking_policy']=='flat_v1'
    with pytest.raises(ValueError,match='policy'):assess(motion,clean,ranking_policy='unknown')
