"""Known fixed chains exercise policy gating, invariance and no-reference access."""
from pathlib import Path
import sys
import copy
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.candidate_eligibility import validate,assess


def fixture():
    config={'base':{'position_trunk_strategy':'observed_chain_proxy_v1','solver':'same'},
            'new':{'position_trunk_strategy':'observed_chain_total_proxy_v1','solver':'same'}}
    rules={'base':{'input_rig':'absent'},'new':{'input_rig':'absent','geometry_alternative_to':'base'}}
    p=np.array([[-.1,0,0],[.1,0,0],[0,.2,0],[0,.35,0],[0,.4,0],[-.2,.52,0],[.2,.52,0]])
    motion=Motion(np.tile(p,(101,1,1)),['left_hip','right_hip','spine1','spine2','spine3','left_shoulder','right_shoulder'],np.arange(101)/30,'position')
    return config,rules,motion


def test_only_exactly_redundant_geometry_is_skipped_and_base_retained():
    config,rules,motion=fixture();validate(config,rules)
    assert assess(motion,'base',config,rules)[0]
    accepted,receipt=assess(motion,'new',config,rules)
    assert not accepted and receipt['reason']=='same_fixed_geometry_as_retained_base'
    motion.positions[:,5:,1]+=np.linspace(-.015,.015,101)[:,None]
    accepted,receipt=assess(motion,'new',config,rules)
    assert accepted and receipt['geometry_target_changed']
    moved=copy.deepcopy(motion);moved.positions+=[10,-2,3];moved.provenance['reference']='nonexistent-and-never-read'
    assert assess(moved,'new',config,rules)[0]
    motion.positions[:]=np.nan
    accepted,receipt=assess(motion,'new',config,rules)
    assert not accepted and receipt['base_proxy']['status']=='model_prior'
    assert assess(motion,'base',config,rules)[0]


def test_declared_rig_and_different_solver_cannot_use_geometry_equivalence():
    config,rules,motion=fixture();motion.provenance['input_rig']={'declared':True}
    assert not assess(motion,'new',config,rules)[0]
    config['new']['solver']='different'
    with pytest.raises(ValueError,match='only in fixed trunk'):validate(config,rules)
    config['new']['solver']='same';rules['new']['window']='001'
    with pytest.raises(ValueError,match='declared input'):validate(config,rules)
