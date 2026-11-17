import json
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k import candidate_pool
from motiogen.evalkit.k.motion import Motion
from motiogen.degradation.metrics import NAMES


def test_candidate_entrypoint_retains_failures_selects_whole_clip_and_keys_cache(tmp_path,monkeypatch):
    positions=np.tile(np.random.default_rng(2).normal(size=(1,len(NAMES),3)),(3,1,1))
    times=np.arange(3)/30
    motion=Motion(positions,NAMES,times,'motionlcm',provenance={'inputs':{'D':'input-hash'}})
    monkeypatch.setattr(candidate_pool,'load',lambda *a,**kw:motion)
    calls=[]
    def fake_fit(run,out,mass,floor,config):
        assert run=='declared-D' and 'reference' not in config
        calls.append(config['label'])
        if config['label']=='failed':raise RuntimeError('Recorded solver failure')
        out.mkdir(parents=True)
        values={'times':times,'names':np.array(NAMES),'positions':positions+config['bias']}
        np.savez_compressed(out/'fit.npz',**values)
        (out/'model.osim').write_text(config['label'])
        receipt={'key':{'inputs':motion.provenance['inputs'],'config':config}}
        (out/'receipt.json').write_text(json.dumps(receipt))
        return {**values,'receipt':receipt}
    monkeypatch.setattr('motiogen.evalkit.k.k4_fitting.fit',fake_fit)
    candidates={name:dict(version='k4-fit-1',export_clock='native_multiple',label=name,bias=bias)
                for name,bias in [('biased',.1),('exact',0),('failed',0)]}
    settings=dict(version='k4-candidate-pool-1',candidates=candidates,
                  selection_thresholds=dict(angle_deg=5,position_cm=5,direction_rate_deg_s=10,added_step_deg=5))
    out=tmp_path/'fit'
    result=candidate_pool.fit('declared-D',out,75,0,settings)
    selection=result['receipt']['candidate_selection']
    assert selection['selected']=='exact' and not selection['per_frame_switching']
    assert 'failed' in selection['failures']
    assert (out/'model.osim').read_text()=='exact'
    assert (out/'candidates/biased/model.osim').read_text()=='biased'
    np.testing.assert_array_equal(result['positions'],positions)
    candidate_pool.fit('declared-D',out,75,0,settings)
    assert len(calls)==3  # Cache preserves selected output and every decision.
    settings['selection_thresholds']['angle_deg']=4
    with pytest.raises(ValueError,match='identity changed'):
        candidate_pool.fit('declared-D',out,75,0,settings)
