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


@pytest.mark.parametrize('early_budget',[False,True])
@pytest.mark.parametrize('compose_export',[False,True])
def test_candidate_entrypoint_retains_failures_selects_whole_clip_and_keys_cache(tmp_path,monkeypatch,compose_export,early_budget):
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
                for name,bias in [('biased',.1),('exact',0),('failed',0),('rig_only',0)]}
    settings=dict(version='k4-candidate-pool-1',candidates=candidates,
                  candidate_requirements={'rig_only':{'input_rig':'required'}},
                  selection_thresholds=dict(angle_deg=5,position_cm=5,direction_rate_deg_s=10,added_step_deg=5))
    if early_budget:
        settings.update(candidate_budget={'version':'first_input_margin_v1','margin':.8},selection_include_export_path=True)
    if compose_export:
        settings['coordinated_export_variants']={'version':'fixed_native_anchors_v1'}
        def fake_export(observed,source,out,policy):
            assert observed is motion
            if source.name=='exact':raise RuntimeError('Export failed; keep the good native candidate')
            out.mkdir(parents=True)
            values={'times':times,'names':np.array(NAMES),'positions':positions+.2}
            np.savez_compressed(out/'fit.npz',**values)
            receipt={'key':{'inputs':motion.provenance['inputs']}}
            (out/'receipt.json').write_text(json.dumps(receipt))
            return {**values,'receipt':receipt}
        monkeypatch.setattr('motiogen.evalkit.k.path_variant.build',fake_export)
    out=tmp_path/'fit'
    result=candidate_pool.fit('declared-D',out,75,0,settings)
    selection=result['receipt']['candidate_selection']
    assert selection['selected']=='exact' and not selection['per_frame_switching']
    if early_budget:
        assert selection['budget']['early_accepted']=='exact'
        assert selection['budget']['unevaluated_native_candidates']==['failed','rig_only']
        assert 'failed' not in selection['failures']
    else:
        assert 'failed' in selection['failures']
        assert selection['inapplicable']['rig_only']['declared_input_rig_present'] is False
    assert 'rig_only' not in calls and 'rig_only' not in selection['failures']
    if compose_export:
        assert 'biased__coordinated' in selection['qualities']
        if not early_budget:
            assert 'exact__coordinated' in selection['failures']
            assert selection['composed_exports']['exact']['base_candidate_retained']
    assert (out/'model.osim').read_text()=='exact'
    assert (out/'candidates/biased/model.osim').read_text()=='biased'
    np.testing.assert_array_equal(result['positions'],positions)
    candidate_pool.fit('declared-D',out,75,0,settings)
    assert len(calls)==(2 if early_budget else 3)  # Cache preserves selected output and every decision.
    settings['selection_thresholds']['angle_deg']=4
    with pytest.raises(ValueError,match='identity changed'):
        candidate_pool.fit('declared-D',out,75,0,settings)
