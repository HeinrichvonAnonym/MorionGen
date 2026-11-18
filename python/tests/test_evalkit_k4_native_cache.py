import json,sys
from pathlib import Path
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.native_cache import FrozenNativeBank


@pytest.mark.parametrize('bind_method',[False,True])
def test_frozen_cache_keeps_all_candidates_and_checks_every_contract(tmp_path,bind_method):
    settings={'version':'k4-candidate-pool-1','candidates':{'old_winner':{'version':'k4-fit-1'},'other':{'version':'k4-fit-1'}}}
    contract={'config':settings,'implementation':{'fitter.py':'frozen-code'},'input_config_sha256':'input-contract',
              'biomech_sha256':'backend','model_sha256':'asset'}
    source=tmp_path/'bank';source.mkdir()
    key={**contract,'inputs':{'D':'hash'},'mass_kg':75.,'floor_m':0.}
    if bind_method:key['config']={**settings,'method':'positions'}
    selection={'selected':'old_winner','qualities':{n:{} for n in settings['candidates']},'failures':{},'inapplicable':{}}
    (source/'receipt.json').write_text(json.dumps(dict(version=settings['version'],key=key,candidate_selection=selection)))
    for name,config in settings['candidates'].items():
        child=source/'candidates'/name;child.mkdir(parents=True)
        (child/'receipt.json').write_text(json.dumps({'key':{**key,'config':dict(config,method='positions')}}))
        np.savez(child/'fit.npz',positions=np.zeros((2,2,3)),names=['a','b'])
    composed={**settings,'coordinated_export_variants':{'version':'fixed_native_anchors_v1'}}
    if bind_method:composed['method']='positions'
    provider=FrozenNativeBank(source,contract,composed)
    assert set(provider.files)=={'old_winner','other'}
    result=provider('other','D',tmp_path/'copied',75.,0.,dict(settings['candidates']['other'],method='positions'))
    np.testing.assert_array_equal(result['positions'],0.)
    with pytest.raises(ValueError,match='contract mismatch'):
        FrozenNativeBank(source,{**contract,'model_sha256':'changed'},composed)
    with pytest.raises(ValueError,match='Only export'):
        FrozenNativeBank(source,contract,{**composed,'new_solver_setting':1})
    with pytest.raises(ValueError,match='mass/floor'):
        provider('other','D',tmp_path/'mass_changed',76.,0.,dict(settings['candidates']['other'],method='positions'))
    (source/'candidates/other/fit.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact changed'):
        provider('other','D',tmp_path/'changed',75.,0.,dict(settings['candidates']['other'],method='positions'))
