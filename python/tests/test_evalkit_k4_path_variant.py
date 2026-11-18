from pathlib import Path
import json,sys,shutil
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.path_variant import build
from motiogen.evalkit.k.output import write_output
from motiogen.evalkit.k.k4_fitting import model_fk,write_coordinates
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.config import digest


def test_export_composition_preserves_native_model_geometry_and_source(tmp_path):
    import opensim as osim
    import biomech
    osim.Logger.setLevelString('error')
    source=tmp_path/'source';source.mkdir();shutil.copyfile(biomech.MODEL,source/'model.osim')
    model=osim.Model(str(biomech.MODEL));state=model.initSystem();cs=model.getCoordinateSet()
    labels=[cs.get(i).getName() for i in range(cs.getSize())]
    row=np.array([cs.get(n).getValue(state)*(180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
    row[labels.index('knee_angle_l')]=60
    native=np.tile(row,(2,1));points,names,native=model_fk(biomech.MODEL,labels,native,direct_couplers=True,return_coordinates=True)
    chest=(points[:,names.index('left_shoulder')]+points[:,names.index('right_shoulder')])/2
    extra=np.stack([chest,chest+[0,.1,0],chest+[0,.3,0]],1)
    motion=Motion(np.concatenate([points,extra],axis=1),names+['Spine3','neck','head'],np.array([10.,10.1]),'known_fk',
                  provenance={'inputs':{'declared-D':'known-fixture'},'reference':'MUST_NOT_BE_READ'})
    settings=dict(version='k4-fit-1',postprocess='native_preserving',export_clock='native_multiple',
                  constraint_policy='direct_couplers',rotation_resampling='whole_body_so3',
                  root_policy='input_hip_midpoint',head_neck='attached_head_neck_v1',resample_min_rate_hz=20)
    write_coordinates(source/'coordinates_refined_native.mot',[0,.1],labels,native)
    values,head,_=write_output(motion,source,labels,np.tile(native[0],(3,1)),np.array([0,.05,.1]),settings)
    receipt=dict(version='k4-fit-1',key=dict(config=settings,inputs=motion.provenance['inputs'],model_sha256=digest(biomech.MODEL)),
                 head_neck_model=head,kinematic_only=True,force_eligible=False)
    (source/'receipt.json').write_text(json.dumps(receipt))
    hashes={p.name:digest(p) for p in source.iterdir() if p.is_file()}
    policy={'version':'fixed_native_anchors_v1','refinement':{'max_nfev':20,'continuity_weight':0}}
    result=build(motion,source,tmp_path/'variant',policy)
    assert result['receipt']['export_variant']['native_fit_recomputed'] is False
    assert result['receipt']['export_variant']['head_geometry_reestimated'] is False
    np.testing.assert_allclose(result['positions'],values['positions'],atol=1e-8)
    assert {p.name:digest(p) for p in source.iterdir() if p.is_file()}==hashes
    assert digest(tmp_path/'variant/model.osim')==hashes['model.osim']
    assert digest(tmp_path/'variant/coordinates_refined_native.mot')==hashes['coordinates_refined_native.mot']
    cached=build(motion,source,tmp_path/'variant',policy)
    np.testing.assert_array_equal(cached['positions'],result['positions'])
    with pytest.raises(ValueError,match='identity changed'):
        build(motion,source,tmp_path/'variant',{**policy,'refinement':{'max_nfev':21}})
    motion.provenance['inputs']={'different-D':'hash'}
    with pytest.raises(ValueError,match='input identity'):
        build(motion,source,tmp_path/'other',policy)


def test_export_variant_cannot_write_inside_source(tmp_path):
    with pytest.raises(ValueError,match='independent peer'):
        build(None,tmp_path,tmp_path/'child',{})
