import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.motion import Motion
from motiogen.evalkit.k.position_centres import estimate,register


def sample():
    names=['left_hip','right_hip','spine1','spine3','left_shoulder','right_shoulder']
    p=np.array([[0,0,-.1],[0,0,.1],[0,.2,0],[0,.4,0],[-.04,.53,-.18],[-.04,.53,.18]])
    return Motion(np.tile(p,(5,1,1)),names,np.arange(5)/30,'positions')


def test_known_geometry_and_global_rigid_transform_invariance():
    m=sample();v,r=estimate(m)
    np.testing.assert_allclose(v['proximal'],[0,.2,0],atol=1e-12)
    np.testing.assert_allclose(v['distal'],[-.04,.33,0],atol=1e-12)
    other=copy.deepcopy(m)
    other.positions=other.positions@Rotation.from_rotvec([.3,-.6,.8]).as_matrix().T+[3,2,-4]
    other.provenance['reference']='must never be opened'
    w,_=estimate(other)
    for key in v:np.testing.assert_allclose(v[key],w[key],atol=1e-12)
    assert not r['neutral_geometry_identified'] and r['fixed_for_clip']


def test_missing_degenerate_prior_and_rig_preserve_unavailable_status():
    m=sample();m.positions[:,m.names.index('spine1')]=np.nan
    assert estimate(m)[0] is None
    m=sample();m.positions[:,m.names.index('spine3')]=m.positions[:,m.names.index('spine1')]
    assert estimate(m)[0] is None
    m=sample();m.provenance['degradation']={'prior_joints':['spine1']}
    assert estimate(m)[1]['reason']=='required_nodes_are_prior'
    m=sample();m.provenance['input_rig']={'declared':True}
    assert estimate(m)[1]['reason']=='declared_rig_requires_declared_centres'


def test_registration_keeps_only_three_parent_translations_and_saved_fk():
    import opensim as osim
    import biomech
    osim.Logger.setLevelString('error')
    with TemporaryDirectory(dir=ROOT/'develop_logs/test_runs/20261118_k4_next') as d:
        paths=[Path(d)/x for x in ['before.osim','after.osim']]
        model=osim.Model(str(biomech.MODEL));model.initSystem();model.printToXML(str(paths[0]))
        receipt=register(model,sample());model.printToXML(str(paths[1]))
        restored=osim.Model(str(paths[1]));state=restored.initSystem()
        for joint,row in receipt['changes'].items():
            np.testing.assert_allclose(biomech._child_origin(restored,state,joint),row['target_neutral_world_m'],atol=1e-9)
        trees=[ET.parse(p).getroot() for p in paths]
        for tree in trees:
            for joint in ['back','acromial_l','acromial_r']:
                element=tree.find('.//CustomJoint[@name="'+joint+'"]')
                frames=element.find('frames')
                parent=element.find('socket_parent_frame').text.rsplit('/',1)[-1]
                offset=next(x for x in frames if x.attrib.get('name')==parent)
                offset.find('translation').text='REGISTERED_TRANSLATION'
        assert ET.tostring(trees[0])==ET.tostring(trees[1])
