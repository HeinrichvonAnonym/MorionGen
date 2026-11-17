"""Independent geometry and parallel-transport invariants for follow-up candidates."""
from pathlib import Path
import sys
import copy
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.branch_continuity import transport_direction,weak_flexion_weight
from motiogen.evalkit.k.position_geometry import trunk_chain_proxy
from motiogen.evalkit.k.motion import Motion


def test_transport_maps_direction_and_is_global_basis_equivariant():
    a=np.array([0.,1,0]);b=np.array([.6,.8,0]);r=transport_direction(a,b)
    np.testing.assert_allclose(r@a,b,atol=1e-12)
    np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-12)
    q=Rotation.from_euler('xyz',[.4,-.3,.8]).as_matrix()
    np.testing.assert_allclose(transport_direction(q@a,q@b),q@r@q.T,atol=1e-12)
    assert transport_direction(a,-a) is None
    assert transport_direction(a,np.zeros(3)) is None
    assert weak_flexion_weight(0,30)==1
    assert weak_flexion_weight(np.deg2rad(40),30)==0


def test_chain_proxy_fixed_invariant_and_keeps_neutral_unknown():
    names=['left_hip','right_hip','spine1','spine2','spine3','left_shoulder','right_shoulder']
    p=np.array([[-.1,0,0],[.1,0,0],[0,.1,0],[0,.2,0],[0,.35,0],[-.2,.5,0],[.2,.5,0]])
    m=Motion(np.tile(p,(3,1,1)),names,np.arange(3,dtype=float),'test')
    value,receipt=trunk_chain_proxy(m)
    assert abs(value-.5)<1e-12 and receipt['neutral_length_identified'] is False
    other=copy.deepcopy(m);other.positions=other.positions@Rotation.from_rotvec([.4,.2,-.3]).as_matrix().T+[5.,-2,8]
    other.provenance['reference']='must not be read'
    assert abs(trunk_chain_proxy(other)[0]-value)<1e-12
    m.positions[:,3]=np.nan
    assert trunk_chain_proxy(m)[0] is None
