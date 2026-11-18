from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.rotation_chart import interior


def test_same_shoulder_rotation_moves_away_from_artificial_chart_boundary():
    q=np.deg2rad([[160,-120,-180],[160,-75,-18],[10,20,30]])
    lo=np.deg2rad([-90,-120,-180]);hi=np.deg2rad([160,120,180])
    result,receipt=interior(q,'ZXY',[1,1,1],lo,hi)
    np.testing.assert_allclose(Rotation.from_euler('ZXY',result).as_matrix(),Rotation.from_euler('ZXY',q).as_matrix(),atol=1e-10)
    assert np.all(result>=lo) and np.all(result<=hi)
    assert min(result[0]-lo)>0.1 and min(hi-result[0])>0.1
    assert min(result[1]-lo)>0.1 and min(hi-result[1])>0.1
    assert receipt['changed_frames']==[0,1]


def test_signed_axes_and_gimbal_keep_physical_rotation_and_domain():
    q=np.deg2rad([[30,90,50],[-120,-35,170]])
    signs=np.array([-1,1,-1]);lo=np.full(3,-np.pi);hi=np.full(3,np.pi)
    result,receipt=interior(q,'XYZ',signs,lo,hi)
    np.testing.assert_allclose(Rotation.from_euler('XYZ',result*signs).as_matrix(),Rotation.from_euler('XYZ',q*signs).as_matrix(),atol=1e-10)
    assert np.all(result>=lo) and np.all(result<=hi)
