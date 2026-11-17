"""Independent native-frame head scoring; never imported by runtime fitting."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import valid_rotation
from .head_neck import interpolate_rotations
from .native_frames import read
from .scoring import summary


def distance(a,b):
    good=valid_rotation(a)&valid_rotation(b);out=np.full(len(a),np.nan)
    out[good]=np.rad2deg(Rotation.from_matrix(a[good].swapaxes(-1,-2)@b[good]).magnitude())
    return out


def compare(reference,degraded,fitted,times,tolerance_mask):
    """R/D determine the same observable samples for all three comparisons."""
    ri={n:i for i,n in enumerate(reference.names)}
    chest='Spine3' if 'Spine3' in ri else 'Spine2'
    rr=reference.rotations[:,ri[chest]].swapaxes(-1,-2)@reference.rotations[:,ri['Head']]
    native=read(degraded)
    unknown=np.full((len(times),3,3),np.nan)
    r=interpolate_rotations(reference.times,rr,times)
    d=(interpolate_rotations(degraded.times,native['head_relative_thorax'],times)
       if native['status']=='declared_native_frames' else unknown.copy())
    f=(interpolate_rotations(fitted['times'],fitted['head_relative_thorax'],times)
       if 'head_relative_thorax' in fitted else unknown.copy())
    observed=valid_rotation(r)&valid_rotation(d)
    for a in (r,d,f):a[~observed]=np.nan
    frames={'R':r,'D':d,'F':f};result={};temporal={}
    strata={'full':np.ones(len(times),bool),'continuous':~tolerance_mask,'source_jump_tolerance':tolerance_mask}
    for label,x,y in [('D-R','R','D'),('F-D','D','F'),('F-R','R','F')]:
        a,b=frames[x],frames[y];error=distance(a,b)
        result[label]={};temporal[label]={}
        # Relative rotation step geodesics are coordinate-chart invariant.
        sa,sb=distance(a[:-1],a[1:]),distance(b[:-1],b[1:])
        good=valid_rotation(a[:-1])&valid_rotation(a[1:])&valid_rotation(b[:-1])&valid_rotation(b[1:])
        velocity_error=np.full(len(times)-1,np.nan)
        va=Rotation.from_matrix(a[1:][good]@a[:-1][good].swapaxes(-1,-2)).as_rotvec()
        vb=Rotation.from_matrix(b[1:][good]@b[:-1][good].swapaxes(-1,-2)).as_rotvec()
        velocity_error[good]=np.rad2deg(np.linalg.norm(vb-va,axis=-1))/np.diff(times)[good]
        for name,mask in strata.items():
            result[label][name]=summary(error[mask],[5,10,20])
            steps=mask[:-1]&mask[1:]&observed[:-1]&observed[1:]
            temporal[label][name]={'reference_step_deg':summary(sa[steps]),'output_step_deg':summary(sb[steps]),
                'angular_speed_error_deg_s':summary(abs(sb-sa)[steps]/np.diff(times)[steps]),
                'angular_velocity_error_deg_s':summary(velocity_error[steps])}
    return {'head_neck_rotation_comparisons':result,'head_neck_rotation_temporal':temporal,
            'head_neck_rotation_scope':'Relative declared source-native head/chest frames, derived from fitted chain FK; no anatomical rotation claim; common R/D support'}
