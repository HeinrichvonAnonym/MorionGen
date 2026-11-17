"""Model-verified hip SO(3) interpolation for independent kinematics.

Euler coordinates describe a chart, not a vector-space trajectory. This optional
path preserves native joint rotations across chart changes; model FK still
generates every exported position. No source/reference positions are used.
"""
import warnings
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation,Slerp


def hip_slerp(model_path,labels,coordinates,times,new_times,interpolated):
    import opensim as osim
    import biomech
    from .constraints import couplers,synchronize
    from .config import digest
    labels=list(labels);times=np.asarray(times);new_times=np.asarray(new_times)
    if len(times)<2 or np.any(np.diff(times)<=0) or new_times[0]<times[0] or new_times[-1]>times[-1]:
        raise ValueError('Hip interpolation requires ordered bounded clocks')
    model=osim.Model(str(model_path));state=model.initSystem();bindings=couplers(model,state)
    root=ET.parse(model_path);result=np.array(interpolated,copy=True);records={}
    for side in ('l','r'):
        joint_name='hip_'+side;joint=model.getJointSet().get(joint_name)
        xml=root.find(f".//CustomJoint[@name='{joint_name}']")
        if xml is None:raise ValueError('Declared hip CustomJoint is missing')
        order='';names=[];slopes=[]
        for i in range(3):
            axis=xml.find(f"./SpatialTransform/TransformAxis[@name='rotation{i+1}']")
            vector=np.fromstring(axis.findtext('axis',''),sep=' ')
            coefficients=np.fromstring(axis.findtext('./LinearFunction/coefficients',''),sep=' ')
            name=axis.findtext('coordinates','').strip()
            if vector.shape!=(3,) or coefficients.shape!=(2,) or abs(coefficients[0])!=1 or coefficients[1]!=0:
                raise ValueError('Unsupported hip rotation function')
            j=int(np.argmax(abs(vector)))
            if not np.array_equal(abs(vector),np.eye(3)[j]) or name not in labels:
                raise ValueError('Unsupported hip rotation axis or coordinate')
            order+='XYZ'[j];names.append(name);slopes.append(coefficients[0]*vector[j])
        if len(set(order))!=3 or len(set(names))!=3:raise ValueError('Hip needs three distinct Euler axes')
        cols=[labels.index(n) for n in names];slopes=np.asarray(slopes)
        rotations=Rotation.from_euler(order,coordinates[:,cols]*slopes,degrees=True)
        error=0.
        for i in sorted(set([0,len(times)//2,len(times)-1])):
            biomech._set(model,state,labels,coordinates[i]);synchronize(state,bindings);model.realizePosition(state)
            def matrix(frame):
                transform=frame.getTransformInGround(state);r=transform.R()
                return np.array([[r.get(j,k) for k in range(3)] for j in range(3)])
            actual=matrix(joint.getParentFrame()).T@matrix(joint.getChildFrame())
            error=max(error,float(np.max(abs(actual-rotations[i].as_matrix()))))
        if error>1e-8:raise ValueError('Declared hip composition disagrees with OpenSim FK')
        target=Slerp(times,rotations)(new_times)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',UserWarning)
            angles=target.as_euler(order,degrees=True)/slopes
        for j,name in enumerate(names):
            c=model.getCoordinateSet().get(name);bounds=np.rad2deg([c.getRangeMin(),c.getRangeMax()])
            if angles[:,j].min()<bounds[0]-1e-7 or angles[:,j].max()>bounds[1]+1e-7:
                raise ValueError('Canonical hip chart exceeds declared independent domain')
        result[:,cols]=angles
        reconstruction=Rotation.from_euler(order,angles*slopes,degrees=True)
        delta=float(np.max((target.inv()*reconstruction).magnitude()))
        if delta>1e-10:raise ValueError('Hip interpolation chart failed to reconstruct target')
        records[joint_name]={'coordinates':names,'order':order,'signs':slopes.tolist(),
            'native_model_matrix_error':error,'interpolated_rotation_error_rad':delta,
            'euler_chart_warnings':[str(w.message) for w in caught]}
    return result,{'version':'hip-joint-slerp-1','model_sha256':digest(model_path),'joints':records,
        'scope':'Joint rotation interpolation only; native poses unchanged; knee couplings and all other coordinates retain their existing interpolation'}
