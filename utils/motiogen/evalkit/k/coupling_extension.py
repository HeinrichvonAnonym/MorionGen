"""Explicit high-flexion continuation for independent kinematics, never force use.

The original primary knee coordinate remains unchanged. Auxiliary rotations
return smoothly to zero at full folding; translations approach their boundary
values. This is a declared geometric extension, not a physiological model.
"""
import numpy as np
from .config import digest


def hermite(q,q0,q1,value,slope,end_value):
    t=(np.asarray(q)-q0)/(q1-q0)
    return (2*t**3-3*t**2+1)*value+(t**3-2*t**2+t)*(q1-q0)*slope+(-2*t**3+3*t**2)*end_value


def apply(path,policy):
    import opensim as osim
    if policy!={'version':'knee_auxiliary_hermite_v1','boundary_deg':140.,'end_deg':180.,'grid_deg':.1}:
        raise ValueError('Unknown frozen knee continuation policy')
    original_hash=digest(path);model=osim.Model(str(path));records={}
    q0,q1=np.deg2rad([policy['boundary_deg'],policy['end_deg']]);grid=np.deg2rad(np.linspace(0,180,1801))
    for s in ('l','r'):
        joint=osim.CustomJoint.safeDownCast(model.updJointSet().get('walker_knee_'+s));st=joint.updSpatialTransform()
        for i in range(1,6):
            axis=st.updTransformAxis(i);fun=axis.getFunction()
            value=lambda q:float(fun.calcValue(osim.Vector([float(q)])))
            y0=value(q0);h=1e-5;slope=(value(q0+h)-value(q0-h))/(2*h);end=0. if i<3 else y0
            values=np.array([value(q) if q<=q0 else hermite(q,q0,q1,y0,slope,end) for q in grid],float)
            spline=osim.SimmSpline()
            for x,y in zip(grid,values):spline.addPoint(float(x),float(y))
            # Independently evaluate between knots, where an interpolation bug shows.
            checks=np.deg2rad(np.arange(.05,140.,.1));error=max(abs(spline.calcValue(osim.Vector([float(q)]))-value(q)) for q in checks)
            left=(spline.calcValue(osim.Vector([q0]))-spline.calcValue(osim.Vector([q0-1e-6])))/1e-6
            right=(spline.calcValue(osim.Vector([q0+1e-6]))-spline.calcValue(osim.Vector([q0])))/1e-6
            if error>1e-5 or abs(left-right)>1e-4:raise ValueError('Original-domain/transition preservation failed')
            records[joint.getName()+'/'+str(i)]={'original_function':fun.getConcreteClassName(),'boundary_value_native':y0,
                'boundary_derivative_native':slope,'end_value_native':end,'original_domain_max_error_native':float(error),
                'boundary_derivative_jump_native':float(abs(left-right)),'units':'rad' if i<3 else 'm'}
            axis.set_function(spline)
    model.finalizeConnections();model.initSystem();model.printToXML(str(path))
    return {'policy':policy,'functions':records,'model_sha256_before':original_hash,'model_sha256_after':digest(path),
        'scope':'Independent geometric high-flexion continuation; primary axis/DOFs unchanged; auxiliary rotations ease to zero and translations to boundary values; no physiological or force validation'}
