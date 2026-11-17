"""Explicit high-flexion continuation for independent kinematics, never force use.

The original primary knee coordinate remains unchanged. Auxiliary rotations
approach declared endpoints (fixed-FK folding in v2); translations approach their
boundary values. This is a geometric extension, not a physiological model.
"""
import numpy as np
from .config import digest


def hermite(q,q0,q1,value,slope,end_value):
    t=(np.asarray(q)-q0)/(q1-q0)
    return (2*t**3-3*t**2+1)*value+(t**3-2*t**2+t)*(q1-q0)*slope+(-2*t**3+3*t**2)*end_value


def endpoint_from_geometry(model,side,boundary,end):
    """Find two auxiliary endpoint rotations by FK, independently of all motion."""
    import opensim as osim
    import biomech
    from scipy.optimize import least_squares
    from .constraints import couplers,synchronize
    probe=osim.Model(model);joint=osim.CustomJoint.safeDownCast(probe.updJointSet().get('walker_knee_'+side));st=joint.updSpatialTransform()
    for i in range(3,6):
        axis=st.updTransformAxis(i);value=axis.getFunction().calcValue(osim.Vector([boundary]));axis.set_function(osim.Constant(value))
    def residual(x):
        for i,value in zip((1,2),x):st.updTransformAxis(i).set_function(osim.Constant(float(value)))
        probe.finalizeConnections();state=probe.initSystem();bindings=couplers(probe,state)
        probe.updCoordinateSet().get('knee_angle_'+side).setValue(state,end,False)
        synchronize(state,bindings);probe.realizePosition(state)
        hip=biomech._child_origin(probe,state,'hip_'+side);knee=biomech._child_origin(probe,state,'walker_knee_'+side);ankle=biomech._child_origin(probe,state,'ankle_'+side)
        u=hip-knee;v=ankle-knee;u/=np.linalg.norm(u);v/=np.linalg.norm(v)
        return np.r_[u-v,1e-6*x]
    fit=least_squares(residual,np.zeros(2),bounds=(-np.pi/4,np.pi/4),diff_step=1e-4,max_nfev=60,ftol=1e-12,xtol=1e-12,gtol=1e-12)
    mismatch=np.linalg.norm(residual(fit.x)[:3]);error=float(np.rad2deg(2*np.arcsin(min(1.,mismatch/2))))
    if error>.01:raise ValueError('FK full-fold endpoint not reached within declared auxiliary bound')
    return fit.x,{'auxiliary_rotations_rad':fit.x.tolist(),'folding_direction_error_deg':error,
                 'coordinate_rad':end,'status':int(fit.status),'nfev':fit.nfev,
                 'source':'Fixed model FK only; no D poses or R; auxiliary search ±45deg numerical guard'}


def apply(path,policy):
    import opensim as osim
    if policy not in ({'version':v,'boundary_deg':140.,'end_deg':180.,'grid_deg':.1} for v in ('knee_auxiliary_hermite_v1','knee_auxiliary_hermite_v2')):
        raise ValueError('Unknown frozen knee continuation policy')
    original_hash=digest(path);model=osim.Model(str(path));records={};endpoints={}
    q0,q1=np.deg2rad([policy['boundary_deg'],policy['end_deg']]);grid=np.deg2rad(np.linspace(0,180,1801))
    for s in ('l','r'):
        endpoint=np.zeros(2)
        if policy['version']=='knee_auxiliary_hermite_v2':endpoint,endpoints[s]=endpoint_from_geometry(model,s,q0,q1)
        joint=osim.CustomJoint.safeDownCast(model.updJointSet().get('walker_knee_'+s));st=joint.updSpatialTransform()
        for i in range(1,6):
            axis=st.updTransformAxis(i);fun=axis.getFunction()
            value=lambda q:float(fun.calcValue(osim.Vector([float(q)])))
            y0=value(q0);h=1e-5;slope=(value(q0+h)-value(q0-h))/(2*h);end=float(endpoint[i-1]) if i<3 else y0
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
    return {'policy':policy,'endpoint_fk':endpoints,'functions':records,'model_sha256_before':original_hash,'model_sha256_after':digest(path),
        'scope':'Independent geometric high-flexion continuation; primary axis/DOFs unchanged; auxiliary rotations ease to the declared FK endpoint and translations to boundary values; no physiological or force validation'}
