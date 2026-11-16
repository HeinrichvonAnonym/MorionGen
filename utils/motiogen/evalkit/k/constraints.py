"""Describe the effective coordinate domain without changing the model state."""
import numpy as np
import xml.etree.ElementTree as ET


def coordinate_contract(model,state,names):
    cs=model.getCoordinateSet();records={}
    for name in names:
        c=cs.get(name);locked=bool(c.getLocked(state));constrained=bool(c.isConstrained(state))
        records[name]={'range_deg':[float(np.degrees(c.getRangeMin())),float(np.degrees(c.getRangeMax()))],
            'locked':locked,'constrained':constrained,'initial_value_deg':float(np.degrees(c.getValue(state))),
            'geometry_role':'fixed_by_model' if locked else 'dependent_or_prescribed' if constrained else 'independent_coordinate',
            'solver_role':'listed_variable_but_geometry_fixed' if locked else 'requested_variable'}
    return {'coordinates':records,'range_source':'supplied model asset; solver uses these as hard box bounds',
        'interpretation':'model domain and modeling constraints; not universal physiological bounds',
        'locked_variables':'legacy optimizer can list these parameters, but OpenSim refuses their changes; no independent observed DOF is gained'}


def synchronize_couplers(model_path,model,labels,data):
    """Honor declared identity couplers in native units after interpolation.

    OpenSim storage converts type-1 rotational coordinates to degrees, while
    type-3 coupled coordinates remain native. Unsupported coupling functions
    fail explicitly rather than silently leaving a stale dependent trajectory.
    """
    out=np.array(data,copy=True);cs=model.getCoordinateSet();pending=[];receipts=[]
    for constraint in ET.parse(model_path).findall('.//CoordinateCouplerConstraint'):
        if constraint.findtext('isEnforced','true').strip()=='false':continue
        sources=constraint.findtext('independent_coordinate_names','').split()
        target=constraint.findtext('dependent_coordinate_name','').strip()
        coefficients=np.fromstring(constraint.findtext('coupled_coordinates_function/LinearFunction/coefficients',''),sep=' ')
        if len(sources)!=1 or coefficients.shape!=(2,) or not np.array_equal(coefficients,[1.,0.]) or float(constraint.findtext('scale_factor','1'))!=1.:
            raise ValueError('unsupported_coordinate_coupler: '+constraint.attrib.get('name',''))
        if sources[0] not in labels or target not in labels:
            raise ValueError('missing_coordinate_coupler_column: '+target)
        pending.append((sources[0],target))
    while pending:
        available=[pair for pair in pending if pair[0] not in {target for _,target in pending}]
        if not available:raise ValueError('cyclic_coordinate_couplers')
        for source,target in available:
            a,b=labels.index(source),labels.index(target)
            native=np.deg2rad(out[:,a]) if cs.get(source).getMotionType()==1 else out[:,a]
            value=np.rad2deg(native) if cs.get(target).getMotionType()==1 else native
            receipts.append({'independent':source,'dependent':target,'function':'identity from model XML',
                'source_storage':'degrees' if cs.get(source).getMotionType()==1 else 'native',
                'dependent_storage':'degrees' if cs.get(target).getMotionType()==1 else 'native',
                'maximum_dependent_change_in_storage_units':float(np.max(abs(out[:,b]-value)))})
            out[:,b]=value;pending.remove((source,target))
    return out,{'status':'model_couplers_synchronized','couplers':receipts,'reference_access':False}
