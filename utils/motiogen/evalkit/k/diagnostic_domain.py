"""Explicit kinematic-only constraint intervention on a per-fit model copy."""


def apply(model_path,policy):
    import opensim as osim
    from .config import digest
    import numpy as np
    ranges={}
    if isinstance(policy,dict):
        if set(policy)!={'base','coordinate_ranges_deg','scope'} or policy['base']!='release_subtalar_original_range':
            raise ValueError('Explicit diagnostic range contract required')
        if policy['scope']!='counterfactual_kinematic_only':raise ValueError('Range expansion is diagnostic only')
        ranges=policy['coordinate_ranges_deg']
    elif policy!='release_subtalar_original_range':raise ValueError('Unknown diagnostic model-domain policy')
    before=digest(model_path);model=osim.Model(str(model_path));changes={}
    for side in ('l','r'):
        name='subtalar_angle_'+side;c=model.updCoordinateSet().get(name)
        changes[name]={'locked_before':c.getDefaultLocked(),'locked_after':False,
                       'range_native_before':[c.getRangeMin(),c.getRangeMax()]}
        c.setDefaultLocked(False)
        changes[name]['range_native_after']=[c.getRangeMin(),c.getRangeMax()]
    for name,bounds in ranges.items():
        c=model.updCoordinateSet().get(name)
        if c.getMotionType()!=1 or len(bounds)!=2 or not np.isfinite(bounds).all() or bounds[0]>=bounds[1]:
            raise ValueError('Invalid diagnostic angular range')
        changes.setdefault(name,{'locked_before':c.getDefaultLocked(),'locked_after':c.getDefaultLocked(),
                                 'range_native_before':[c.getRangeMin(),c.getRangeMax()]})
        c.setRangeMin(float(np.deg2rad(bounds[0])));c.setRangeMax(float(np.deg2rad(bounds[1])))
        changes[name]['range_native_after']=[c.getRangeMin(),c.getRangeMax()]
    model.finalizeConnections();model.initSystem();model.printToXML(str(model_path))
    return {'policy':policy,'changes':changes,'model_sha256_before':before,'model_sha256_after':digest(model_path),
            'scope':('Counterfactual range expansion on diagnostic copy; no anatomical range validation, structural change, or force qualification' if ranges else 'Diagnostic copy only: existing subtalar DOFs released inside original asset ranges; no structural change or force qualification')}
