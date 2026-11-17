"""Explicit kinematic-only constraint intervention on a per-fit model copy."""


def apply(model_path,policy):
    import opensim as osim
    from .config import digest
    if policy!='release_subtalar_original_range':raise ValueError('Unknown diagnostic model-domain policy')
    before=digest(model_path);model=osim.Model(str(model_path));changes={}
    for side in ('l','r'):
        name='subtalar_angle_'+side;c=model.updCoordinateSet().get(name)
        changes[name]={'locked_before':c.getDefaultLocked(),'locked_after':False,
                       'range_native_before':[c.getRangeMin(),c.getRangeMax()]}
        c.setDefaultLocked(False)
        changes[name]['range_native_after']=[c.getRangeMin(),c.getRangeMax()]
    model.finalizeConnections();model.initSystem();model.printToXML(str(model_path))
    return {'policy':policy,'changes':changes,'model_sha256_before':before,'model_sha256_after':digest(model_path),
            'scope':'Diagnostic copy only: existing subtalar DOFs released inside original asset ranges; no structural change or force qualification'}
