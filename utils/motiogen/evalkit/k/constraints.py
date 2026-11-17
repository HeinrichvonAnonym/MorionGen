"""Evaluate declared coordinate couplers without changing independent pose."""


def couplers(model,state):
    import opensim as osim
    result=[];constraints=model.getConstraintSet();coordinates=model.getCoordinateSet()
    for i in range(constraints.getSize()):
        constraint=constraints.get(i)
        if not constraint.isEnforced(state):continue
        c=osim.CoordinateCouplerConstraint.safeDownCast(constraint)
        if c is None:raise ValueError('Direct coordinate update requires coordinate-coupler constraints only')
        factor=float(c.getPropertyByName('scale_factor').toString())
        if factor!=1.:raise ValueError('Non-unit coupling scale needs a separately verified convention')
        names=c.getIndependentCoordinateNames()
        result.append((coordinates.get(c.getDependentCoordinateName()),
                       [coordinates.get(names.get(j)) for j in range(names.getSize())],c.getFunction()))
    return result


def synchronize(state,bindings):
    import opensim as osim
    for dependent,independent,function in bindings:
        value=function.calcValue(osim.Vector([c.getValue(state) for c in independent]))
        dependent.setValue(state,value,False)
