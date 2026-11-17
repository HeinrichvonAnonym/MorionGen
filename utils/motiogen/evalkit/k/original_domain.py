"""Original asset limits remain an independent diagnostic, never a fit selector."""
import numpy as np
from .config import digest


def assess(original_model,labels,coordinates):
    import opensim as osim
    model=osim.Model(str(original_model));state=model.initSystem();cs=model.getCoordinateSet();rows={}
    for j,name in enumerate(labels):
        c=cs.get(name);angular=c.getMotionType()==1;factor=180/np.pi if angular else 1.
        lo,hi=c.getRangeMin()*factor,c.getRangeMax()*factor;values=coordinates[:,j]
        finite=np.isfinite(values);beyond=np.maximum(np.maximum(lo-values,values-hi),0)
        departure=np.abs(values-c.getDefaultValue()*factor)
        rows[str(name)]={'units':'deg' if angular else 'native','original_range':[lo,hi],
            'original_clamped':c.getDefaultClamped(),'original_locked':c.getDefaultLocked(),'dependent':c.isDependent(state),
            'frames':len(values),'finite_frames':int(finite.sum()),'outside_range_frames':int(np.sum(finite&(beyond>1e-6))),
            'max_range_exceedance':float(np.max(beyond[finite])) if finite.any() else None,
            'moved_from_original_lock_frames':int(np.sum(finite&(departure>1e-6))) if c.getDefaultLocked() else 0}
    return {'source_sha256':digest(original_model),'coordinates':rows,
            'scope':'Original asset range/lock diagnostics; no physiological classification or force qualification; no clipping of the independent kinematic output'}
