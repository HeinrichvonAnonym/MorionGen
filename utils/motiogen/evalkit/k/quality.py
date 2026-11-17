"""K4 runtime input fidelity; no reference-based accuracy qualification."""
import numpy as np
from .timebase import supported


def assess(motion,times,positions,names):
    source_names=list(motion.legacy()[1]);source=motion.positions
    goal=np.stack([np.stack([np.interp(times,motion.times,source[:,source_names.index(name),axis])
                            for axis in range(3)],1) for name in names],1)
    time_valid=supported(times,motion.times[0],motion.times[-1])
    observed=np.isfinite(goal).all(-1)&time_valid[:,None]
    finite=np.isfinite(positions).all(-1)&observed
    error=np.linalg.norm(positions-goal,axis=-1)
    error[~finite]=np.nan
    idx={n:i for i,n in enumerate(names)}
    pairs={}
    for side in ('left','right'):
        for part,a,b in [('upper_arm','shoulder','elbow'),('forearm','elbow','wrist'),
                         ('thigh','hip','knee'),('shank','knee','ankle'),('foot_axis','ankle','foot')]:
            pairs[side+'_'+part]=(idx[side+'_'+a],idx[side+'_'+b])
    errors=[]
    for a,b in pairs.values():
        x=goal[:,b]-goal[:,a];y=positions[:,b]-positions[:,a]
        den=np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)
        angle=np.rad2deg(np.arccos(np.clip(np.sum(x*y,axis=-1)/np.maximum(den,1e-15),-1,1)))
        angle[(den<1e-12)|~time_valid]=np.nan;errors.append(angle)
    angles=np.stack(errors,1)
    return {'names':np.asarray(names),'position_error_m':error,'position_observed':observed,
            'position_scorable':finite,'position_input_fidelity_5cm':finite&(error<=.05),
            'direction_names':np.asarray(list(pairs)), 'direction_error_deg':angles,
            'direction_scorable':np.isfinite(angles),'direction_input_fidelity_5deg':angles<=5}
