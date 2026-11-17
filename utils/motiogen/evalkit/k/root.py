"""Input-only global translation after pose fitting; no reference offsets."""
import numpy as np


def compensate(model_path, times, labels, coordinates, motion):
    from .k4_fitting import model_fk
    # Pose has already been constraint-assembled. Reassembling at this stage
    # can numerically move coordinates lying on a range boundary; translation
    # leaves all internal constraints unchanged and does not require it.
    before,names=model_fk(model_path,labels,coordinates,assemble=False)
    index={n:i for i,n in enumerate(motion.names)}
    observed=(motion.positions[:,index['left_hip']]+motion.positions[:,index['right_hip']])/2
    target=np.stack([np.interp(times,motion.times,observed[:,j]) for j in range(3)],1)
    valid=(times>=motion.times[0])&(times<=motion.times[-1])&np.isfinite(target).all(-1)
    model_mid=(before[:,names.index('left_hip')]+before[:,names.index('right_hip')])/2
    shift=np.where(valid[:,None],target-model_mid,0.)
    changed=np.array(coordinates,copy=True)
    for j,name in enumerate(('pelvis_tx','pelvis_ty','pelvis_tz')):changed[:,list(labels).index(name)]+=shift[:,j]
    after,_,consistent=model_fk(model_path,labels,changed,assemble=False,return_coordinates=True)
    # Verify the model's translation convention instead of assuming offsets
    # can be applied directly to scored endpoints without FK.
    residual=after-before-shift[:,None]
    if not np.isfinite(residual).all() or np.max(abs(residual))>1e-6:
        raise ValueError('Model root is not the declared world-translation convention')
    return consistent,{'before':before,'after':after,'shift_m':shift,'valid':valid,
                       'source':'D observed left/right hip midpoint; post-pose world translation'}
