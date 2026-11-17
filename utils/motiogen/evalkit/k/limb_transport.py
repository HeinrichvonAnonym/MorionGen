"""Preserve accepted limb world orientations after changing a parent pose.

Uses two fitted models and the observed input only. It supplies a candidate,
not an independent rotation observation; source-fidelity guards decide promotion.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def transport(old_path, new_path, labels, old, candidate, limbs=('legs',)):
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.constraints import synchronize_couplers
    models=[osim.Model(str(p)) for p in (old_path,new_path)]
    states=[m.initSystem() for m in models];model=models[1];state=states[1]
    cs=model.getCoordinateSet();out=candidate.copy();reports={}
    def rotation(m,s,body):
        r=m.getBodySet().get(body).getTransformInGround(s).R()
        return np.array([[r.get(i,j) for j in range(3)] for i in range(3)])
    for kind in limbs:
        for side in ('l','r'):
            if kind=='legs':
                free=['hip_flexion','hip_adduction','hip_rotation'];fixed=['knee_angle','ankle_angle','subtalar_angle'];body='femur_'+side
            elif kind=='arms':
                free=['arm_flex','arm_add','arm_rot'];fixed=['elbow_flex','pro_sup'];body='humerus_'+side
            else:raise ValueError(kind)
            columns=[labels.index(n+'_'+side) for n in free];coords=[cs.get(labels[c]) for c in columns]
            if any(c.getLocked(state) or c.isConstrained(state) for c in coords):
                raise ValueError('transport requires independent parent coordinates')
            lo=np.array([c.getRangeMin() for c in coords])+1e-7;hi=np.array([c.getRangeMax() for c in coords])-1e-7
            kept=[labels.index(n+'_'+side) for n in fixed if n+'_'+side in labels];out[:,kept]=old[:,kept]
            previous=None;errors=[]
            for i in range(len(out)):
                if not (np.isfinite(out[i]).all() and np.isfinite(old[i]).all()):
                    previous=None;errors.append(None);continue
                biomech._set(models[0],states[0],labels,old[i]);wanted=rotation(models[0],states[0],body)
                biomech._set(model,state,labels,out[i])
                def residual(x):
                    for c,v in zip(coords,x):c.setValue(state,float(v),False)
                    model.realizePosition(state)
                    return Rotation.from_matrix(rotation(model,state,body).T@wanted).as_rotvec()
                seed=np.clip(np.deg2rad(out[i,columns]),lo,hi)
                seeds=[seed] if previous is None else [np.clip(previous,lo,hi),seed]
                best=None
                for initial in seeds:
                    sol=least_squares(residual,initial,bounds=(lo,hi),max_nfev=60)
                    if best is None or np.linalg.norm(sol.fun)<np.linalg.norm(best.fun):best=sol
                    if np.linalg.norm(best.fun)<1e-7:break
                out[i,columns]=np.rad2deg(best.x);previous=best.x
                errors.append(float(np.degrees(np.linalg.norm(best.fun))))
            finite=[v for v in errors if v is not None]
            reports[kind+'_'+side]={'orientation_error_p95_deg':float(np.percentile(finite,95)) if finite else None,
                'orientation_error_max_deg':max(finite) if finite else None,'valid':len(finite),'total':len(out)}
    out,coupling=synchronize_couplers(new_path,model,labels,out)
    return out,{'source':'accepted fitted limb world orientation, not independent observation',
        'geometry':'new fixed model retained','limbs':reports,'couplers':coupling}
