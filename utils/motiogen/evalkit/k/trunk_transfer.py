"""Fixed-rig trunk-direction transfer with endpoint segment frames held fixed.

This is an explicit retargeting objective, not a source rotation convention.
Only internal spine frames receive a minimum swing; pelvis, thorax, head and
limb world rotations remain unchanged. Infeasible target geometry is reported.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def minimum_swing(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    cross = np.cross(a, b); sine = np.linalg.norm(cross)
    cosine = float(np.clip(a @ b, -1., 1.))
    if sine < 1e-12:
        if cosine > 0: return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(abs(a))])
        return Rotation.from_rotvec(np.pi * axis / np.linalg.norm(axis)).as_matrix()
    return Rotation.from_rotvec(cross / sine * np.arctan2(sine, cosine)).as_matrix()


def transfer(reference, source_names, names, source_for, rest, parents, rotations, root):
    indices = {n:i for i,n in enumerate(source_names)}
    required = ['left_hip','right_hip','left_shoulder','right_shoulder']
    if any(n not in indices or n not in source_for for n in required):
        raise ValueError('trunk_transfer_requires_bilateral_hip_and_shoulder_observations')
    r = rotations.copy()
    internal = [j for j,s in enumerate(source_for) if s in {'spine1','spine2','spine_mid'}]
    if not internal: raise ValueError('trunk_transfer_requires_internal_spine')
    def fk(rot):
        p = np.zeros(rot.shape[:2]+(3,))
        for j,par in enumerate(parents):
            p[:,j] = root if par < 0 else p[:,par] + rot[:,par] @ (rest[j]-rest[par])
        return p
    def trunk(p, ids):
        return .5*(p[:,ids[2]]+p[:,ids[3]]-p[:,ids[0]]-p[:,ids[1]])
    original = fk(r); ti = [source_for.index(n) for n in required]
    wanted = trunk(reference.positions,[indices[n] for n in required])
    # Linear FK lets us isolate the aggregate displacement controlled by the
    # internal frames without modifying any rest offsets or endpoint frames.
    zeroed = r.copy(); zeroed[:,internal] = 0
    fixed = trunk(fk(zeroed),ti); mobile = trunk(original,ti)-fixed
    feasible=[];swings=[]
    for i,(w,c,v) in enumerate(zip(wanted,fixed,mobile)):
        length=np.linalg.norm(w); radius=np.linalg.norm(v)
        if length<1e-6 or radius<1e-6:
            feasible.append(False);swings.append(0.);continue
        d=w/length; dot=float(c@d); disc=radius**2-(c@c-dot**2)
        if disc < -1e-12:
            feasible.append(False);swings.append(0.);continue
        candidates=[s*d-c for s in [dot+np.sqrt(max(0.,disc)),dot-np.sqrt(max(0.,disc))] if s>1e-6]
        if not candidates:
            feasible.append(False);swings.append(0.);continue
        dest=max(candidates,key=lambda x:float(x@v));q=minimum_swing(v,dest)
        r[i,internal]=q@r[i,internal]
        feasible.append(True);swings.append(float(np.degrees(Rotation.from_matrix(q).magnitude())))
    return fk(r),r,{'objective':'source hip-midpoint to shoulder-midpoint direction on fixed target geometry',
        'source':'project retargeting objective; derived, not independent rotation observation',
        'corrected_frames':feasible,'swing_deg':swings,'internal_nodes':[names[j] for j in internal],
        'preserved':'target rest offsets; pelvis, thorax, head and limb world frames',
        'infeasible_policy':'retain topology_v2 result; record unsupported direction transfer'}
