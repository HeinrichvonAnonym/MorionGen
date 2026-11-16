"""Eight actual output readers. Internal world basis: right-handed Y-up, metres."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import chain, valid_rotation, coordinate_precision_valid
from .config import CONFIG, digest
try:
    from ...motion import HUMANML_NAMES, SOMA77_NAMES, Y_TO_Z
except ImportError:
    from motion import HUMANML_NAMES, SOMA77_NAMES, Y_TO_Z

PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19]


@dataclass
class Motion:
    positions: np.ndarray
    names: list
    times: np.ndarray
    method: str
    rotations: np.ndarray | None = None
    rotation_names: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        original = np.asarray(self.positions)
        precision_valid = coordinate_precision_valid(original)
        self.positions = np.array(original, dtype=float, copy=True)
        # Natural missing points stay point-wise missing. Only precision failure
        # of finite world coordinates invalidates the complete positional frame.
        rejected = np.isfinite(original).all(axis=(-2,-1)) & ~precision_valid
        if rejected.any():
            self.positions[rejected] = np.nan
            self.provenance['coordinate_precision']={'status':'world_positions_unreliable',
                'rejected_frames':np.flatnonzero(rejected).tolist(),'resolution_m':1e-4,
                'rotation_policy':'native rotations retained independently'}
        self.times = np.asarray(self.times, float)
        if self.positions.shape != (len(self.times), len(self.names), 3) or len(set(self.names)) != len(self.names):
            raise ValueError('K position layout mismatch')
        if len(self.times) < 2 or not np.isfinite(self.times).all() or np.any(np.diff(self.times)<=0):
            raise ValueError('K needs a finite strictly increasing clock with at least two frames')
        if self.rotations is not None and self.rotations.shape != (len(self.times),len(self.rotation_names),3,3):
            raise ValueError('K rotation layout mismatch')

    @property
    def fps(self):
        return float(1/np.median(np.diff(self.times)))

    def legacy(self):
        if not np.allclose(np.diff(self.times), 1/self.fps, rtol=1e-3, atol=1e-6):
            raise ValueError('Full-body fitting requires an explicitly resampled uniform clock')
        names = self.names.copy()
        if 'left_wrist_body' in names:
            rename = {'left_wrist':'left_hand_point','right_wrist':'right_hand_point','left_wrist_body':'left_wrist','right_wrist_body':'right_wrist'}
            names = [rename.get(n,n) for n in names]
        if 'Spine' not in names:
            names = [{'spine1':'Spine','spine3':'Spine3'}.get(n,n) for n in names]
        return self.positions, names, self.fps


def load(source, method=None, config=None):
    from .config import resolve_config, ROOT
    settings = resolve_config(config)
    if isinstance(source, Motion):
        return source
    src = Path(source)
    file = src if src.suffix == '.npz' else src/'motion.npz'
    used = []
    def read(p):
        used.append(Path(p))
        with np.load(p, allow_pickle=False) as z:
            return {k:z[k] for k in z.files if k not in ('metadata','config')}
    m = read(file)
    meta = json.loads(str(m.get('metadata_json', '{}')))
    method = (method or meta.get('baseline') or meta.get('model','')).lower()
    if 'posed_joints' in m: method = 'kimodo'
    if 'pose' in m or 'factorial' in str(src): method = 'primal'
    if not method:
        candidates = [n for n in settings['profiles'] if n in str(src).lower()]
        method = candidates[0] if len(candidates)==1 else 'unknown'
    fps = float(m.get('fps', settings['profiles'].get(method,{}).get('fps',30)))
    r = None; rn = []; native_y = method == 'kimodo'
    if method == 'kimodo':
        p = np.asarray(m['posed_joints']); p = p[0] if p.ndim==4 else p
        names = list(SOMA77_NAMES); r = m.get('global_rot_mats'); rn = names
    elif method == 'primal':
        native = Path(str(file).replace('_canonical.npz','.npz'))
        a = m if 'pose' in m else read(native)
        pose = a['pose'][0]
        p = m['joints'] if 'joint_names' in m else a['joints'][0]
        names = list(HUMANML_NAMES)
        local = Rotation.from_rotvec(pose[:,3:].reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
        r = chain(local,PARENTS); rn = names
        if 'joint_names' in m:
            p = np.asarray(p) @ Y_TO_Z
        native_y = True
    else:
        p = m['joints']; names = [str(n) for n in m['joint_names']]
        if method == 'ardy': r = m.get('global_rotations'); rn = names
        elif method == 'chois' and (file.parent/'raw/native_prediction.npz').exists():
            a = read(file.parent/'raw/native_prediction.npz'); r = a['global_rotations']; rn = list(HUMANML_NAMES)
        elif method == 'dart' and (file.parent/'smplx_params.npz').exists():
            a = read(file.parent/'smplx_params.npz')
            r = chain(np.concatenate([a['global_orient'][:,None], a['body_pose']],axis=1),PARENTS); rn = list(HUMANML_NAMES)
        elif method == 'closd' and (file.parent/'native_physics.npz').exists():
            a = read(file.parent/'native_physics.npz')
            if 'rotations_xyzw' in a:
                q = a['rotations_xyzw']; good = np.isfinite(q).all(-1)&(np.abs(np.linalg.norm(q,axis=-1)-1)<2e-3)
                q = q.copy(); q[~good] = [0,0,0,1]
                r = Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(*q.shape[:-1],3,3); r[~good] = np.nan
                rn = [str(n) for n in a['body_names']]
    if not native_y:
        p = np.asarray(p) @ Y_TO_Z
        if r is not None: r = np.einsum('ij,tkjl->tkil',Y_TO_Z.T,r)
    if r is not None:
        r = np.asarray(r,float).copy(); r[~valid_rotation(r)] = np.nan
    times = np.asarray(m.get('times',m.get('timestamps',np.arange(len(p))/fps)),float)
    motion = Motion(p,names,times,method,r,rn,{'inputs':{str(x.resolve()):digest(x) for x in used},
                  'orientation_source':'native' if r is not None else 'unavailable',
                  'anatomical_calibration':'native_segment_axes_only', 'axes':'right-handed Y-up','units':'m'})
    sidecar = file.parent/'k_provenance.json'
    if sidecar.exists():
        declared = json.loads(sidecar.read_text())
        motion.provenance['degradation'] = declared
        motion.provenance['inputs'][str(sidecar.resolve())] = digest(sidecar)
    from .calibration import model_basis
    bases,receipt=model_basis(motion,src if src.is_dir() else src.parent)
    motion.provenance['calibration']=receipt
    if bases is not None:motion.provenance['model_bases']={k:v.tolist() for k,v in bases.items()}
    registry = settings.get('convention_registry')
    if registry:
        path = ROOT/registry
        motion.provenance['convention'] = {'version': settings['version'], 'registry': registry,
            'sha256': digest(path), 'anatomical_scope': 'source model convention; project assumptions retained explicitly'}
        motion.provenance['inputs'][str(path)] = digest(path)
    return motion
