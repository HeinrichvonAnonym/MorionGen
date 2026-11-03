"""Native MotionCorrection for SMPL-joint outputs (DART, OmniControl) through Kimodo's SMPL-X-22 rig.

Runs in the Kimodo environment. Joints come in canonical Z-up metres; the rig is
Y-up like Kimodo. Rest offsets are SMPL-X neutral bone directions scaled to the
run's median bone lengths, with the children of multi-child joints re-estimated
from the run's mean local geometry so that FK reproduces the input.
"""
import numpy as np
import torch

from .motion import Y_TO_Z

CONTACT_HEIGHT_M = .05
CONTACT_SPEED_M_S = .3
FK_GATE_M = .01


def _kabsch(rest, observed):
    u, _, vt = np.linalg.svd(rest.T @ observed)
    d = np.eye(3); d[2, 2] = np.sign(np.linalg.det(vt.T @ u.T))
    return vt.T @ d @ u.T


def _align(a, b):
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if c < -.999999:
        axis = np.cross(a, [1., 0, 0] if abs(a[0]) < .9 else [0, 1., 0]); axis /= np.linalg.norm(axis)
        return 2*np.outer(axis, axis)-np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3)+k+k @ k/(1+c)


class Rig:
    def __init__(self, joints, names, fps):
        from kimodo.skeleton import SMPLXSkeleton22
        self.names, self.fps = list(names), float(fps)
        self.skeleton = SMPLXSkeleton22()
        self.order = [self.names.index(n) for n in self.skeleton.bone_order_names]
        x = (np.asarray(joints, np.float64) @ Y_TO_Z)[:, self.order]
        parents = self.skeleton.joint_parents.numpy()
        rest = self.skeleton.neutral_joints.numpy().astype(np.float64)
        offsets = np.zeros_like(rest)
        for j in range(1, len(parents)):
            direction = rest[j]-rest[parents[j]]
            offsets[j] = direction/np.linalg.norm(direction)*np.median(np.linalg.norm(x[:, j]-x[:, parents[j]], axis=-1))
        children = {j: [c for c in range(len(parents)) if parents[c] == j] for j in range(len(parents))}
        for j, ch in children.items():
            if len(ch) > 1:
                frames = np.stack([_kabsch(offsets[ch], x[t, ch]-x[t, j]) for t in range(len(x))])
                offsets[ch] = np.einsum('tji,tcj->ci', frames, x[:, ch]-x[:, j:j+1])/len(x)
        neutral = np.zeros_like(rest)
        for j in range(1, len(parents)):
            neutral[j] = neutral[parents[j]]+offsets[j]
        self.skeleton.neutral_joints = torch.tensor(neutral+rest[0], dtype=torch.float32)
        glob = np.zeros((len(x), len(parents), 3, 3)); local = np.zeros_like(glob)
        for t in range(len(x)):
            for j in range(len(parents)):
                parent = np.eye(3) if parents[j] < 0 else glob[t, parents[j]]
                ch = children[j]
                if not ch:
                    glob[t, j] = parent
                elif len(ch) == 1:
                    glob[t, j] = _align(parent @ offsets[ch[0]], x[t, ch[0]]-x[t, j]) @ parent
                else:
                    glob[t, j] = _kabsch(offsets[ch], x[t, ch]-x[t, j])
                local[t, j] = parent.T @ glob[t, j]
        self.x = x
        self.local = torch.tensor(local, dtype=torch.float32)[None]
        self.root = torch.tensor(x[:, 0], dtype=torch.float32)[None]
        feet = self.skeleton.foot_joint_idx
        height = x[:, feet, 1]
        speed = np.zeros_like(height)
        speed[1:] = np.linalg.norm(np.diff(x[:, feet][:, :, [0, 2]], axis=0), axis=-1)*self.fps
        self.contacts = torch.tensor((height < height.min(0)+CONTACT_HEIGHT_M) & (speed < CONTACT_SPEED_M_S), dtype=torch.float32)[None]
        error = np.linalg.norm(self.joints_y(self.local, self.root)-x, axis=-1)
        self.fk = {'rms_m': float(np.sqrt((error**2).mean())), 'max_m': float(error.max()), 'gate_m': FK_GATE_M,
                   'pass': bool(np.sqrt((error**2).mean()) <= FK_GATE_M),
                   'contact_fraction': self.contacts[0].mean(0).tolist()}

    def joints_y(self, local, root):
        from kimodo.skeleton.kinematics import fk
        return fk(local, root, self.skeleton)[1][0].numpy().astype(np.float64)

    def to_canonical(self, joints_y):
        out = np.empty_like(joints_y)
        out[:, self.order] = joints_y @ Y_TO_Z.T
        return out

    @staticmethod
    def native_xz(xy):
        xy = np.asarray(xy, np.float64)
        return np.stack([xy[..., 0], -xy[..., 1]], -1)

    def correct(self, frames, targets_xy, start=0, stop=None, local=None, root=None):
        """Native post_process_motion with Root2D at frames (absolute), on [start, stop)."""
        from kimodo.constraints import Root2DConstraintSet
        from kimodo.postprocess import post_process_motion
        local = self.local if local is None else local
        root = self.root if root is None else root
        stop = local.shape[1] if stop is None else stop
        frames = torch.tensor([f-start for f in frames])
        constraint = Root2DConstraintSet(self.skeleton, frames, torch.tensor(self.native_xz(targets_xy), dtype=torch.float32))
        out = post_process_motion(local[:, start:stop], root[:, start:stop], self.contacts[:, start:stop],
                                  self.skeleton, [constraint], root_margin=.04)
        local, root = local.clone(), root.clone()
        local[:, start:stop] = out['local_rot_mats'].cpu().float()
        root[:, start:stop] = out['root_positions'].cpu().float()
        return local, root

    def crossfade(self, local, root, boundaries):
        """ARDY X0 output-only crossfade at every boundary (linear ramp, 0.168 s half-decay)."""
        from .ardy_boundary import apply
        local, root = local.clone(), root.clone()
        edges = list(boundaries)+[root.shape[1]]
        for b, nb in zip(edges[:-1], edges[1:]):
            if b < 2:
                continue
            r, q = apply('crossfade', root[0, b-2:b], local[0, b-2:b], root[0, b:nb], local[0, b:nb], self.fps)
            root[0, b:nb], local[0, b:nb] = r.float(), q.float()
        return local, root
