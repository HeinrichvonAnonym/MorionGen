"""Published BVH reference, without retargeting, filtering or anatomical claims."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from motion_retarget.bvh import parse_bvh
from motiogen.evalkit.k.motion import Motion


def read_bvh(payload):
    """CoLLaboratE dialect: animated translations replace rest OFFSET axes.

    Keep the generic/legacy BVH reader unchanged. CHANNEL order is intrinsic;
    units (cm) and world up (Y) are dataset geometry conventions, not BVH fields.
    """
    bvh = parse_bvh(payload)
    count = bvh.frame_count
    p = np.zeros((count, len(bvh.joints), 3))
    r = np.zeros((count, len(bvh.joints), 3, 3))
    for j, joint in enumerate(bvh.joints):
        local_p = np.broadcast_to(joint.offset, (count, 3)).copy()
        order, columns = [], []
        for k, channel in enumerate(joint.channels):
            if channel.lower().endswith('position'):
                local_p[:, 'XYZ'.index(channel[0].upper())] = bvh.frames[:, joint.channel_start+k]
            elif channel.lower().endswith('rotation'):
                order.append(channel[0].upper())
                columns.append(joint.channel_start+k)
        local_r = (Rotation.from_euler(''.join(order), bvh.frames[:, columns], degrees=True).as_matrix()
                   if order else np.broadcast_to(np.eye(3), (count, 3, 3)))
        if joint.parent < 0:
            p[:, j], r[:, j] = local_p, local_r
        else:
            par = joint.parent
            p[:, j] = p[:, par] + np.einsum('tij,tj->ti', r[:, par], local_p)
            r[:, j] = r[:, par] @ local_r
    rest=np.zeros((len(bvh.joints),3))
    for j,joint in enumerate(bvh.joints):
        if joint.parent>=0:rest[j]=rest[joint.parent]+joint.offset*.01
    return Motion(p*.01, [j.name for j in bvh.joints], np.arange(count)/bvh.fps,
                  'reference', r, [j.name for j in bvh.joints],
                  {'scope': 'published_bvh_native_frames', 'translation': 'absolute_channels',
                   'anatomical_axes': 'not_established', 'units': 'cm_to_m_geometry_convention',
                   'rest_positions':rest.tolist(),'parents':[j.parent for j in bvh.joints]})


def sample(motion, times):
    """Interpolation only; do not silently smooth the comparison reference."""
    times = np.asarray(times)
    if times[0] < motion.times[0]-1e-9 or times[-1] > motion.times[-1]+1e-9:
        raise ValueError('Reference extrapolation prohibited')
    p = np.stack([np.interp(times, motion.times, c)
                  for c in motion.positions.reshape(len(motion.times), -1).T], 1)
    r = None if motion.rotations is None else np.stack([
        Slerp(motion.times, Rotation.from_matrix(motion.rotations[:, j]))(times).as_matrix()
        for j in range(len(motion.rotation_names))], 1)
    return Motion(p.reshape(len(times), -1, 3), motion.names, times, motion.method,
                  r, motion.rotation_names, dict(motion.provenance))


def window(motion, bounds, fps=None):
    start, end = bounds
    if fps is None:
        times = motion.times[(motion.times >= start) & (motion.times <= end)]
    else:
        times = start+np.arange(int(np.floor((end-start)*fps+1e-9))+1)/fps
        times = times[times <= motion.times[-1]]
    return sample(motion, times)
