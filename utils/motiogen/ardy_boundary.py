"""Window-boundary treatments and corrected-window renoising for native ARDY (M1 / V3).

Boundary treatments change only the frames of the new window, so "output only"
and "write back to history" differ solely in what the next window conditions on.
The continuity target of the first new frame is the one-frame linear
extrapolation of the last two committed output frames.
"""
import math

import torch

TREATMENTS = ('none', 'inertialization', 'crossfade', 'lowpass')


def _log_so3(rotation):
    cos = ((rotation.diagonal(dim1=-2, dim2=-1).sum(-1)-1.)/2.).clamp(-1., 1.)
    angle = torch.acos(cos)
    axis = torch.stack([rotation[..., 2, 1]-rotation[..., 1, 2], rotation[..., 0, 2]-rotation[..., 2, 0],
                        rotation[..., 1, 0]-rotation[..., 0, 1]], -1)
    scale = torch.where(angle > 1e-6, angle/(2.*torch.sin(angle).clamp_min(1e-12)), torch.full_like(angle, .5))
    return axis*scale[..., None]


def _exp_so3(vector):
    angle = vector.norm(dim=-1, keepdim=True)
    axis = vector/angle.clamp_min(1e-12)
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*x.shape, 3, 3)
    eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(skew.shape)
    sin, cos = torch.sin(angle)[..., None], torch.cos(angle)[..., None]
    return eye+sin*skew+(1.-cos)*skew@skew


def decay_weights(treatment, frames, fps, omega=10.):
    """Fraction of the boundary offset removed at each new frame (1 at the first frame)."""
    t = torch.arange(frames, dtype=torch.float64)/fps
    if treatment == 'inertialization':
        # PRIMAL primal/utils/mop_repr.py inertial_offset with zero offset velocity.
        return (1.+omega*t)*torch.exp(-omega*t)
    if treatment == 'crossfade':
        # Linear ramp with the same half-decay time as the critically damped offset.
        half = _half_decay_seconds(omega)
        return (1.-t/(2.*half)).clamp_min(0.)
    raise ValueError(treatment)


def _half_decay_seconds(omega):
    x = 1.
    for _ in range(60):
        x -= ((1.+x)*math.exp(-x)-.5)/(-x*math.exp(-x))
    return x/omega


def apply(treatment, previous_root, previous_rot, root, rot, fps, omega=10.):
    """previous_*: last two committed output frames [2,3] / [2,J,3,3]; root [H,3], rot [H,J,3,3]."""
    if treatment == 'none':
        return root, rot
    dtype = root.dtype
    root64, rot64 = root.double(), rot.double()
    prev_root, prev_rot = previous_root.double(), previous_rot.double()
    if treatment == 'lowpass':
        return _lowpass(prev_root, prev_rot, root64, rot64, fps, omega, dtype)
    target_root = 2*prev_root[-1]-prev_root[-2]
    target_rot = prev_rot[-1]@(prev_rot[-2].transpose(-1, -2)@prev_rot[-1])
    weights = decay_weights(treatment, root.shape[0], fps, omega).to(root64)
    root_offset = root64[0]-target_root
    rot_offset = _log_so3(rot64[0]@target_rot.transpose(-1, -2))
    new_root = root64-weights[:, None]*root_offset
    new_rot = _exp_so3(-weights[:, None, None]*rot_offset)@rot64
    return new_root.to(dtype), new_rot.to(dtype)


def _lowpass(prev_root, prev_rot, root, rot, fps, omega, dtype):
    sigma = fps/omega
    radius = int(math.ceil(3*sigma))
    offsets = torch.arange(-radius, radius+1, dtype=torch.float64)
    kernel = torch.exp(-.5*(offsets/sigma)**2)
    roots = torch.cat([prev_root, root]); rots = torch.cat([prev_rot, rot])
    count = roots.shape[0]; first = prev_root.shape[0]
    out_root, out_rot = root.clone(), rot.clone()
    for k in range(root.shape[0]):
        index = (first+k+offsets.long()).clamp(0, count-1)
        w = (kernel/kernel.sum()).to(root)
        out_root[k] = (w[:, None]*roots[index]).sum(0)
        mean = (w[:, None, None, None]*rots[index]).sum(0)
        u, _, v = torch.linalg.svd(mean)
        d = torch.det(u@v).sign()
        fix = torch.ones_like(u[..., 0]); fix[..., -1] = d
        out_rot[k] = u@torch.diag_embed(fix)@v
    return out_root.to(dtype), out_rot.to(dtype)


def renoise(model, motion, history, n, fraction, text_feat, text_pad_mask, mask, observed, cfg, seed):
    """Noise the corrected window to an intermediate level and denoise it again with the same
    history, text and constraint conditioning (deterministic DDIM, eta = 0).

    motion: corrected explicit normalized motion [1, n+H, D] (history frames included).
    Returns the new explicit motion of the same length.
    """
    from unittest import mock
    horizon = model.gen_horizon_len
    batch = 1
    steps = int(model.diffusion.num_base_steps)
    k = max(1, min(steps, int(round(fraction*steps))))
    steps_tensor = torch.tensor([steps], device=model.device)
    use_timesteps = model.diffusion.space_timesteps(steps_tensor[0])[0]
    model.diffusion.calc_diffusion_vars(use_timesteps)
    if history is not None:
        history_tokens, global_transl, heading = model._encode_init_history(history, batch)
    else:
        history_tokens = None
        global_transl = torch.zeros((batch, model.motion_rep.nfeats_dict['root_pos']), device=model.device)
        heading = torch.zeros(batch, device=model.device)
    full, _ = model.hybrid.get_hybrid_motion_from_explicit(
        motion=motion, motion_len=torch.full((batch,), n+horizon, device=model.device, dtype=torch.long),
        motion_pad_mask=torch.ones(batch, n+horizon, device=model.device, dtype=torch.bool))
    if n:
        full, _, _ = model._recenter_history(full, torch.full((batch,), n-1, device=model.device, dtype=torch.long), requantize=False)
    clean = full[:, n//model.num_frames_per_token:]
    generator = torch.Generator(device=model.device).manual_seed(int(seed))
    noise = torch.randn(clean.shape, generator=generator, device=model.device)
    level = torch.full((batch,), k-1, device=model.device, dtype=torch.long)
    start = model.diffusion.q_sample(clean, level, noise=noise)
    original = torch.randn
    hits = [0]

    def initial(*shape, **kwargs):
        size = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list, torch.Size)) else tuple(shape)
        if size == tuple(start.shape):
            hits[0] += 1
            return start.clone()
        return original(*shape, **kwargs)
    with mock.patch('torch.randn', side_effect=initial):
        sequence = model._generate_window(history_tokens, global_transl, 0, n, n+horizon, text_feat, text_pad_mask,
                                          heading, mask, observed, steps_tensor, cfg, list(range(k))[::-1],
                                          progress_bar=lambda iterable: iterable, target_motion=None, cfg_type=None)
    if hits[0] != 1:
        raise RuntimeError(f'Renoise start replaced {hits[0]} noise draws, expected 1')
    length = n+horizon
    unchanged = clean if history_tokens is None else torch.cat([history_tokens, clean], 1)
    roundtrip = _explicit(model, unchanged, global_transl, length, mask)
    reference = model.motion_rep.inverse(motion, is_normalized=True)['root_positions'][0, n:]
    error = torch.linalg.vector_norm(model.motion_rep.inverse(roundtrip, is_normalized=True)['root_positions'][0, n:]-reference, dim=-1)
    info = {'steps': k, 'signal_scale': float(model.diffusion.sqrt_alphas_cumprod[k-1]),
            'roundtrip_root_max_m': float(error.max())}
    return _explicit(model, sequence, global_transl, length, mask), info


def _explicit(model, sequence, global_transl, length, mask):
    from ardy.model.ardy_model import translate_normalized_root_motion
    root_motion, latent = model.hybrid.get_root_and_latent_body_motion_from_hybrid(sequence)
    root_motion = translate_normalized_root_motion(root_motion, global_transl, model.motion_rep)
    if model.autoencoder.encode_with_quantization:
        latent = model.autoencoder.requantize(latent)
    sequence = model.hybrid.get_hybrid_motion_from_root_and_latent_body_motion(root_motion, latent)
    return model.hybrid.get_explicit_motion_from_hybrid(
        sequence, torch.ones(1, length, device=sequence.device, dtype=torch.bool),
        torch.full((1,), length, device=sequence.device, dtype=torch.long), motion_mask=mask)
