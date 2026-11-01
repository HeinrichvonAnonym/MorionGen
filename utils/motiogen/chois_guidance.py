"""CHOIS geometric guidance decomposition (20261101 V2).

Native guidance (trainer_chois.apply_different_guidance_loss and
ObjectCondGaussianDiffusion.p_sample_loop_guided): loss = 1e4 * hand-object
+ 3e5 * feet-floor, reconstruction guidance on the predicted clean sample at
denoising steps 1..9 of 1000.  Absent condition keys keep the native settings.
"""

NATIVE_LAST_STEP = 10
TERMS = ('both', 'hand_object', 'feet_floor')
DIRECTIONS = ('loss', 'random_equal_norm')


def options(condition):
    result = {'terms': condition.get('terms', 'both'), 'scale': float(condition.get('scale', 1.)),
              'last_step': int(condition.get('last_step', NATIVE_LAST_STEP)),
              'direction': condition.get('direction', 'loss')}
    if result['terms'] not in TERMS or result['direction'] not in DIRECTIONS:
        raise ValueError('Unknown CHOIS guidance terms or direction')
    if result['last_step'] < 2 or result['scale'] < 0:
        raise ValueError('CHOIS guidance needs at least step 1 and a non-negative scale')
    return result


def equal_norm_direction(gradient, generator):
    """Gaussian direction on the support of the loss gradient, rescaled to its norm."""
    import torch
    support = gradient != 0
    noise = torch.randn(gradient.shape, generator=generator, device=gradient.device, dtype=gradient.dtype)*support
    norm = noise.norm()
    return (noise*(gradient.norm()/norm) if norm > 0 else noise), float(support.float().mean())


def guided_loop(model, last_step):
    """Native p_sample_loop_guided with the guided step range 1..last_step-1."""
    import torch

    def loop(shape, x_cond, guidance_fn=None, language_embedding=None, opt_fn=None,
             rest_human_offsets=None, data_dict=None, contact_labels=None, cond_mask=None, padding_mask=None):
        device = model.betas.device
        b = shape[0]
        x = torch.randn(shape, device=device)
        for i in reversed(range(0, model.num_timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            if guidance_fn is not None and 0 < i < last_step:
                x = model.p_sample_guided_reconstruction_guidance(
                    x, t, x_cond, guidance_fn, language_embedding=language_embedding, opt_fn=opt_fn,
                    rest_human_offsets=rest_human_offsets, data_dict=data_dict, contact_labels=contact_labels,
                    cond_mask=cond_mask, padding_mask=padding_mask)
            else:
                x = model.p_sample(x, t, x_cond, language_embedding, padding_mask=padding_mask)
        return x
    return loop


def install(adapter, condition, evidence, count):
    import torch
    opt = options(condition)
    trainer, model = adapter.trainer, adapter.model
    use_hand, use_feet = opt['terms'] in ('both', 'hand_object'), opt['terms'] in ('both', 'feet_floor')
    steps = []
    evidence['guidance'] = {**opt, 'steps': steps}
    generator = None
    if opt['direction'] == 'random_equal_norm':
        generator = torch.Generator(device='cuda')
        generator.manual_seed(int(adapter.seed))

    def grad(value, x):
        result = torch.autograd.grad(value, x, retain_graph=True, allow_unused=True)[0]
        return torch.zeros_like(x) if result is None else result

    def guided(noise_level, pred_clean_x, x_pose_cond, cond_mask, rest_human_offsets, data_dict, **kwargs):
        count('geometry_guidance')
        feet = trainer.apply_feet_floor_contact_guidance(pred_clean_x, rest_human_offsets, data_dict, **kwargs)
        hand = trainer.apply_hand_object_interaction_guidance_loss(pred_clean_x, rest_human_offsets, data_dict, **kwargs)
        weighted_hand, weighted_feet = hand * 10000, feet * 100000 * 3
        if use_hand and use_feet:
            loss = weighted_hand + weighted_feet
        else:
            loss = weighted_hand if use_hand else weighted_feet
        if opt['scale'] != 1.:
            loss = loss * opt['scale']
        g_hand, g_feet = grad(weighted_hand, pred_clean_x), grad(weighted_feet, pred_clean_x)
        applied = (g_hand*use_hand + g_feet*use_feet)*opt['scale']
        t = int(noise_level.reshape(-1)[0])
        record = {'t': t, 'hand_object_loss': float(hand), 'feet_floor_loss': float(feet),
                  'hand_object_grad_norm': float(g_hand.norm()), 'feet_floor_grad_norm': float(g_feet.norm()),
                  'applied_grad_norm': float(applied.norm()),
                  'posterior_variance': float(model.posterior_variance[t])}
        steps.append(record)
        if generator is None:
            return loss
        direction, support = equal_norm_direction(applied, generator)
        record.update(random_support_fraction=support, random_direction_norm=float(direction.norm()))
        return (pred_clean_x * direction.detach()).sum()

    trainer.apply_different_guidance_loss = guided
    if opt['last_step'] != NATIVE_LAST_STEP:
        model.p_sample_loop_guided = guided_loop(model, opt['last_step'])
        count('guided_loop_replaced')
    return opt
