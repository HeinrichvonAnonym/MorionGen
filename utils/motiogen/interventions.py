"""Instrument native inference components without removing task information."""


def install(adapter, condition):
    evidence = {'condition': condition, 'calls': {}, 'scope': 'fixed pretrained component intervention'}
    def count(key):
        evidence['calls'][key] = evidence['calls'].get(key, 0) + 1
    if adapter.name == 'motionlcm':
        gain = float(condition['control_residual'])
        def hook(module, inputs, output):
            count('controlnet')
            def scale(value):
                if isinstance(value, (tuple, list)):
                    return type(value)(scale(x) for x in value)
                return value * gain
            return (scale(output[0]), *output[1:])
        adapter.model.controlnet.register_forward_hook(hook)
    elif adapter.name == 'omnicontrol':
        model = adapter.model.model
        original = model.cmdm_forward
        def realism(*args, **kwargs):
            count('realism')
            return original(*args, **kwargs) * float(condition['realism'])
        model.cmdm_forward = realism
        guide = adapter.diffusion.guide
        def spatial(x, *args, **kwargs):
            count('spatial')
            return guide(x, *args, **kwargs) if condition['spatial'] else x
        adapter.diffusion.guide = spatial
    elif adapter.name == 'chois':
        enabled = bool(condition['guidance'])
        adapter.trainer.use_guidance_in_denoising = enabled
        adapter.trainer.opt.use_guidance_in_denoising = enabled
        # 20261101 V2 decomposition keys (terms, scale, last_step, direction) route guidance
        # through chois_guidance; the 20261024 matrix conditions keep the native function.
        from .chois_guidance import install as install_guidance, options as guidance_options
        if enabled and set(condition) - {'guidance'}:
            install_guidance(adapter, condition, evidence, count)
        else:
            if set(condition) - {'guidance'}:
                guidance_options(condition)
                raise ValueError('CHOIS guidance decomposition requires guidance enabled')
            guide = adapter.trainer.apply_different_guidance_loss
            def guided(*args, **kwargs):
                count('geometry_guidance')
                return guide(*args, **kwargs)
            adapter.trainer.apply_different_guidance_loss = guided
        sample = adapter.model.sample
        def sampled(*args, **kwargs):
            count('native_sampler')
            if (kwargs.get('guidance_fn') is not None) != enabled:
                raise RuntimeError('Native CHOIS sampler guidance does not match intervention')
            return sample(*args, **kwargs)
        adapter.model.sample = sampled
    elif adapter.name == 'ardy':
        from .ardy_native import install as install_ardy
        install_ardy(adapter, condition, evidence)
    else:
        raise ValueError(f'No implemented intervention for {adapter.name}')
    return evidence
