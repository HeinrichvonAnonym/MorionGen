"""Explicit effective rigid-trunk proxy from observed positional spine nodes.

This is a fixed geometric approximation, not an identified neutral/anatomical
length. Missing or unstable chains retain the original model prior.
"""
import numpy as np


def trunk_chain_proxy(motion,policy='observed_chain_proxy_v1'):
    if policy not in ('observed_chain_proxy_v1','observed_chain_total_proxy_v1'):
        raise ValueError('Unknown effective trunk proxy')
    index = {n: i for i, n in enumerate(motion.names)}
    required = ['left_hip', 'right_hip', 'spine1', 'spine2', 'spine3',
                'left_shoulder', 'right_shoulder']
    receipt = {'policy': policy, 'neutral_length_identified': False,
               'fixed_for_clip': True, 'scope': 'Effective rigid-model length estimate; no neutral-pose or anatomical claim',
               'assumption': 'Sum of observed chain segments approximates a straightened rigid torso',
               'relative_spread_limit': 0.10}
    if any(n not in index for n in required):
        return None, dict(receipt, status='model_prior', reason='required_chain_nodes_missing')
    if policy=='observed_chain_total_proxy_v1':
        prior=set(motion.provenance.get('degradation',{}).get('prior_joints',[]))
        if prior.intersection(required):
            return None,dict(receipt,status='model_prior',reason='required_chain_nodes_are_prior')
    p = lambda n: motion.positions[:, index[n]]
    chain = [(p('left_hip') + p('right_hip')) / 2,
             p('spine1'), p('spine2'), p('spine3'),
             (p('left_shoulder') + p('right_shoulder')) / 2]
    lengths = np.stack([np.linalg.norm(b-a, axis=1) for a, b in zip(chain[:-1], chain[1:])], axis=1)
    valid = np.isfinite(lengths).all(1) & (lengths > 1e-6).all(1)
    receipt.update(observed_frames=int(valid.sum()), total_frames=len(lengths))
    if valid.sum() < 2:
        return None, dict(receipt, status='model_prior', reason='insufficient_non_degenerate_chain')
    quantiles = np.percentile(lengths[valid], [5, 50, 95], axis=0)
    spread = (quantiles[2]-quantiles[0])/quantiles[1]
    receipt.update(segment_quantiles_m=quantiles.tolist(), relative_segment_spread=spread.tolist())
    unstable=np.any(spread > receipt['relative_spread_limit'])
    if policy=='observed_chain_total_proxy_v1':
        total=np.percentile(lengths[valid].sum(1),[5,50,95])
        total_spread=float((total[2]-total[0])/total[1])
        receipt.update(total_length_quantiles_m=total.tolist(),relative_total_spread=total_spread,
                       terminal_endpoint='Derived bilateral shoulder midpoint; not a single observed rigid bone',
                       stability_rule='First three spine links each <=10% spread; effective total chain <=10%. Same fixed median estimator; terminal variation propagated to total.',
                       uncertainty_scope='Observed5/50/95 percentiles of the effective proxy, not a confidence interval or neutral anatomical length')
        unstable=np.any(spread[:-1]>receipt['relative_spread_limit']) or total_spread>receipt['relative_spread_limit']
    if unstable:
        return None, dict(receipt, status='model_prior', reason='chain_geometry_unstable')
    value = float(np.median(lengths[valid].sum(1)))
    return value, dict(receipt, status='estimated_effective_length', length_m=value)
