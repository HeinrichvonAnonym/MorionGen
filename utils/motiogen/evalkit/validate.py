"""Validation statistics of evalkit against the synthetic truth (config/evalkit_v1_20261107.json, validation)."""
import numpy as np


def match(t_truth, t_tool):
    """Index into t_tool of the nearest tool frame for every truth frame."""
    i = np.clip(np.searchsorted(t_tool, t_truth), 1, len(t_tool)-1)
    return np.where(np.abs(t_tool[i-1]-t_truth) <= np.abs(t_tool[i]-t_truth), i-1, i)


def clopper_pearson_upper(k, n, level=.95):
    """One-sided upper bound of a binomial proportion."""
    from scipy.stats import beta
    return 1. if n == 0 or k == n else float(beta.ppf(level, k+1, n-k))


def rate(k, n):
    return {'k': int(k), 'n': int(n), 'rate': float(k/n) if n else None, 'upper95': clopper_pearson_upper(k, n)}


def kill_rates(truth_pos, truth_neg, tool_pos, clip=None, boot=1000, seed=0):
    """False-kill (tool positive | truth negative) and missed-kill (tool negative | truth positive) with Clopper-Pearson
    upper bounds over frames and a clip-level bootstrap 95% interval when clip ids are given."""
    out = {'false_kill': rate((tool_pos & truth_neg).sum(), truth_neg.sum()),
           'missed_kill': rate((~tool_pos & truth_pos).sum(), truth_pos.sum())}
    if clip is not None and len(clip):
        ids = np.unique(clip); rng = np.random.default_rng(seed)
        groups = {c: np.flatnonzero(clip == c) for c in ids}
        fk, mk = [], []
        for _ in range(boot):
            idx = np.concatenate([groups[c] for c in rng.choice(ids, len(ids))])
            n, p = truth_neg[idx].sum(), truth_pos[idx].sum()
            fk.append((tool_pos[idx] & truth_neg[idx]).sum()/n if n else np.nan)
            mk.append((~tool_pos[idx] & truth_pos[idx]).sum()/p if p else np.nan)
        for k, v in (('false_kill', fk), ('missed_kill', mk)):
            out[k]['clip_bootstrap_95'] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return out


def passes(r, line=.05, upper=.10):
    return all(r[k]['rate'] is not None and r[k]['rate'] <= line and r[k]['upper95'] <= upper for k in ('false_kill', 'missed_kill'))
