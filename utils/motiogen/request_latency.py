"""Load recorded request samples for report inspection, without timing inference."""
import json
import math
from functools import lru_cache
from .config import ROOT


def matched_ardy_steady_metrics(samples):
    """Same request positions for 32/160-frame histories in the 20s protocol.

    This is a post-hoc measurement refinement, not a new sample or rerun.
    Preserve the original condition-specific full-buffer summaries separately.
    """
    import numpy as np
    if len(samples) != 50 or not all(math.isfinite(v) and v >= 0 for v in samples):
        raise ValueError('Expected all 50 finite ARDY requests from the registered 400-frame rollout')
    matched = samples[20:]
    return {'matched_full_history_request_count': len(matched),
            'matched_full_history_request_mean_seconds': float(np.mean(matched)),
            'matched_full_history_request_p50_seconds': float(np.quantile(matched, .5)),
            'matched_full_history_request_p95_seconds': float(np.quantile(matched, .95)),
            'matched_full_history_request_rtf': float(np.mean(matched) / .4)}


@lru_cache(maxsize=None)
def recorded_requests(method, run):
    folder=ROOT/run
    if method=='ardy':
        source=folder/'worker_result.json'
        result=json.loads(source.read_text())
        samples=result['timing']['replan']['samples_seconds']
        scope='Native replan wall compute; includes initial short-history requests; excludes pacing and queue wait.'
    elif method in ('dart','closd'):
        source=folder/'trace.json'
        result=json.loads(source.read_text())
        samples=[r['seconds'] for r in result['steps']]
        scope=('Policy action plus native kinematic environment step; excludes pacing and queue wait.' if method=='dart' else
               'Native env_step including triggered DiP and PhysX; excludes separately timed PHC action, pacing and queue wait.')
    else:
        return None
    if not samples or not all(isinstance(v,(float,int)) and math.isfinite(v) and v>=0 for v in samples):
        raise ValueError('Recorded request durations must be finite and nonnegative: '+run)
    return {'samples_seconds':samples,'source':str(source.relative_to(ROOT)),'scope':scope,
            'includes_startup_requests':True,'observations_are_not_independent_scenes':True}

