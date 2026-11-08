#!/usr/bin/env python
"""Does a with-replacement window resample give a finite FID, and how should the reference set be resampled?

Inputs: the evalkit distribution self-check embeddings (develop_logs/test_runs/20261107_evalkit/distribution/<E>/
embeddings.npz; the 2 cm noise perturbation stands in for a generated set) and CHOIS's official FID code
(chois_release/t2m_eval/utils/metrics.py; T2M uses the same function body). No method output is read.
Reference variants per resample: 'paired' (same indices on the reference, windows paired with ground truth),
'fixed' (full reference), 'independent' (reference resampled with its own indices).
Output: develop_logs/test_runs/20261109_inference_premises/fid_bootstrap_check.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.evalkit import distribution as D  # noqa: E402

SRC = ROOT/'develop_logs/test_runs/20261107_evalkit/distribution'
OUT = ROOT/'develop_logs/test_runs/20261109_inference_premises/fid_bootstrap_check.json'
RESAMPLES, SEED = 200, 20261109


def main():
    spec = importlib.util.spec_from_file_location('chois_metrics', D.CHOIS_EVAL/'utils/metrics.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

    def fid(a, b):
        try:
            return float(m.calculate_frechet_distance(*m.calculate_activation_statistics(a), *m.calculate_activation_statistics(b)))
        except ValueError:
            return float('nan')

    report = {'resamples': RESAMPLES, 'seed': SEED, 'generated_stand_in': 'perturbed_noise_0.02', 'evaluators': {}}
    for e in ('omomo', 't2m'):
        z = np.load(SRC/e/'embeddings.npz')
        gen = z['perturbed_noise_0.02']; n = len(gen)
        ref = z['ground_truth'][:n] if e == 'omomo' else z['ground_truth']
        r = np.random.default_rng(SEED)
        out = {'paired': [], 'fixed': [], 'independent': []}
        for _ in range(RESAMPLES):
            idx = r.integers(0, n, n); other = r.integers(0, len(ref), len(ref))
            if e == 'omomo':
                out['paired'].append(fid(gen[idx], ref[idx]))
            out['fixed'].append(fid(gen[idx], ref))
            out['independent'].append(fid(gen[idx], ref[other]))
        report['evaluators'][e] = {
            'n_generated': n, 'n_reference': len(ref), 'paired_with_reference': e == 'omomo',
            'embedding_dim': int(gen.shape[1]), 'fid_without_resampling': fid(gen, ref),
            **{k: {'failed': int(np.isnan(v).sum()), 'median': float(np.nanmedian(v)),
                   'p2.5': float(np.nanpercentile(v, 2.5)), 'p97.5': float(np.nanpercentile(v, 97.5))}
               for k, v in out.items() if v}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
