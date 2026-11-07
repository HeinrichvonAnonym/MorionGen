"""Upper-body load (config/evalkit_upper_body_20261107.json): validation against the synthetic truth and floor robustness.

  validate   [--splits dev heldout2]            control clips, tool vs truth -> <out>/validation.json
  robustness --rows A --compare B               the same rows evaluated twice (e.g. main vs self-floor) -> <out>/robustness.json
OpenSim python (~/.local/share/opensim-api/bin/python).
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
sys.path.insert(0, str(ROOT/'utils/motiogen'))
from motiogen.evalkit import muscle  # noqa: E402
from motiogen.evalkit.validate import rate  # noqa: E402

DATA = Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])/'outputs/evalkit_20261107'
PROTOCOL = 'config/evalkit_upper_body_20261107.json'
OUT = ROOT/'develop_logs/test_runs/20261107_evalkit/upper_body'
METRICS = [f'{p}_moment_nm_kg_{s}' for p in muscle.UPPER_BODY for s in ('mean', 'p95')]
RATIOS = (1.1, 1.2, 1.5)


def agreement(tool, truth):
    from scipy.stats import spearmanr
    t, g = np.asarray(tool), np.asarray(truth)
    rel = np.abs(t-g)/g
    high = g >= np.median(g)
    out = {'clips': len(g), 'spearman': float(spearmanr(t, g)[0]),
           'median_relative_error': float(np.median(rel)), 'median_absolute_error': float(np.median(np.abs(t-g))),
           'median_signed_relative_error': float(np.median((t-g)/g)),
           'median_relative_error_truth_upper_half': float(np.median(rel[high])),
           'median_relative_error_truth_lower_half': float(np.median(rel[~high])), 'pair_order': {}}
    for ratio in RATIOS:
        pairs = [(i, j) for i, j in combinations(range(len(g)), 2) if max(g[i], g[j])/min(g[i], g[j]) >= ratio]
        out['pair_order'][str(ratio)] = rate(int(sum(np.sign(t[i]-t[j]) == np.sign(g[i]-g[j]) for i, j in pairs)), len(pairs))
    return out


def validate(args):
    clips = []
    for split in args.splits:
        for truth in sorted((DATA/'truth_eval'/split).glob('*__control')):
            tool = DATA/'tool'/split/truth.name/'work'
            if not (truth/'so/so_so_force.sto').exists() or not (tool/'so/so_so_force.sto').exists():
                continue
            g, t = muscle.upper_body(truth/'so', truth/'model_so.osim'), muscle.upper_body(tool/'so', tool/'model_so.osim')
            clips.append({'clip': truth.name, 'split': split, 'mass_kg': {'truth': g['mass_kg'], 'tool': t['mass_kg']},
                          'so_frames': {'truth': len(g['times']), 'tool': len(t['times'])},
                          **{m: {'truth': g[m], 'tool': t[m]} for m in METRICS}})
    stats = {m: agreement([c[m]['tool'] for c in clips], [c[m]['truth'] for c in clips]) for m in METRICS}
    po = lambda m: stats[m]['pair_order']['1.5']
    gate = {m: bool(stats[m]['spearman'] >= .8 and po(m)['rate'] >= .9) for m in METRICS if m.endswith('_mean')}
    out = {'protocol': PROTOCOL, 'splits': args.splits, 'statistics': stats,
           'integration_gate': {'rule': 'clip mean: Spearman >= 0.8 and pair order at truth ratio >= 1.5 >= 0.9', 'pass': gate},
           'clips': clips}
    write(out, args.out/'validation.json')
    for m in METRICS:
        s = stats[m]
        print(f"{m:32s} n={s['clips']} rho={s['spearman']:.3f} med|rel|={s['median_relative_error']:.3f} "
              f"signed={s['median_signed_relative_error']:+.3f} pair1.5={po(m)['rate']:.3f} ({po(m)['n']})")
    print('gate', gate)


def robustness(args):
    rows = []
    for a in sorted(Path(args.rows).glob('*/evalkit.json')):
        b = Path(args.compare)/f'{a.parent.name}{args.suffix}'/'evalkit.json'
        if not b.exists():
            continue
        ra, rb = (json.loads(p.read_text()) for p in (a, b))
        if not all((p.parent/'work/so/so_so_force.sto').exists() for p in (a, b)):
            continue
        ua, ub = (muscle.upper_body(p.parent/'work/so', p.parent/'work/model_so.osim') for p in (a, b))
        leg = [r['muscle'].get('muscle_effort_sum_a2_mean') for r in (ra, rb)]
        rows.append({'id': a.parent.name, 'method': ra['receipt'].get('method'), 'leg_effort': leg,
                     **{m: [ua[m], ub[m]] for m in METRICS}})
    change = lambda k: [abs(r[k][1]-r[k][0])/abs(r[k][0]) for r in rows if r[k][0]]
    out = {'protocol': PROTOCOL, 'rows_a': str(args.rows), 'rows_b': str(args.compare), 'rows': rows,
           'median_abs_relative_change': {k: float(np.median(change(k))) for k in ['leg_effort', *METRICS]},
           'p90_abs_relative_change': {k: float(np.percentile(change(k), 90)) for k in ['leg_effort', *METRICS]}}
    write(out, args.out/'robustness.json')
    print(json.dumps({k: out[k] for k in ('median_abs_relative_change', 'p90_abs_relative_change')}, indent=1), len(rows))


def write(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('validate'); v.add_argument('--splits', nargs='+', default=['dev', 'heldout2'])
    r = sub.add_parser('robustness'); r.add_argument('--rows', type=Path, required=True); r.add_argument('--compare', type=Path, required=True)
    r.add_argument('--suffix', default='__selffloor')
    for s in (v, r): s.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    {'validate': validate, 'robustness': robustness}[a.cmd](a)


if __name__ == '__main__':
    main()
