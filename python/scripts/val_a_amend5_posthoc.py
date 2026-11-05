#!/usr/bin/env python
"""Amendment 5 post-hoc record (diagnostic only; selects nothing, gates nothing): V3 over a v_on grid on the development
walks and both held-out sets, all valid frames and walking frames only (body-progress proxy >= 0.2 m/s, as in
val_a_amend5_standing.py), with the share of false contacts that fall in standing frames.

Writes develop_logs/test_runs/20261105_val_a/amend5_posthoc.json. OpenSim python.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'python/scripts'), str(ROOT/'utils/motiogen')]
import val_a_amend4_tune as T  # noqa: E402
import val_a_amend5 as A  # noqa: E402
import val_a_amend5_standing as S  # noqa: E402

S.RULES = [('height', None)]+[('hmax', v) for v in (.3, .5, .8, 1.2)]+[('onset', v) for v in (.1, .2, .3, .4, .5, .6, .7, .8, 1.)]


def summarize(per):
    st = [p['standing_share'] for p in per if p['standing_share'] is not None]
    out = {'trials': len(per), 'standing_share_median': float(np.median(st)), 'standing_share_mean': float(np.mean(st)), 'rules': {}}
    for rule, v in S.RULES:
        a = [p[f'{rule} {v} all'] for p in per if p[f'{rule} {v} all'] is not None]
        w = [p[f'{rule} {v} walking'] for p in per if p[f'{rule} {v} walking'] is not None]
        out['rules'][f'{rule} {v}'] = {'v3_all_median': float(np.median(a)), 'v3_walking_median': float(np.median(w)), 'n_walking': len(w),
                                       'false_contacts_in_standing_share': sum(p[f'{rule} {v} fp_stand'] for p in per)/max(sum(p[f'{rule} {v} fp'] for p in per), 1)}
    return out


def main():
    walk, _ = T.development()
    sets = {'development_walk': [r['id'] for r, _ in walk]}
    for fn, name in (('amend4_heldout_selection.json', 'amend4_heldout'), ('amend5_heldout_selection.json', 'amend5_heldout')):
        sets[name] = [r['id'] for r in json.loads((A.LOG/fn).read_text())['trials']]
    out = {'status': 'diagnostic, recorded after the amendment-5 held-out gates failed; no rule is selected or changed by it', 'registered_gate': A.SPEC['gate']['pass_if'],
           'sets': {}}
    for name, ids in sets.items():
        out['sets'][name] = summarize([S.trial(*A.jointcenters(i)) for i in ids])
    (A.LOG/'amend5_posthoc.json').write_text(json.dumps(out, indent=1))
    for name, s in out['sets'].items():
        print(name, s['trials'], 'standing', round(s['standing_share_median'], 3))
        for k, r in s['rules'].items():
            print(f"   {k:12s} all {r['v3_all_median']:.3f} walking {r['v3_walking_median']:.3f} fp_standing {r['false_contacts_in_standing_share']:.2f}")


if __name__ == '__main__':
    main()
