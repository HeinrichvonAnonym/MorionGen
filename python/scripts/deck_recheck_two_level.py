#!/usr/bin/env python
"""Two-level (text -> task) re-check of the deck's MotionLCM and OmniControl paired effects (design notes section 1,
analysis plan deck_recheck). Existing native metrics only: the paused-clock rows of
develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json and the 0.1 s step jerk of
develop_logs/test_runs/20261030_next_round/v5/per_run.json; no motion is regenerated or re-evaluated.

The 12 reach tasks share two texts (left / right hand). Control-off motions depend on text and seed only, so the six
tasks of a text are not independent. Per effect: the deck's task-cluster bootstrap (reproduced, gate), a two-level
bootstrap (texts with replacement, then tasks within each drawn text), the per-text means, and the number of distinct
control-off inputs.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.native_analysis import clustered_interval  # noqa: E402

ANALYSIS = ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json'
V5 = ROOT/'develop_logs/test_runs/20261030_next_round/v5'
OUT = ROOT/'develop_logs/test_runs/20261108_deck_recheck'
METRICS = ['hand_position_error', 'position_success_rate', 'jerk_rms', 'jerk_common_0p1s', 'trunk_inclination',
           'right_upper_arm_elevation', 'left_upper_arm_elevation', 'foot_sliding_given_5cm_contact_10cms',
           'foot_floor_penetration_fraction', 'model_rtf']
REPS, SEED = 10000, 20261108
EFFECTS = {'motionlcm': [('control_residual', {}, None)],
           'omnicontrol': [('spatial', {}, None), ('realism', {}, None),
                           ('spatial', {'realism': 0}, None), ('spatial', {'realism': 1}, None),
                           ('realism', {'spatial': 0}, None), ('realism', {'spatial': 1}, None),
                           ('realism', {}, 'spatial')]}


def rows():
    jerk = {r['run']: r['jerk_common_0p1s'] for r in json.loads((V5/'per_run.json').read_text())}
    out = []
    for r in json.loads(ANALYSIS.read_text())['rows']:
        if r['clock'] != 'paused' or r['method'] not in EFFECTS:
            continue
        r = {**r, 'values': {**r['values'], 'jerk_common_0p1s': jerk.get(r['run'])}, 'text': r['task_spec']['prompt']}
        out.append(r)
    return out


def deltas(rs, factor, fixed, interaction):
    """(task, text, seed, delta): on - off within (task, seed), averaging the other factor unless fixed; interaction:
    (factor on - off at interaction 1) - (at interaction 0)."""
    groups = defaultdict(list)
    for r in rs:
        if all(r['condition'][k] == v for k, v in fixed.items()):
            groups[r['task'], r['seed']].append(r)
    out = {}
    for metric in METRICS:
        d = []
        for (task, seed), g in sorted(groups.items()):
            def eff(sub):
                on = [x['values'].get(metric) for x in sub if x['condition'][factor] == 1]
                off = [x['values'].get(metric) for x in sub if x['condition'][factor] == 0]
                if not on or len(on) != len(off) or any(v is None or not np.isfinite(v) for v in on+off):
                    return None
                return float(np.mean(on)-np.mean(off))
            if interaction:
                a = eff([x for x in g if x['condition'][interaction] == 1]); b = eff([x for x in g if x['condition'][interaction] == 0])
                v = None if a is None or b is None else a-b
            else:
                v = eff(g)
            if v is not None:
                d.append((task, g[0]['text'], seed, v))
        out[metric] = d
    return out


def two_level(d, reps=REPS, seed=SEED):
    task_mean = defaultdict(list)
    for task, text, _, v in d:
        task_mean[(text, task)].append(v)
    by_text = defaultdict(list)
    for (text, task), vs in sorted(task_mean.items()):
        by_text[text].append(np.mean(vs))
    texts = sorted(by_text); arr = [np.array(by_text[t]) for t in texts]
    point = float(np.mean(np.concatenate(arr)))
    rng = np.random.default_rng(seed); draws = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(texts), len(texts))
        draws[i] = np.mean(np.concatenate([arr[k][rng.integers(0, len(arr[k]), len(arr[k]))] for k in pick]))
    ci = [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
    return {'mean': point, 'ci95': ci, 'texts': len(texts), 'per_text_mean': {t: float(np.mean(a)) for t, a in zip(texts, arr)},
            'same_sign_across_texts': bool(len({np.sign(np.mean(a)) for a in arr}) == 1)}


def excludes(ci):
    return bool(ci and (ci[0] > 0 or ci[1] < 0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rs = rows()
    a = json.loads(ANALYSIS.read_text())
    ref = {}
    for e in a['contrasts']:
        if e['clock'] == 'paused' and len(e['factors']) == 1:
            ref[(e['method'], e['factors'][0], '', e['metric'])] = e
        if e['clock'] == 'paused' and len(e['factors']) == 2:
            ref[(e['method'], 'x'.join(e['factors']), '', e['metric'])] = e
    for e in a['simple_effects']:
        if e['clock'] == 'paused':
            ref[(e['method'], e['factor'], json.dumps(e['fixed'], sort_keys=True), e['metric'])] = e
    results, gate = [], []
    for method, effects in EFFECTS.items():
        mine = [r for r in rs if r['method'] == method]
        for factor, fixed, inter in effects:
            off_inputs = len({r['motion_sha256'] for r in mine if all(r['condition'][k] == 0 for k in r['condition'])})
            for metric, d in deltas(mine, factor, fixed, inter).items():
                if len({x[0] for x in d}) < 2:
                    continue
                deck = clustered_interval([x[3] for x in d], [str(x[0]) for x in d])
                name = f'{factor}x{inter}' if inter else factor
                key = (method, 'x'.join(sorted([factor, inter])) if inter else factor, json.dumps(fixed, sort_keys=True) if fixed else '', metric)
                r0 = ref.get(key)
                if r0 is not None:
                    gate.append({'effect': f'{method} {name} {fixed or ""} {metric}', 'deck_mean': r0['mean_equal_group_weight'],
                                 'mine': deck['mean_equal_group_weight'], 'deck_ci': r0['ci95'], 'mine_ci': deck['ci95'],
                                 'equal': bool(np.isclose(r0['mean_equal_group_weight'], deck['mean_equal_group_weight'])
                                               and r0['ci95'] is not None and np.allclose(r0['ci95'], deck['ci95']))})
                tl = two_level(d)
                results.append({'method': method, 'effect': name, 'fixed': fixed, 'metric': metric, 'pairs': len(d),
                                'tasks': len({x[0] for x in d}), 'control_off_distinct_inputs': off_inputs,
                                'task_cluster': {'mean': deck['mean_equal_group_weight'], 'ci95': deck['ci95'], 'excludes_0': excludes(deck['ci95'])},
                                'two_level': {**tl, 'excludes_0': excludes(tl['ci95'])},
                                'conclusion_changes': excludes(deck['ci95']) != excludes(tl['ci95'])})
    res = {'source': {'analysis': str(ANALYSIS.relative_to(ROOT)), 'jerk_0p1s': str((V5/'per_run.json').relative_to(ROOT))},
           'bootstrap': {'reps': REPS, 'seed': SEED, 'two_level': 'texts with replacement, then tasks within each drawn text with replacement; equal task weight'},
           'reproduction_gate': {'checked': len(gate), 'equal': sum(g['equal'] for g in gate), 'mismatches': [g for g in gate if not g['equal']]},
           'effects': results}
    (OUT/'recheck.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res['reproduction_gate'], indent=1)[:2000])
    for r in results:
        t, w = r['task_cluster'], r['two_level']
        flag = 'CHANGES' if r['conclusion_changes'] else ''
        print(f"{r['method']:11s} {r['effect']:15s} {json.dumps(r['fixed']):16s} {r['metric']:36s} {t['mean']:+9.3f} "
              f"task[{t['ci95'][0]:+.3f},{t['ci95'][1]:+.3f}] two[{w['ci95'][0]:+.3f},{w['ci95'][1]:+.3f}] "
              f"texts {'same' if w['same_sign_across_texts'] else 'DIFF'} {flag}")


if __name__ == '__main__':
    main()
