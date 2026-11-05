#!/usr/bin/env python
"""Summary of E v2 (config/e_v2_protocol_20261106.json): tau, Moco test split, static and injection validation against
the registered pass criteria, and the v1-v2 consistency of the bracket comparisons.

Reads develop_logs/test_runs/20261106_e_v2/{moco_refs,static,inject,rescore} and the v1 bracket rows, writes analysis.json
there; sections whose runs are missing are reported as incomplete. OpenSim python.
"""
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils/motiogen'), str(ROOT/'utils'), str(ROOT/'python/scripts')]
import analyze_e_domain as v1  # noqa: E402

BASE = ROOT/'develop_logs/test_runs/20261106_e_v2'
PROTOCOL = json.loads((ROOT/'config/e_v2_protocol_20261106.json').read_text())
REFS = json.loads((ROOT/'config/e_v2_moco_refs_20261106.json').read_text())
PASS = PROTOCOL['pass_criteria']
RULES = v1.RULES
DETECT, DEPTH = .10, .03
rate = v1.rate


def envelope(z, rule, tau):
    bound = np.zeros(len(z[f'{rule}_residual_bw']))
    for state, value in tau.items():
        bound[z[f'{rule}_state_{state}']] = value
    return z[f'{rule}_residual_bw'] > bound


# ---------------------------------------------------------------- moco
def moco():
    tau = PROTOCOL['tau_envelope_bw']
    out = {'tau_envelope_bw': tau, 'tau_source': PROTOCOL['tau_source'], 'tau_frozen_at': PROTOCOL['tau_frozen_at'], 'refs': {}}
    by_split = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in REFS['refs']:
        status = BASE/'moco_refs'/'status'/f"{r['trial']}.json"
        rec = json.loads(status.read_text()) if status.exists() else None
        row = {'split': r['split'], 'class': r['class'], 'status': rec['status'] if rec else 'not_run',
               'attempts': rec['attempts'] if rec else None}
        if rec and rec['status'] == 'ok':
            d = ROOT/rec['dir']; ev = json.loads((d/'evaluation.json').read_text()); z = np.load(d/'frames_v2.npz')
            sel = z['sel']; row['frames'] = int(sel.sum()); row['rules'] = {}
            for rule in RULES:
                v2 = ev['v2']['rules'][rule]
                x = {'v1_envelope_infeasible_share': ev['rules'][rule]['envelope_infeasible_share'],
                     'v1_residual_force_rms_bw': ev['rules'][rule]['residual_force_rms_bw'],
                     'v2_residual_force_rms_bw': v2['residual_force_rms_bw'], 'v2_contact_agreement': v2['contact_agreement'],
                     'v2_vertical_grf': v2.get('vertical_grf'), 'v2_lowest_elevation_m': v2.get('lowest_elevation_m'),
                     'v2_hover_share': float(np.mean(z[f'{rule}_lowest_elevation'][sel] > DEPTH)),
                     'v2_penetration_share': float(np.mean(z[f'{rule}_lowest_elevation'][sel] < -DEPTH))}
                if tau:
                    x['v2_envelope_infeasible_share'] = float(envelope(z, rule, tau[rule])[sel].mean())
                row['rules'][rule] = x
                for k, val in x.items():
                    if isinstance(val, float):
                        by_split[r['split']][(rule, k)].append(val)
        out['refs'][r['trial']] = row
    out['status_counts'] = {sp: dict(collections.Counter(v['status'] for v in out['refs'].values() if v['split'] == sp))
                            for sp in ('calibration', 'test')}
    out['mean_by_split'] = {sp: {f'{rule} {k}': float(np.mean(v)) for (rule, k), v in sorted(d.items())} for sp, d in by_split.items()}
    test = by_split.get('test', {})
    out['pass_false_alarm'] = {rule: None if not test.get((rule, 'v2_envelope_infeasible_share')) else
                               {'mean': float(np.mean(test[(rule, 'v2_envelope_infeasible_share')])),
                                'n': len(test[(rule, 'v2_envelope_infeasible_share')]), 'max': PASS['moco_test_false_alarm_max'],
                                'pass': bool(np.mean(test[(rule, 'v2_envelope_infeasible_share')]) <= PASS['moco_test_false_alarm_max'])}
                               for rule in RULES}
    return out


# ---------------------------------------------------------------- static
def static():
    poses = json.loads((BASE/'static_poses.json').read_text())
    rows = [json.loads((BASE/'static'/f"{p['id']}.json").read_text()) for p in poses if (BASE/'static'/f"{p['id']}.json").exists()]
    ok = [r for r in rows if r['status'] == 'success']
    out = {'poses': len(poses), 'done': len(rows), 'success': len(ok), 'failed': [r['id'] for r in rows if r['status'] != 'success']}
    flag = lambda r, rule: r['rules'][rule]['envelope_infeasible_share'] > .5
    pen = lambda r, rule: r['rules'][rule]['penetration_share'] > .5
    kind = lambda r: 'hover05' if (r.get('offset_m') or 0) > 0 else 'sink05' if (r.get('offset_m') or 0) < 0 else r['source']
    for tol in ('primary', 'support_tol_0.01'):
        res = {}
        for rule in RULES:
            t = lambda r: r['truth'][tol]
            strict = [r for r in ok if t(r)['label'] == 'infeasible' and t(r)['support_feet'] and t(r)['margin_m'] is not None
                      and t(r)['margin_m'] <= -.05]
            x = {'pass_subset_detection': rate(flag(r, rule) for r in strict),
                 'detection_truth_infeasible': rate(flag(r, rule) for r in ok if t(r)['label'] == 'infeasible'),
                 'false_alarm_truth_feasible': rate(flag(r, rule) for r in ok if t(r)['label'] == 'feasible'),
                 'flag_indeterminate': rate(flag(r, rule) for r in ok if t(r)['label'] == 'indeterminate')}
            sub = x['pass_subset_detection']
            x['pass'] = None if not sub['n'] else bool(sub['share'] >= PASS['static_detection_min']['value'])
            groups = collections.defaultdict(list)
            for r in ok:
                m = t(r)['margin_m']
                band = 'no_support' if m is None else next(f'[{a:g},{b:g})' for a, b in v1.BANDS if a <= m < b)
                groups[('band', band)].append(flag(r, rule))
                groups[('category', f"{kind(r)} {t(r)['category']}")].append(flag(r, rule))
                groups[('kind_label', f"{kind(r)} {t(r)['label']}")].append(flag(r, rule))
                groups[('method', r['class'])].append(flag(r, rule))
            for (k, name), v in sorted(groups.items()):
                x.setdefault(f'flag_by_{k}', {})[name] = rate(v)
            x['penetration_flag_by_kind'] = {k: rate(pen(r, rule) for r in ok if kind(r) == k) for k in ('human', 'generated', 'hover05', 'sink05')}
            x['hover_share_median_by_kind'] = {k: float(np.median([r['rules'][rule]['hover_share'] for r in ok if kind(r) == k]))
                                               for k in ('human', 'generated', 'hover05', 'sink05') if any(kind(r) == k for r in ok)}
            res[rule] = x
        res['truth_counts'] = dict(collections.Counter(f"{kind(r)} {r['truth'][tol]['label']} {r['truth'][tol]['category']}" for r in ok))
        out[tol] = res
    return out


# ---------------------------------------------------------------- inject
def inject():
    jobs = json.loads((BASE/'inject_jobs.json').read_text())
    d = BASE/'inject'
    rows = {j['id']: json.loads((d/f"{j['id']}.json").read_text()) for j in jobs if (d/f"{j['id']}.json").exists()}
    out = {'jobs': len(jobs), 'done': len(rows), 'status': dict(collections.Counter(r['status'] for r in rows.values())),
           'failed': [k for k, r in rows.items() if r['status'] == 'failed']}

    def metrics(z, rule, sel):
        if not sel.any():
            return None
        m = v1.window_metrics(z, rule, sel)
        m['hover_share'] = float(z[f'{rule}_hover'][sel].mean()); m['penetration_share'] = float(z[f'{rule}_penetration'][sel].mean())
        return m

    deltas = collections.defaultdict(list)
    for j in jobs:
        r = rows.get(j['id']); c = rows.get(f"{j['base']}__control")
        if j['defect'] == 'control' or not r or not c or r['status'] != 'success' or c['status'] != 'success':
            continue
        z = np.load(d/f"{j['id']}.npz"); zc = np.load(d/f"{j['base']}__control.npz")
        t, tc = z['times'], zc['times']
        if r['window_s']:
            a, b = r['window_s']; sel, selc = (t >= a) & (t < b), (tc >= a) & (tc < b)
        else:
            s = .25 if j['defect'] == 'time_compression' else .5
            sel, selc = v1.trim_sel(t, s), v1.trim_sel(tc, s)
        for rule in RULES:
            m, mc = metrics(z, rule, sel), metrics(zc, rule, selc)
            if m is None or mc is None:
                continue
            dd = {k: (None if m[k] is None or mc[k] is None else m[k]-mc[k]) for k in m}
            dd['detected'] = dd['envelope_infeasible_share'] >= DETECT
            dd['depth_detected'] = dd['hover_share'] >= DETECT or dd['penetration_share'] >= DETECT
            deltas[(j['defect'], j['size'], rule)].append((j['kind'], j['class'], dd))
    table = {}
    for (defect, size, rule), v in sorted(deltas.items()):
        row = {}
        for grp in ('all', 'moco', 'human', 'walk', 'sts', 'squat', 'wholebody'):
            vv = [x for kind, cls, x in v if grp in ('all', kind, cls)]
            if not vv:
                continue
            med = lambda k: (float(np.median([x[k] for x in vv if x[k] is not None])) if any(x[k] is not None for x in vv) else None)
            row[grp] = {'n': len(vv), 'detection_rate': float(np.mean([x['detected'] for x in vv])),
                        'depth_detection_rate': float(np.mean([x['depth_detected'] for x in vv])),
                        'hover_detection_rate': float(np.mean([x['hover_share'] >= DETECT for x in vv])),
                        'penetration_detection_rate': float(np.mean([x['penetration_share'] >= DETECT for x in vv])),
                        **{f'median_delta_{k}': med(k) for k in ('envelope_infeasible_share', 'residual_force_rms_bw', 'foot_skate_share',
                                                                  'landmark_contact_share', 'hover_share', 'penetration_share')}}
        table.setdefault(f'{defect} {size:g}', {})[rule] = row
    out['table'] = table
    hover = table.get('float_whole_m 0.05', {}); sink = table.get('penetrate_whole_m 0.05', {})
    out['pass'] = {rule: {'whole_hover_0.05_envelope': None if rule not in hover else
                          {'rate': hover[rule]['all']['detection_rate'], 'n': hover[rule]['all']['n'],
                           'pass': hover[rule]['all']['detection_rate'] >= PASS['whole_clip_hover_0.05_detection_min']['value']},
                          'whole_penetration_0.05_depth': None if rule not in sink else
                          {'rate': sink[rule]['all']['penetration_detection_rate'], 'n': sink[rule]['all']['n'],
                           'pass': sink[rule]['all']['penetration_detection_rate'] >= PASS['whole_clip_penetration_0.05_detection_min']['value']}}
                   for rule in RULES}
    ctl = collections.defaultdict(list)
    for k, r in rows.items():
        if r['defect'] == 'control' and r['status'] == 'success':
            z = np.load(d/f'{k}.npz'); sel = v1.trim_sel(z['times'], .5)
            for rule in RULES:
                m = metrics(z, rule, sel)
                for key in ('envelope_infeasible_share', 'hover_share', 'penetration_share'):
                    ctl[(rule, r['kind'], key)].append(m[key])
    out['control_median'] = {f'{a} {b} {c}': float(np.median(v)) for (a, b, c), v in sorted(ctl.items())}
    return out


# ---------------------------------------------------------------- rescore
METRICS = ('residual_force_rms_bw', 'unsupported_fraction_5pct_bw', 'envelope_infeasible_share', 'reba_r3_certain_acceptable_gated')


def flat(rule_row):
    out = {k: rule_row[k] for k in METRICS[:3]}
    out['reba_r3_certain_acceptable_gated'] = rule_row['reba_r3_certain_acceptable']['gated_envelope']
    return out


def load(path):
    rows = {}
    for f in path.glob('*.json'):
        r = json.loads(f.read_text())
        if r['status'] == 'success':
            rows[r['id']] = r
    return rows


def versus(c1, c2):
    out = {}
    for key in sorted(set(c1) & set(c2)):
        a, b = c1[key]['R0'], c2[key]['R0']
        if a.get('mean') is None or b.get('mean') is None:
            out[key] = 'not_testable'; continue
        s1 = v1.np.sign(a['mean']) if c1[key]['R0_excludes_zero'] else 0
        s2 = v1.np.sign(b['mean']) if c2[key]['R0_excludes_zero'] else 0
        out[key] = ('same_sign_conclusion' if s1 == s2 != 0 else 'null_under_both' if s1 == s2 == 0 else
                    'opposite_sign' if s1*s2 < 0 else 'v1_only' if s1 else 'v2_only')
    return out


def rescore():
    v1.METRICS, v1.flat = METRICS, flat
    out = {}
    for set_name in ('e4', 'x3', 'm1', 'e1'):
        rows2 = load(BASE/'rescore'/set_name); rows1 = load(v1.BASE/'bracket'/set_name)
        if not rows2:
            out[set_name] = {'runs': 0}; continue
        desc = {}
        for rule in RULES:
            by_m = collections.defaultdict(list)
            for r in rows2.values():
                if r['valid']:
                    by_m[r['method']].append(r['rules'][rule])
            desc[rule] = {m: {k: float(np.median([x[k] for x in v if x.get(k) is not None]))
                              for k in ('residual_force_rms_bw', 'envelope_infeasible_share', 'foot_skate_share', 'landmark_contact_share')} |
                          {'lowest_elevation_median_m': float(np.median([x['lowest_elevation_m']['median'] for x in v]))}
                          for m, v in sorted(by_m.items())}
        common = set(rows1) & set(rows2)
        c1 = v1.consistency(v1.comparisons(set_name, {k: rows1[k] for k in common}))
        c2 = v1.consistency(v1.comparisons(set_name, {k: rows2[k] for k in common}))
        table = versus(c1, c2)
        out[set_name] = {'runs_v2': len(rows2), 'runs_v1': len(rows1), 'paired_runs': len(common), 'descriptive_valid_median': desc,
                         'v1_v2_R0': dict(collections.Counter(table.values())),
                         'v1_v2_R0_by_metric': {m: dict(collections.Counter(v for k, v in table.items() if k.endswith(f':: {m}'))) for m in METRICS},
                         'changed': {k: v for k, v in table.items() if v not in ('same_sign_conclusion', 'null_under_both')},
                         'v2_verdict_counts': dict(collections.Counter(v['verdict'] for v in c2.values())),
                         'v2_consistency': c2, 'v1_consistency_same_runs': c1}
    return out


def main():
    sections = sys.argv[1:] or ['moco', 'static', 'inject', 'rescore']
    path = BASE/'analysis.json'
    result = json.loads(path.read_text()) if path.exists() else {}
    result['protocol'] = PROTOCOL['protocol']
    for s in sections:
        result[s] = globals()[s]()
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    for s in sections:
        print(s, json.dumps(result[s], ensure_ascii=False, default=float)[:3000])


if __name__ == '__main__':
    main()
