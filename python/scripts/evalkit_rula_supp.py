#!/usr/bin/env python
"""RULA supplement (config/evalkit_rula_supp_20261108.json) on the heldout4 split, after evalkit_validate.py truth / tool /
selffloor --dims residual,ergo:

  evalkit_rula_supp.py rates      per-subject RULA / REBA kill rates (both floors), subject-cluster bootstrap, floor check,
                                  10-subject pooled view (heldout2 + 3 + 4, method floor)
  evalkit_rula_supp.py decompose  false-kill sources (heldout2 + 3 + 4, method floor): RULA inputs recomputed from the saved
                                  coordinates, one input group at a time swapped from truth into the tool  [--workers N]
"""
import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'python/scripts')]
import evalkit_validate as V  # noqa: E402
from evalkit.validate import rate  # noqa: E402

REG = json.loads((ROOT/'config/evalkit_rula_supp_20261108.json').read_text())
OUT = V.LOG/'rula_supp'
SPLITS = ('heldout2', 'heldout3', 'heldout4')
GROUPS = {'upper_arm': ('upper_arm_flexion_deg', 'arm_abducted_or_rotated'), 'elbow': ('elbow_flexion_deg',),
          'trunk': ('trunk_flexion_deg', 'trunk_side_bent', 'trunk_twisted'), 'neck': ('neck_flexion_deg', 'neck_side_bent'),
          'shoulder_raised': ('shoulder_raised',), 'legs': ('leg_support_score', 'legs_score', 'knee_flexion_deg'),
          'unstable': ('rapid_change_or_unstable',)}
NUMERIC = ('upper_arm_flexion_deg', 'elbow_flexion_deg', 'trunk_flexion_deg', 'neck_flexion_deg', 'knee_flexion_deg')
load = lambda p: dict(np.load(p, allow_pickle=True)) if Path(p).exists() else None
subject = lambda cid: cid.split('_')[1]
base_task = lambda cid: cid.split('__')[0].split('_', 2)[2]


def rows(split, floor):
    out = []
    for c in V.clips(split):
        c = {**c, 'split': split}; m = V.meta(c)
        if not V.ergo_pass(m):
            continue
        tool = DATA_TOOL[floor]/split/c['id']
        if floor == 'self' and not (tool/'selffloor.json').exists():
            continue
        out.append({'id': c['id'], 'meta': m, 'tool': load(tool/'frames.npz'),
                    'truth_eval': load(V.DATA/'truth_eval'/split/c['id']/'truth_eval.npz')})
    return [r for r in out if r['tool'] is not None and r['truth_eval'] is not None]


DATA_TOOL = {'method': V.DATA/'tool', 'self': V.DATA/'tool_selffloor'}


def frame_arrays(rs, m):
    """Truth and tool classes of the matched ergo frames with their subject labels (evalkit_validate.ergo_frames)."""
    d = V.ergo_frames(rs, lambda meta: True)[m]
    subj = np.array([subject(rs[i]['id']) for i in d['clip']]) if len(d['clip']) else np.array([])
    return d['truth'], d['tool'], subj


def fk_mk(t, x):
    neg, pos, kill = t == 3, t == 1, (x == 1) | (x == 0)
    return {'false_kill': rate((kill & neg).sum(), neg.sum()), 'missed_kill': rate(((x == 3) & pos).sum(), pos.sum())}


def subject_bootstrap(t, x, subj, boot, seed):
    ids = np.unique(subj); rng = np.random.default_rng(seed)
    k = {s: int((((x == 1) | (x == 0)) & (t == 3) & (subj == s)).sum()) for s in ids}
    n = {s: int(((t == 3) & (subj == s)).sum()) for s in ids}
    draws = []
    for _ in range(boot):
        pick = rng.choice(ids, len(ids))
        den = sum(n[s] for s in pick)
        draws.append(sum(k[s] for s in pick)/den if den else np.nan)
    return {'subjects': len(ids), 'interval95': [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]}


def rates(args):
    OUT.mkdir(parents=True, exist_ok=True)
    boot = 4000
    res = {'registration': 'config/evalkit_rula_supp_20261108.json', 'split': args.split, 'clips': REG['clips']}
    for floor in ('self', 'method'):
        rs = rows(args.split, floor)
        block = {'clips': len(rs), 'line': {m.lower(): V.ergo_rates(rs, lambda meta: True)[m.lower()] for m in ('RULA', 'REBA')}}
        for m in ('rula', 'reba'):
            block['line'][m] = {k: v for k, v in block['line'][m].items() if k != 'confusion'}
            t, x, subj = frame_arrays(rs, m)
            block[f'{m}_per_subject'] = {s: fk_mk(t[subj == s], x[subj == s]) for s in np.unique(subj)}
            block[f'{m}_subject_bootstrap'] = subject_bootstrap(t, x, subj, boot, 0)
            block[f'{m}_subjects_over_5pct'] = sum(v['false_kill']['rate'] is not None and v['false_kill']['rate'] > .05
                                                   for v in block[f'{m}_per_subject'].values())
        res[f'{floor}_floor'] = block
    res['primary'] = {'rula': res['self_floor']['line']['rula'], 'pass': res['self_floor']['line']['rula']['pass']}
    same = {}
    for c in rows(args.split, 'self'):
        mt = load(V.DATA/'tool'/args.split/c['id']/'frames.npz')
        if mt is None:
            continue
        for m in ('rula', 'reba'):
            a, b = c['tool'][f'ergo_{m}_class'], mt[f'ergo_{m}_class']
            same.setdefault(m, []).append(bool(len(a) == len(b) and np.array_equal(a, b)))
    res['floor_check'] = {m: {'clips_identical': int(sum(v)), 'clips': len(v)} for m, v in same.items()}
    pooled = {}
    for split in [s for s in SPLITS if (V.SYNTH/s).exists()]:
        rs = rows(split, 'method'); t, x, subj = frame_arrays(rs, 'rula')
        for s in np.unique(subj):
            pooled[f'{split}_{s}'] = fk_mk(t[subj == s], x[subj == s])['false_kill']
    res['pooled_10_subjects_method_floor'] = {'splits': [s for s in SPLITS if (V.SYNTH/s).exists()], 'per_subject': pooled,
                                              'over_5pct': sum(v['rate'] > .05 for v in pooled.values() if v['rate'] is not None),
                                              'pooled': rate(sum(v['k'] for v in pooled.values()), sum(v['n'] for v in pooled.values()))}
    (OUT/f'rates_{args.split}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({'primary': res['primary'], 'floor_check': res['floor_check'],
                      'self_rula_per_subject': {s: v['false_kill']['rate'] for s, v in res['self_floor']['rula_per_subject'].items()},
                      'self_rula_bootstrap': res['self_floor']['rula_subject_bootstrap'],
                      'pooled_10': {k: v for k, v in res['pooled_10_subjects_method_floor'].items() if k != 'per_subject'}}, indent=1))


def capture_score(model, times, labels, coords, masks, sel, src, zero, free, unmeasurable):
    """ergo.score with the per-frame, per-side RULA inputs recorded (score_union is called frame -> method -> side)."""
    from evalkit import ergo
    calls, orig = [], ergo.score_union

    def rec(inputs, method):
        calls.append((method, dict(inputs)))
        return orig(inputs, method)
    ergo.score_union = rec
    try:
        e = ergo.score(model, times, labels, coords, masks, sel, src, zero, free, unmeasurable)
    finally:
        ergo.score_union = orig
    rula = [c[1] for c in calls if c[0] == 'RULA']
    sides = len(ergo.SIDES)
    return e, [dict(zip(ergo.SIDES, rula[i*sides:(i+1)*sides])) for i in range(len(e['times']))]


def rula_class(side_inputs):
    from evalkit import ergo
    lo = hi = None
    for inp in side_inputs.values():
        a, b = ergo.score_union(inp, 'RULA')
        lo, hi = (a, b) if lo is None else (max(lo, a), max(hi, b))
    risk = ergo.RISK['RULA']
    return 1 if lo >= risk else 2 if hi >= risk else 3


def decompose_job(clip):
    """One clip of the method-floor run: recomputed tool and truth RULA inputs, reproduction check, attribution of the
    scored-risk false kills."""
    try:
        V._quiet()
        import biomech
        import biomech_sources
        from evalkit import ergo, fit
        from evalkit.validate import match
        split, cid = clip['split'], clip['id']
        d = Path(clip['dir']); te_dir = V.DATA/'truth_eval'/split/cid; tl_dir = V.DATA/'tool'/split/cid
        tr = np.load(d/'truth/truth.npz')
        t_times, t_labels, t_coords = biomech.read_storage(d/'truth/coordinates.mot')
        t_masks = {s: {k: tr[f'contact_{s}_{k}'] for k in V.LANDMARKS} for s in V.SIDES}
        te, t_in = capture_score(te_dir/'model.osim', t_times, t_labels, t_coords, t_masks, fit.scored(t_times),
                                 biomech_sources.load(d/'truth/clean_joints.npz'), ergo.neutral('synthetic'), True, None)
        f = np.load(tl_dir/'frames.npz', allow_pickle=True)
        x_times, x_labels, x_coords = biomech.read_storage(tl_dir/'work/coordinates.mot')
        if len(x_times) != len(f['times']) or not np.allclose(x_times, f['times'], atol=1e-6):
            return {'id': cid, 'error': 'tool coordinates do not match frames.npz times'}
        x_masks = {s: {k: f[f'contact_{s}_{k}'] for k in V.LANDMARKS} for s in V.SIDES}
        run = d/'input'
        xe, x_in = capture_score(tl_dir/'work/model.osim', x_times, x_labels, x_coords, x_masks, f['scored'],
                                 biomech_sources.load(run), ergo.neutral('synthetic', run), ergo.object_free(None), f['unmeasurable'])
        st, sx = np.load(te_dir/'truth_eval.npz', allow_pickle=True), f
        repro = {'truth': bool(np.array_equal(te['rula_class'], st['ergo_rula_class'])),
                 'tool': bool(np.array_equal(xe['rula_class'], sx['ergo_rula_class']))}
        j = match(te['times'], xe['times']); ok = np.abs(xe['times'][j]-te['times']) <= .02
        frames = []
        for k in np.flatnonzero(ok):
            t, x = int(te['rula_class'][k]), int(xe['rula_class'][j[k]])
            if t != 3:
                continue
            row = {'truth_class': t, 'tool_class': x,
                   'diff': {n: float(np.mean([x_in[j[k]][s][n]-t_in[k][s][n] for s in x_in[j[k]]])) for n in NUMERIC}}
            if x == 1:
                row['fixed_by'] = [g for g, keys in GROUPS.items()
                                   if rula_class({s: {**x_in[j[k]][s], **{n: t_in[k][s][n] for n in keys}} for s in x_in[j[k]]}) != 1]
            frames.append(row)
        return {'id': cid, 'split': split, 'reproduced': repro, 'frames': frames}
    except Exception as e:
        import traceback
        return {'id': clip['id'], 'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-1500:]}


def pool_results(fn, items, workers):
    out = []
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        for n, r in enumerate(p.map(fn, items), 1):
            out.append(r)
            if n % 50 == 0 or n == len(items):
                print(json.dumps({'done': n, 'of': len(items), 'errors': sum('error' in x for x in out)}), flush=True)
    return out


def decompose(args):
    OUT.mkdir(parents=True, exist_ok=True)
    splits = [s for s in SPLITS if (V.SYNTH/s).exists()]
    items = [{**c, 'split': s} for s in splits for c in V.clips(s) if V.ergo_pass(V.meta({**c, 'split': s}))]
    items = [c for c in items if (V.DATA/'tool'/c['split']/c['id']/'frames.npz').exists()][:args.limit]
    results = pool_results(decompose_job, items, args.workers)
    errors = [r for r in results if 'error' in r]; ok = [r for r in results if 'error' not in r]
    repro = {s: {'truth': sum(r['reproduced']['truth'] for r in ok), 'tool': sum(r['reproduced']['tool'] for r in ok), 'clips': len(ok)}
             for s in ('all',)}
    reliable = [r for r in ok if r['reproduced']['truth'] and r['reproduced']['tool']]
    fr = [dict(f, id=r['id']) for r in reliable for f in r['frames']]
    fk_unmeasurable = sum(f['tool_class'] == 0 for f in fr); fk_risk = [f for f in fr if f['tool_class'] == 1]
    correct = [f for f in fr if f['tool_class'] == 3]
    share = lambda fs, g: float(np.mean([g in f['fixed_by'] for f in fs])) if fs else None
    combined = lambda fs: float(np.mean([not f['fixed_by'] for f in fs])) if fs else None
    med = lambda fs, n: float(np.median([f['diff'][n] for f in fs])) if fs else None
    tasks = sorted({base_task(f['id']) for f in fk_risk})
    res = {'registration': 'config/evalkit_rula_supp_20261108.json', 'splits': splits,
           'reproduction': {**repro['all'], 'reliable_clips': len(reliable), 'reliable': len(reliable) == len(ok)},
           'errors': [{k: r[k] for k in ('id', 'error')} for r in errors],
           'truth_acceptable_frames': len(fr), 'false_kill_unmeasurable': int(fk_unmeasurable), 'false_kill_risk': len(fk_risk),
           'attribution': {**{g: share(fk_risk, g) for g in GROUPS}, 'combined': combined(fk_risk)},
           'by_task': {b: {'false_kill_risk': sum(base_task(f['id']) == b for f in fk_risk),
                           'truth_acceptable': sum(base_task(f['id']) == b for f in fr),
                           **{g: share([f for f in fk_risk if base_task(f['id']) == b], g) for g in GROUPS},
                           'combined': combined([f for f in fk_risk if base_task(f['id']) == b])} for b in tasks},
           'input_diff_median_tool_minus_truth': {n: {'false_kill': med(fk_risk, n), 'correct': med(correct, n)} for n in NUMERIC}}
    (OUT/'decompose.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'by_task'}, indent=1))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('step', choices=('rates', 'decompose')); a.add_argument('--workers', type=int, default=6)
    a.add_argument('--limit', type=int); a.add_argument('--split', default='heldout4', help='rates: the registered split is heldout4')
    args = a.parse_args()
    {'rates': rates, 'decompose': decompose}[args.step](args)


if __name__ == '__main__':
    main()
