#!/usr/bin/env python
"""evalkit validation on the synthetic set (config/evalkit_v1_20261107.json, validation). OpenSim python.

  calibrate            synthetic ergo neutral (skeleton definition) and floor reference (development stand controls);
                       writes the calibrated fields of the registration (refused once the version is frozen)
  tool --split S       evalkit on every clip of the split (muscle on control clips only)
  truth --split S      the same ergo scoring and static optimization on the truth (truth model, coordinates, contact)
  score --split S      kill rates; on dev also the tau / depth tuning grid (writes nothing into the registration)
Outputs: per clip on the data disk (evalkit_20261107/{tool,truth_eval}/<split>/<clip>), summaries under
develop_logs/test_runs/20261107_evalkit/validation.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
from evalkit.spec import SPEC, SPEC_PATH, SIDES, LANDMARKS  # noqa: E402

DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/evalkit_20261107')
SYNTH = DATA/'synth'
LOG = ROOT/'develop_logs/test_runs/20261107_evalkit/validation'
SKIP_KILL = ('bone_stretch', 'slide_m_s')


def clips(split):
    out = []
    for base in sorted((SYNTH/split).iterdir()):
        if not (base/'done.json').exists():
            continue
        info = json.loads((base/'done.json').read_text())['info']
        out += [{'dir': str(base/c), 'id': c, 'base_dir': str(base), 'info': info} for c in json.loads((base/'done.json').read_text())['clips']]
    return out


def meta(clip):
    return json.loads((Path(clip['dir'])/'input/meta.json').read_text())


def _quiet():
    os.nice(19)
    import opensim as osim
    osim.Logger.setLevelString('error')


# ---------------------------------------------------------------- calibrate

def neutral_reading(base_dir, info):
    import opensim as osim
    from evalkit import synth, ergo
    model = osim.Model(str(Path(base_dir)/'model.osim')); state = model.initSystem(); cs = model.getCoordinateSet()
    labels = synth.coordinate_names(model)
    d = np.array([cs.get(n).getDefaultValue()*(180/np.pi if cs.get(n).getMotionType() == 1 else 1.) for n in labels])
    ex = synth.export(Path(base_dir)/'model.osim', {'labels': labels, 'coords': np.repeat(d[None], 2, 0), 'fps': synth.FPS,
                                                     'floor': 0., 'stretch': None}, info, seed=0)
    return ergo.rest_reading(ex['clean_joints'][0], ex['joint_names'])


def calibrate(args):
    if SPEC['version'] != 'v1-dev':
        raise SystemExit('registration is frozen')
    _quiet()
    from evalkit import run
    stand = [c for c in clips('dev') if c['id'].endswith('stand__control')]
    readings = [neutral_reading(c['base_dir'], c['info']) for c in stand]
    neck = np.mean([r[0] for r in readings], 0); shoulder = np.mean([r[1] for r in readings], 0)
    heights = {s: {k: [] for k in LANDMARKS} for s in SIDES}
    zero = {s: {k: 0. for k in LANDMARKS} for s in SIDES}
    for c in stand:
        m = meta(c); out = DATA/'calibration'/c['id']
        run.evaluate(Path(c['dir'])/'input', out, m['floor_height_m'], zero, m['mass_kg'], dims=('residual',))
        f = np.load(out/'frames.npz'); tr = np.load(Path(c['dir'])/'truth/truth.npz')
        from evalkit.validate import match
        j = match(f['times'], tr['times'])
        for s in SIDES:
            both = (tr[f'contact_{s}_heel'] & tr[f'contact_{s}_toe'])[j] & f['scored']
            for k in LANDMARKS:
                heights[s][k] += list(f[f'elevation_{s}_{k}'][both])
    ref = {s: {k: float(np.median(heights[s][k])) for k in LANDMARKS} for s in SIDES}
    spec = json.loads(SPEC_PATH.read_text())
    spec['floor']['calibrated']['synthetic'] = ref
    spec['ergo']['neutral_calibrated']['synthetic'] = {'neck_deg': neck.tolist(), 'shoulder_elevation_deg': shoulder.tolist()}
    spec['calibration_receipt'] = {'clips': [c['id'] for c in stand], 'reference_frames': {s: len(heights[s]['heel']) for s in SIDES},
                                   'neutral_subjects': [c['info']['seed'] for c in stand]}
    SPEC_PATH.write_text(json.dumps(spec, indent=2))
    print(json.dumps({'reference': ref, 'neutral_neck_deg': neck.tolist(), 'neutral_shoulder_deg': shoulder.tolist()}, indent=1))


# ---------------------------------------------------------------- tool and truth runs

def tool_job(clip):
    out = DATA/'tool'/clip['split']/clip['id']
    if (out/'evalkit.json').exists():
        return 'skip'
    try:
        _quiet()
        from evalkit import run
        m = meta(clip)
        dims = ('residual', 'ergo', 'muscle') if m['defect'] == 'control' else ('residual', 'ergo')
        run.evaluate(Path(clip['dir'])/'input', out, m['floor_height_m'], 'synthetic', m['mass_kg'], method='synthetic', dims=dims)
        return 'ok'
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        (out/'error.json').write_text(json.dumps({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-2000:]}))
        return 'failed'


def truth_model(clip, path):
    """The truth model with the neck-base station neck_c (the export's neck station) for the trunk angle."""
    import opensim as osim
    from evalkit import synth
    model = osim.Model(str(Path(clip['base_dir'])/'model.osim'))
    loc = np.array(synth.TORSO['neck'])*clip['info']['torso_factor']
    model.addMarker(osim.Marker('neck_c', model.getBodySet().get('torso'), osim.Vec3(*loc)))
    model.finalizeConnections(); model.printToXML(str(path))


def truth_job(clip):
    out = DATA/'truth_eval'/clip['split']/clip['id']
    if (out/'truth_eval.npz').exists():
        return 'skip'
    try:
        _quiet()
        import biomech
        import biomech_sources
        from evalkit import ergo, fit, muscle
        out.mkdir(parents=True, exist_ok=True)
        d = Path(clip['dir']); m = meta(clip); tr = np.load(d/'truth/truth.npz')
        truth_model(clip, out/'model.osim')
        times, labels, coords = biomech.read_storage(d/'truth/coordinates.mot')
        sel = fit.scored(times)
        masks = {s: {k: tr[f'contact_{s}_{k}'] for k in LANDMARKS} for s in SIDES}
        src = biomech_sources.load(d/'truth/clean_joints.npz')
        e = ergo.score(out/'model.osim', times, labels, coords, masks, sel, src, ergo.neutral('synthetic'), True, None)
        arrays = {f'ergo_{k}': v for k, v in e.items() if k != 'source_mode'}
        if m['defect'] == 'control':
            for name in ('coordinates.mot', 'grf.xml', 'grf.mot'):
                link = out/name
                if not link.exists():
                    link.symlink_to(d/'truth'/name)
            mu = muscle.evaluate(out, float(times[sel][0]), float(times[sel][-1]))
            arrays.update(muscle_times=mu['times'], muscle_effort=mu['effort'], muscle_reserve_abs_sum_nm=mu['reserve_abs_sum_nm'])
            (out/'muscle.json').write_text(json.dumps(mu['summary'], indent=1))
        np.savez_compressed(out/'truth_eval.npz', times=times, scored=sel, source_mode=e['source_mode'], **arrays)
        return 'ok'
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        (out/'error.json').write_text(json.dumps({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-2000:]}))
        return 'failed'


def pool(fn, items, workers):
    counts = {}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        for s in p.map(fn, items):
            counts[s] = counts.get(s, 0)+1
            print(json.dumps(counts), flush=True)
    return counts


# ---------------------------------------------------------------- score

def load_clip(clip):
    d = Path(clip['dir']); m = meta(clip)
    tool, truth = DATA/'tool'/clip['split']/clip['id'], DATA/'truth_eval'/clip['split']/clip['id']
    if not (tool/'frames.npz').exists():
        return None
    return {'meta': m, 'truth': dict(np.load(d/'truth/truth.npz')), 'tool': dict(np.load(tool/'frames.npz', allow_pickle=True)),
            'truth_eval': dict(np.load(truth/'truth_eval.npz', allow_pickle=True)) if (truth/'truth_eval.npz').exists() else None,
            'truth_muscle': json.loads((truth/'muscle.json').read_text()) if (truth/'muscle.json').exists() else None,
            'tool_summary': json.loads((tool/'evalkit.json').read_text())}


def residual_frames(rows):
    """Concatenated per-frame arrays over the kill-rate clips: truth label, tool residual, lowest elevation, state,
    unmeasurable, clip index."""
    from evalkit.validate import match
    out = {k: [] for k in ('label', 'residual', 'lowest', 'state', 'unmeasurable', 'clip')}
    for n, r in enumerate(rows):
        if r['meta']['defect'] in SKIP_KILL:
            continue
        tr, tl = r['truth'], r['tool']
        j = match(tr['times'], tl['times']); keep = tl['scored'][j]
        out['label'].append(tr['e_label'][keep]); out['residual'].append(tl['residual_bw'][j][keep])
        out['lowest'].append(tl['lowest_elevation_m'][j][keep]); out['state'].append(tl['state'][j][keep])
        out['unmeasurable'].append(tl['unmeasurable'][j][keep]); out['clip'].append(np.full(keep.sum(), n))
    return {k: np.concatenate(v) for k, v in out.items()}


def residual_positive(f, tau, depth):
    bound = np.select([f['state'] == 'event', f['state'] == 'contact'], [tau['event'], tau['contact']], tau['flight'])
    return (f['residual'] > bound) | (f['lowest'] < -depth) | f['unmeasurable']


def tune(f):
    """Registered grid search on the development split: per-state tau and the depth minimizing the larger kill rate."""
    pos, neg = f['label'] == 1, f['label'] == 0
    grid = np.round(np.arange(.02, .505, .01), 3); depths = np.round(np.arange(.01, .0605, .005), 3)
    best = None
    tau = {'event': .5, 'contact': .5, 'flight': .5}
    for depth in depths:
        t = dict(tau)
        for _ in range(3):
            for state in ('contact', 'event', 'flight'):
                worst = []
                for g in grid:
                    p = residual_positive(f, {**t, state: g}, depth)
                    worst.append(max((p & neg).sum()/max(neg.sum(), 1), (~p & pos).sum()/max(pos.sum(), 1)))
                worst = np.array(worst)
                t[state] = float(grid[np.flatnonzero(worst == worst.min())[-1]])
        p = residual_positive(f, t, depth)
        fk = (p & neg).sum()/max(neg.sum(), 1); mk = (~p & pos).sum()/max(pos.sum(), 1)
        cand = (max(fk, mk), -depth, t, float(depth), fk, mk)
        if best is None or cand[:2] < best[:2]:
            best = cand
    return {'tau_bw': best[2], 'penetration_depth_m': best[3], 'dev_false_kill': best[4], 'dev_missed_kill': best[5]}


def ergo_frames(rows):
    from evalkit.validate import match
    out = {m: {'truth': [], 'tool': [], 'clip': []} for m in ('reba', 'rula')}
    for n, r in enumerate(rows):
        if r['meta']['defect'] in SKIP_KILL or r['truth_eval'] is None:
            continue
        te, tl = r['truth_eval'], r['tool']
        j = match(te['ergo_times'], tl['ergo_times'])
        ok = np.abs(tl['ergo_times'][j]-te['ergo_times']) <= .02
        for m in out:
            out[m]['truth'].append(te[f'ergo_{m}_class'][ok]); out[m]['tool'].append(tl[f'ergo_{m}_class'][j][ok])
            out[m]['clip'].append(np.full(ok.sum(), n))
    return {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in out.items()}


def score(args):
    from evalkit import validate
    from evalkit.ergo import CLASSES
    items = clips(args.split)
    rows = [r for r in (load_clip({**c, 'split': args.split}) for c in items) if r is not None]
    res = {'split': args.split, 'clips': len(rows), 'spec_version': SPEC['version']}
    f = residual_frames(rows)
    if args.split == 'dev':
        res['tuning'] = tune(f)
    tau, depth = (res['tuning']['tau_bw'], res['tuning']['penetration_depth_m']) if args.split == 'dev' else (SPEC['residual']['tau_bw'], SPEC['residual']['penetration_depth_m'])
    if tau:
        p = residual_positive(f, tau, depth)
        res['residual'] = validate.kill_rates(f['label'] == 1, f['label'] == 0, p, f['clip'])
        res['residual']['frames'] = {'positive': int((f['label'] == 1).sum()), 'negative': int((f['label'] == 0).sum()), 'ambiguous': int((f['label'] == -1).sum())}
        res['residual']['pass'] = validate.passes(res['residual'])
    ef = ergo_frames(rows)
    res['ergo'] = {}
    for m, d in ef.items():
        t, x = d['truth'], d['tool']
        r = validate.kill_rates(t == 1, t == 3, (x == 1) | (x == 0), d['clip'])
        r['missed_kill'] = validate.rate(((x == 3) & (t == 1)).sum(), (t == 1).sum())
        certain = (t == 1) | (t == 3)
        r['inconclusive_share'] = float(np.mean(x[certain] == 2)) if certain.any() else None
        r['confusion'] = {CLASSES[a]: {CLASSES[b]: int(((t == a) & (x == b)).sum()) for b in range(4)} for a in range(4)}
        r['pass'] = validate.passes(r)
        res['ergo'][m] = r
    det = {}
    for r in rows:
        if r['meta']['defect'] == 'bone_stretch':
            from evalkit.validate import match
            tr, tl = r['truth'], r['tool']; j = match(tr['times'], tl['times']); keep = tl['scored'][j]
            lab, um = tr['unmeasurable_label'][keep], tl['unmeasurable'][j][keep]
            d = det.setdefault(str(r['meta']['size']), [0, 0]); d[0] += int((um & (lab == 1)).sum()); d[1] += int((lab == 1).sum())
    other = [r['tool']['unmeasurable'][r['tool']['scored']] for r in rows if r['meta']['defect'] != 'bone_stretch']
    res['unmeasurable'] = {'detection': {k: validate.rate(*v) for k, v in det.items()},
                           'false_unmeasurable': validate.rate(int(sum(o.sum() for o in other)), int(sum(len(o) for o in other)))}
    mus = []
    for r in rows:
        if r['meta']['defect'] == 'control' and r['truth_muscle'] and 'muscle' in r['tool_summary']:
            a, b = r['tool_summary']['muscle']['muscle_effort_sum_a2_mean'], r['truth_muscle']['muscle_effort_sum_a2_mean']
            mus.append({'id': r['meta']['id'], 'tool': a, 'truth': b, 'relative_error': abs(a-b)/b})
    errs = np.array([m['relative_error'] for m in mus])
    res['muscle'] = {'clips': mus, 'median_relative_error': float(np.median(errs)) if len(errs) else None,
                     'share_within_10pct': float(np.mean(errs <= .1)) if len(errs) else None,
                     'pass': bool(len(errs) and np.median(errs) <= .1)}
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG/f'{args.split}_score.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'muscle'}, indent=1)[:6000])
    print(json.dumps({k: v for k, v in res['muscle'].items() if k != 'clips'}))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('step', choices=('calibrate', 'tool', 'truth', 'score'))
    a.add_argument('--split', choices=('dev', 'heldout'), default='dev'); a.add_argument('--workers', type=int, default=6)
    a.add_argument('--limit', type=int); a.add_argument('--only')
    args = a.parse_args()
    if args.step == 'calibrate':
        return calibrate(args)
    if args.step == 'score':
        return score(args)
    if args.split == 'heldout' and SPEC['version'] == 'v1-dev':
        raise SystemExit('freeze the registration (version v1) before the held-out split')
    items = [{**c, 'split': args.split} for c in clips(args.split) if not args.only or args.only in c['id']][:args.limit]
    print(json.dumps(pool(tool_job if args.step == 'tool' else truth_job, items, args.workers)))


if __name__ == '__main__':
    main()
