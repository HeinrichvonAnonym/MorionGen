#!/usr/bin/env python
"""evalkit validation on the synthetic set (config/evalkit_v11_20261107.json, validation). OpenSim python.

  calibrate            synthetic ergo neutral (skeleton definition) and floor reference (development stand controls);
                       writes the calibrated fields of the registration (refused once the version is frozen)
  tool --split S       evalkit on every clip of the split (muscle on control clips only)
  truth --split S      the same ergo scoring and static optimization on the truth (truth model, coordinates, contact)
  score --split S      kill rates; on dev also the tau / depth tuning grid (writes nothing into the registration)
  kinematic --split S  only the kinematic check of score (injected size against the kinematic metrics)
  videos --split S     the registered 10 + 10 acceptance draw (validation.videos), static optimization where the run
                       had none, evalkit_render videos and an index page
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
MARGINS = tuple(float(x) for x in np.round(np.arange(.005, .0305, .005), 3))
MIN_PASS_SIZE_M = .05


def residual_pass(m):
    """Clips of the residual pass line (user decision 11:14): float / penetrate below 5 cm are reported separately."""
    return m['defect'] not in SKIP_KILL and not (m['defect'].startswith(('float', 'penetrate')) and m['size'] < MIN_PASS_SIZE_M)


def ergo_pass(m):
    """Clips of the ergonomics pass line (user decision 11:14): whole-clip float has no truth contact."""
    return m['defect'] not in SKIP_KILL and m['defect'] != 'float_whole_m'


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
    model = osim.Model(str(Path(base_dir)/'model.osim')); model.initSystem(); cs = model.getCoordinateSet()
    labels = synth.coordinate_names(model)
    d = np.array([cs.get(n).getDefaultValue()*(180/np.pi if cs.get(n).getMotionType() == 1 else 1.) for n in labels])
    ex = synth.export(Path(base_dir)/'model.osim', {'labels': labels, 'coords': np.repeat(d[None], 2, 0), 'fps': synth.FPS,
                                                     'floor': 0., 'stretch': None}, info, seed=0)
    return ergo.rest_reading(ex['clean_joints'][0], ex['joint_names'])


CALIBRATION_DEFECTS = ('control', 'lean_deg', 'time_compression', 'bone_stretch', 'slide_m_s')


def calib_job(clip):
    """Fit and residual with zero reference offsets (the elevations are then landmark height above the input floor);
    the fit is written where the tool run reuses it."""
    _quiet()
    from evalkit import run
    from evalkit.validate import match
    m = meta(clip); out = DATA/'tool'/'dev'/clip['id']
    zero = {s: {k: 0. for k in LANDMARKS} for s in SIDES}
    run.evaluate(Path(clip['dir'])/'input', out, m['floor_height_m'], zero, m['mass_kg'], dims=('residual',))
    f = np.load(out/'frames.npz'); tr = np.load(Path(clip['dir'])/'truth/truth.npz')
    j = match(f['times'], tr['times'])
    heights = {}
    for s in SIDES:
        both = (tr[f'contact_{s}_heel'] & tr[f'contact_{s}_toe'])[j] & f['scored']
        heights[s] = {k: f[f'elevation_{s}_{k}'][both].tolist() for k in LANDMARKS}
    return heights


def calibrate(args):
    if not SPEC['version'].endswith('-dev'):
        raise SystemExit('registration is frozen')
    _quiet()
    items = [c for c in clips('dev') if meta(c)['defect'] in CALIBRATION_DEFECTS]
    stand = [c for c in items if c['id'].endswith('stand__control')]
    readings = [neutral_reading(c['base_dir'], c['info']) for c in stand]
    neck = np.mean([r[0] for r in readings], 0); shoulder = np.mean([r[1] for r in readings], 0)
    heights = {s: {k: [] for k in LANDMARKS} for s in SIDES}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        for h in p.map(calib_job, items):
            for s in SIDES:
                for k in LANDMARKS:
                    heights[s][k] += h[s][k]
    ref = {s: {k: float(np.median(heights[s][k])) for k in LANDMARKS} for s in SIDES}
    spec = json.loads(SPEC_PATH.read_text())
    spec['floor']['calibrated']['synthetic'] = ref
    spec['ergo']['neutral_calibrated']['synthetic'] = {'neck_deg': neck.tolist(), 'shoulder_elevation_deg': shoulder.tolist()}
    spec['calibration_receipt'] = {'clips': len(items), 'defects': list(CALIBRATION_DEFECTS),
                                   'reference_frames': {s: len(heights[s]['heel']) for s in SIDES},
                                   'reference_iqr_m': {s: {k: np.percentile(heights[s][k], [25, 75]).tolist() for k in LANDMARKS} for s in SIDES},
                                   'neutral_subjects': [c['info']['seed'] for c in stand]}
    SPEC_PATH.write_text(json.dumps(spec, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'reference': ref, 'neutral_neck_deg': neck.tolist(), 'neutral_shoulder_deg': shoulder.tolist()}, indent=1))


# ---------------------------------------------------------------- tool and truth runs

def offset_job(clip):
    """Fit and foot offset of one clip (floor.per_clip_correction), cached as foot_offset.json."""
    out = DATA/'tool'/clip['split']/clip['id']
    from evalkit.fit import PREPARE
    path = out/'foot_offset.json'
    if path.exists() and json.loads(path.read_text()).get('fit_prepare') == PREPARE:
        return 'skip'
    try:
        _quiet()
        from evalkit import run
        m = meta(clip)
        d = run.foot_offset(Path(clip['dir'])/'input', out/'work', m['mass_kg'], m['floor_height_m'])[0]
        path.write_text(json.dumps({'foot_offset_m': d, 'fit_prepare': PREPARE}))
        return 'ok'
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        (out/'error_offset.json').write_text(json.dumps({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-2000:]}))
        return 'failed'


def offset_reference(items):
    """D of floor.per_clip_correction for a split: median foot offset over its clips (method 'synthetic')."""
    d = [json.loads((DATA/'tool'/c['split']/c['id']/'foot_offset.json').read_text())['foot_offset_m']['mean'] for c in items]
    return float(np.median(d))


def tool_job(task):
    clip, ref = task
    out = DATA/'tool'/clip['split']/clip['id']
    from evalkit.fit import PREPARE
    if (out/'evalkit.json').exists():
        r = json.loads((out/'evalkit.json').read_text())['receipt']
        if (r['reference'] == 'synthetic' and r.get('fit_prepare') == PREPARE and r.get('spec_version') == SPEC['version']
                and r.get('contact_margin_m') == SPEC['residual']['contact_margin_m'] and r.get('offset_reference_m') == ref):
            return 'skip'
    try:
        _quiet()
        from evalkit import run
        m = meta(clip)
        dims = ('residual', 'ergo', 'muscle') if m['defect'] == 'control' else ('residual', 'ergo')
        run.evaluate(Path(clip['dir'])/'input', out, m['floor_height_m'], 'synthetic', m['mass_kg'], method='synthetic', dims=dims,
                     offset_reference=ref)
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


def margins_job(clip):
    """Residual of one development clip at every contact margin of the tuning grid (the fit is shared)."""
    out = DATA/'tool'/clip['split']/clip['id']
    from evalkit.fit import PREPARE
    if (out/'margins.npz').exists() and str(np.load(out/'margins.npz').get('prepare', '')) == PREPARE:
        return 'skip'
    try:
        _quiet()
        import opensim as osim
        import biomech
        from evalkit import fit, residual
        from evalkit.spec import absolute_reference, reference_offsets
        m = meta(clip)
        f = fit.fit(Path(clip['dir'])/'input', out/'work', m['mass_kg'], m['floor_height_m'])
        kin = biomech.body_kinematics(osim.Model(str(out/'work/model.osim')), f['labels'], f['coords'])
        ref = absolute_reference(reference_offsets('synthetic'), m['floor_height_m'])
        arrays = {'times': f['times'], 'scored': fit.scored(f['times']), 'unmeasurable': f['unmeasurable'], 'prepare': PREPARE}
        for g in MARGINS:
            res = residual.evaluate(kin, f['times'], m['mass_kg'], m['floor_height_m'], ref, margin=g)
            arrays.update({f'{g}_residual_bw': res['residual_bw'], f'{g}_lowest': res['lowest_elevation'], f'{g}_state': res['state']})
        np.savez_compressed(out/'margins.npz', **arrays)
        return 'ok'
    except Exception as e:
        out.mkdir(parents=True, exist_ok=True)
        (out/'error_margins.json').write_text(json.dumps({'error': f'{type(e).__name__}: {e}', 'trace': traceback.format_exc()[-2000:]}))
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
    if not (tool/'frames.npz').exists() and not (tool/'margins.npz').exists():
        return None
    return {'meta': m, 'dir': str(d), 'truth': dict(np.load(d/'truth/truth.npz')),
            'tool': dict(np.load(tool/'frames.npz', allow_pickle=True)) if (tool/'frames.npz').exists() else None,
            'margins': dict(np.load(tool/'margins.npz', allow_pickle=True)) if (tool/'margins.npz').exists() else None,
            'truth_eval': dict(np.load(truth/'truth_eval.npz', allow_pickle=True)) if (truth/'truth_eval.npz').exists() else None,
            'truth_muscle': json.loads((truth/'muscle.json').read_text()) if (truth/'muscle.json').exists() else None,
            'tool_summary': json.loads((tool/'evalkit.json').read_text()) if (tool/'evalkit.json').exists() else None}


def residual_frames(rows, include=residual_pass, margin=None):
    """Concatenated per-frame arrays over the included clips: truth label, tool residual, lowest elevation, state,
    unmeasurable, clip index; margin: from the margin grid (margins step) instead of the tool run."""
    from evalkit.validate import match
    out = {k: [] for k in ('label', 'residual', 'lowest', 'state', 'unmeasurable', 'clip')}
    for n, r in enumerate(rows):
        if not include(r['meta']):
            continue
        tr = r['truth']
        if margin is None:
            tl = r['tool']; tl = {**tl, 'lowest': tl['lowest_elevation_m']}
        else:
            g = r['margins']; tl = {k: g[k] for k in ('times', 'scored', 'unmeasurable')}
            tl.update(residual_bw=g[f'{margin}_residual_bw'], lowest=g[f'{margin}_lowest'], state=g[f'{margin}_state'])
        j = match(tr['times'], tl['times']); keep = tl['scored'][j]
        out['label'].append(tr['e_label'][keep]); out['residual'].append(tl['residual_bw'][j][keep])
        out['lowest'].append(tl['lowest'][j][keep]); out['state'].append(tl['state'][j][keep])
        out['unmeasurable'].append(tl['unmeasurable'][j][keep]); out['clip'].append(np.full(keep.sum(), n))
    return {k: np.concatenate(v) for k, v in out.items()}


def residual_positive(f, tau, depth):
    bound = np.select([f['state'] == 'event', f['state'] == 'contact'], [tau['event'], tau['contact']], tau['flight'])
    return (f['residual'] > bound) | (f['lowest'] < -depth) | f['unmeasurable']


def tune_margins(rows):
    """Registered grid search on the development split (amendment 11:48): contact margin jointly with tau and depth,
    minimizing the larger kill rate over the pass clips; ties to the larger margin."""
    best = None
    for g in MARGINS:
        t = tune(residual_frames(rows, margin=g))
        cand = (max(t['dev_false_kill'], t['dev_missed_kill']), -g)
        print(json.dumps({'margin': g, **t}), flush=True)
        if best is None or cand < best[0]:
            best = (cand, {'contact_margin_m': g, **t})
    return best[1]


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
        if not ergo_pass(r['meta']) or r['truth_eval'] is None or r['tool'] is None:
            continue
        te, tl = r['truth_eval'], r['tool']
        j = match(te['ergo_times'], tl['ergo_times'])
        ok = np.abs(tl['ergo_times'][j]-te['ergo_times']) <= .02
        for m in out:
            out[m]['truth'].append(te[f'ergo_{m}_class'][ok]); out[m]['tool'].append(tl[f'ergo_{m}_class'][j][ok])
            out[m]['clip'].append(np.full(ok.sum(), n))
    return {m: {k: np.concatenate(v) for k, v in d.items()} for m, d in out.items()}


def muscle_summary(mus, high_load=2.):
    """Descriptive muscle agreement (no pass line, spec muscle.criterion): relative and absolute error overall and by
    truth load (effort >= high_load, 约定，未经推演), Spearman, and the share of clip pairs whose order the tool keeps."""
    from itertools import combinations
    from scipy.stats import spearmanr
    from evalkit.validate import rate
    if not mus:
        return {'clips': mus, 'criterion': SPEC['muscle'].get('criterion')}
    t, g = np.array([m['tool'] for m in mus]), np.array([m['truth'] for m in mus])
    rel, err = np.abs(t-g)/g, np.abs(t-g)
    out = {'clips': mus, 'criterion': SPEC['muscle'].get('criterion'), 'high_load_effort': high_load}
    for name, k in (('all', np.ones(len(g), bool)), ('high_load', g >= high_load), ('low_load', g < high_load)):
        out[name] = {'clips': int(k.sum()),
                     'median_relative_error': float(np.median(rel[k])) if k.any() else None,
                     'median_absolute_error': float(np.median(err[k])) if k.any() else None,
                     'mean_signed_relative_error': float(np.mean((t-g)[k]/g[k])) if k.any() else None,
                     'share_within_10pct': float(np.mean(rel[k] <= .1)) if k.any() else None}
    out['spearman'] = float(spearmanr(t, g)[0]) if len(mus) > 2 else None
    out['pair_order'] = {}
    for ratio in (1.1, 1.2, 1.5):
        pairs = [(i, j) for i, j in combinations(range(len(g)), 2) if max(g[i], g[j])/min(g[i], g[j]) >= ratio]
        kept = sum(np.sign(t[i]-t[j]) == np.sign(g[i]-g[j]) for i, j in pairs)
        out['pair_order'][str(ratio)] = rate(int(kept), len(pairs))
    out['median_relative_error'] = out['all']['median_relative_error']
    out['pass'] = None if SPEC['muscle'].get('criterion') == 'descriptive' else bool(out['median_relative_error'] <= .1)
    return out


def score(args):
    from evalkit import validate
    from evalkit.ergo import CLASSES
    items = clips(args.split)
    rows = [r for r in (load_clip({**c, 'split': args.split}) for c in items) if r is not None]
    res = {'split': args.split, 'clips': len(rows), 'spec_version': SPEC['version']}
    if args.split == 'dev' and args.tune:
        res['tuning'] = tune_margins(rows)
        margin, tau, depth = res['tuning']['contact_margin_m'], res['tuning']['tau_bw'], res['tuning']['penetration_depth_m']
    else:
        margin, tau, depth = None, SPEC['residual']['tau_bw'], SPEC['residual']['penetration_depth_m']
    if tau:
        res['residual'] = {}
        below = lambda m: m['defect'].startswith(('float', 'penetrate')) and m['size'] < MIN_PASS_SIZE_M
        for name, include in (('pass_clips', residual_pass), ('below_resolution_2cm', below),
                              ('all_kill_clips', lambda m: m['defect'] not in SKIP_KILL)):
            f = residual_frames(rows, include, margin)
            p = residual_positive(f, tau, depth)
            r = validate.kill_rates(f['label'] == 1, f['label'] == 0, p, f['clip'])
            r['frames'] = {'positive': int((f['label'] == 1).sum()), 'negative': int((f['label'] == 0).sum()), 'ambiguous': int((f['label'] == -1).sum())}
            res['residual'][name] = r
        res['residual']['pass'] = validate.passes(res['residual']['pass_clips'])
    if args.tune:  # tool frames of the other dimensions may predate the tuned fit; score them after the tool run
        LOG.mkdir(parents=True, exist_ok=True)
        (LOG/f'{args.split}_tune.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1)[:6000])
        return
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
    rows = [r for r in rows if r['tool'] is not None]
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
        if r['meta']['defect'] == 'control' and r['truth_muscle'] and 'muscle' in (r['tool_summary'] or {}):
            a, b = r['tool_summary']['muscle']['muscle_effort_sum_a2_mean'], r['truth_muscle']['muscle_effort_sum_a2_mean']
            mus.append({'id': r['meta']['id'], 'tool': a, 'truth': b, 'signed_error': a-b, 'relative_error': abs(a-b)/b})
    res['muscle'] = muscle_summary(mus)
    res['kinematic'] = kinematic_check(rows)
    LOG.mkdir(parents=True, exist_ok=True)
    (LOG/f'{args.split}_score.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ('muscle', 'kinematic')}, indent=1)[:6000])
    print(json.dumps({k: v for k, v in res['muscle'].items() if k != 'clips'}))
    print(json.dumps({k: v for k, v in res['kinematic'].items() if k != 'clips'}, indent=1))


def kinematic_check(rows):
    """Kinematic metrics against the injected size: slide (slip over the injected displacement, size m/s for one third
    of the clip), bone_stretch (right-arm stretch), whole-clip penetration / hover depth, time compression (jerk over
    the control of the same base, expected factor**3)."""
    from evalkit import kinematic
    per = {}
    for r in rows:
        m = r['meta']; mo = np.load(Path(r['dir'])/'input/motion.npz')
        k = kinematic.evaluate(mo['joints'], mo['joint_names'], float(mo['fps']), r['tool'])
        tr, cj = r['truth'], np.load(Path(r['dir'])/'truth/clean_joints.npz')
        truth_frames = {'times': tr['times'], 'scored': np.ones(len(tr['times']), bool),
                        **{f'contact_{s}_{l}': tr[f'contact_{s}_{l}'] for s in SIDES for l in LANDMARKS}}
        truth_slip = kinematic.skating(cj['joints'], cj['joint_names'], float(cj['fps']), truth_frames)['max_slip_m']
        per[m['id']] = {'base': m['id'].split('__')[0], 'defect': m['defect'], 'size': m['size'],
                        'duration_s': (len(mo['joints'])-1)/float(mo['fps']), 'max_slip_m': k['skating']['max_slip_m'],
                        'truth_max_slip_m': truth_slip,
                        'bone_stretch_max': k['bone_stretch_max'], 'jerk_0p1s': k['jerk_0p1s'], **k['floor']}
    control = {v['base']: v for v in per.values() if v['defect'] == 'control'}
    ctrl_slip = [v['max_slip_m'] for v in control.values() if v['max_slip_m'] is not None]
    ctrl_stretch = [v['bone_stretch_max'] for v in control.values()]
    out = {'control': {'max_slip_m': {'max': max(ctrl_slip), 'median': float(np.median(ctrl_slip))},
                       'bone_stretch_max': {'max': max(ctrl_stretch), 'median': float(np.median(ctrl_stretch))}}}

    def by_size(defect, fn):
        sizes = sorted({v['size'] for v in per.values() if v['defect'] == defect})
        return {str(s): fn([v for v in per.values() if v['defect'] == defect and v['size'] == s], s) for s in sizes}

    def slide(vs, s):
        slip = np.array([v['max_slip_m'] for v in vs if v['max_slip_m'] is not None]); injected = s*vs[0]['duration_s']/3
        return {'clips': len(vs), 'injected_m': injected, 'median_slip_m': float(np.median(slip)), 'min_slip_m': float(slip.min()),
                'above_control_max': float(np.mean(slip > max(ctrl_slip))),
                'auc_vs_controls': float(np.mean([(a > b)+.5*(a == b) for a in slip for b in ctrl_slip]))}
    out['slide'] = by_size('slide_m_s', slide)
    paired = [v for v in per.values() if v['defect'] in ('control', 'slide_m_s') and v['max_slip_m'] is not None
              and v['truth_max_slip_m'] is not None]
    groups = {'walk_control': lambda v: v['defect'] == 'control' and 'demo_walk' in v['base'],
              'constructed_control': lambda v: v['defect'] == 'control' and 'demo_walk' not in v['base'],
              **{f'slide_{s}': (lambda v, s=s: v['defect'] == 'slide_m_s' and v['size'] == s) for s in (.1, .3, .6)}}
    out['slip_vs_truth'] = {g: {'clips': len(vs), 'truth_median_m': float(np.median([v['truth_max_slip_m'] for v in vs])),
                                'tool_minus_truth_median_m': float(np.median([v['max_slip_m']-v['truth_max_slip_m'] for v in vs])),
                                'abs_error_max_m': float(max(abs(v['max_slip_m']-v['truth_max_slip_m']) for v in vs))}
                            for g, f in groups.items() for vs in [[v for v in paired if f(v)]] if vs}
    out['bone_stretch'] = by_size('bone_stretch', lambda vs, s: {
        'clips': len(vs), 'median_estimate': float(np.median([v['bone_stretch_max'] for v in vs])),
        'above_control_max': float(np.mean([v['bone_stretch_max'] > max(ctrl_stretch) for v in vs]))})
    for defect, key in (('penetrate_whole_m', 'penetration_depth_p95_m'), ('float_whole_m', 'hover_height_median_m')):
        out[defect] = by_size(defect, lambda vs, s, key=key: {
            'clips': len(vs), 'median_estimate_m': float(np.median([v[key] for v in vs])),
            'control_median_m': float(np.median([control[v['base']][key] for v in vs if v['base'] in control]))})
    out['time_compression'] = by_size('time_compression', lambda vs, s: {
        'clips': len(vs), 'expected_ratio': s**3,
        'median_jerk_ratio': float(np.median([v['jerk_0p1s']/control[v['base']]['jerk_0p1s'] for v in vs if v['base'] in control]))})
    out['clips'] = per
    return out


# ---------------------------------------------------------------- acceptance videos

DIM_NAME = {'residual': '残余力', 'unmeasurable': '不可测', 'reba': 'REBA', 'rula': 'RULA'}


def video_draw(split):
    """The registered draw (validation.videos): 10 positive clips round-robin over defect types, 10 negative clips
    round-robin over base motions, numpy default_rng(crc32('evalkit_v1_videos')); written once, never redrawn."""
    import zlib
    from evalkit import fit
    path = LOG/f'{split}_videos.json'
    if path.exists():
        return json.loads(path.read_text())
    pos, neg = {}, {}
    for c in clips(split):
        m = meta(c)
        if m['defect'] == 'slide_m_s':
            continue
        tr = np.load(Path(c['dir'])/'truth/truth.npz'); sel = fit.scored(tr['times'])
        te = np.load(DATA/'truth_eval'/split/c['id']/'truth_eval.npz', allow_pickle=True)
        why = [k for k, hit in (('residual', (tr['e_label'][sel] == 1).any()), ('unmeasurable', (tr['unmeasurable_label'][sel] == 1).any()),
                                ('reba', (te['ergo_reba_class'] == 1).any()), ('rula', (te['ergo_rula_class'] == 1).any())) if hit]
        rec = {'id': c['id'], 'defect': m['defect'], 'size': m['size'], 'base': Path(c['base_dir']).name, 'truth_positive': why}
        (pos.setdefault(m['defect'], []) if why else neg.setdefault(rec['base'], [])).append(rec)
    rng = np.random.default_rng(zlib.crc32(b'evalkit_v1_videos'))

    def round_robin(groups, n):
        queues = {k: [groups[k][i] for i in rng.permutation(len(groups[k]))] for k in sorted(groups)}
        out = []
        while len(out) < n and any(queues.values()):
            for k in sorted(queues):
                if queues[k] and len(out) < n:
                    out.append(queues[k].pop(0))
        return out
    draw = {'split': split, 'rule': SPEC['validation']['videos'], 'pool': {'positive': sum(map(len, pos.values())),
                                                                          'negative': sum(map(len, neg.values()))},
            'positive': round_robin(pos, 10), 'negative': round_robin(neg, 10)}
    path.write_text(json.dumps(draw, indent=1, ensure_ascii=False))
    return draw


def video_job(task):
    """Static optimization on the tool and truth side of one drawn clip (if its run had none), then the render."""
    clip, kind, out = task
    try:
        _quiet()
        import subprocess
        from evalkit import fit, muscle
        m = meta(clip); tool, truth = DATA/'tool'/clip['split']/clip['id'], DATA/'truth_eval'/clip['split']/clip['id']
        t = np.load(Path(clip['dir'])/'truth/truth.npz')['times']; sel = fit.scored(t)
        for work, dest, links in ((tool/'work', tool, ()), (truth, truth, ('coordinates.mot', 'grf.xml', 'grf.mot'))):
            if m['defect'] == 'control' or (dest/'video_muscle.npz').exists():
                continue
            for name in links:
                if not (work/name).exists():
                    (work/name).symlink_to(Path(clip['dir'])/'truth'/name)
            mu = muscle.evaluate(work, float(t[sel][0]), float(t[sel][-1]))
            np.savez_compressed(dest/'video_muscle.npz', muscle_times=mu['times'], muscle_effort=mu['effort'],
                                muscle_reserve_abs_sum_nm=mu['reserve_abs_sum_nm'])
        title = (f"{'阳性' if kind == 'positive' else '阴性'}：{clip['id']}（缺陷 {m['defect']} {m['size']}；"
                 f"真值阳性维度 {'、'.join(DIM_NAME[d] for d in clip['truth_positive']) or '无'}）")
        r = subprocess.run([sys.executable, str(ROOT/'python/scripts/evalkit_render.py'), '--clip', clip['dir'], '--tool', str(tool),
                            '--truth', str(truth), '--out', str(out), '--title', title], capture_output=True, text=True)
        return clip['id'], 'ok' if r.returncode == 0 else f'render failed: {r.stderr[-800:]}'
    except Exception as e:
        return clip['id'], f'{type(e).__name__}: {e} {traceback.format_exc()[-800:]}'


def videos(args):
    draw = video_draw(args.split)
    dirs = {c['id']: c for c in clips(args.split)}
    out = LOG/f'{args.split}_videos'
    tasks = [({**dirs[r['id']], 'split': args.split, 'truth_positive': r['truth_positive']}, kind, out/f"{kind}_{r['id']}.mp4")
             for kind in ('positive', 'negative') for r in draw[kind]]
    status = {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), max_tasks_per_child=1) as p:
        for cid, s in p.map(video_job, tasks):
            status[cid] = s; print(json.dumps({cid: s}, ensure_ascii=False), flush=True)
    score_path = LOG/f'{args.split}_score.json'
    rows = []
    for kind in ('positive', 'negative'):
        for r in draw[kind]:
            v = f"{kind}_{r['id']}"
            rows.append(f"<tr><td>{'阳性' if kind == 'positive' else '阴性'}</td><td>{r['id']}</td><td>{r['defect']} {r['size']}</td>"
                        f"<td>{'、'.join(DIM_NAME[d] for d in r['truth_positive']) or '无'}</td><td><a href='{v}.mp4'><img src='{v}.png' width='480'></a></td></tr>")
    (out/'index.html').write_text(
        "<html><head><meta charset='utf-8'><title>evalkit 验收视频</title></head><body>"
        f"<h2>evalkit v1 验收视频（{args.split}，登记抽样 10 阳 + 10 阴）</h2>"
        f"<p>抽样规则：{SPEC['validation']['videos']['draw']}。池：阳性 {draw['pool']['positive']} 条，阴性 {draw['pool']['negative']} 条。"
        f"{'汇总见 ' + score_path.name if score_path.exists() else ''}</p>"
        "<table border='1' cellpadding='4'><tr><th>类</th><th>片段</th><th>缺陷</th><th>真值阳性维度</th><th>视频（点击）</th></tr>"
        + ''.join(rows) + "</table></body></html>\n")
    print(json.dumps({'status': status, 'index': str(out/'index.html')}, ensure_ascii=False, indent=1))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('step', choices=('calibrate', 'margins', 'tool', 'truth', 'score', 'kinematic', 'videos'))
    a.add_argument('--tune', action='store_true', help='score on dev: run the registered margin / tau / depth grid')
    a.add_argument('--split', choices=('dev', 'heldout', 'heldout2'), default='dev'); a.add_argument('--workers', type=int, default=6)
    a.add_argument('--limit', type=int); a.add_argument('--only')
    args = a.parse_args()
    if args.step == 'calibrate':
        return calibrate(args)
    if args.step == 'score':
        return score(args)
    if args.step == 'videos':
        return videos(args)
    if args.step == 'kinematic':
        rows = [r for r in (load_clip({**c, 'split': args.split}) for c in clips(args.split)) if r is not None and r['tool'] is not None]
        res = kinematic_check(rows)
        (LOG/f'{args.split}_kinematic.json').write_text(json.dumps(res, indent=1))
        return print(json.dumps({k: v for k, v in res.items() if k != 'clips'}, indent=1))
    if args.split.startswith('heldout') and SPEC['version'].endswith('-dev'):
        raise SystemExit(f"freeze the registration ({SPEC['version']}) before a held-out split")
    if args.split == 'heldout' and SPEC['version'] != 'v1':
        raise SystemExit('the held-out split of v1 is spent; later versions validate on their own split')
    items = [{**c, 'split': args.split} for c in clips(args.split) if not args.only or args.only in c['id']][:args.limit]
    if args.step == 'tool':
        print(json.dumps(pool(offset_job, items, args.workers)))
        ref = offset_reference(items)
        print(json.dumps({'offset_reference_m': ref}), flush=True)
        return print(json.dumps(pool(tool_job, [(c, ref) for c in items], args.workers)))
    print(json.dumps(pool({'truth': truth_job, 'margins': margins_job}[args.step], items, args.workers)))


if __name__ == '__main__':
    main()
