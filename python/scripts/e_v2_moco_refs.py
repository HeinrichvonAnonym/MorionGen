"""E v2 dynamic references: 24 kin30 clips (walk, squat, sts, wholebody x 6) tracked with Moco, split by subject.

--select writes config/e_v2_moco_refs_20261106.json (seed 20261106; subjects are split globally, so no subject is in both
calibration and test). --run solves them with the smoke-tested configuration, at most 3 at a time, 2 h each; a clip that
times out is retried once at 1.5 s and otherwise recorded as failed, never replaced.
"""
import argparse, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'python/scripts'))
from e_domain_moco import KIN30, VAL, clip  # noqa: E402

CONFIG = ROOT/'config/e_v2_moco_refs_20261106.json'
OUT_REL = 'develop_logs/test_runs/20261106_e_v2/moco_refs'
DISK = '/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/e_v2_20261106/moco_refs'
CLASSES = ('walk', 'squat', 'sts', 'wholebody')
PER_SPLIT = 3
SEED = 20261106
MIN_SECONDS = 1.5
TIMEOUT_S = 7200
RETRY_SECONDS = 1.5


def select():
    rows = [r for r in json.loads((VAL/'selection.json').read_text())['trials'] if r['class'] in CLASSES]
    rows = [r for r in rows if (KIN30/r['id']/'coordinates.mot').exists()]
    for r in rows:
        t0, t1 = clip(r['id'], 2.)
        r['clip'] = (t0, t1)
    rows = [r for r in rows if r['clip'][1]-r['clip'][0] >= MIN_SECONDS-1e-9]
    rng = np.random.default_rng(SEED)
    subjects = sorted({r['subject'] for r in rows})
    for attempt in range(1, 1001):
        order = list(rng.permutation(subjects)); half = len(order)//2
        side = {s: ('calibration' if i < half else 'test') for i, s in enumerate(order)}
        pools = {(c, sp): sorted({r['subject'] for r in rows if r['class'] == c and side[r['subject']] == sp})
                 for c in CLASSES for sp in ('calibration', 'test')}
        if all(len(v) >= PER_SPLIT for v in pools.values()):
            break
    else:
        raise SystemExit('no subject split gives 3 subjects per class and split')
    chosen = []
    for (c, sp), subs in pools.items():
        for s in rng.choice(subs, PER_SPLIT, replace=False):
            cands = sorted((r for r in rows if r['class'] == c and r['subject'] == s), key=lambda r: r['id'])
            r = cands[int(rng.integers(len(cands)))]
            chosen.append({'trial': r['id'], 'class': c, 'split': sp, 'subject': s, 'study': r['study'], 'task': r['task'],
                           'mass_kg': r['mass_kg'], 'trial_seconds': r['seconds'], 'clip_s': [round(x, 4) for x in r['clip']]})
    spec = {'created': time.strftime('%Y-%m-%d %H:%M'), 'seed': SEED, 'split_attempts': attempt,
            'rule': 'eligible: kin30 trials of the four classes whose centred clip min(2 s, duration - 1 s) is >= 1.5 s; '
                    'subjects shuffled and halved (redrawn until each class has >= 3 eligible subjects per half); per class '
                    'and half 3 subjects drawn, one trial per subject drawn',
            'solve': {'continuation': True, 'mesh_interval_s': .02, 'max_parallel': 3, 'timeout_s': TIMEOUT_S,
                      'retry_on_timeout_seconds': RETRY_SECONDS, 'failed_entries_are_not_replaced': True},
            'subjects': {sp: sorted(s for s, v in side.items() if v == sp) for sp in ('calibration', 'test')},
            'refs': sorted(chosen, key=lambda x: (x['split'], x['class'], x['trial']))}
    assert not set(spec['subjects']['calibration']) & set(spec['subjects']['test'])
    CONFIG.write_text(json.dumps(spec, indent=1))
    print(json.dumps({k: spec[k] for k in ('seed', 'split_attempts')}),
          *(f"{r['split']:11s} {r['class']:9s} {r['trial']} {r['clip_s'][1]-r['clip_s'][0]:.2f}s" for r in spec['refs']), sep='\n')


def one(ref, args):
    log_dir = ROOT/OUT_REL/'logs'; log_dir.mkdir(parents=True, exist_ok=True)
    rec = {'trial': ref['trial'], 'attempts': []}
    for seconds, tag in ((2., None), (RETRY_SECONDS, 'short')):
        cmd = [sys.executable, str(ROOT/'python/scripts/e_domain_moco.py'), '--trial', ref['trial'], '--solve', '--evaluate',
               '--continuation', '--mesh', str(args.mesh), '--threads', str(args.threads), '--seconds', str(seconds),
               '--out', OUT_REL, '--disk', DISK] + (['--implicit'] if args.implicit else []) + (['--tag', tag] if tag else [])
        log = log_dir/f"{ref['trial']}{'__'+tag if tag else ''}.log"
        start = time.time()
        with open(log, 'w') as fh:
            try:
                code = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=TIMEOUT_S, cwd=ROOT).returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        out = ROOT/OUT_REL/(ref['trial']+('__'+tag if tag else ''))
        solve = json.loads((out/'solve.json').read_text()) if (out/'solve.json').exists() else None
        rec['attempts'].append({'seconds': seconds, 'exit': code, 'wall_s': round(time.time()-start),
                                'success': bool(solve and solve['success']), 'evaluated': (out/'evaluation.json').exists()})
        if code != 'timeout':
            break
    last = rec['attempts'][-1]
    rec['status'] = 'ok' if last['success'] and last['evaluated'] else 'failed'
    rec['dir'] = str(out.relative_to(ROOT))
    (ROOT/OUT_REL/'status').mkdir(parents=True, exist_ok=True)
    (ROOT/OUT_REL/'status'/f"{ref['trial']}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)
    return rec


def run(args):
    spec = json.loads(CONFIG.read_text())
    refs = [r for r in spec['refs'] if not args.only or r['trial'] in args.only]
    refs = [r for r in refs if not (ROOT/OUT_REL/'status'/f"{r['trial']}.json").exists()]
    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(lambda r: one(r, args), refs))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--select', action='store_true'); a.add_argument('--run', action='store_true')
    a.add_argument('--parallel', type=int, default=3); a.add_argument('--threads', type=int, default=4)
    a.add_argument('--mesh', type=float, default=.02); a.add_argument('--implicit', action='store_true')
    a.add_argument('--only', nargs='*')
    args = a.parse_args()
    if args.select:
        select()
    if args.run:
        run(args)


if __name__ == '__main__':
    main()
