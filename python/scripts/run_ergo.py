"""Score RULA/REBA frames on finished E runs (config/ergo_rula_reba_20261105.json).

--set e4 | x3 | m1 reads the E metrics.json files of that experiment and writes one JSON per run to
develop_logs/test_runs/20261105_ergo/<set>/<id>.json; finished runs are skipped.
"""
import argparse
import json
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))


def json_write(path, data):
    tmp = Path(path).with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)

OUT = ROOT/'develop_logs/test_runs/20261105_ergo'
SETS = {'e4': ROOT/'develop_logs/test_runs/20261103_e4_cross_method/runs',
        'x3': ROOT/'develop_logs/test_runs/20261104_x3/e/runs',
        'm1': ROOT/'develop_logs/test_runs/20261105_ardy_fair/e/runs'}


def prompt(source):
    p = Path(source)
    d = p if p.is_dir() else p.parent.parent
    for name, get in (('meta.json', lambda c: c.get('text')), ('manifest.json', lambda c: (c.get('scenario') or {}).get('prompt')),
                      ('scenario.json', lambda c: c.get('prompt'))):
        f = d/name
        if f.exists():
            v = get(json.loads(f.read_text()))
            if v:
                return v
    return None


def object_free(set_name, metrics):
    import ergo_frames
    if set_name in ('x3', 'm1'):
        return True
    if metrics['method'] == 'chois':
        return False
    if metrics['method'] == 'kimodo':
        return ergo_frames.object_free(prompt(metrics['source']))
    return True


def job(args):
    set_name, path = args
    os.nice(19)
    import ergo_frames
    metrics = json.loads(Path(path).read_text())
    out = OUT/set_name/f"{metrics['id']}.json"
    if out.exists():
        return 'skip'
    row = {k: metrics.get(k) for k in ('id', 'method', 'task', 'seed', 'arm', 'condition', 'offset_m', 'source', 'work')}
    row['valid'] = bool(metrics.get('valid'))
    row['object_free'] = object_free(set_name, metrics)
    try:
        row.update(ergo_frames.score_run(metrics['work'], metrics['source'], row['object_free'],
                                         float(metrics.get('scaling', {}).get('mass_kg', 75.))))
        row['status'] = 'success'
    except Exception as e:
        row.update(status='failed', error=f'{type(e).__name__}: {e}', trace=traceback.format_exc()[-1500:])
    json_write(out, row)
    return row['status']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--set', required=True, choices=sorted(SETS))
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--only', nargs='*', help='run ids to score (smoke test)')
    args = p.parse_args()
    (OUT/args.set).mkdir(parents=True, exist_ok=True)
    files = sorted(SETS[args.set].glob('*/metrics.json'))
    if args.only:
        files = [f for f in files if f.parent.name in args.only]
    counts = {}
    with Pool(args.workers) as pool:
        for status in pool.imap_unordered(job, [(args.set, str(f)) for f in files]):
            counts[status] = counts.get(status, 0)+1
    print(json.dumps({'set': args.set, 'runs': len(files), **counts}))


if __name__ == '__main__':
    main()
