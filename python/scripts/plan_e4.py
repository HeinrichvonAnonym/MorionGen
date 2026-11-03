#!/usr/bin/env python
"""Register the E4 job list (config/e4_cross_method_20261103.json): existing experiment-1 outputs, read only."""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen')
SPEC = json.loads((ROOT/'config/e4_cross_method_20261103.json').read_text())
OUT = ROOT/'develop_logs/test_runs/20261103_e4_cross_method'
KIMODO = DATA/'benchmarks/kimodo-stratified-20261008'
SEEDS = 2


def key(condition):
    return '_'.join(f'{k}{v}' for k, v in sorted(condition.items()))


def source(method, run, condition):
    if method == 'primal':
        c = condition
        return ROOT/run/'factorial'/f'ground{c["ground_snap"]}_reproject{c["reprojection"]}_inertialize{c["inertialization"]}_canonical.npz'
    return ROOT/run


def matrix_jobs():
    rows = json.loads((ROOT/'develop_logs/test_runs/20261024_native_matrix_analysis/analysis.json').read_text())['rows']
    jobs = []
    for method, spec in SPEC['selection']['methods'].items():
        if method == 'kimodo':
            continue
        conditions = [key(c) for c in spec['conditions']]
        found = defaultdict(dict)
        for r in rows:
            if r['method'] != method or r['clock'] != 'paused':
                continue
            c = {k: r['condition'][k] for k in spec['conditions'][0]}
            if key(c) in conditions:
                found[r['task'], r['seed']][key(c)] = (c, r['run'])
        by_task = defaultdict(list)
        for (task, seed), cells in found.items():
            if len(cells) == len(conditions):
                by_task[task].append(seed)
        for task in sorted(by_task):
            for seed in sorted(by_task[task])[:SEEDS]:
                for name in conditions:
                    c, run = found[task, seed][name]
                    jobs.append({'id': f'{method}_{name}_t{task}_s{seed}', 'method': method, 'task': str(task), 'cluster': str(task),
                                 'seed': seed, 'condition': c, 'source': str(source(method, run, c)), 'run': run})
    return jobs


def kimodo_jobs():
    rows = json.loads((ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json').read_text())['rows']
    first = {}
    for r in sorted(rows, key=lambda r: r['case']):
        first.setdefault(r['family'], r)
    jobs = []
    for family, r in sorted(first.items()):
        cluster = family.split('/', 1)[1]
        for flag, folder in ((0, 'generated_raw'), (1, 'generated_postprocess')):
            c = {'correction': flag}
            jobs.append({'id': f'kimodo_{key(c)}_{family.replace("/", "-")}', 'method': 'kimodo', 'task': family, 'cluster': cluster,
                         'seed': r['seed'], 'condition': c, 'source': str(KIMODO/folder/r['case']), 'case': r['case']})
    return jobs


def main():
    jobs = matrix_jobs()+kimodo_jobs()
    missing = [j['id'] for j in jobs if not (Path(j['source']) if j['source'].endswith('.npz') else Path(j['source'])/'motion.npz').exists()]
    if missing:
        raise SystemExit(f'missing sources: {missing[:5]} ({len(missing)})')
    OUT.mkdir(parents=True, exist_ok=True)
    count = defaultdict(int)
    for j in jobs:
        count[j['method']] += 1
    (OUT/'plan.json').write_text(json.dumps({'protocol': SPEC['protocol'], 'counts': count, 'jobs': jobs}, indent=1))
    print(json.dumps(count), len(jobs))


if __name__ == '__main__':
    main()
