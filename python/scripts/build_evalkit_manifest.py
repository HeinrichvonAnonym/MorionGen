#!/usr/bin/env python
"""evalkit manifests of the large batch: generated runs and linked earlier runs (python/scripts/build_batch_plan.py) plus the
416 reused Kimodo pairs (split deck_recheck; develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json, as E4).

  build_evalkit_manifest.py --batch DIR
Reads DIR/plans/plan_*.json, DIR/plans/links.json and DIR/queue*/state.json (the successful attempt of each job). Writes
DIR/evalkit/manifest_development.json (development rows and the Kimodo pairs) and DIR/evalkit/manifest_held_out.json:
held-out rows are evaluated apart and never summarised (analysis plan split_use; evalkit_batch.py check and the run
median diagnostic summarise every row of their manifest).

Row fields (evaluation protocol rows.manifest_fields): id, method, source, condition, split, task_id, seed, paraphrase_id,
text_condition, target_zone, reach_task (MotionLCM / OmniControl wrist target), world (config/motion_world/<method>.json
path and sha256), input_sha256, origin (generated / linked / rehearsal) and run; PRIMAL: one row per switch combination
(factorial/<switch>_canonical.npz). CHOIS rows of the windows not in config/evalkit_floor_reference_20261107.json carry
reference (landmark offsets above the floor) and floor_m from config/evalkit_floor_reference_chois_added_20261109.json.
Kimodo rows have no task_id: each case is its own lumbar stratum (protocol lumbar_subsample); cluster = source motion.
Rows whose generation did not succeed are listed under missing, never replaced.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = json.loads((ROOT/'config/inference_tasks_20261109.json').read_text())
DATA = Path(json.loads((ROOT/'config/baselines.json').read_text())['data_root'])
KIMODO_AUDIT = ROOT/'develop_logs/test_runs/20261024_kimodo_pair_audit/audit.json'
KIMODO_ROOT = DATA/'benchmarks/kimodo-stratified-20261008'
FLOOR_FROZEN = json.loads((ROOT/'config/evalkit_floor_reference_20261107.json').read_text())['methods']['chois']['windows']
FLOOR_ADDED_PATH = ROOT/'config/evalkit_floor_reference_chois_added_20261109.json'
PRIMAL_SWITCH = 'ground{ground_snap}_reproject{reprojection}_inertialize{inertialization}'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def world(method):
    p = ROOT/f'config/motion_world/{method}.json'
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p)}


def successful_runs(batch):
    out = {}
    for state in sorted(Path(batch).glob('queue*/state.json')):
        for job, rec in json.loads(state.read_text())['jobs'].items():
            for a in rec.get('attempts', []):
                if a.get('status') == 'success' and a.get('run_dir'):
                    out[job] = Path(a['run_dir'])
    return out


def last_status(batch):
    out = {}
    for state in sorted(Path(batch).glob('queue*/state.json')):
        for job, rec in json.loads(state.read_text())['jobs'].items():
            out[job] = rec.get('status')
    return out


def rows_for(row, run, origin, added_floor):
    m = row['method']
    spec = TASKS['methods'][m]
    task = next(t for t in spec['tasks'] if t['id'] == row['task'])
    base = {'method': m, 'split': row['split'], 'task_id': row['task'], 'seed': row['seed'], 'paraphrase_id': row['paraphrase'],
            'text_condition': row['text_level'], 'world': world(m), 'origin': origin, 'run': str(run),
            'inference_row': [row['method'], row['task'], row['seed'], row['condition'], row['text_level']]}
    if m in ('motionlcm', 'omnicontrol'):
        base['target_zone'] = task['zone'] if task['tier'] == 'risk' else 'ppt_original'
        base['reach_task'] = {'position_m': task['position_m'], 'time_seconds': task['time_seconds'], 'joint': task['joint']}
    else:
        base['target_zone'] = {'primal': 'initial_state', 'chois': 'official_window'}.get(m, 'path')
    text = f"_{row['text_level']}_p{row['paraphrase']}" if row['text_level'] else ''
    if m == 'primal':
        out = []
        for cond in spec['conditions']:
            src = run/'factorial'/(PRIMAL_SWITCH.format(**cond) + '_canonical.npz')
            name = '_'.join(f'{k}{v}' for k, v in cond.items())
            out.append({'id': f"primal_{name}_t{row['task']}_s{row['seed']}", 'source': str(src), 'condition': cond,
                        'input_sha256': sha(src), **base})
        return out
    cond = spec['conditions'][row['condition']]
    name = '_'.join(f'{k}{v}' for k, v in cond.items())
    r = {'id': f"{m}_{name}{text}_t{row['task']}_s{row['seed']}", 'source': str(run), 'condition': cond,
         'input_sha256': sha(run/'motion.npz'), **base}
    if m == 'chois' and str(task['native_window_index']) not in FLOOR_FROZEN:
        rec = added_floor['windows'][str(task['native_window_index'])]
        r['reference'] = {s: {k: rec['reference_m'][s][k] - rec['world_floor_m'] for k in ('heel', 'toe')} for s in 'rl'}
        r['floor_m'] = rec['world_floor_m']
        r['floor_reference_source'] = str(FLOOR_ADDED_PATH.relative_to(ROOT))
    return [r]


def kimodo_rows():
    audit = json.loads(KIMODO_AUDIT.read_text())
    out = []
    for a in audit['rows']:
        for c, sub in ((0, 'generated_raw'), (1, 'generated_postprocess')):
            src = KIMODO_ROOT/sub/a['case']
            out.append({'id': f"kimodo_correction{c}_{a['case'].replace('/', '-')}", 'method': 'kimodo', 'source': str(src),
                        'condition': {'correction': c}, 'split': 'deck_recheck', 'seed': a['seed'], 'cluster': a['cluster'],
                        'case': a['case'], 'text_condition': None, 'target_zone': 'none', 'world': world('kimodo'),
                        'input_sha256': sha(src/'motion.npz'), 'origin': 'linked', 'run': str(src)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--batch', required=True, type=Path)
    p.add_argument('--kimodo', default='all', help="'all' (416 pairs) or a number of pairs (rehearsal)")
    a = p.parse_args()
    plans = Path(a.batch)/'plans'
    jobs = [j for f in sorted(plans.glob('plan_*.json')) for j in json.loads(f.read_text())['jobs']]
    links = json.loads((plans/'links.json').read_text())['links']
    done, status = successful_runs(a.batch), last_status(a.batch)
    added_floor = json.loads(FLOOR_ADDED_PATH.read_text())
    rows, missing = [], []
    for j in jobs:
        if j['id'] in done:
            rows += rows_for(j['inference_row'], done[j['id']], 'generated', added_floor)
        else:
            missing.append({'job': j['id'], 'inference_row': j['inference_row'], 'last_status': status.get(j['id'])})
    for x in links:
        rows += rows_for(x['row'], ROOT/x['run'], 'rehearsal' if x.get('origin') == 'rehearsal' else 'linked', added_floor)
    kim = kimodo_rows()
    if a.kimodo != 'all':
        cases = sorted({r['case'] for r in kim})[:int(a.kimodo)]
        kim = [r for r in kim if r['case'] in cases]
    rows += kim
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit('duplicate evalkit row ids')
    out = Path(a.batch)/'evalkit'
    out.mkdir(parents=True, exist_ok=True)
    meta = {'created': time.strftime('%Y-%m-%d %H:%M'), 'builder': 'python/scripts/build_evalkit_manifest.py', 'batch': str(a.batch),
            'missing': missing}
    for name, keep in (('manifest_development.json', lambda r: r['split'] != 'held_out'),
                       ('manifest_held_out.json', lambda r: r['split'] == 'held_out')):
        sel = [r for r in rows if keep(r)]
        (out/name).write_text(json.dumps({**meta, 'rows': sel}, indent=1, ensure_ascii=False) + '\n')
    counts = {}
    for r in rows:
        c = counts.setdefault(r['method'], {'development': 0, 'held_out': 0, 'deck_recheck': 0})
        c[r['split']] += 1
    print(json.dumps({'rows': counts, 'missing': len(missing)}))


if __name__ == '__main__':
    main()
