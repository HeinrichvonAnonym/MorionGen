"""Descriptive leave-one-seed-out sensitivity on a balanced task subset.

No extra sampling, confidence intervals, or claims of independent subjects.
This is a report-side companion to the registered scene-cluster estimator.
"""
import math
from collections import defaultdict
from statistics import mean


def seed_sensitivity(values, tasks, seeds, expected_seeds=(20, 21, 22, 23, 24)):
    if not len(values) == len(tasks) == len(seeds):
        raise ValueError('Each paired delta requires one task and seed')
    expected = tuple(expected_seeds)
    if len(set(expected)) != len(expected) or len(expected) < 2:
        raise ValueError('At least two distinct registered seeds required')
    grouped = defaultdict(dict)
    for value, task, seed in zip(values, tasks, seeds):
        if not math.isfinite(value) or seed not in expected:
            raise ValueError('Finite deltas and registered seeds required')
        if seed in grouped[task]:
            raise ValueError('Duplicate task/seed paired effect')
        grouped[task][seed] = float(value)
    balanced = {task: row for task, row in grouped.items() if set(row) == set(expected)}
    per_seed = [{'seed': seed, 'equal_task_mean_delta': mean(row[seed] for row in balanced.values())}
                for seed in expected] if balanced else []
    omissions = [{'omitted_seed': seed, 'equal_task_mean_delta': mean(
        mean(value for other, value in row.items() if other != seed) for row in balanced.values())}
        for seed in expected] if balanced else []
    return {'registered_seeds': list(expected), 'observed_tasks': len(grouped),
            'balanced_tasks': sorted(balanced), 'excluded_incomplete_tasks': sorted(set(grouped) - set(balanced)),
            'balanced_pairs': len(balanced) * len(expected), 'per_seed': per_seed,
            'balanced_mean_delta': mean(r['equal_task_mean_delta'] for r in per_seed) if per_seed else None,
            'leave_one_seed_out': omissions,
            'scope': 'Descriptive sensitivity on tasks with all five computable paired seeds. '
                     'Keep the same tasks for every omission. Not a confidence interval, '
                     'not extra experiments, and not a replacement for the full eligible estimator.'}


def annotate_seed_sensitivity(effects, blocks):
    indexed = defaultdict(list)
    for block in blocks:
        if not block['complete']:
            continue
        for contrast in block['contrasts']:
            key = (block.get('method', 'omnicontrol'), block.get('clock', 'paused'),
                   tuple(contrast['factors']), contrast['metric'])
            indexed[key].append((contrast['delta'], block['task'], block['seed']))
    for effect in effects:
        key = (effect['method'], effect['clock'], tuple(effect['factors']), effect['metric'])
        rows = indexed[key]
        effect['seed_sensitivity'] = seed_sensitivity(
            [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows])

