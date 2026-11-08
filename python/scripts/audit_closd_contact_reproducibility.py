#!/usr/bin/env python
"""CLoSD contact-force reproducibility against the earlier analyses that used per-frame forces.

Groups of runs with identical input (same entry point, task, seed, condition, clock) are compared pairwise:
  - joints equal bit for bit (the motion itself),
  - per-frame vertical total force differences,
  - E4 contact validation quantities computed between the reruns themselves (run_e4_cross_method.contact_validation:
    Pearson r of the per-frame vertical total, 0.5 s trimmed; analyze_e4.lowpass_r: 6 Hz Butterworth, filtfilt),
    i.e. the ceiling a perfect force estimate could reach against a single simulator run,
  - the native matrix force metrics (native_analysis.physics_contact_sensitivity, 5 N), whose execution-context
    contrasts are compared with the rerun differences.
No new generation; reads native_physics.npz of finished runs. Output: develop_logs/test_runs/20261109_closd_contact_audit/audit.json
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.native_analysis import physics_contact_sensitivity  # noqa: E402

RUNS = ROOT/'develop_logs/test_runs'
OUT = RUNS/'20261109_closd_contact_audit'
FPS, TRIM, FC = 30., .5, 6.
CONTRACT = json.loads((ROOT/'config/native_metric_contract.json').read_text())
METRICS = {'native_force_foot_contact_fraction_5n': 'contact_fraction',
           'native_force_contact_foot_body_motion_fraction_5n_10cms': 'foot_body_translation_over_10cms_given_contact',
           'native_nonfoot_contact_frame_fraction_5n': 'nonfoot_contact_frame_fraction'}


def load(run):
    with np.load(RUNS/run/'native_physics.npz') as d:
        return {k: d[k] for k in d.files}


def vertical(d):
    return d['contact_forces'][..., 2].sum(1).astype(float)


def r_pair(a, b, low):
    t = np.arange(len(a))/FPS
    keep = (t >= TRIM) & (t <= t[-1]-TRIM)
    if low:
        bb, aa = butter(4, FC*2/FPS)
        a, b = filtfilt(bb, aa, a), filtfilt(bb, aa, b)
    return float(np.corrcoef(a[keep], b[keep])[0, 1])


def metrics(d):
    rows = physics_contact_sensitivity(d['joints'], d['contact_forces'], d['body_names'], FPS, CONTRACT)['rows']
    row = next(r for r in rows if r['force_threshold_n'] == 5.)
    return {k: row[v] for k, v in METRICS.items()}


def groups():
    out = {'runtime_entry task_a seed 1 paused': ['20261109_closd_rotation_smoke/20261109_010349_closd_runtime_reveal_smoke',
                                                  '20261109_closd_rotation_smoke/20261109_010946_closd_runtime_reveal_smoke',
                                                  '20261029_012927_closd_runtime_reveal_efc357a3']}
    archive = json.loads((RUNS/'20261024_closd_matrix/queue/state.json').read_text())['jobs']
    for queue in ('before_queue', 'after_queue'):
        state = RUNS/'20261109_closd_native_rotation'/queue/'state.json'
        if not state.exists():
            continue
        for job, s in json.loads(state.read_text())['jobs'].items():
            if s.get('status') != 'success':
                continue
            source = job.split('_', 1)[1]
            g = out.setdefault(f'native_entry {source}', [str(Path(archive[source]['run_dir']).relative_to(RUNS))])
            g.append(str(Path(s['run_dir']).relative_to(RUNS)))
    return out


def main():
    result = {'created': time.strftime('%Y-%m-%d %H:%M'), 'groups': {}}
    for name, runs in groups().items():
        data = {r: load(r) for r in runs}
        pairs = []
        for a, b in itertools.combinations(runs, 2):
            fa, fb = vertical(data[a]), vertical(data[b])
            diff = np.abs(fa-fb)
            win = int(FPS*.5)
            wdiff = np.abs(np.convolve(fa-fb, np.ones(win)/win, mode='valid'))
            ma, mb = metrics(data[a]), metrics(data[b])
            pairs.append({'runs': [a, b], 'joints_bitwise_equal': bool(np.array_equal(data[a]['joints'], data[b]['joints'])),
                          'vertical_force_abs_diff_n': {'median': float(np.median(diff)), 'p95': float(np.percentile(diff, 95)), 'max': float(diff.max()),
                                                        'frames_differing_fraction': float(np.mean(diff > 0)), 'window_0p5s_max': float(wdiff.max()),
                                                        'run_means': [float(fa.mean()), float(fb.mean())]},
                          'rerun_pearson_r_vertical': r_pair(fa, fb, False), 'rerun_pearson_r_vertical_lowpass_6hz': r_pair(fa, fb, True),
                          'force_metrics_5n': {k: [ma[k], mb[k]] for k in METRICS}, 'force_metric_abs_diff_5n': {k: abs(ma[k]-mb[k]) for k in METRICS}})
        result['groups'][name] = {'runs': runs, 'pairs': pairs}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(result, indent=1))
    for name, g in result['groups'].items():
        for p in g['pairs']:
            print(name, [Path(r).name[:22] for r in p['runs']], 'joints_eq', p['joints_bitwise_equal'],
                  'r', round(p['rerun_pearson_r_vertical'], 3), 'r6hz', round(p['rerun_pearson_r_vertical_lowpass_6hz'], 3),
                  'maxdiff', round(p['vertical_force_abs_diff_n']['max']), 'frac_diff', round(p['vertical_force_abs_diff_n']['frames_differing_fraction'], 3),
                  {k[13:30]: round(v, 4) for k, v in p['force_metric_abs_diff_5n'].items()})


if __name__ == '__main__':
    main()
