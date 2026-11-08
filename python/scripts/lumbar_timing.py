#!/usr/bin/env python
"""Per-row cost of the evalkit upper (lumbar) dimension on an idle CPU (design notes section 14 item 6).

Each row is copied without its upper outputs and run.add_upper is called in this single process, one row at a time,
on the existing fit and contact masks (no refit), as the batch's add-upper path does. Wall and CPU seconds, frames and
the load average are recorded; the lumbar metrics are compared with the row's stored result.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
ROWS = ROOT/'develop_logs/test_runs/20261107_evalkit/v11/smoke8x5/rows'
DEFAULT = ['motionlcm_control_residual0_t0_s20', 'omnicontrol_realism0_spatial0_t4_s21', 'ardy_correction0_history_long0_t2_s21']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', nargs='*', default=DEFAULT)
    ap.add_argument('--out', default=str(ROOT/'develop_logs/test_runs/20261109_lumbar_timing'))
    a = ap.parse_args()
    from evalkit import run
    import numpy as np
    out_root = Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    results = []
    for rid in a.rows:
        src = ROWS/rid; dst = out_root/rid
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns('upper', 'upper_e3'))
        rec = json.loads((dst/'evalkit.json').read_text()); rec.pop('upper', None)
        (dst/'evalkit.json').write_text(json.dumps(rec, indent=1))
        stored = json.loads((src/'evalkit.json').read_text())['upper']['lumbar']
        times = np.load(dst/'frames.npz', allow_pickle=True)['times']; frames = len(times)
        load0 = os.getloadavg(); w0 = time.perf_counter(); c0 = time.process_time()
        run.add_upper(rec['receipt']['run'], dst)
        wall = time.perf_counter()-w0; cpu = time.process_time()-c0; load1 = os.getloadavg()
        new = json.loads((dst/'evalkit.json').read_text())['upper']['lumbar']
        same = {k: isinstance(new['metrics'].get(k), (int, float)) and bool(np.isclose(new['metrics'][k], v, rtol=1e-6, equal_nan=True))
                for k, v in stored['metrics'].items() if isinstance(v, (int, float))}
        r = {'row': rid, 'method': rec['receipt']['method'], 'frames': frames, 'motion_seconds': round(float(times[-1]-times[0]), 2),
             'wall_s': round(wall, 1), 'cpu_s': round(cpu, 1), 'wall_s_per_frame': round(wall/frames, 3),
             'stored_seconds_contended': stored.get('seconds'), 'lumbar_status': new['status'],
             'stored_status': stored['status'], 'metrics_equal_stored': all(same.values()), 'metrics_compared': len(same),
             'loadavg_before': load0, 'loadavg_after': load1}
        results.append(r)
        print(json.dumps(r), flush=True)
        (out_root/'timing.json').write_text(json.dumps({'rows': results, 'cpu_count': os.cpu_count(),
                                                        'source_rows': str(ROWS.relative_to(ROOT))}, indent=1))


if __name__ == '__main__':
    main()
