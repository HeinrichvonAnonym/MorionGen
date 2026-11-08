#!/usr/bin/env python
"""Input premise check of the section-14 inference registration: MotionLCM / OmniControl control hints.

Both adapters feed hint = (native_positions(target) - Mean_raw[joint]) / Std_raw[joint] without clipping
(utils/motiogen/adapters.py). This recomputes that value for every reach task of config/inference_tasks_20261109.json
with each method's own Mean_raw / Std_raw and reports |z| per axis, the legacy tasks (already run) as the reference.
Reads no generated motion. Output: develop_logs/test_runs/20261109_inference_premises/hint_zscores.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils'))
from motiogen.adapters import require  # noqa: E402
from motiogen.config import load_config  # noqa: E402
from motiogen.motion import HUMANML_NAMES  # noqa: E402
from motiogen.motion import native_positions  # noqa: E402

OUT = ROOT/'develop_logs/test_runs/20261109_inference_premises'


def main():
    cfg = load_config()
    tasks = json.loads((ROOT/'config/inference_tasks_20261109.json').read_text())
    out = {'source': 'config/inference_tasks_20261109.json', 'formula': 'utils/motiogen/adapters.py: (native_positions(p, Y-up) - Mean_raw[j]) / Std_raw[j], no clipping', 'methods': {}}
    for method in ('motionlcm', 'omnicontrol'):
        mean = np.load(require(cfg, method, 'Mean_raw.npy')).reshape(22, 3)
        std = np.load(require(cfg, method, 'Std_raw.npy')).reshape(22, 3)
        rows = []
        for t in tasks['methods'][method]['tasks']:
            side = t['joint'].split('_')[0]
            j = HUMANML_NAMES.index(f'{side}_wrist')
            z = (native_positions(t['position_m'], 'Y-up')-mean[j])/std[j]
            rows.append({'task': t['id'], 'tier': t['tier'], 'zone': t.get('zone'), 'direction': t.get('direction'), 'side': side,
                         'native_xyz_m': [round(float(v), 4) for v in native_positions(t['position_m'], 'Y-up')],
                         'z': [round(float(v), 2) for v in z], 'max_abs_z': round(float(np.abs(z).max()), 2)})
        legacy = max(r['max_abs_z'] for r in rows if r['tier'] == 'legacy')
        out['methods'][method] = {'wrist_mean_m': {s: mean[HUMANML_NAMES.index(f'{s}_wrist')].round(4).tolist() for s in ('left', 'right')},
                                  'wrist_std_m': {s: std[HUMANML_NAMES.index(f'{s}_wrist')].round(4).tolist() for s in ('left', 'right')},
                                  'legacy_max_abs_z': legacy, 'risk_max_abs_z': max(r['max_abs_z'] for r in rows if r['tier'] == 'risk'),
                                  'tasks': rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'hint_zscores.json').write_text(json.dumps(out, indent=1))
    for m, v in out['methods'].items():
        print(m, 'mean', v['wrist_mean_m'], 'std', v['wrist_std_m'], 'legacy max|z|', v['legacy_max_abs_z'], 'risk max|z|', v['risk_max_abs_z'])
        for r in v['tasks']:
            print(' ', r['task'], r['tier'], r['side'], r['zone'], r['direction'], r['native_xyz_m'], r['z'])


if __name__ == '__main__':
    main()
