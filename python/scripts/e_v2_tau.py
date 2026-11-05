#!/usr/bin/env python
"""Freeze the E v2 envelope tau into config/e_v2_protocol_20261106.json from the Moco calibration split.

tau[rule][state] = 99th percentile of residual_bw over the scored frames (0.5 s trim) of the successful calibration
references pooled in that state; states with fewer than 30 pooled frames keep the v1 tau (flagged); fewer than 6 successful
calibration references: not calibratable, nothing is frozen. Refuses to overwrite a frozen tau.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'utils/motiogen'))
import e_rules  # noqa: E402

PROTOCOL = ROOT/'config/e_v2_protocol_20261106.json'
REFS = ROOT/'config/e_v2_moco_refs_20261106.json'
MOCO = ROOT/'develop_logs/test_runs/20261106_e_v2/moco_refs'
QUANTILE, MIN_FRAMES, MIN_REFS = 99, 30, 6


def main():
    protocol = json.loads(PROTOCOL.read_text())
    if protocol['tau_envelope_bw'] is not None:
        raise SystemExit(f"tau already frozen at {protocol['tau_frozen_at']}")
    calib = [r for r in json.loads(REFS.read_text())['refs'] if r['split'] == 'calibration']
    used, missing = [], []
    for r in calib:
        status = MOCO/'status'/f"{r['trial']}.json"
        rec = json.loads(status.read_text()) if status.exists() else None
        if rec is None:
            raise SystemExit(f"calibration reference {r['trial']} has not run yet")
        (used if rec['status'] == 'ok' else missing).append((r, rec))
    if len(used) < MIN_REFS:
        print(json.dumps({'calibratable': False, 'successful': len(used), 'failed': [r['trial'] for r, _ in missing]}))
        return
    frames = [np.load(ROOT/rec['dir']/'frames_v2.npz') for _, rec in used]
    tau, source = {}, {}
    for rule in e_rules.RULES:
        tau[rule], source[rule] = {}, {}
        for state in e_rules.TAU[rule]:
            pooled = np.concatenate([f[f'{rule}_residual_bw'][f['sel'] & f[f'{rule}_state_{state}']] for f in frames])
            if len(pooled) < MIN_FRAMES:
                tau[rule][state] = e_rules.TAU[rule][state]
                source[rule][state] = {'source': 'v1 (sparse state)', 'frames': int(len(pooled))}
            else:
                tau[rule][state] = float(np.percentile(pooled, QUANTILE))
                source[rule][state] = {'source': f'p{QUANTILE} calibration', 'frames': int(len(pooled)),
                                       'v1': e_rules.TAU[rule][state]}
    protocol['tau_envelope_bw'] = tau
    protocol['tau_source'] = {'calibration_refs': [r['trial'] for r, _ in used],
                              'failed_calibration_refs': [r['trial'] for r, _ in missing], 'states': source}
    protocol['tau_frozen_at'] = time.strftime('%Y-%m-%d %H:%M')
    PROTOCOL.write_text(json.dumps(protocol, indent=1, ensure_ascii=False))
    print(json.dumps({'tau': tau, 'source': source}, indent=1))


if __name__ == '__main__':
    main()
