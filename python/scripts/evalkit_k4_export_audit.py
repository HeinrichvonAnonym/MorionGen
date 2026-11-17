"""Separate coordinate-file roundtrip from interpolation and native fitting.

Each replay's in-memory coordinates are saved as OpenSim MOT, read back, and
evaluated through the same model FK. This audit never reads R or changes a fit.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'utils'), str(ROOT/'utils/motiogen')]
import numpy as np
import opensim as osim
import biomech
from motiogen.evalkit.k.k4_fitting import model_fk, write_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    osim.Logger.setLevelString('error')
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = json.loads((args.replay/'receipt.json').read_text())
    model = Path(receipt['source_fit'])/'model.osim'
    rows = []
    for path in sorted((args.replay/'views').glob('*/*/*/coordinates.npz')):
        condition = path.relative_to(args.replay/'views').parts[0]
        with np.load(path) as z:
            time, labels, q = z['times'], list(z['labels']), z['coords']
        target = args.output/(condition+'.mot')
        write_coordinates(target, time, labels, q)
        read_time, read_labels, read_q = biomech.read_storage(target)
        if read_labels != labels or read_q.shape != q.shape:
            raise ValueError('MOT roundtrip changed labels or shape')
        before, names = model_fk(model, labels, q, direct_couplers=True)
        after, _ = model_fk(model, labels, read_q, direct_couplers=True)
        with np.load(path.parent/'fk.npz') as z:
            stored = z['positions'][:, [list(z['names']).index(n) for n in names]]
        coordinate_error = float(abs(read_q-q).max())
        time_error = float(abs(read_time-time).max())
        fk_error = float(abs(after-before).max())
        saved_error = float(abs(before-stored).max())
        row = dict(condition=condition, frames=len(time), time_max_abs_s=time_error,
                   coordinate_max_abs_native_units=coordinate_error,
                   roundtrip_fk_max_abs_m=fk_error, saved_body_fk_max_abs_m=saved_error,
                   coordinate_file_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        row['passed'] = time_error < 1e-6 and coordinate_error < 1e-6 and fk_error < 1e-7 and saved_error < 1e-7
        rows.append(row)
    result = dict(source_replay=str(args.replay), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  rows=rows, passed=bool(rows) and all(r['passed'] for r in rows),
                  scope='MOT numerical roundtrip at identical timestamps; separate from sampling/readback interpolation and native solver error. Body FK only; head chain already stored separately.',
                  implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
