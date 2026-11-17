"""D-only IK initialization diagnostic; preserve geometry, targets and joint domain.

This does not change production fitting or select a deployed configuration.
Original R is deliberately absent; score the exported FK in a separate process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'utils'), str(ROOT / 'utils/motiogen')]
import opensim as osim
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.evalkit.k.timebase import native_clock
from motiogen.evalkit.k.initialization import flexion, structure, seed_definitions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'input', 'output'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--window', required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise ValueError('Preserve prior diagnostics; choose a new output directory')
    a.output.mkdir(parents=True)
    osim.Logger.setLevelString('error')
    motion = load(a.input, a.method, config=ROOT/'config/evalkit_k4_20261117/input.json')
    baseline = json.loads((a.fit/'receipt.json').read_text())
    if baseline['key']['inputs'] != motion.provenance['inputs']:
        raise ValueError('Input differs from frozen common baseline')
    p, names, fps = motion.legacy()
    source = a.fit/'model.osim'
    seeds, definition = seed_definitions(source, p, names)
    if set(seeds) != {'nominal','knee','heading','heading_knee'}:
        raise ValueError('Four-seed diagnostic unavailable: '+str(definition['unavailable']))
    conditions = {'default_replay': seeds['nominal'], 'knee_only': seeds['knee'],
                  'heading_only': seeds['heading'], 'heading_knee': seeds['heading_knee']}
    yaw = seeds['heading']['pelvis_rotation']
    scans = definition['knee_mapping']
    receipt = {'source_model_sha256': sha(source), 'input_hashes': motion.provenance['inputs'],
               'source_trc_sha256': sha(a.fit/'markers.trc'), 'code_sha256': sha(__file__),
               'scope': 'D-only initialization diagnostic. Same model, marker weights, TRC and IK tool; no R, no geometry/range/lock changes.',
               'heading_seed_deg': yaw, 'knee_mapping': scans, 'conditions': {}}
    geometry = baseline['geometry']
    weights = {n:w for n,w in biomech.WEIGHTS.items() if n in biomech.MARKERS or n in geometry['registered_axial_markers']}
    for name, values in conditions.items():
        out = a.output/name; out.mkdir()
        probe = osim.Model(str(source))
        for coordinate, value in values.items():
            probe.updCoordinateSet().get(coordinate).setDefaultValue(np.deg2rad(value))
        path = out/'model.osim'; probe.printToXML(str(path))
        if structure(source) != structure(path):
            raise ValueError('Model changed beyond coordinate default values')
        (out/'ik').mkdir()
        biomech.inverse_kinematics(path, a.fit/'markers.trc', out/'ik/ik.mot', 0., (len(p)-1)/fps, weights)
        times, columns, q = biomech.read_storage(out/'ik/ik.mot')
        times, clock = native_clock(times, motion.times)
        fk, fn = model_fk(path, columns, q, direct_couplers=True)
        dest = a.output/'views'/name/a.window/a.method; dest.mkdir(parents=True)
        np.savez_compressed(dest/'fk.npz', times=times+motion.times[0], positions=fk, names=fn)
        receipt['conditions'][name] = {'defaults_deg': values, 'model_sha256': sha(path), 'clock': clock,
            'structure_unchanged_except_defaults': True,
            'knee_coordinate_range_deg': {side: [float(q[:,columns.index('knee_angle_'+s)].min()),float(q[:,columns.index('knee_angle_'+s)].max())] for side,s in [('left','l'),('right','r')]}}
        (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(name,receipt['conditions'][name]['knee_coordinate_range_deg'],flush=True)
    if sha(source)!=receipt['source_model_sha256']:
        raise ValueError('Source model changed')


if __name__=='__main__':
    main()
