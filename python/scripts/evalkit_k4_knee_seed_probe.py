"""D-only IK initialization diagnostic; preserve geometry, targets and joint domain.

This does not change production fitting or select a deployed configuration.
Original R is deliberately absent; score the exported FK in a separate process.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'utils'), str(ROOT / 'utils/motiogen')]
import opensim as osim
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.k4_fitting import model_fk
from motiogen.evalkit.k.timebase import native_clock


def flexion(hip, knee, ankle):
    """Independent atan2 convention: straight=0, right angle=90 degrees."""
    a, b = knee - hip, ankle - knee
    value = np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                 np.sum(a * b, axis=-1)))
    valid = np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)>1e-12
    return np.where(valid,value,np.nan)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def structure(path):
    tree = ET.parse(path)
    for coordinate in tree.findall('.//Coordinate'):
        value = coordinate.find('default_value')
        if value is not None:
            value.text = 'INITIALIZATION_ONLY'
    return ET.tostring(tree.getroot())


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
    idx = {n: i for i, n in enumerate(names)}
    source = a.fit/'model.osim'
    model = osim.Model(str(source)); state = model.initSystem()
    cs = model.getCoordinateSet()
    labels = [cs.get(i).getName() for i in range(cs.getSize())]
    defaults = np.array([cs.get(n).getDefaultValue() * (180/np.pi if cs.get(n).getMotionType()==1 else 1) for n in labels])
    # Scan the model's own knee coordinate -> geometric knee angle mapping.
    scans = {}; knee_seed = {}
    for side, suffix in [('left', 'l'), ('right', 'r')]:
        target = flexion(*[p[0, idx[side+'_'+joint]] for joint in ('hip', 'knee', 'ankle')])
        if not np.isfinite(target):
            raise ValueError('First-frame knee geometry is missing or degenerate')
        c = cs.get('knee_angle_'+suffix)
        grid = np.linspace(c.getRangeMin(), c.getRangeMax(), 281)*180/np.pi
        q = np.tile(defaults, (len(grid), 1)); q[:, labels.index(c.getName())] = grid
        fk, fn = model_fk(source, labels, q, direct_couplers=True)
        angles = flexion(*[fk[:, fn.index(side+'_'+joint)] for joint in ('hip', 'knee', 'ankle')])
        chosen = int(np.argmin(abs(angles-target)))
        knee_seed[c.getName()] = float(grid[chosen])
        scans[side] = {'input_initial_flexion_deg': float(target), 'coordinate_deg': grid.tolist(),
                       'geometric_flexion_deg': angles.tolist(), 'seed_deg': float(grid[chosen]),
                       'seed_mapping_error_deg': float(abs(angles[chosen]-target))}
    h = p[0, idx['right_hip']] - p[0, idx['left_hip']]
    neutral = biomech._child_origin(model,state,'hip_r') - biomech._child_origin(model,state,'hip_l')
    yaw = np.arctan2(h[0],h[2]) - np.arctan2(neutral[0],neutral[2])
    yaw = float(np.rad2deg(np.arctan2(np.sin(yaw),np.cos(yaw))))
    if not np.isfinite(h).all() or np.linalg.norm(h[[0,2]]) < 1e-6:
        raise ValueError('Horizontal hip line cannot initialize heading')
    conditions = {'default_replay': {}, 'knee_only': knee_seed,
                  'heading_only': {'pelvis_rotation': yaw},
                  'heading_knee': {**knee_seed, 'pelvis_rotation': yaw}}
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
