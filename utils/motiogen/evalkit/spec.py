"""Registered evalkit configuration and the per-method floor reference offsets."""
import json
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
if str(PACKAGE.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE.parent))

ROOT = PACKAGE.parents[2]
SPEC_PATH = ROOT/'config/evalkit_v11_20261107.json'
SPEC = json.loads(SPEC_PATH.read_text())
FLOOR_REFERENCES = json.loads((ROOT/SPEC['floor']['method_references']).read_text())
SIDES = ('r', 'l')
LANDMARKS = ('heel', 'toe')


def reference_offsets(variant):
    """Flat-foot landmark heights above the method's own floor {side: {landmark: m}} for a floor-reference variant
    (config/e_v2_floor_reference_20261106.json methods, or the evalkit-calibrated entries of the registration)."""
    own = SPEC['floor'].get('calibrated', {})
    if variant in own:
        return own[variant]
    if variant.startswith('chois_w'):
        rec = FLOOR_REFERENCES['methods']['chois']['windows'][variant[len('chois_w'):]]
    else:
        rec = FLOOR_REFERENCES['methods'][variant]
    return {s: {k: rec['reference_m'][s][k]-rec['world_floor_m'] for k in LANDMARKS} for s in SIDES}


def absolute_reference(offsets, floor):
    return {s: {k: floor+offsets[s][k] for k in LANDMARKS} for s in SIDES}
