"""Per-run inputs of a generated motion that are not in its joints: the floor-reference variant (run_mapping of the
floor reference file), the task text and the object-free judgment of the ergonomics dimension."""
import json
import re
from pathlib import Path

from .spec import FLOOR_REFERENCES

SMPL_FAMILY = ('motionlcm', 'omnicontrol')


def chois_window(source):
    """CHOIS test window of a run: scenario native_window_index, else the one window whose sequence the stdout names."""
    source = Path(source)
    scenario = json.loads((source/'scenario.json').read_text())
    if scenario.get('native_window_index') is not None:
        return str(scenario['native_window_index'])
    sequences = set(re.findall(r'sub\d+_[a-z]+_\d+', (source/'stdout.log').read_text(errors='replace')))
    windows = [w for w, rec in FLOOR_REFERENCES['methods']['chois']['windows'].items() if rec['sequence'] in sequences]
    if len(windows) != 1:
        raise ValueError(f'CHOIS window not determined from {source}: {sorted(sequences)}')
    return windows[0]


def floor_variant(method, source, condition=None):
    """Floor-reference variant key (spec.reference_offsets) and world floor (m) of one run.
    condition: the E4-style condition dict; ARDY / Kimodo read their correction flag from it, else from the run config."""
    condition = condition or {}
    if method == 'ardy':
        corr = condition['correction'] if 'correction' in condition else \
            json.loads((Path(source)/'config.json').read_text())['_intervention']['correction']
        key = 'ardy__correction' if corr else 'ardy'
    elif method == 'kimodo':
        key = 'kimodo__postprocess' if condition.get('correction') else 'kimodo'
    elif method == 'chois':
        w = chois_window(source)
        return f'chois_w{w}', FLOOR_REFERENCES['methods']['chois']['windows'][w]['world_floor_m']
    elif method in SMPL_FAMILY:
        key = 'humanml3d'
    else:
        key = method
    return key, FLOOR_REFERENCES['methods'][key]['world_floor_m']


def prompt(source):
    """Task text of a run (meta.json text, manifest scenario prompt or scenario.json prompt), None if there is none."""
    p = Path(source)
    d = p if p.is_dir() else p.parent.parent
    for name, get in (('meta.json', lambda c: c.get('text')), ('manifest.json', lambda c: (c.get('scenario') or {}).get('prompt')),
                      ('scenario.json', lambda c: c.get('prompt'))):
        f = d/name
        if f.exists():
            v = get(json.loads(f.read_text()))
            if v:
                return v
    return None


def object_free(method, source):
    """Ergonomics object-free judgment of an E4-style run: CHOIS always carries an object, Kimodo by its prompt
    (ergo.object_free), the other methods' tasks are object-free locomotion / reaching."""
    from .ergo import object_free as by_text
    if method == 'chois':
        return False
    if method == 'kimodo':
        return by_text(prompt(source))
    return True
