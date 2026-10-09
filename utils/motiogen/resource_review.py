"""Reviewed bounds are distinct from provisional cost estimates and watchdog telemetry."""
import json
from pathlib import Path

from .config import ROOT, repo_root
from .runtime import budgets,checksum,json_write


def inference_review(adapter):
    from .adapters import ResourceGate
    path=Path(adapter.cfg['_source']).parent/'inference_evidence.json'
    if not path.exists():
        raise ResourceGate('No reviewed inference memory bound: provisional estimates cannot authorize allocation')
    entry=json.loads(path.read_text()).get(adapter.name)
    if entry and 'profiles' in entry:
        profile='reconstruction' if adapter.scene.get('kind')=='reconstruction' else 'full'
        entry=entry['profiles'].get(profile)
    if not entry or not entry.get('reviewed') or not entry.get('source_basis'):
        raise ResourceGate('Inference loading/runtime peak remains unverified for this configuration')
    from .assets import find_asset
    for file in entry.get('files',[]):
        if 'baseline_asset' in file:
            resolved=find_asset(adapter.cfg,adapter.name,file['baseline_asset'])
        elif 'repo_file' in file:
            resolved=repo_root(adapter.cfg,adapter.name)/file['repo_file']
        elif 'framework_file' in file:
            resolved=ROOT/file['framework_file']
        else:
            resolved=Path(file['path'])
        if resolved is None or not Path(resolved).is_file() or checksum(resolved)!=file['sha256']:
            raise ResourceGate('Inference evidence asset/source fingerprint is stale')
    current=budgets(adapter.cfg)
    if entry['ram_bound_bytes']>current['ram_bytes'] or entry['gpu_bound_bytes']>current['gpu_bytes']:
        raise ResourceGate('Reviewed inference peak exceeds current RAM/VRAM budget')
    if adapter.scene['duration_seconds']*adapter.spec['fps']>entry['max_frames'] and adapter.scene['kind']!='online':
        raise ResourceGate('Scene exceeds reviewed frame bound')
    json_write(adapter.path/'resource_review.json',{'evidence':entry,'current_budget':current})
    return entry
