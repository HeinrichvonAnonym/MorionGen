"""Pinned official baseline source checkouts."""
import json
import subprocess
from pathlib import Path

from .config import ROOT, asset_root, repo_root
from .runtime import checksum, json_write


class SourceError(RuntimeError):
    pass


def source_specs():
    path = ROOT / 'config/baseline_sources.json'
    payload = json.loads(path.read_text())
    if payload.get('schema_version') != 1:
        raise SourceError('Unsupported baseline source schema')
    return payload['sources']


def _git(args, cwd=None, check=True):
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    if check and result.returncode:
        raise SourceError(result.stderr.strip() or result.stdout.strip() or f'git exited {result.returncode}')
    return result


def prepare_source(cfg, baseline):
    spec = source_specs()[baseline]
    target = repo_root(cfg, baseline)
    target.parent.mkdir(parents=True, exist_ok=True)
    created = False
    if target.exists():
        if not (target / '.git').is_dir():
            raise SourceError(f'Existing baseline target is not a git checkout: {target}')
        dirty = _git(['status', '--porcelain'], target).stdout.strip()
        if dirty:
            raise SourceError(f'Refuse to change dirty baseline checkout: {target}')
        remote = _git(['remote', 'get-url', 'origin'], target).stdout.strip()
        if remote.rstrip('/') != spec['url'].rstrip('/'):
            raise SourceError(f'Baseline origin mismatch: {remote}')
    else:
        _git(['clone', '--filter=blob:none', spec['url'], str(target)])
        created = True
    current = _git(['rev-parse', 'HEAD'], target, check=False)
    if current.returncode or current.stdout.strip() != spec['commit']:
        _git(['fetch', '--depth=1', 'origin', spec['commit']], target)
        _git(['checkout', '--detach', spec['commit']], target)
    elif created:
        _git(['checkout', '--detach', spec['commit']], target)
    actual = _git(['rev-parse', 'HEAD'], target).stdout.strip()
    if actual != spec['commit']:
        raise SourceError(f'Pinned checkout mismatch for {baseline}: {actual}')
    receipt = {
        'baseline': baseline,
        'url': spec['url'],
        'commit': actual,
        'tree': _git(['rev-parse', 'HEAD^{tree}'], target).stdout.strip(),
        'clean': not bool(_git(['status', '--porcelain'], target).stdout.strip()),
        'source_manifest_sha256': checksum(ROOT / 'config/baseline_sources.json'),
        'path': str(target),
    }
    json_write(asset_root(cfg) / 'sources' / f'{baseline}.json', receipt)
    return receipt
