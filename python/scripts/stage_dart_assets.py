"""Stage official DART inference assets on the configured external data volume."""
import json
import argparse
import pathlib
import shutil
import sys
import time
import subprocess
from types import SimpleNamespace

import gdown

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = pathlib.Path(json.loads((ROOT / 'config/baselines.json').read_text())['data_root'])
DEST = DATA / 'outputs/assets/dart-official'
LOG = ROOT / 'develop_logs/test_runs/20261024_dart_assets.json'


def main():
    global DEST, LOG
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['dart', 'primal'], default='dart')
    args = parser.parse_args()
    folder = {'dart': '1vJg3GFVPT6kr6cA0HrQGmiAEBE2dkaps',
              'primal': '1it78M-gojouMZNlmmHm5PbjJdt7bvbVb'}[args.source]
    DEST = DATA / f'outputs/assets/{args.source}-official'
    LOG = ROOT / f'develop_logs/test_runs/20261024_{args.source}_assets.json'
    reserve = 80 * 1024**3
    if shutil.disk_usage(DATA).free < reserve + 30 * 1024**3:
        raise RuntimeError('Insufficient space: reserve 80 GiB plus provisional 30 GiB staging allowance')
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = DEST / 'download_manifest.json'
    if manifest.exists():
        files = [SimpleNamespace(**x) for x in json.loads(manifest.read_text())]
    else:
        files = gdown.download_folder(
            url=f'https://drive.google.com/drive/folders/{folder}',
            output=str(DEST), skip_download=True, quiet=True,
        )
        if files:
            manifest.write_text(json.dumps([{'id': x.id, 'path': x.path} for x in files], indent=2))
    if not files:
        raise RuntimeError('Official Google Drive listing unavailable; check access/authentication')
    receipt = {'source': f'official {args.source} README Google Drive', 'started': time.time(),
               'destination': str(DEST), 'status': 'running', 'files': []}
    LOG.parent.mkdir(parents=True, exist_ok=True)
    for item in files:
        relative = pathlib.PurePosixPath(item.path)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe remote path')
        # Scene meshes are unnecessary for the first text/history and reach smoke.
        if str(relative).startswith('data/scenes/'):
            continue
        if relative.name == 'args_read.yaml':
            continue  # Human-readable duplicate; runtime loads args.yaml.
        if args.source == 'dart' and str(relative).startswith(('mld_denoiser/', 'mvae/')):
            if not str(relative).startswith(('mld_denoiser/mld_fps_clip_repeat_euler/', 'mvae/mvae_fps_clip/')):
                continue
        if shutil.disk_usage(DATA).free < reserve:
            raise RuntimeError('Disk reserve reached')
        target = DEST / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size:
            receipt['files'].append({'path': str(relative), 'bytes': target.stat().st_size,
                                     'source': 'existing_download_requires_content_check'})
            LOG.write_text(json.dumps(receipt, indent=2))
            continue
        print(f'Downloading {relative}', flush=True)
        result = None
        if target.suffix in ('.pt', '.pth', '.ckpt', '.pkl', '.npz'):
            temporary = target.with_suffix(target.suffix + '.part')
            completed = subprocess.run(['curl', '-fL', '--retry', '2', '--max-time', '600', '-sS',
                f'https://drive.google.com/uc?export=download&id={item.id}', '-o', str(temporary)])
            if completed.returncode == 0:
                with temporary.open('rb') as stream:
                    prefix = stream.read(4)
                if prefix.startswith((b'PK\x03\x04', b'\x80')):
                    temporary.replace(target)
                    result = str(target)
        if target.suffix == '.yaml':
            temporary = target.with_suffix('.yaml.part')
            completed = subprocess.run(['curl', '-fL', '--retry', '2', '--max-time', '60', '-sS',
                f'https://drive.google.com/uc?export=download&id={item.id}', '-o', str(temporary)])
            if completed.returncode == 0:
                import yaml
                try:
                    document = yaml.safe_load(temporary.read_text())
                    if isinstance(document, dict) or (isinstance(document, str) and document.startswith('# tyro YAML.')):
                        temporary.replace(target)
                        result = str(target)
                except (ValueError, yaml.YAMLError):
                    pass
        for attempt in range(3):
            if result:
                break
            try:
                options = {'id': item.id} if attempt == 0 else {
                    'url': f'https://drive.usercontent.google.com/download?id={item.id}&export=download&confirm=t'}
                result = gdown.download(**options, output=str(target), quiet=True,
                                        resume=True, timeout=30)
                if result:
                    break
            except Exception as exc:
                receipt['last_error'] = {'path': str(relative), 'type': type(exc).__name__,
                                         'attempt': attempt + 1}
                LOG.write_text(json.dumps(receipt, indent=2))
                print(f'Retry {attempt + 1}: {type(exc).__name__}', flush=True)
        if not result:
            receipt['status'] = 'download_failed'
            LOG.write_text(json.dumps(receipt, indent=2))
            raise RuntimeError(f'Download failed: {relative}; check access/authentication')
        receipt['files'].append({'path': str(relative), 'bytes': target.stat().st_size})
        LOG.write_text(json.dumps(receipt, indent=2))
    receipt['status'] = 'downloaded_not_validated'
    receipt['finished'] = time.time()
    LOG.write_text(json.dumps(receipt, indent=2))
    print(f'Receipt: {LOG}', flush=True)


if __name__ == '__main__':
    main()
