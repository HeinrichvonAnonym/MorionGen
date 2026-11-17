"""Isolated source-position codec process; the legacy K exporter is unchanged."""
import atexit
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
_worker=None


def encode(path):
    global _worker
    if _worker is None or _worker.poll() is not None:
        root=Path(json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())['local_roots']['andy']).parents[2]
        _worker=subprocess.Popen([str(root/'envs/motiogen-omnicontrol/bin/python'),
            str(ROOT/'python/scripts/degradation_humanml_codec.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
        atexit.register(_worker.terminate)
    _worker.stdin.write(json.dumps({'motion':str(Path(path).resolve())})+'\n');_worker.stdin.flush()
    for line in _worker.stdout:
        if line.startswith('K_CODEC '):
            result=json.loads(line[8:])
            if result['status']!='ok':raise ValueError(result['error'])
            return
    raise RuntimeError('Codec process exited')
