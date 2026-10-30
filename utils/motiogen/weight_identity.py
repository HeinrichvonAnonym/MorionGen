"""Stat-bound content hashes of reviewed native inference dependencies.

Retrospective receipts explicitly require filesystem change times to predate
the run; they are never represented as hashes observed at historical load time.
"""
import fcntl
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
from .config import ROOT,asset_root
from .runtime import checksum,json_write

def signature(path):
    stat=path.stat()
    return {'device':stat.st_dev,'inode':stat.st_ino,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns}

def dependencies(cfg,method):
    registry=json.loads((ROOT/'config/native_weight_bindings.json').read_text());base=asset_root(cfg);paths=[]
    patterns = cfg['baselines'].get(method, {}).get('weight_patterns')
    if patterns is None: patterns = registry[method]
    for pattern in patterns:
        found=sorted(base.glob(pattern))
        if not found:raise FileNotFoundError('Registered inference dependency absent: '+pattern)
        paths.extend(found)
    if method in ('ardy','kimodo'):
        text=base/'text/llm2vec';policy=json.loads((text/'local_base_receipt.json').read_text())
        paths.extend(Path(row['path']) for row in policy['files'])
        for name in [policy['mntp_repo'],policy['supervised_repo']]:
            found=list((text/name).rglob('*.safetensors'))
            if not found:raise FileNotFoundError('LLM adapter absent: '+name)
            paths.extend(found)
    return sorted(set(path.resolve(strict=True) for path in paths))

def snapshot(cfg,method):
    cache=asset_root(cfg)/'native_identity_cache';cache.mkdir(exist_ok=True)
    rows=[]
    for path in dependencies(cfg,method):
        key=hashlib.sha256(str(path).encode()).hexdigest();receipt=cache/(key+'.json')
        with (cache/(key+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            before=signature(path)
            old=json.loads(receipt.read_text()) if receipt.exists() else None
            if old is not None and old['stat']==before:row=old
            else:
                digest=checksum(path)
                if signature(path)!=before:raise RuntimeError('Dependency changed during hashing: '+str(path))
                row={'path':str(path),'sha256':digest,'stat':before,'hash_observed_at':datetime.now(timezone.utc).isoformat()}
                json_write(receipt,row)
            rows.append(row)
    identity=hashlib.sha256(json.dumps({r['path']:r['sha256'] for r in rows},sort_keys=True).encode()).hexdigest()
    return {'method':method,'identity':identity,'files':rows,'scope':'Registered native generator, decoder/body, policy/tracker and text-weight dependencies; not a dynamic file-open trace'}

def bind(cfg,method,directory):
    result=snapshot(cfg,method);result['binding']='before native model load'
    result.update(bound_at=datetime.now(timezone.utc).isoformat(),after_run_unchanged=False,
                  receipt_path=str(Path(directory)/'weight_binding.json'))
    json_write(result['receipt_path'],result)
    return result

def verify_unchanged(binding):
    for row in binding['files']:
        if signature(Path(row['path']))!=row['stat']:raise RuntimeError('Inference dependency changed during run: '+row['path'])
    binding.update(after_run_unchanged=True,checked_after_run_at=datetime.now(timezone.utc).isoformat())
    if binding.get('receipt_path'):json_write(binding['receipt_path'],binding)
