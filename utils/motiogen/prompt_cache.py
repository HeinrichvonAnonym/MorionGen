"""Identity-bound exact text embeddings for repeated fixed-prompt experiments.

Not a language-model replacement: misses execute the existing encoder. Cache
creation hashes all source artifacts; hits require unchanged inode/ctime/mtime/
size and encoder implementation/environment. Cold misses remain reported.
"""
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .config import ROOT,asset_root,repo_root
from .runtime import checksum,json_write

def install(cfg,baseline,out):
    import torch
    from . import local_llm
    original=local_llm.precompute_encoder
    root=asset_root(cfg)/'text/llm2vec'
    policy=json.loads((root/'local_base_receipt.json').read_text())
    paths={Path(r['path']) for r in policy['files']}
    for directory in [root/'original_base',root/policy['mntp_repo'],root/policy['supervised_repo']]:
        paths.update(p for p in directory.rglob('*') if p.is_file() and p.suffix in ['.json','.safetensors','.bin','.model'])
    paths.update([Path(local_llm.__file__),Path(__file__)])
    paths.update((repo_root(cfg,baseline)/baseline/'model/llm2vec').rglob('*.py'))
    def encode(config,name,prompts,cache_path=None):
        if name!=baseline or cache_path is not None:raise ValueError('Unexpected encoder/cache scope')
        files={str(p.resolve()):{'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
                               'ctime_ns':p.stat().st_ctime_ns,'inode':p.stat().st_ino} for p in sorted(paths)}
        identity={'baseline':baseline,'prompts':sorted(set(prompts)),'files':files,
                  'packages':{k:importlib.metadata.version(k) for k in ['torch','transformers','peft']},
                  'implementation':checksum(Path(local_llm.__file__)),'dtype':'bfloat16','schema':2}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        directory=root/'exact_prompt_cache_v2'/key;directory.mkdir(parents=True,exist_ok=True)
        tensor_path=directory/'embeddings.pt';receipt_path=directory/'receipt.json'
        with (directory/'cache.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            hit=tensor_path.exists() and receipt_path.exists()
            if hit:
                receipt=json.loads(receipt_path.read_text())
                if receipt['identity']!=identity or checksum(tensor_path)!=receipt['tensor_sha256']:raise ValueError('Prompt cache identity/content mismatch')
                encoded=torch.load(tensor_path,map_location='cpu',weights_only=True)
            else:
                sources={str(p.resolve()):checksum(p) for p in sorted(paths)}
                encoder=original(config,name,prompts)
                encoded=encoder.encoded
                temporary=directory/'embeddings.pt.tmp';torch.save(encoded,temporary);temporary.replace(tensor_path)
                receipt={'identity':identity,'source_sha256':sources,'tensor_sha256':checksum(tensor_path)}
                json_write(receipt_path,receipt)
            for prompt in prompts:
                tensor=encoded[prompt]
                if tuple(tensor.shape)!=(1,1,4096) or not torch.isfinite(tensor).all():raise ValueError('Invalid cached text embedding')
        json_write(out/'prompt_cache.json',{'hit':hit,'identity':key,'receipt':str(receipt_path),
                   'tensor_sha256':receipt['tensor_sha256'],'scope':'Exact fixed-prompt embedding reuse; cache-miss text loading is separate from warm-cache model timing'})
        print(json.dumps({'stage':'identity_bound_text_cache','hit':hit,'identity':key}),flush=True)
        return local_llm.FrozenTextEncoder(encoded,'cuda')
    local_llm.precompute_encoder=encode
