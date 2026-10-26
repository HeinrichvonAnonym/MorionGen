"""A low-memory profile may only run with an already-proven exact cache hit."""
import importlib.metadata
import json
from pathlib import Path
from .config import ROOT
from .runtime import checksum

def check(baseline,prompts):
    proof=json.loads((ROOT/'develop_logs/test_runs/20261024_ardy_cache_check/equivalence.json').read_text())
    if proof['status']!='passed' or baseline!='ardy':raise ValueError('No exact cache equivalence proof for this profile')
    reference=proof['cache'][-1];path=Path(reference['receipt']);receipt=json.loads(path.read_text())
    identity=receipt['identity']
    if identity['baseline']!=baseline or identity['prompts']!=sorted(set(prompts)):raise ValueError('Prompts differ from the reviewed cached workload')
    for name,expected in identity['packages'].items():
        if importlib.metadata.version(name)!=expected:raise ValueError('Cached encoder environment changed')
    for name,expected in identity['files'].items():
        stat=Path(name).stat();actual={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns,'inode':stat.st_ino}
        if actual!=expected:raise ValueError('Cached encoder source/weight changed: '+name)
    tensor=path.parent/'embeddings.pt'
    if checksum(tensor)!=receipt['tensor_sha256'] or receipt['tensor_sha256']!=reference['tensor_sha256']:raise ValueError('Cached tensor hash changed')
    return {'identity':reference['identity'],'tensor_sha256':reference['tensor_sha256'],'equivalence_proof':str(ROOT/'develop_logs/test_runs/20261024_ardy_cache_check/equivalence.json')}

def install_guard(cfg,baseline,scene,out):
    from . import local_llm
    from .prompt_cache import install
    prompts=[scene['prompt'],*[r['prompt'] for r in scene.get('updates',[])]]
    cfg['_validated_prompt_cache']=check(baseline,prompts)
    def forbidden_miss(*a,**kw):
        raise RuntimeError('Required-hit low-memory profile cannot instantiate a text model on cache miss')
    # install captures this fallback; even deletion between check and cache lock
    # cannot accidentally allocate the full LLM under the smaller memory bound.
    local_llm.precompute_encoder=forbidden_miss
    install(cfg,baseline,out)
    cached_encoder=local_llm.precompute_encoder
    def required_hit(config,name,texts,cache_path=None):
        check(name,texts)
        return cached_encoder(config,name,texts,cache_path)
    local_llm.precompute_encoder=required_hit
