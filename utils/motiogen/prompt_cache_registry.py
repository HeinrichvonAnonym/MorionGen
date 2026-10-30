"""Required-hit exact prompt cache for registered prompt sets beyond the reviewed pair.

prompt_cache_review only admits the prompt pair of its equivalence proof. The
proof concerns the encoder implementation (miss and hit give the same motion),
not prompt content, so other prompt sets may use the low-memory profile once
their embeddings exist as exact cache entries. Entries are created by
python/scripts/populate_prompt_cache.py with the unmodified encoder, one batch-1
call per prompt, exactly as a cache miss would. A miss at run time raises.
"""
import json
from .config import ROOT

PROOF = ROOT/'develop_logs/test_runs/20261024_ardy_cache_check/equivalence.json'


def install_guard(cfg, baseline, scene, out):
    from . import local_llm
    from .prompt_cache import install
    if baseline != 'ardy':
        raise ValueError('Registered prompt caches are reviewed for ARDY only')
    proof = json.loads(PROOF.read_text())
    if proof['status'] != 'passed':
        raise ValueError('Encoder cache equivalence proof is not passed')
    prompts = [scene['prompt'], *[r['prompt'] for r in scene.get('updates', [])]]

    def forbidden_miss(*a, **kw):
        raise RuntimeError('Registered prompt set has no exact cache entry; run populate_prompt_cache.py first')
    local_llm.precompute_encoder = forbidden_miss
    install(cfg, baseline, out)
    cfg['_validated_prompt_cache'] = {'profile': 'registered_prompt_set', 'prompts': sorted(set(prompts)),
                                      'equivalence_proof': str(PROOF)}
